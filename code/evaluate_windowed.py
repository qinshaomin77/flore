# -*- coding: utf-8 -*-
"""
evaluate_windowed.py
====================

Unified windowed evaluation entry point for:

* deterministic MGMQ-DDQN (``--controller rl``);
* SUMO native actuated control (``--controller actuated``);
* E2 movement Max-Pressure (``--controller max_pressure``);
* fixed truck-weighted Max-Pressure
  (``--controller truck_weighted_max_pressure``).

Examples
--------

RL traffic-only::

    python code/evaluate_windowed.py \
      --controller rl \
      --case-name m0_traffic \
      --reward-mode traffic_only \
      --config configs/grid36/objective/m0_traffic.yaml \
      --checkpoint models/grid36/m0_traffic.pt \
      --eval-episodes 20 --seed 19 --workers 3

RL multi-objective::

    python code/evaluate_windowed.py \
      --controller rl \
      --case-name multi_objective \
      --reward-mode multi_objective \
      --config configs/grid36/base.yaml \
      --checkpoint models/grid36/multi_objective.pt \
      --eval-episodes 20 --seed 19 --workers 3

SUMO actuated::

    python code/evaluate_windowed.py \
      --controller actuated \
      --case-name actuated \
      --config configs/grid36/calibration/sumo_actuated_eval.yaml \
      --eval-episodes 20 --seed 19 --workers 3

Max-Pressure::

    python code/evaluate_windowed.py \
      --controller max_pressure \
      --case-name max_pressure \
      --detector-map data/grid36/maxpressure_detector_map.json \
      --eval-episodes 20 --seed 19 --workers 3

For each scenario, the script writes SUMO ``tripinfo`` XML files, detailed
``emission`` CSV files, one directory per episode for windowed metrics, and
merged ``all_episodes`` CSV files.  It does not update replay memory or model
parameters.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from agent import MGMQAgentManager, set_global_seed
from config import MasterConfig, build_sumo_seed_list, get_config
from emission_lookup import EmissionEpisodeRecorder, build_emission_lookup
from env import SumoEnv
from evaluation_window_logger import (
    WindowEvaluationConfig,
    WindowEvaluationLogger,
)
from max_pressure_controller import MaxPressureController
from max_pressure_truck_weighted import TruckWeightedMaxPressureController
from network_parser import parse_network
from vehicle_state import consolidate_vehicle_state_outputs
from obs_reward import ObsRewardBuilder


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EPS = 1.0e-9
DEFAULT_RL_CHECKPOINT = (
    PROJECT_ROOT
    / "best_checkpoints"
    / "grid36"
    / "multi_objective.pt"
)


RISK_SERVICE_LANE_FIELDS = (
    "episode",
    "sumo_seed",
    "step",
    "sim_time",
    "tl_id",
    "lane_id",
    "demand",
    "nox_tau_mg",
    "nox_pressure",
    "green_service_s",
)


class RiskServiceLaneLogger:
    """Stream the minimal lane data needed for Exceed-service diagnostics."""

    def __init__(
        self,
        *,
        output_root: Path,
        episode: int,
        sumo_seed: int,
        net_info: Any,
        start_s: float,
        end_s: float,
        decision_interval_s: float,
        compression: str,
    ) -> None:
        self.episode = int(episode)
        self.sumo_seed = int(sumo_seed)
        self.net_info = net_info
        self.start_s = float(start_s)
        self.end_s = float(end_s)
        self.decision_interval_s = float(decision_interval_s)
        self.compression = str(compression)
        self.row_count = 0
        self.min_sim_time_s = math.inf
        self.max_sim_time_s = -math.inf
        self._committed = False
        self._closed = False

        episode_dir = (
            Path(output_root).resolve()
            / "physical"
            / f"episode_{self.episode:04d}"
        )
        episode_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            "lane_step.csv.gz"
            if self.compression == "gzip"
            else "lane_step.csv"
        )
        self.path = episode_dir / filename
        if self.path.exists():
            raise FileExistsError(
                f"Risk-service lane log already exists: {self.path}"
            )
        self._temporary_path = self.path.with_name(
            f"{self.path.name}.tmp.{os.getpid()}"
        )
        if self.compression == "gzip":
            self._handle = gzip.open(
                self._temporary_path,
                mode="wt",
                encoding="utf-8",
                newline="",
                compresslevel=1,
            )
        else:
            self._handle = self._temporary_path.open(
                mode="w",
                encoding="utf-8",
                newline="",
            )
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=list(RISK_SERVICE_LANE_FIELDS),
            extrasaction="raise",
        )
        self._writer.writeheader()

    @staticmethod
    def _require_values(
        values: Any,
        *,
        field: str,
        expected: int,
        tl_id: str,
    ) -> Any:
        try:
            actual = len(values)
        except TypeError as exc:
            raise ValueError(
                f"{field} is not lane-aligned for TLS {tl_id}"
            ) from exc
        if actual != expected:
            raise ValueError(
                f"{field} lane count mismatch for TLS {tl_id}: "
                f"expected {expected}, got {actual}"
            )
        return values

    @staticmethod
    def _finite_float(value: Any, *, field: str, tl_id: str, lane_id: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(
                f"Non-finite {field} for TLS {tl_id}, lane {lane_id}: {value!r}"
            )
        return result

    def record(
        self,
        *,
        decision_step: int,
        raw_observations: Mapping[str, Any],
        observations: Mapping[str, Any],
    ) -> None:
        if self._closed:
            raise RuntimeError("Cannot record to a closed risk-service lane log")
        rows: list[dict[str, Any]] = []
        for tl_id, obs in observations.items():
            raw = raw_observations[str(tl_id)]
            sim_time = float(raw.sim_time)
            if sim_time < self.start_s - EPS or sim_time > self.end_s + EPS:
                continue

            lanes = tuple(
                self.net_info.get_intersection(str(tl_id)).all_inc_lanes_flat()
            )
            lane_count = len(lanes)
            features = obs.lane_features
            if len(features) != lane_count:
                raise ValueError(
                    f"lane_features row count mismatch for TLS {tl_id}: "
                    f"expected {lane_count}, got {len(features)}"
                )
            feature_names = tuple(getattr(obs, "lane_feature_names", ()))
            feature_index = {name: index for index, name in enumerate(feature_names)}
            demand_index = feature_index.get("demand", 0)
            if demand_index >= int(features.shape[1]):
                raise ValueError(f"Demand feature is unavailable for TLS {tl_id}")

            debug = obs.debug if isinstance(obs.debug, Mapping) else {}
            lane_debug = debug.get("lane", {})
            if not isinstance(lane_debug, Mapping):
                raise ValueError(f"Lane risk debug data are unavailable for TLS {tl_id}")
            nox_tau_mg = self._require_values(
                lane_debug.get("nox_tau_mg", ()),
                field="nox_tau_mg",
                expected=lane_count,
                tl_id=str(tl_id),
            )
            nox_pressure = self._require_values(
                lane_debug.get("nox_pressure", ()),
                field="nox_pressure",
                expected=lane_count,
                tl_id=str(tl_id),
            )
            green_service_s = self._require_values(
                getattr(raw, "green_service_s", ()),
                field="green_service_s",
                expected=lane_count,
                tl_id=str(tl_id),
            )

            for lane_index, lane_id in enumerate(lanes):
                demand = self._finite_float(
                    features[lane_index, demand_index],
                    field="demand",
                    tl_id=str(tl_id),
                    lane_id=str(lane_id),
                )
                tau = self._finite_float(
                    nox_tau_mg[lane_index],
                    field="nox_tau_mg",
                    tl_id=str(tl_id),
                    lane_id=str(lane_id),
                )
                pressure = self._finite_float(
                    nox_pressure[lane_index],
                    field="nox_pressure",
                    tl_id=str(tl_id),
                    lane_id=str(lane_id),
                )
                green = self._finite_float(
                    green_service_s[lane_index],
                    field="green_service_s",
                    tl_id=str(tl_id),
                    lane_id=str(lane_id),
                )
                if green < -EPS or green > self.decision_interval_s + EPS:
                    raise ValueError(
                        f"green_service_s outside [0, {self.decision_interval_s:g}] "
                        f"for TLS {tl_id}, lane {lane_id}: {green}"
                    )
                rows.append(
                    {
                        "episode": self.episode,
                        "sumo_seed": self.sumo_seed,
                        "step": int(decision_step),
                        "sim_time": sim_time,
                        "tl_id": str(tl_id),
                        "lane_id": str(lane_id),
                        "demand": demand,
                        "nox_tau_mg": tau,
                        "nox_pressure": pressure,
                        "green_service_s": min(
                            self.decision_interval_s,
                            max(0.0, green),
                        ),
                    }
                )

            self.min_sim_time_s = min(self.min_sim_time_s, sim_time)
            self.max_sim_time_s = max(self.max_sim_time_s, sim_time)

        if rows:
            self._writer.writerows(rows)
            self.row_count += len(rows)

    def commit(self) -> dict[str, Any]:
        if self._committed:
            raise RuntimeError("Risk-service lane log was already committed")
        if not self._closed:
            self._handle.close()
            self._closed = True
        if self.row_count <= 0:
            self._temporary_path.unlink(missing_ok=True)
            raise ValueError(
                f"No risk-service lane rows fell within "
                f"[{self.start_s:g}, {self.end_s:g}] s"
            )
        os.replace(self._temporary_path, self.path)
        self._committed = True
        return {
            "path": str(self.path.resolve()),
            "row_count": int(self.row_count),
            "min_sim_time_s": float(self.min_sim_time_s),
            "max_sim_time_s": float(self.max_sim_time_s),
        }

    def abort(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True
        if not self._committed:
            self._temporary_path.unlink(missing_ok=True)


def resolve_path(
    path: Optional[str | os.PathLike[str]],
    *,
    must_exist: bool = False,
) -> Optional[str]:
    if path is None or str(path).strip() == "":
        return None
    candidate = Path(path).expanduser()
    candidates = (
        [candidate]
        if candidate.is_absolute()
        else [
            PROJECT_ROOT / candidate,
            SCRIPT_DIR / candidate,
            Path.cwd() / candidate,
        ]
    )
    chosen = candidates[0]
    for item in candidates:
        if item.exists():
            chosen = item
            break
    chosen = chosen.resolve()
    if must_exist and not chosen.exists():
        raise FileNotFoundError(str(chosen))
    return str(chosen)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run RL, SUMO actuated, or Max-Pressure evaluation with "
            "10-second and 300-second outputs."
        )
    )
    parser.add_argument(
        "--controller",
        choices=(
            "rl",
            "actuated",
            "max_pressure",
            "truck_weighted_max_pressure",
        ),
        required=True,
    )
    parser.add_argument("--case-name", required=True)
    parser.add_argument(
        "--reward-mode",
        default="",
        help="Metadata label, e.g. traffic_only or multi_objective.",
    )
    parser.add_argument("--config", default="")
    parser.add_argument(
        "--reward-risk-mode",
        "--risk_reward_mode",
        dest="reward_risk_mode",
        choices=("full", "no_tail", "sum_only"),
        default=None,
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_RL_CHECKPOINT),
        help=(
            "RL checkpoint path. Defaults to "
            "models/grid36/ddqn_multi_objective.pt."
        ),
    )
    parser.add_argument("--sumocfg", default="")
    parser.add_argument(
        "--sumocfg-dir",
        default="",
        help=(
            "Batch mode: evaluate every *.sumocfg in this directory. "
            "Each scenario uses --eval-episodes episodes."
        ),
    )
    parser.add_argument("--net-xml", default="")
    parser.add_argument("--add-xml", default="")
    parser.add_argument("--groups-json", default="")
    parser.add_argument("--emission-factor-csv", default="")
    parser.add_argument("--thresholds-json", default="")
    parser.add_argument(
        "--evaluation-e2-add",
        default="",
        help=(
            "Additional file containing the evaluation E2 detectors. "
            "Empty auto-detects truck_sensitive_grid36_all_lanes_e2.add.xml "
            "beside the effective sumocfg."
        ),
    )
    parser.add_argument(
        "--repair-manifest",
        default="",
        help=(
            "JSON manifest containing policy/demand/scenario episode lists. "
            "Strictly run every listed episode without output-completeness checks. "
            "Unlisted scenarios and summary-only tasks are skipped."
        ),
    )
    parser.add_argument("--repair-episode", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--repair-finalize", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--repair-run-id", default="",
                        help=argparse.SUPPRESS)
    parser.add_argument(
        "--evaluation-exit-e1-add",
        default="",
        help=(
            "Additional file containing perimeter exit E1 detectors. "
            "Empty auto-detects truck_sensitive_grid36_boundary_exit_e1.add.xml."
        ),
    )
    parser.add_argument("--evaluation-e2-prefix", default="e2_all_")
    parser.add_argument("--evaluation-exit-e1-prefix", default="e1_exit_")
    parser.add_argument("--expected-e2-count", type=int, default=504)
    parser.add_argument("--expected-exit-e1-count", type=int, default=72)
    parser.add_argument(
        "--detector-map",
        default="",
        help=(
            "Required for --controller max_pressure or "
            "truck_weighted_max_pressure."
        ),
    )
    parser.add_argument(
        "--exit-e1-json",
        default="",
        help=(
            "JSON list, or object containing exit_e1_ids, for physical "
            "perimeter-exit E1 detectors."
        ),
    )
    parser.add_argument(
        "--exit-e1-id",
        action="append",
        default=[],
        help="Repeat to supply perimeter-exit E1 detector IDs directly.",
    )
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--port", type=int, default=8813)
    parser.add_argument("--port-stride", type=int, default=10)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-gui", action="store_true")
    parser.add_argument(
        "--output-root",
        default="results/evaluation/windowed",
    )
    parser.add_argument(
        "--output-name",
        default="",
        help=(
            "Batch-mode result directory below --output-root. "
            "Empty keeps the existing reward-mode/case-name behavior."
        ),
    )
    parser.add_argument("--decision-seconds", type=int, default=10)
    parser.add_argument("--window-seconds", type=int, default=300)
    parser.add_argument("--analysis-start", type=int, default=300)
    parser.add_argument("--analysis-end", type=int, default=3300)
    parser.add_argument("--save-vehicle-10s", action="store_true")
    parser.add_argument(
        "--network-300s-only",
        action="store_true",
        help=(
            "Save only per-episode network_300s.csv under "
            "windows_epsiode/episode_XXXX, plus compact run metadata."
        ),
    )
    parser.add_argument(
        "--no-window-output",
        action="store_true",
        help=(
            "Disable 10-second/300-second window collection and dedicated "
            "evaluation detectors. Tripinfo, TLS phase output, optional FCD, "
            "and optional lightweight mechanism logs are still saved."
        ),
    )
    parser.add_argument(
        "--save-mechanism-log",
        action="store_true",
        help="Save pre-action TLS decisions for mechanism analysis.",
    )
    parser.add_argument(
        "--save-risk-service-log",
        action="store_true",
        help=(
            "Save the minimal lane-level demand, NOx pressure, and effective "
            "green-service data needed for Exceed-state diagnostics."
        ),
    )
    parser.add_argument(
        "--risk-log-start",
        type=float,
        default=0.0,
        help="First lane-state timestamp to retain; defaults to simulation start.",
    )
    parser.add_argument(
        "--risk-log-end",
        type=float,
        default=0.0,
        help=(
            "Last lane-state timestamp to retain. Zero (the default) uses "
            "the configured episode duration."
        ),
    )
    parser.add_argument(
        "--risk-log-compression",
        choices=("gzip", "none"),
        default="none",
        help="Per-episode lane-log compression; defaults to plain CSV.",
    )
    parser.add_argument(
        "--save-fcd",
        action="store_true",
        help="Save SUMO FCD vehicle trajectories.",
    )
    parser.add_argument(
        "--save-vehicle-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write one per-episode Parquet file with per-second vehicle state, "
            "SUMO HBEFA4 NOx, and MOVES NOx (default: enabled)."
        ),
    )
    parser.add_argument(
        "--verbose-output",
        action="store_true",
        help="Print detailed runtime paths and batch status messages.",
    )
    parser.add_argument(
        "--episode-duration",
        type=int,
        default=0,
        help="Optional override; zero keeps the YAML/default duration.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.repair_episode is not None or args.repair_finalize:
        if not args.repair_manifest or not args.repair_run_id:
            raise ValueError("Global repair stage requires --repair-manifest and --repair-run-id")
        if args.repair_episode is not None and args.repair_finalize:
            raise ValueError("--repair-episode and --repair-finalize cannot be combined")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.repair_run_id):
            raise ValueError("Invalid --repair-run-id")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be > 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.port <= 0 or args.port > 65535:
        raise ValueError("--port is outside the valid TCP port range")
    if args.port_stride <= 0:
        raise ValueError("--port-stride must be > 0")
    if args.controller == "rl" and not args.checkpoint:
        raise ValueError("--checkpoint is required for --controller rl")
    if args.controller in {
        "max_pressure",
        "truck_weighted_max_pressure",
    } and not args.detector_map:
        raise ValueError(
            "--detector-map is required for Max-Pressure controllers"
        )
    if args.save_risk_service_log and args.controller != "rl":
        raise ValueError("--save-risk-service-log currently requires --controller rl")
    if args.risk_log_start < 0:
        raise ValueError("--risk-log-start must be >= 0")
    if args.risk_log_end < 0:
        raise ValueError("--risk-log-end must be >= 0")
    if 0 < args.risk_log_end <= args.risk_log_start:
        raise ValueError("--risk-log-end must be greater than --risk-log-start")
    if args.use_gui and args.workers > 1:
        raise ValueError("--use-gui requires --workers 1")
    if args.workers > 1 and str(args.device).lower() != "cpu":
        raise ValueError("Parallel evaluation currently requires --device cpu")
    if args.expected_e2_count < 0 or args.expected_exit_e1_count < 0:
        raise ValueError("Expected detector counts must be >= 0")
    if args.network_300s_only:
        if args.no_window_output:
            raise ValueError(
                "--network-300s-only cannot be combined with --no-window-output"
            )
        incompatible = [
            option
            for enabled, option in (
                (args.save_vehicle_10s, "--save-vehicle-10s"),
                (args.save_mechanism_log, "--save-mechanism-log"),
                (args.save_risk_service_log, "--save-risk-service-log"),
                (args.save_fcd, "--save-fcd"),
                (bool(args.repair_manifest), "--repair-manifest"),
            )
            if enabled
        ]
        if incompatible:
            raise ValueError(
                "--network-300s-only cannot be combined with: "
                + ", ".join(incompatible)
            )

    max_port = args.port + (min(args.workers, args.eval_episodes) - 1) * (
        args.port_stride
    )
    if max_port > 65535:
        raise ValueError("Requested worker ports exceed 65535")

    if not args.no_window_output:
        WindowEvaluationConfig(
            decision_interval_s=args.decision_seconds,
            aggregate_window_s=args.window_seconds,
            analysis_start_s=args.analysis_start,
            analysis_end_s=args.analysis_end,
        ).validate()


def load_exit_e1_ids(args: argparse.Namespace) -> tuple[str, ...]:
    detector_ids = [str(x) for x in args.exit_e1_id if str(x).strip()]
    if args.exit_e1_json:
        path = Path(
            resolve_path(args.exit_e1_json, must_exist=True)
            or args.exit_e1_json
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, Mapping):
            values = payload.get("exit_e1_ids", payload.get("detector_ids", []))
        else:
            raise ValueError(
                "exit E1 JSON must be a list or an object containing "
                "exit_e1_ids"
            )
        if not isinstance(values, list):
            raise ValueError("exit_e1_ids must be a JSON list")
        detector_ids.extend(str(x) for x in values if str(x).strip())
    return tuple(dict.fromkeys(detector_ids))


def _runtime_detector_copy(source: Path, destination: Path) -> int:
    """Copy a detector additional file while disabling its XML output."""
    tree = ET.parse(source)
    count = 0
    for node in tree.getroot().iter():
        tag = _local_name(node.tag).lower()
        if tag in {
            "laneareadetector",
            "e2detector",
            "inductionloop",
            "e1detector",
        }:
            node.set("file", "NUL" if os.name == "nt" else "/dev/null")
            count += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return count


def prepare_evaluation_sumo_config(
    *,
    source_sumocfg: str,
    case_root: Path,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    """Create a self-contained SUMO config that loads only the new eval sensors."""
    source_cfg = Path(source_sumocfg).resolve()
    source_dir = source_cfg.parent
    default_e2_source = source_dir / (
        "truck_sensitive_grid36_all_lanes_e2.add.xml"
    )
    default_exit_source = source_dir / (
        "truck_sensitive_grid36_boundary_exit_e1.add.xml"
    )
    if not default_e2_source.is_file():
        default_e2_source = source_dir.parent / default_e2_source.name
    if not default_exit_source.is_file():
        default_exit_source = source_dir.parent / default_exit_source.name
    e2_source = Path(
        resolve_path(args.evaluation_e2_add, must_exist=True)
        if args.evaluation_e2_add
        else default_e2_source
    ).resolve()
    exit_source = Path(
        resolve_path(args.evaluation_exit_e1_add, must_exist=True)
        if args.evaluation_exit_e1_add
        else default_exit_source
    ).resolve()
    for source in (e2_source, exit_source):
        if not source.is_file():
            raise FileNotFoundError(
                f"Evaluation detector additional file not found: {source}"
            )

    runtime_dir = case_root / "runtime_inputs"
    if getattr(args, "repair_episode", None) is not None:
        runtime_dir = runtime_dir / f"repair_ep{int(args.repair_episode):04d}"
    runtime_e2 = runtime_dir / "evaluation_all_lanes_e2.add.xml"
    runtime_exit = runtime_dir / "evaluation_boundary_exit_e1.add.xml"
    e2_count = _runtime_detector_copy(e2_source, runtime_e2)
    exit_count = _runtime_detector_copy(exit_source, runtime_exit)

    if args.expected_e2_count and e2_count != args.expected_e2_count:
        raise ValueError(
            f"Evaluation E2 file contains {e2_count} detectors; "
            f"expected {args.expected_e2_count}: {e2_source}"
        )
    if (
        args.expected_exit_e1_count
        and exit_count != args.expected_exit_e1_count
    ):
        raise ValueError(
            f"Evaluation exit E1 file contains {exit_count} detectors; "
            f"expected {args.expected_exit_e1_count}: {exit_source}"
        )

    tree = ET.parse(source_cfg)
    root = tree.getroot()
    input_node = next(
        (node for node in root if _local_name(node.tag) == "input"),
        None,
    )
    if input_node is None:
        input_node = ET.SubElement(root, "input")

    additional_node = next(
        (
            node
            for node in input_node
            if _local_name(node.tag) == "additional-files"
        ),
        None,
    )
    if additional_node is None:
        additional_node = ET.SubElement(input_node, "additional-files")

    # The runtime sumocfg lives in the result tree, so make every referenced
    # input absolute before moving it away from the network directory.
    for node in input_node.iter():
        tag = _local_name(node.tag)
        if tag not in {
            "net-file",
            "route-files",
            "additional-files",
            "weight-files",
        }:
            continue
        values = [
            item.strip()
            for item in str(node.get("value", "")).split(",")
            if item.strip()
        ]
        resolved_values: list[str] = []
        for value in values:
            path = Path(value)
            if not path.is_absolute():
                path = source_dir / path
            resolved_values.append(str(path.resolve()).replace("\\", "/"))
        node.set("value", ",".join(resolved_values))

    additional_values = [
        item.strip()
        for item in str(additional_node.get("value", "")).split(",")
        if item.strip()
    ]
    for path in (runtime_e2, runtime_exit):
        value = str(path.resolve()).replace("\\", "/")
        if value not in additional_values:
            additional_values.append(value)
    additional_node.set("value", ",".join(additional_values))

    runtime_cfg = runtime_dir / "evaluation_runtime.sumocfg"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    tree.write(runtime_cfg, encoding="utf-8", xml_declaration=True)
    return str(runtime_cfg.resolve()), {
        "source_sumocfg": str(source_cfg),
        "runtime_sumocfg": str(runtime_cfg.resolve()),
        "evaluation_e2_add": str(e2_source),
        "evaluation_exit_e1_add": str(exit_source),
        "evaluation_e2_count": e2_count,
        "evaluation_exit_e1_count": exit_count,
    }


def prepare_config(
    args: argparse.Namespace,
    *,
    output_root: Path,
) -> MasterConfig:
    config_path = resolve_path(args.config, must_exist=True) if args.config else None
    cfg = get_config(config_path)
    cfg.seed = int(args.seed)
    cfg.device = str(args.device)

    if args.controller == "actuated":
        cfg.env.baseline_control_mode = "sumo_actuated"
        if not args.sumocfg:
            cfg.env.sumo_cfg = cfg.env.baseline_sumo_cfg
        if not args.net_xml and getattr(cfg.env, "baseline_net_xml", ""):
            cfg.env.net_xml = cfg.env.baseline_net_xml

    if args.sumocfg:
        cfg.env.sumo_cfg = str(args.sumocfg)
    if args.net_xml:
        cfg.env.net_xml = str(args.net_xml)
    if args.add_xml:
        cfg.env.add_xml = str(args.add_xml)
    if args.groups_json:
        cfg.env.intersection_groups_json = str(args.groups_json)
    if args.emission_factor_csv:
        cfg.env.emission_factor_csv = str(args.emission_factor_csv)
    if args.thresholds_json:
        cfg.emission_risk.thresholds_json = str(args.thresholds_json)
    if args.reward_risk_mode is not None:
        cfg.emission_risk.reward_risk_mode = str(args.reward_risk_mode)
    if args.episode_duration:
        cfg.env.episode_duration = int(args.episode_duration)

    cfg.env.sumo_cfg = (
        resolve_path(cfg.env.sumo_cfg, must_exist=True) or cfg.env.sumo_cfg
    )
    cfg.env.net_xml = (
        resolve_path(cfg.env.net_xml, must_exist=True) or cfg.env.net_xml
    )
    cfg.env.add_xml = (
        resolve_path(cfg.env.add_xml, must_exist=True) or cfg.env.add_xml
    )
    cfg.env.intersection_groups_json = (
        resolve_path(cfg.env.intersection_groups_json, must_exist=True)
        or cfg.env.intersection_groups_json
    )
    cfg.env.emission_factor_csv = (
        resolve_path(cfg.env.emission_factor_csv, must_exist=True)
        or cfg.env.emission_factor_csv
    )
    cfg.env.tripinfo_dir = str(output_root / "tripinfo")
    cfg.env.save_detector_outputs = False
    cfg.env.save_sumo_aux_outputs = False
    cfg.env.save_fcd_output = bool(args.save_fcd)
    cfg.env.save_vehicle_state_output = bool(
        args.save_vehicle_state and not args.network_300s_only
    )
    cfg.env.save_vehicle_second_output = False
    cfg.env.save_tls_phase_outputs = not bool(args.network_300s_only)
    cfg.env.verbose_runtime_output = bool(args.verbose_output)
    if args.network_300s_only:
        cfg.log.save_emission_step = False

    if (
        not args.no_window_output
        and int(cfg.env.decision_interval) != int(args.decision_seconds)
    ):
        raise ValueError(
            "The logger decision interval must equal cfg.env.decision_interval: "
            f"{args.decision_seconds} != {cfg.env.decision_interval}"
        )
    if (
        not args.no_window_output
        and int(cfg.env.episode_duration) % int(args.window_seconds) != 0
    ):
        raise ValueError(
            "episode_duration must be divisible by --window-seconds"
        )
    if (
        not args.no_window_output
        and int(args.analysis_end) > int(cfg.env.episode_duration)
    ):
        raise ValueError(
            "--analysis-end exceeds cfg.env.episode_duration"
        )
    cfg.validate()
    return cfg


# ----------------------------------------------------------------------
# Nominal route demand parsing
# ----------------------------------------------------------------------


def _local_name(tag: str) -> str:
    return str(tag).split("}", 1)[-1]


def _read_xml_root(path: Path) -> ET.Element:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rb") as file_obj:
            return ET.parse(file_obj).getroot()
    return ET.parse(path).getroot()


def _classify_route_type(type_id: str, vclass: str = "") -> str:
    text = f"{type_id} {vclass}".strip().lower()
    if any(token in text for token in ("truck", "lorry", "hdv", "freight", "hgv")):
        return "truck"
    if any(token in text for token in ("sedan", "passenger", "car", "auto")):
        return "sedan"
    return "unknown"


def _sumocfg_route_files(sumocfg_path: str) -> list[Path]:
    path = Path(sumocfg_path).resolve()
    root = _read_xml_root(path)
    route_value = ""
    for element in root.iter():
        if _local_name(element.tag) == "route-files":
            route_value = str(element.get("value", ""))
            break
    route_paths: list[Path] = []
    for item in route_value.split(","):
        item = item.strip()
        if not item:
            continue
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        route_paths.append(candidate.resolve())
    return route_paths


def _finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _flow_departures(
    node: ET.Element,
    *,
    episode_duration: float,
) -> tuple[list[tuple[float, float]], bool]:
    """Return (depart_time, weight) pairs and whether weights are expected."""
    begin = max(0.0, _finite_float(node.get("begin"), 0.0))
    end = min(
        episode_duration,
        _finite_float(node.get("end"), episode_duration),
    )
    if end <= begin:
        return [], False

    raw_number = node.get("number")
    if raw_number is not None:
        number = max(0, int(_finite_float(raw_number, 0.0)))
        if number == 0:
            return [], False
        spacing = (end - begin) / number
        return [
            (begin + index * spacing, 1.0) for index in range(number)
        ], False

    raw_period = node.get("period")
    if raw_period is not None:
        try:
            period = float(raw_period)
        except ValueError as exc:
            raise ValueError(
                "Non-numeric SUMO flow period is not supported by the "
                f"nominal scheduler: flow={node.get('id', '')!r}, "
                f"period={raw_period!r}. Convert it to number, period, "
                "vehsPerHour, or probability first."
            ) from exc
        if period > 0.0:
            count = int(math.ceil((end - begin) / period - EPS))
            return [
                (begin + index * period, 1.0)
                for index in range(max(0, count))
                if begin + index * period < end - EPS
            ], False

    raw_vph = node.get("vehsPerHour")
    if raw_vph is not None:
        vph = _finite_float(raw_vph, 0.0)
        if vph > 0.0:
            period = 3600.0 / vph
            count = int(math.ceil((end - begin) / period - EPS))
            return [
                (begin + index * period, 1.0)
                for index in range(max(0, count))
                if begin + index * period < end - EPS
            ], False

    raw_probability = node.get("probability")
    if raw_probability is not None:
        probability = min(
            1.0, max(0.0, _finite_float(raw_probability, 0.0))
        )
        return [
            (second, probability)
            for second in range(int(math.floor(begin)), int(math.ceil(end)))
            if begin <= second < end
        ], True

    return [(begin, 1.0)], False


def parse_nominal_schedule(
    *,
    sumocfg_path: str,
    episode_duration_s: int,
    interval_s: int,
) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    schedule: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "vehicle_count": 0.0,
            "truck_count": 0.0,
            "sedan_count": 0.0,
            "unknown_count": 0.0,
            "contains_expectation": 0.0,
        }
    )
    route_files = _sumocfg_route_files(sumocfg_path)
    if not route_files:
        raise ValueError(
            f"No route-files entry found in SUMO config: {sumocfg_path}"
        )

    total_explicit = 0.0
    total_expected = 0.0
    probabilistic_flow_count = 0
    distribution_reference_count = 0
    for route_path in route_files:
        if not route_path.is_file():
            raise FileNotFoundError(str(route_path))
        root = _read_xml_root(route_path)
        type_class: dict[str, str] = {}
        for node in root.iter():
            if _local_name(node.tag) == "vType":
                type_id = str(node.get("id", ""))
                type_class[type_id] = _classify_route_type(
                    type_id, str(node.get("vClass", ""))
                )

        type_shares: dict[str, dict[str, float]] = {}
        for node in root.iter():
            if _local_name(node.tag) != "vTypeDistribution":
                continue
            distribution_id = str(node.get("id", "")).strip()
            members = str(node.get("vTypes", "")).split()
            probabilities = str(node.get("probabilities", "")).split()
            weighted_members: list[tuple[str, float]] = []
            if members:
                raw_weights = [
                    _finite_float(value, 0.0) for value in probabilities
                ]
                if len(raw_weights) != len(members):
                    raw_weights = [1.0] * len(members)
                weighted_members.extend(zip(members, raw_weights))
            else:
                for child in node:
                    if _local_name(child.tag) != "vType":
                        continue
                    member_id = str(
                        child.get("id", child.get("refId", ""))
                    ).strip()
                    if member_id:
                        weighted_members.append(
                            (
                                member_id,
                                _finite_float(
                                    child.get("probability", 1.0), 1.0
                                ),
                            )
                        )
            total_weight = sum(max(0.0, weight) for _, weight in weighted_members)
            if not distribution_id or total_weight <= 0.0:
                continue
            shares = {name: 0.0 for name in ("truck", "sedan", "unknown")}
            for member_id, weight in weighted_members:
                vehicle_class = type_class.get(
                    member_id, _classify_route_type(member_id)
                )
                shares[vehicle_class] += max(0.0, weight) / total_weight
            type_shares[distribution_id] = shares

        for node in root.iter():
            tag = _local_name(node.tag)
            if tag not in {"vehicle", "trip", "flow"}:
                continue
            type_id = str(node.get("type", ""))
            shares = type_shares.get(type_id)
            if shares is None:
                vehicle_type = type_class.get(
                    type_id, _classify_route_type(type_id)
                )
                shares = {
                    name: 1.0 if name == vehicle_type else 0.0
                    for name in ("truck", "sedan", "unknown")
                }
                uses_distribution = False
            else:
                uses_distribution = True
                distribution_reference_count += 1
            if tag in {"vehicle", "trip"}:
                raw_depart = str(node.get("depart", ""))
                try:
                    depart_pairs = [(float(raw_depart), 1.0)]
                except ValueError:
                    continue
                contains_expectation = False
            else:
                depart_pairs, contains_expectation = _flow_departures(
                    node,
                    episode_duration=float(episode_duration_s),
                )
                probabilistic_flow_count += int(contains_expectation)

            for depart, weight in depart_pairs:
                if depart < 0.0 or depart >= episode_duration_s:
                    continue
                interval_index = int(math.floor((depart + 1.0e-9) / interval_s))
                item = schedule[interval_index]
                item["vehicle_count"] += weight
                for vehicle_type, share in shares.items():
                    item[f"{vehicle_type}_count"] += weight * share
                if contains_expectation or uses_distribution:
                    item["contains_expectation"] = 1.0
                    total_expected += weight
                else:
                    total_explicit += weight

    n_intervals = int(math.ceil(episode_duration_s / interval_s))
    finalized = {
        index: dict(schedule[index]) for index in range(n_intervals)
    }
    metadata = {
        "route_files": [str(path) for path in route_files],
        "nominal_explicit_vehicle_count": total_explicit,
        "nominal_expected_vehicle_count": total_expected,
        "probabilistic_flow_count": probabilistic_flow_count,
        "vehicle_type_distribution_reference_count": (
            distribution_reference_count
        ),
        "schedule_interval_s": interval_s,
    }
    return finalized, metadata


# ----------------------------------------------------------------------
# Episode execution
# ----------------------------------------------------------------------


def _load_detector_map(path: str) -> dict[str, Any]:
    resolved = resolve_path(path, must_exist=True)
    if not resolved:
        raise ValueError("Missing detector map path")
    payload = json.loads(Path(resolved).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Max-Pressure detector map must be a JSON object")
    for key in ("upstream", "downstream"):
        if key not in payload:
            raise ValueError(f"Detector map is missing {key!r}")
    return payload


def _collect_initial_raw(env: SumoEnv, tl_ids: Iterable[str]) -> dict[str, Any]:
    return {
        tl_id: env._collect_obs(tl_id, None)  # type: ignore[attr-defined]
        for tl_id in tl_ids
    }


def _build_runtime(
    args: argparse.Namespace,
    *,
    output_root: Path,
) -> dict[str, Any]:
    cfg = prepare_config(args, output_root=output_root)
    set_global_seed(int(cfg.seed))
    net_info = parse_network(
        cfg.env.net_xml,
        cfg.env.add_xml,
        cfg.env.intersection_groups_json,
    )
    emission_lookup = build_emission_lookup(
        cfg.env.emission_factor_csv,
        default_pollutant=cfg.env.emission_pollutants[0],
    )
    if args.no_window_output:
        nominal_schedule, nominal_meta = {}, {}
    else:
        nominal_schedule, nominal_meta = parse_nominal_schedule(
            sumocfg_path=cfg.env.sumo_cfg,
            episode_duration_s=int(cfg.env.episode_duration),
            interval_s=int(args.decision_seconds),
        )

    runtime: dict[str, Any] = {
        "cfg": cfg,
        "net_info": net_info,
        "emission_lookup": emission_lookup,
        "nominal_schedule": nominal_schedule,
        "nominal_meta": nominal_meta,
        "builder": None,
        "agent": None,
        "max_pressure": None,
    }

    if args.controller == "rl":
        builder = ObsRewardBuilder(cfg, net_info)
        agent = MGMQAgentManager(cfg, net_info, device=str(args.device))
        checkpoint_path = (
            resolve_path(args.checkpoint, must_exist=True) or args.checkpoint
        )
        agent.load_checkpoint(
            checkpoint_path,
            load_optimizer=False,
            allow_threshold_mismatch=bool(args.thresholds_json),
        )
        agent.online_bank.eval()
        runtime["builder"] = builder
        runtime["agent"] = agent
    elif args.controller in {
        "max_pressure",
        "truck_weighted_max_pressure",
    }:
        detector_map = _load_detector_map(args.detector_map)
        runtime["detector_map"] = detector_map
        controller_class = (
            TruckWeightedMaxPressureController
            if args.controller == "truck_weighted_max_pressure"
            else MaxPressureController
        )
        runtime["max_pressure"] = controller_class(
            net_info, detector_map, cfg.max_pressure
        )
    return runtime


def _record_lightweight_mechanism_rows(
    *,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    episode: int,
    sumo_seed: int,
    decision_step: int,
    raw_observations: Mapping[str, Any],
    actions: Mapping[str, int],
    action_infos: Optional[Mapping[str, Any]] = None,
) -> None:
    """Record only decision-level signal/action data outside window mode."""
    if not args.save_mechanism_log:
        return
    action_infos = action_infos or {}
    for tl_id, action in sorted(actions.items()):
        raw = raw_observations[tl_id]
        info = action_infos.get(tl_id)
        action_mask = list(getattr(info, "action_mask", []) or [])
        valid_action_count: Any = getattr(info, "valid_action_count", "")
        rows.append(
            {
                "episode": int(episode),
                "sumo_seed": int(sumo_seed),
                "case_name": str(args.case_name),
                "controller": str(args.controller),
                "reward_mode": str(args.reward_mode),
                "decision_step": int(decision_step),
                "simulation_time_s": float(raw.sim_time),
                "tls_id": str(tl_id),
                "current_green_phase": int(raw.current_phase),
                "selected_action": int(action),
                "target_green_phase": int(action),
                "is_phase_switch": int(int(action) != int(raw.current_phase)),
                "valid_action_count": valid_action_count,
                "action_mask_json": json.dumps(action_mask),
                "in_yellow": int(bool(raw.in_yellow)),
                "pending_green_phase": int(raw.pending_phase),
                "sumo_green_phase": int(raw.sumo_green_phase),
                "sumo_raw_phase": int(raw.sumo_raw_phase),
                "sumo_state": str(raw.sumo_state),
            }
        )


def _print_episode_completed(
    *,
    args: argparse.Namespace,
    episode: int,
    wall_time_s: float,
) -> None:
    _console(
        f"scenario={args.case_name} "
        f"episode={int(episode)}/{int(args.eval_episodes)} "
        f"wall_time={float(wall_time_s):.1f}s"
    )


def _format_elapsed(seconds: float) -> str:
    """Return a compact HH:MM:SS representation of elapsed wall time."""
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _console(message: str, *, file=None) -> None:
    """Print one concise evaluation line with the current system time."""
    target = sys.stdout if file is None else file
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}",
        file=target,
        flush=True,
    )


def run_episode(
    *,
    args: argparse.Namespace,
    runtime: Mapping[str, Any],
    episode: int,
    sumo_seed: int,
    port: int,
    exit_e1_ids: tuple[str, ...],
    output_root: Path,
) -> dict[str, Any]:
    cfg: MasterConfig = runtime["cfg"]
    net_info = runtime["net_info"]
    emission_lookup = runtime["emission_lookup"]
    recorder: Optional[EmissionEpisodeRecorder] = None
    if not args.network_300s_only:
        recorder = EmissionEpisodeRecorder(
            emission_lookup,
            pollutants=cfg.env.emission_pollutants,
            output_unit="mg",
        )
    env: Optional[SumoEnv] = None
    window_logger: Optional[WindowEvaluationLogger] = None
    risk_service_logger: Optional[RiskServiceLaneLogger] = None
    mechanism_rows: list[dict[str, Any]] = []
    risk_service_metadata: dict[str, Any] = {}
    started = time.time()

    try:
        if not args.no_window_output:
            window_logger = WindowEvaluationLogger(
                config=WindowEvaluationConfig(
                    decision_interval_s=int(args.decision_seconds),
                    aggregate_window_s=int(args.window_seconds),
                    analysis_start_s=int(args.analysis_start),
                    analysis_end_s=int(args.analysis_end),
                    save_network_10s=not bool(args.network_300s_only),
                    save_e2_detector_10s=not bool(args.network_300s_only),
                    save_network_300s=True,
                    save_e2_detector_300s=not bool(args.network_300s_only),
                    save_e2_vehicle_300s=not bool(args.network_300s_only),
                    save_vehicle_10s=bool(args.save_vehicle_10s),
                    save_decision_actions=bool(args.save_mechanism_log),
                    exit_e1_ids=exit_e1_ids,
                    e2_id_prefix=str(args.evaluation_e2_prefix),
                    exit_e1_id_prefix=str(args.evaluation_exit_e1_prefix),
                    expected_e2_count=int(args.expected_e2_count),
                    expected_exit_e1_count=int(args.expected_exit_e1_count),
                    pollutant=str(cfg.env.emission_pollutants[0]),
                    output_unit="mg",
                ),
                output_root=output_root,
                emission_lookup=emission_lookup,
                controller=str(args.controller),
                case_name=str(args.case_name),
                reward_mode=str(args.reward_mode),
                nominal_by_interval=runtime["nominal_schedule"],
            )
        env = SumoEnv(
            cfg.env,
            net_info,
            port=int(port),
            use_gui=bool(args.use_gui),
            emission_lookup=emission_lookup,
            emission_recorder=recorder,
            evaluation_logger=window_logger,
        )
        control_tls = args.controller != "actuated"
        env.start(
            episode_id=int(episode),
            seed=int(sumo_seed),
            control_tls=control_tls,
        )
        if args.network_300s_only and window_logger is not None:
            default_episode_dir = window_logger.episode_dir
            window_logger.episode_dir = (
                output_root
                / "windows_epsiode"
                / f"episode_{int(episode):04d}"
            )
            window_logger.episode_dir.mkdir(parents=True, exist_ok=True)
            try:
                default_episode_dir.rmdir()
            except OSError:
                pass
        if args.save_risk_service_log:
            risk_service_logger = RiskServiceLaneLogger(
                output_root=output_root,
                episode=int(episode),
                sumo_seed=int(sumo_seed),
                net_info=net_info,
                start_s=float(args.risk_log_start),
                end_s=float(args.risk_log_end),
                decision_interval_s=float(args.decision_seconds),
                compression=str(args.risk_log_compression),
            )

        raw = _collect_initial_raw(env, net_info.intersection_ids)
        if args.controller == "rl":
            builder: ObsRewardBuilder = runtime["builder"]
            agent: MGMQAgentManager = runtime["agent"]
            observations = builder.build_observations(raw)
        else:
            observations = None

        max_pressure = runtime.get("max_pressure")
        if max_pressure is not None:
            max_pressure.reset()

        for step in range(int(cfg.env.steps_per_episode)):
            if args.controller == "rl":
                actions, action_infos = agent.act(
                    observations,
                    epsilon=0.0,
                    deterministic=True,
                    record_q_values=False,
                )
                if window_logger is not None:
                    window_logger.record_decision_actions(
                        decision_step=step,
                        raw_observations=raw,
                        actions=actions,
                    )
                else:
                    _record_lightweight_mechanism_rows(
                        rows=mechanism_rows,
                        args=args,
                        episode=episode,
                        sumo_seed=sumo_seed,
                        decision_step=step,
                        raw_observations=raw,
                        actions=actions,
                        action_infos=action_infos,
                    )
                next_raw = env.step(actions, decision_step=step)
                next_observations = builder.build_observations(next_raw)
                if risk_service_logger is not None:
                    risk_service_logger.record(
                        decision_step=step,
                        raw_observations=next_raw,
                        observations=next_observations,
                    )
                observations = next_observations
            elif args.controller == "actuated":
                next_raw = env.step_passive(decision_step=step)
            else:
                actions, _, _ = max_pressure.act(
                    env,
                    raw,
                    int(episode),
                    int(step),
                )
                if window_logger is not None:
                    window_logger.record_decision_actions(
                        decision_step=step,
                        raw_observations=raw,
                        actions=actions,
                    )
                else:
                    _record_lightweight_mechanism_rows(
                        rows=mechanism_rows,
                        args=args,
                        episode=episode,
                        sumo_seed=sumo_seed,
                        decision_step=step,
                        raw_observations=raw,
                        actions=actions,
                    )
                next_raw = env.step(actions, decision_step=step)

            if window_logger is not None:
                window_logger.end_decision_step(
                    decision_step=step,
                    step_emission=env._last_step_emission,  # type: ignore[attr-defined]
                )
            raw = next_raw

        tripinfo_path = env.tripinfo_output_path
        tls_switch_states_path = env.tls_switch_states_output_path
        tls_switch_times_path = env.tls_switch_times_output_path
        fcd_output_path = env.fcd_output_path
        stats = env.close()
        env = None
        emission_dir: Optional[Path] = None
        if recorder is not None and (
            window_logger is not None or bool(cfg.log.save_emission_step)
        ):
            emission_result = recorder.finalize_episode()
            emission_dir = write_episode_emission_outputs(
                output_root=output_root,
                episode=int(episode),
                emission_result=emission_result,
            )
        if window_logger is not None:
            summary = window_logger.finalize_episode(
                tripinfo_path=tripinfo_path,
                traffic_stats=stats,
            )
        else:
            mechanism_path = ""
            if args.save_mechanism_log:
                path = (
                    output_root
                    / "mechanism"
                    / f"decision_action_ep{int(episode):04d}.csv"
                )
                write_csv(path, mechanism_rows)
                mechanism_path = str(path.resolve())
            summary = {
                "episode": int(episode),
                "sumo_seed": int(sumo_seed),
                "case_name": str(args.case_name),
                "controller": str(args.controller),
                "reward_mode": str(args.reward_mode),
                "avg_delay_s": float(stats.avg_delay_s),
                "avg_travel_time_s": float(stats.avg_travel_time_s),
                "total_arrived": int(stats.total_arrived),
                "completion_rate": float(stats.completion_rate),
                "mechanism_log_path": mechanism_path,
            }
        summary["emission_dir"] = (
            str(emission_dir.resolve()) if emission_dir is not None else ""
        )
        summary["tripinfo_path"] = str(Path(tripinfo_path).resolve())
        summary["tls_switch_states_path"] = (
            str(Path(tls_switch_states_path).resolve())
            if tls_switch_states_path
            else ""
        )
        summary["tls_switch_times_path"] = (
            str(Path(tls_switch_times_path).resolve())
            if tls_switch_times_path
            else ""
        )
        summary["fcd_output_path"] = (
            str(Path(fcd_output_path).resolve()) if fcd_output_path else ""
        )
        if risk_service_logger is not None:
            risk_service_metadata = risk_service_logger.commit()
            summary["risk_service_log_path"] = risk_service_metadata["path"]
            summary["risk_service_log_rows"] = risk_service_metadata["row_count"]
            summary["risk_service_log_min_time_s"] = risk_service_metadata[
                "min_sim_time_s"
            ]
            summary["risk_service_log_max_time_s"] = risk_service_metadata[
                "max_sim_time_s"
            ]
        if args.network_300s_only:
            Path(tripinfo_path).unlink(missing_ok=True)
            summary["tripinfo_path"] = ""
            if window_logger is not None:
                (window_logger.episode_dir / "quality_control.csv").unlink(
                    missing_ok=True
                )
            runtime_dir = output_root / "sumo_runtime"
            for runtime_name in (
                f"runtime_add_{int(episode):04d}.add.xml",
                f"runtime_sumocfg_{int(episode):04d}.sumocfg",
            ):
                (runtime_dir / runtime_name).unlink(missing_ok=True)
        summary["port"] = int(port)
        summary["wall_time_s"] = time.time() - started
        _print_episode_completed(
            args=args,
            episode=episode,
            wall_time_s=float(summary["wall_time_s"]),
        )
        return summary
    finally:
        if risk_service_logger is not None:
            risk_service_logger.abort()
        if env is not None:
            try:
                env.close(parse_tripinfo=False)
            except Exception:
                pass


def _run_batch(payload: Mapping[str, Any]) -> dict[str, Any]:
    os.environ["SUMO_FORCE_TRACI"] = "1"
    args = argparse.Namespace(**dict(payload["args"]))
    output_root = Path(payload["output_root"]).resolve()
    runtime = _build_runtime(args, output_root=output_root)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for episode, sumo_seed in payload["episodes"]:
        try:
            rows.append(
                run_episode(
                    args=args,
                    runtime=runtime,
                    episode=int(episode),
                    sumo_seed=int(sumo_seed),
                    port=int(payload["port"]),
                    exit_e1_ids=tuple(payload["exit_e1_ids"]),
                    output_root=output_root,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "episode": int(episode),
                    "sumo_seed": int(sumo_seed),
                    "port": int(payload["port"]),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    return {
        "rows": rows,
        "errors": errors,
        "nominal_meta": runtime["nominal_meta"],
    }


# ----------------------------------------------------------------------
# Shared output merging
# ----------------------------------------------------------------------


def write_episode_emission_outputs(
    *,
    output_root: Path,
    episode: int,
    emission_result: Any,
) -> Path:
    """Write one episode's detailed emission tables beside tripinfo/output."""
    emission_dir = (
        Path(output_root).resolve()
        / "emission"
        / f"episode_{int(episode):04d}"
    )
    emission_dir.mkdir(parents=True, exist_ok=True)

    outputs = (
        ("vehicle_df", "vehicle_emission.csv"),
        ("step_df", "decision_step_emission.csv"),
        ("lane_step_df", "lane_emission_step.csv"),
        ("edge_step_df", "edge_emission_step_raw.csv"),
    )
    for attribute, filename in outputs:
        dataframe = getattr(emission_result, attribute, None)
        if dataframe is None or not hasattr(dataframe, "to_csv"):
            continue
        dataframe.to_csv(
            emission_dir / filename,
            index=False,
            encoding="utf-8-sig",
        )
    return emission_dir


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        return list(csv.DictReader(file_obj))


def _manifest_policy(value: str) -> str:
    text = str(value).strip().lower()
    if text.startswith("multi"):
        return "multi_objective"
    if text.startswith("traffic"):
        return "traffic_only"
    raise ValueError(
        "--repair-manifest requires reward-mode starting with "
        "'traffic' or 'multi'"
    )


def _load_manifest_episodes(
    args: argparse.Namespace,
    all_episode_items: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return the manifest episodes for this single scenario."""
    if not args.repair_manifest:
        return list(all_episode_items)
    manifest_value = resolve_path(args.repair_manifest, must_exist=True)
    manifest_path = Path(manifest_value or args.repair_manifest).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "vehicle_state" in payload.get("required_outputs", []) and not args.save_vehicle_state:
        raise ValueError("This repair manifest requires --save-vehicle-state")
    if int(payload.get("global_seed", -1)) != int(args.seed):
        raise ValueError(
            "Manifest global_seed does not match --seed: "
            f"{payload.get('global_seed')} != {args.seed}"
        )
    if int(payload.get("eval_episodes", -1)) != int(args.eval_episodes):
        raise ValueError(
            "Manifest eval_episodes does not match --eval-episodes: "
            f"{payload.get('eval_episodes')} != {args.eval_episodes}"
        )

    canonical = {episode: seed for episode, seed in all_episode_items}
    manifest_map = {
        int(episode): int(seed)
        for episode, seed in dict(payload.get("episode_seed_map", {})).items()
    }
    if manifest_map != canonical:
        raise ValueError(
            "Manifest episode_seed_map does not match the deterministic "
            "mapping generated by --seed/--eval-episodes"
        )

    policy = _manifest_policy(args.reward_mode)
    scenario = str(args.case_name)
    demand_match = re.search(r"eval_(D(?:20|40|50|60|70|80))_P\d+$", scenario)
    if demand_match is None:
        raise ValueError(
            f"Cannot identify demand from manifest scenario name: {scenario}"
        )
    demand = demand_match.group(1)
    matches = [
        task
        for task in list(payload.get("tasks", []))
        if str(task.get("policy", "")).lower() == policy
        and str(task.get("demand", "")).upper() == demand
        and str(task.get("scenario", "")) == scenario
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Manifest contains duplicate task groups for {policy}/{scenario}"
        )
    if not matches:
        return []

    task = matches[0]
    episodes = [int(value) for value in task.get("episodes", [])]
    seeds = [int(value) for value in task.get("sumo_seeds", [])]
    if not episodes or len(episodes) != len(set(episodes)):
        raise ValueError(f"Manifest has empty or duplicate episodes: {policy}/{scenario}")
    expected_seeds = []
    for episode in episodes:
        if episode not in canonical:
            raise ValueError(
                f"Manifest episode is outside the canonical range: {episode}"
            )
        expected_seeds.append(canonical[episode])
    if seeds != expected_seeds:
        raise ValueError(
            f"Manifest episode/seed mismatch for {policy}/{scenario}: "
            f"{seeds} != {expected_seeds}"
        )
    return list(zip(episodes, seeds))


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _episode_analysis_complete(
    case_root: Path,
    episode: int,
    sumo_seed: int,
    existing_summary: Mapping[int, Mapping[str, Any]],
    require_vehicle_state: bool = False,
    episode_duration_s: float = 4200,
) -> bool:
    """Check the minimum outputs jointly required by the analysis scripts."""
    emission_dir = case_root / "emission" / f"episode_{episode:04d}"
    lane_plain = case_root / "physical" / f"episode_{episode:04d}" / "lane_step.csv"
    lane_gzip = lane_plain.with_suffix(".csv.gz")
    row = existing_summary.get(int(episode))
    try:
        summary_matches = row is None or (
            int(row.get("sumo_seed", -1)) == int(sumo_seed)
            and float(row.get("final_sim_time_s", 0)) >= episode_duration_s
        )
    except (TypeError, ValueError):
        summary_matches = False
    if require_vehicle_state:
        parquet = case_root / "vehicle_state" / f"vehicle_state_ep{episode:04d}.parquet"
        if not _nonempty_file(parquet) or parquet.with_suffix(".parquet.incomplete").exists():
            return False
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(str(parquet))
            try:
                required = {"episode", "sumo_seed", "simulation_time_s", "vehicle_id",
                            "speed_mps", "acceleration_mps2", "nox_hbefa4_mg_s", "nox_moves_mg_s"}
                if not required.issubset(parquet_file.schema_arrow.names):
                    return False
            finally:
                parquet_file.close()
        except ImportError as exc:
            raise RuntimeError("Vehicle-state repair validation requires pyarrow") from exc
        except (OSError, ValueError):
            return False
    return bool(
        summary_matches
        and _nonempty_file(
            case_root / "tripinfo" / f"tripinfo_ep{episode:04d}.xml"
        )
        and _nonempty_file(emission_dir / "vehicle_emission.csv")
        and _nonempty_file(emission_dir / "decision_step_emission.csv")
        and (_nonempty_file(lane_plain) or _nonempty_file(lane_gzip))
    )


def _archive_incomplete_episode(case_root: Path, episode: int) -> Optional[Path]:
    """Move partial per-episode artifacts to a recoverable backup directory."""
    candidates = [
        case_root / "vehicle_state" / f"vehicle_state_ep{episode:04d}.parquet",
        case_root / "vehicle_state" / f"vehicle_state_ep{episode:04d}.parquet.incomplete",
        case_root / "physical" / f"episode_{episode:04d}",
        case_root / "emission" / f"episode_{episode:04d}",
        case_root / "windows_episode" / f"episode_{episode:04d}",
        case_root / "tripinfo" / f"tripinfo_ep{episode:04d}.xml",
        case_root / "fcd" / f"fcd_ep{episode:04d}.xml",
        case_root / "fcd" / f"fcd_ep{episode:04d}.xml.gz",
        case_root
        / "phase_state"
        / "tls_switch_states"
        / f"tls_switch_states_ep{episode:04d}.xml",
        case_root
        / "phase_state"
        / "tls_switch_times"
        / f"tls_switch_times_ep{episode:04d}.xml",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_root = (
        case_root
        / "_incomplete_backup"
        / stamp
        / f"episode_{episode:04d}"
    )
    for source in existing:
        relative = source.relative_to(case_root)
        destination = backup_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    return backup_root


def _merge_rows_by_episode(
    existing_rows: Iterable[Mapping[str, Any]],
    replacement_rows: Iterable[Mapping[str, Any]],
    replaced_episodes: set[int],
) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for row in existing_rows:
        try:
            episode = int(row["episode"])
        except (KeyError, TypeError, ValueError):
            continue
        if episode not in replaced_episodes:
            merged[episode] = dict(row)
    for row in replacement_rows:
        merged[int(row["episode"])] = dict(row)
    return [merged[episode] for episode in sorted(merged)]


def merge_episode_csvs(
    *,
    output_root: Path,
    include_vehicle_10s: bool,
    network_300s_only: bool = False,
) -> None:
    names = (
        ["network_300s.csv"]
        if network_300s_only
        else [
            "network_10s.csv",
            "e2_detector_10s.csv",
            "network_300s.csv",
            "e2_detector_300s.csv",
            "e2_vehicle_300s.csv",
            "quality_control.csv",
        ]
    )
    if include_vehicle_10s and not network_300s_only:
        names.append("vehicle_10s.csv")

    merged_dir = output_root / "all_episodes"
    episode_root = output_root / (
        "windows_epsiode" if network_300s_only else "windows_episode"
    )
    for name in names:
        rows: list[dict[str, Any]] = []
        for episode_dir in sorted(
            episode_root.glob(
                "episode_[0-9][0-9][0-9][0-9]"
            )
        ):
            rows.extend(read_csv_rows(episode_dir / name))
        merged_name = f"{Path(name).stem}_all.csv"
        write_csv(merged_dir / merged_name, rows)


def _partition_episodes(
    episode_items: list[tuple[int, int]],
    worker_count: int,
) -> list[list[tuple[int, int]]]:
    partitions: list[list[tuple[int, int]]] = [
        [] for _ in range(worker_count)
    ]
    for index, item in enumerate(episode_items):
        partitions[index % worker_count].append(item)
    return [partition for partition in partitions if partition]


def _remove_cli_value_options(
    argv: list[str],
    option_names: set[str],
) -> list[str]:
    """Remove selected one-value options from a command-line argument list."""
    filtered: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        option = argument.split("=", 1)[0]
        if option in option_names:
            index += 1 if "=" in argument else 2
            continue
        filtered.append(argument)
        index += 1
    return filtered


def _discover_scenario_sumocfgs(directory: Path) -> list[Path]:
    """Validate either one demand's 5 or all demands' 20 scenario mappings."""
    configs = sorted(
        directory.rglob("*.sumocfg"),
        key=lambda path: str(path.relative_to(directory)).lower(),
    )
    if len(configs) not in {5, 20}:
        raise ValueError(
            f"Expected 5 or 20 scenario sumocfg files in {directory}, "
            f"found {len(configs)}"
        )

    for config_path in configs:
        root = _read_xml_root(config_path)
        route_value = ""
        for node in root.iter():
            if _local_name(node.tag) == "route-files":
                route_value = str(node.get("value", "")).strip()
                break
        route_items = [
            value.strip() for value in route_value.split(",") if value.strip()
        ]
        if len(route_items) != 1:
            raise ValueError(
                f"{config_path.name} must reference exactly one route file"
            )
        route_path = Path(route_items[0])
        if not route_path.is_absolute():
            route_path = config_path.parent / route_path
        route_path = route_path.resolve()
        if not route_path.is_file():
            raise FileNotFoundError(str(route_path))
        # Scenario identity comes from sumocfg; identical routes may be shared.
    return configs


def _run_sumocfg_directory(args: argparse.Namespace) -> None:
    """Run the existing single-case evaluator once for each scenario config."""
    directory_value = resolve_path(args.sumocfg_dir, must_exist=True)
    scenario_dir = Path(directory_value or args.sumocfg_dir).resolve()
    if not scenario_dir.is_dir():
        raise NotADirectoryError(str(scenario_dir))
    scenario_configs = _discover_scenario_sumocfgs(scenario_dir)
    if args.repair_manifest:
        manifest_path = Path(resolve_path(args.repair_manifest, must_exist=True) or args.repair_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        policy = _manifest_policy(args.reward_mode)
        selected_scenarios = {str(task["scenario"]) for task in manifest.get("tasks", [])
                              if task.get("policy") == policy and task.get("episodes")}
        scenario_configs = [path for path in scenario_configs if path.stem in selected_scenarios]

    output_value = resolve_path(args.output_root, must_exist=False)
    output_root = Path(output_value or args.output_root).resolve()
    mode_name = str(
        args.output_name or args.reward_mode or args.case_name
    ).strip()
    if not mode_name or Path(mode_name).name != mode_name:
        raise ValueError(
            "output-name/reward-mode/case-name must be a simple directory name"
        )
    mode_root = output_root / mode_name
    mode_root.mkdir(parents=True, exist_ok=True)

    child_args = _remove_cli_value_options(
        list(sys.argv[1:]),
        {"--sumocfg-dir", "--sumocfg", "--case-name", "--output-root"},
    )
    failures: list[dict[str, Any]] = []
    completed: list[str] = []
    started = time.time()

    for scenario_index, config_path in enumerate(scenario_configs, start=1):
        scenario_name = config_path.stem
        if args.verbose_output:
            _console(
                "batch "
                f"scenario={scenario_name} ({scenario_index}/{len(scenario_configs)}) "
                "status=starting"
            )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *child_args,
            "--sumocfg",
            str(config_path),
            "--case-name",
            scenario_name,
            "--output-root",
            str(mode_root),
        ]
        result = subprocess.run(command, cwd=str(SCRIPT_DIR), check=False)
        if result.returncode == 0:
            completed.append(scenario_name)
        else:
            failures.append(
                {"scenario": scenario_name, "returncode": result.returncode}
            )

    total_wall_time_s = time.time() - started
    batch_metadata = {
        "system_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sumocfg_dir": str(scenario_dir),
        "reward_mode": mode_name,
        "eval_episodes_per_scenario": int(args.eval_episodes),
        "global_seed": int(args.seed),
        "workers": int(args.workers),
        "scenario_count": len(scenario_configs),
        "completed_scenarios": completed,
        "failed_scenarios": failures,
        "wall_time_s": total_wall_time_s,
    }
    (mode_root / "batch_metadata.json").write_text(
        json.dumps(batch_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _console(
        "batch completed "
        f"setting={mode_name} scenarios={len(completed)}/{len(scenario_configs)} "
        f"total_time={total_wall_time_s:.1f}s "
        f"({_format_elapsed(total_wall_time_s)})"
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} scenario(s) failed; inspect "
            f"{mode_root / 'batch_metadata.json'}"
        )
    if args.verbose_output:
        _console(
            f"batch output={mode_root}"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.sumocfg_dir:
        _run_sumocfg_directory(args)
        return
    if args.repair_manifest:
        manifest_path = Path(resolve_path(args.repair_manifest, must_exist=True) or args.repair_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not any(task.get("policy") == _manifest_policy(args.reward_mode)
                   and task.get("scenario") == args.case_name and task.get("episodes")
                   for task in manifest.get("tasks", [])):
            _console(f"repair-skip scenario={args.case_name} reason=no_manifest_task")
            return
    exit_e1_ids = () if args.no_window_output else load_exit_e1_ids(args)

    case_root = (
        Path(
            resolve_path(args.output_root, must_exist=False)
            or args.output_root
        )
        / str(args.case_name)
    ).resolve()
    case_root.mkdir(parents=True, exist_ok=True)

    # Resolve and validate the effective config once in the parent, then use
    # its deterministic seed list for every controller case.
    parent_cfg = prepare_config(args, output_root=case_root)
    if args.save_risk_service_log:
        episode_duration = float(parent_cfg.env.episode_duration)
        effective_risk_log_end = float(args.risk_log_end)
        if effective_risk_log_end <= 0:
            effective_risk_log_end = episode_duration
        if effective_risk_log_end <= float(args.risk_log_start):
            raise ValueError(
                "The effective risk-log end must be greater than "
                "--risk-log-start"
            )
        if effective_risk_log_end > episode_duration + EPS:
            raise ValueError(
                "--risk-log-end exceeds the configured episode duration: "
                f"{effective_risk_log_end:g} > {episode_duration:g}"
            )
        # Workers and metadata receive the resolved full-episode endpoint.
        args.risk_log_end = effective_risk_log_end
    if args.no_window_output:
        runtime_sumocfg = str(parent_cfg.env.sumo_cfg)
        runtime_detector_meta: dict[str, Any] = {}
    else:
        runtime_sumocfg, runtime_detector_meta = prepare_evaluation_sumo_config(
            source_sumocfg=parent_cfg.env.sumo_cfg,
            case_root=case_root,
            args=args,
        )
    # Workers receive either the original config (no-window mode) or the
    # generated config containing dedicated evaluation sensors.
    args.sumocfg = runtime_sumocfg
    sumo_seeds = build_sumo_seed_list(
        global_seed=int(parent_cfg.seed),
        total_episodes=int(args.eval_episodes),
        mode=str(parent_cfg.env.sumo_seed_mode),
    )
    if args.save_risk_service_log and len(set(sumo_seeds)) != len(sumo_seeds):
        duplicates = sorted(
            seed for seed in set(sumo_seeds) if sumo_seeds.count(seed) > 1
        )
        raise ValueError(
            "Risk-service paired analysis requires unique SUMO seeds; "
            f"duplicates={duplicates}. Reduce --eval-episodes or provide a "
            "unique deterministic seed configuration."
        )
    all_episode_items = [
        (index + 1, int(seed)) for index, seed in enumerate(sumo_seeds)
    ]
    episode_items = _load_manifest_episodes(args, all_episode_items)
    if args.repair_manifest and not episode_items:
        _console(
            f"repair-skip scenario={args.case_name} reason=no_manifest_task"
        )
        return
    if args.repair_episode is not None:
        episode_items = [item for item in episode_items
                         if item[0] == args.repair_episode]
        if len(episode_items) != 1:
            raise ValueError(f"Episode {args.repair_episode} is not selected by repair manifest")

    existing_summary_rows = read_csv_rows(case_root / "episode_summary.csv")
    existing_summary = {}
    for row in existing_summary_rows:
        try:
            existing_summary[int(row["episode"])] = row
        except (KeyError, TypeError, ValueError):
            continue

    skipped_complete: list[tuple[int, int]] = []
    if args.repair_manifest:
        if not bool(parent_cfg.log.save_emission_step):
            raise ValueError(
                "Manifest repair requires cfg.log.save_emission_step=true"
            )
        if not args.save_risk_service_log:
            raise ValueError(
                "Manifest repair requires --save-risk-service-log"
            )
        if not args.repair_finalize:
            for episode, seed in episode_items:
                backup = _archive_incomplete_episode(case_root, episode)
                _console(
                    f"repair-select scenario={args.case_name} episode={episode} "
                    f"sumo_seed={seed} mode=strict_manifest backup={backup or ''}"
                )

    rerun_episodes = {episode for episode, _seed in episode_items}
    worker_count = min(int(args.workers), len(episode_items))
    partitions = _partition_episodes(episode_items, worker_count)
    ports = [
        int(args.port) + index * int(args.port_stride)
        for index in range(worker_count)
    ]

    args_payload = dict(vars(args))
    tasks = [
        {
            "args": args_payload,
            "output_root": str(case_root),
            "episodes": partition,
            "port": ports[index],
            "exit_e1_ids": list(exit_e1_ids),
        }
        for index, partition in enumerate(partitions)
    ]

    separator = "=" * 78
    _console(separator)
    _console(f"Start evaluation: {args.case_name}")
    _console(f"controller   : {args.controller}")
    if args.controller == "rl":
        _console(f"checkpoint   : {args.checkpoint}")
    _console(
        f"episodes     : {args.eval_episodes}, "
        f"duration={parent_cfg.env.episode_duration}s, seed={args.seed}"
    )
    _console(
        f"parallel     : {worker_count > 1}, workers={worker_count}, "
        f"ports={ports}"
    )
    _console(
        "vehicle_state: "
        + ("enabled (Parquet)" if args.save_vehicle_state else "disabled")
    )
    _console(f"output       : {case_root}")
    _console(separator)

    started = time.time()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    nominal_meta: dict[str, Any] = {}

    if args.repair_finalize:
        result_dir = case_root / "_global_repair_results" / args.repair_run_id
        for episode, seed in episode_items:
            result_path = result_dir / f"ep{episode:04d}.json"
            if not result_path.is_file():
                raise FileNotFoundError(f"Global repair result missing: {result_path}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("run_id") != args.repair_run_id or int(result.get("episode", -1)) != episode or int(result.get("sumo_seed", -1)) != seed:
                raise ValueError(f"Global repair identity mismatch: {result_path}")
            result_rows = result.get("rows", [])
            result_errors = result.get("errors", [])
            if result_errors or len(result_rows) != 1:
                raise ValueError(f"Global repair episode result is incomplete: {result_path}")
            if (int(result_rows[0].get("episode", -1)) != episode
                    or int(result_rows[0].get("sumo_seed", -1)) != seed):
                raise ValueError(f"Global repair row identity mismatch: {result_path}")
            rows.extend(result_rows)
            errors.extend(result_errors)
            if not nominal_meta:
                nominal_meta = dict(result["nominal_meta"])
        if errors:
            raise RuntimeError(f"Global repair contains {len(errors)} failed episode(s); summary left unchanged")
    elif worker_count == 1:
        result = _run_batch(tasks[0])
        rows.extend(result["rows"])
        errors.extend(result["errors"])
        nominal_meta = dict(result["nominal_meta"])
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as executor:
            futures = [executor.submit(_run_batch, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                rows.extend(result["rows"])
                errors.extend(result["errors"])
                if not nominal_meta:
                    nominal_meta = dict(result["nominal_meta"])

    if args.repair_episode is not None:
        episode, seed = episode_items[0]
        result_dir = case_root / "_global_repair_results" / args.repair_run_id
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"ep{episode:04d}.json"
        result_tmp = result_dir / f"ep{episode:04d}.json.tmp"
        result_tmp.write_text(json.dumps({
            "run_id": args.repair_run_id, "episode": episode, "sumo_seed": seed,
            "rows": rows, "errors": errors, "nominal_meta": nominal_meta,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(result_tmp, result_path)
        if errors or len(rows) != 1:
            raise RuntimeError(f"Global repair episode failed: {args.case_name} ep{episode:04d}; see {result_path}")
        _console(f"global-repair episode complete scenario={args.case_name} episode={episode}")
        return

    rows.sort(key=lambda row: int(row["episode"]))
    errors.sort(key=lambda row: int(row["episode"]))
    if args.repair_manifest:
        merged_rows = _merge_rows_by_episode(
            existing_summary_rows, rows, rerun_episodes
        )
        existing_error_rows = read_csv_rows(case_root / "errors.csv")
        merged_errors = _merge_rows_by_episode(
            existing_error_rows, errors, rerun_episodes
        )
    else:
        merged_rows = rows
        merged_errors = errors
    write_csv(case_root / "episode_summary.csv", merged_rows)
    errors_path = case_root / "errors.csv"
    if merged_errors:
        write_csv(errors_path, merged_errors)
    elif errors_path.exists():
        errors_path.unlink()
    if not args.no_window_output:
        merge_episode_csvs(
            output_root=case_root,
            include_vehicle_10s=bool(args.save_vehicle_10s),
            network_300s_only=bool(args.network_300s_only),
        )
    if args.save_vehicle_state and not args.network_300s_only:
        consolidate_vehicle_state_outputs(
            case_root,
            replace_episodes=rerun_episodes if args.repair_manifest else None,
        )

    seed_metadata = {
        "global_seed": int(parent_cfg.seed),
        "sumo_seed_mode": str(parent_cfg.env.sumo_seed_mode),
        "unique_sumo_seed_count": int(len(set(sumo_seeds))),
        "episode_seed_map": [
            {"episode": episode, "sumo_seed": seed}
            for episode, seed in all_episode_items
        ],
    }
    (case_root / "sumo_seeds.json").write_text(
        json.dumps(seed_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    scenario_wall_time_s = time.time() - started
    metadata = {
        "controller": args.controller,
        "short_name": (
            "TW-MP"
            if args.controller == "truck_weighted_max_pressure"
            else ""
        ),
        "truck_priority_weight": (
            2.0
            if args.controller == "truck_weighted_max_pressure"
            else ""
        ),
        "sedan_priority_weight": (
            1.0
            if args.controller == "truck_weighted_max_pressure"
            else ""
        ),
        "uses_rl_agent": args.controller == "rl",
        "uses_checkpoint": args.controller == "rl",
        "uses_reward_for_control": args.controller == "rl",
        "uses_nox_for_control": args.controller == "rl",
        "case_name": args.case_name,
        "reward_mode": args.reward_mode,
        "config": resolve_path(args.config) if args.config else "",
        "checkpoint": (
            resolve_path(args.checkpoint)
            if args.controller == "rl" and args.checkpoint
            else ""
        ),
        "sumocfg": parent_cfg.env.sumo_cfg,
        "runtime_sumocfg": runtime_sumocfg,
        "net_xml": parent_cfg.env.net_xml,
        "add_xml": parent_cfg.env.add_xml,
        "groups_json": parent_cfg.env.intersection_groups_json,
        "emission_factor_csv": parent_cfg.env.emission_factor_csv,
        "thresholds_json": parent_cfg.emission_risk.thresholds_json,
        "threshold_override": bool(args.thresholds_json),
        "detector_map": (
            resolve_path(args.detector_map)
            if args.detector_map
            else ""
        ),
        "exit_e1_ids": list(exit_e1_ids),
        "exit_e1_id_prefix": args.evaluation_exit_e1_prefix,
        "e2_id_prefix": args.evaluation_e2_prefix,
        "window_output_enabled": not bool(args.no_window_output),
        "network_300s_only": bool(args.network_300s_only),
        "window_episode_dir_name": (
            "windows_epsiode"
            if args.network_300s_only
            else "windows_episode"
        ),
        "emission_output_enabled": bool(parent_cfg.log.save_emission_step),
        "vehicle_state_output_enabled": bool(
            args.save_vehicle_state and not args.network_300s_only
        ),
        "throughput_source": (
            "perimeter_exit_e1" if not args.no_window_output else ""
        ),
        "evaluation_detectors": runtime_detector_meta,
        "decision_interval_s": int(args.decision_seconds),
        "aggregate_window_s": (
            int(args.window_seconds) if not args.no_window_output else ""
        ),
        "analysis_start_s": (
            int(args.analysis_start) if not args.no_window_output else ""
        ),
        "analysis_end_s": (
            int(args.analysis_end) if not args.no_window_output else ""
        ),
        "episode_duration_s": int(parent_cfg.env.episode_duration),
        "risk_service_log": {
            "enabled": bool(args.save_risk_service_log),
            "start_s": (
                float(args.risk_log_start)
                if args.save_risk_service_log
                else ""
            ),
            "end_s": (
                float(args.risk_log_end)
                if args.save_risk_service_log
                else ""
            ),
            "compression": (
                str(args.risk_log_compression)
                if args.save_risk_service_log
                else ""
            ),
            "fields": (
                list(RISK_SERVICE_LANE_FIELDS)
                if args.save_risk_service_log
                else []
            ),
            "rho": (
                float(parent_cfg.emission_risk.rho)
                if args.save_risk_service_log
                else ""
            ),
            "green_service_source": (
                "logged" if args.save_risk_service_log else ""
            ),
        },
        "workers": worker_count,
        "ports": ports,
        "wall_time_s": scenario_wall_time_s,
        "successful_episode_count": len(merged_rows),
        "failed_episode_count": len(merged_errors),
        "repair_manifest": (
            resolve_path(args.repair_manifest) if args.repair_manifest else ""
        ),
        "repair_selected_episode_count": len(rerun_episodes),
        "repair_skipped_complete_count": len(skipped_complete),
        "nominal_schedule": nominal_meta,
    }
    (case_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _console(
        "Evaluation completed: "
        f"scenario={args.case_name} episodes={len(merged_rows)}/{args.eval_episodes} "
        f"total_time={scenario_wall_time_s:.1f}s "
        f"({_format_elapsed(scenario_wall_time_s)})"
    )

    if errors:
        raise RuntimeError(
            f"{len(errors)} episode(s) failed; inspect {case_root / 'errors.csv'}"
        )
    if args.verbose_output:
        _console(
            f"output={case_root}"
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()
