# -*- coding: utf-8 -*-
"""Deterministic independent evaluation for PressLight checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from presslight_config import discover_project_root, get_presslight_config  # noqa: E402

sys.path.insert(0, str(SCRIPT_DIR.parent))
PROJECT_ROOT = discover_project_root(SCRIPT_DIR)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import build_sumo_seed_list, get_config  # noqa: E402
from env import SumoEnv  # noqa: E402
from logger import TrainingLogger  # noqa: E402
from obs_reward import ObsRewardBuilder  # noqa: E402
try:
    from vehicle_state import consolidate_vehicle_state_outputs  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - older project roots may not provide it.
    consolidate_vehicle_state_outputs = None
from presslight_agent import (  # noqa: E402
    PressLightAgentManager,
    q_step_rows,
    set_global_seed,
)
from presslight_network import (  # noqa: E402
    build_presslight_network_spec,
    canonical_actions_to_env_actions,
    parse_base_network,
    prepare_presslight_runtime_sumo_cfg,
    save_network_spec,
)
from presslight_observation import (  # noqa: E402
    PressLightObservationBuilder,
    observation_step_rows,
)
from presslight_reward import PressLightRewardBuilder, pressure_step_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one PressLight checkpoint")
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--presslight-config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--trusted-checkpoint",
        action="store_true",
        help="Allow a trusted legacy/full training checkpoint (uses weights_only=False)",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--result-prefix",
        default="",
        help="Optional filename prefix for the main evaluation CSV/JSON outputs",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--port", type=int, default=8813)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--port-stride", type=int, default=10)
    parser.add_argument("--use-gui", action="store_true")
    parser.add_argument("--no-detail", action="store_true")
    parser.add_argument("--no-fcd", action="store_true")
    parser.add_argument(
        "--save-vehicle-state",
        action="store_true",
        help=(
            "Save per-second per-vehicle Parquet output using the shared "
            "vehicle_state mechanism from the selected MGMQ project root"
        ),
    )
    return parser.parse_args()


def _resolve(path: str, *, required: bool = False) -> str:
    if not path:
        return ""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if required and not resolved.exists():
            raise FileNotFoundError(str(resolved))
        return str(resolved)

    candidates = [
        (Path.cwd() / candidate).resolve(),
        (SCRIPT_DIR / candidate).resolve(),
        (PROJECT_ROOT / candidate).resolve(),
    ]
    if required:
        for resolved in candidates:
            if resolved.exists():
                return str(resolved)
        raise FileNotFoundError(
            f"Unable to resolve required path {path!r}; "
            f"checked={[str(item) for item in candidates]}"
        )
    return str(candidates[-1])


def _resolve_emission_factor_csv(path: str) -> str:
    """Resolve the shared emission-factor table across both MGMQ projects."""
    if not path:
        raise ValueError("base_cfg.env.emission_factor_csv is empty")

    configured = Path(path).expanduser()
    candidates = (
        [configured]
        if configured.is_absolute()
        else [
            PROJECT_ROOT / configured,
            PROJECT_ROOT.parent / configured,
            SCRIPT_DIR.parent / configured,
        ]
    )
    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if resolved.is_file():
            return str(resolved)
    raise FileNotFoundError(
        "Emission factor CSV was not found. "
        f"configured={path!r}; checked={checked}"
    )


def _write_csv(path: str, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _mean(values: Iterable[float]) -> float:
    data = [float(x) for x in values]
    return float(np.mean(data)) if data else 0.0


def _tripinfo_extra(path: str) -> dict[str, float]:
    waits: list[float] = []
    waiting_counts: list[float] = []
    stops: list[float] = []
    if not path or not os.path.exists(path):
        return {
            "avg_waiting_time_s": 0.0,
            "avg_waiting_count": 0.0,
            "avg_stop_time_s": 0.0,
        }
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return {
            "avg_waiting_time_s": 0.0,
            "avg_waiting_count": 0.0,
            "avg_stop_time_s": 0.0,
        }
    for trip in root.findall("tripinfo"):
        if str(trip.get("arrival", "-1")) == "-1":
            continue
        waits.append(float(trip.get("waitingTime", 0.0)))
        waiting_counts.append(float(trip.get("waitingCount", 0.0)))
        stops.append(float(trip.get("stopTime", 0.0)))
    return {
        "avg_waiting_time_s": _mean(waits),
        "avg_waiting_count": _mean(waiting_counts),
        "avg_stop_time_s": _mean(stops),
    }


def _vehicle_emission_metrics(emission_result: Any) -> dict[str, float]:
    zero = {
        "vehicle_nox_total_mg": 0.0,
        "vehicle_nox_mean_mg": 0.0,
        "vehicle_nox_p95_mg": 0.0,
        "vehicle_emission_vehicle_count": 0.0,
    }
    frame = getattr(emission_result, "vehicle_df", None)
    if frame is None or not hasattr(frame, "columns") or len(frame) == 0:
        return zero
    subset = frame
    if "pollutant" in frame.columns:
        subset = frame[frame["pollutant"].astype(str).str.lower() == "nox"]
    mass_column = next(
        (
            name
            for name in ("total_NOx_mg", "total_nox_mg", "total_emission_mg")
            if name in subset.columns
        ),
        None,
    )
    if mass_column is None or len(subset) == 0:
        return zero
    values = np.asarray(subset[mass_column], dtype=np.float64)
    vehicle_count = (
        float(subset["veh_id"].astype(str).nunique())
        if "veh_id" in subset.columns
        else float(len(values))
    )
    return {
        "vehicle_nox_total_mg": float(values.sum()),
        "vehicle_nox_mean_mg": float(values.mean()) if values.size else 0.0,
        "vehicle_nox_p95_mg": float(np.percentile(values, 95)) if values.size else 0.0,
        "vehicle_emission_vehicle_count": vehicle_count,
    }


def _summary(rows: list[dict[str, Any]], checkpoint: str, seeds: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "checkpoint": checkpoint,
        "eval_episodes": len(rows),
        "sumo_seeds": seeds,
    }
    excluded = {"eval_episode", "sumo_seed"}
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in excluded and isinstance(value, (int, float, np.number))
        }
    )
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
        summary[key] = float(values.mean()) if values.size else 0.0
        summary[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return summary


def _load_agent(
    pl_cfg: Any,
    network_spec: Any,
    checkpoint: str,
    trusted_checkpoint: bool,
) -> tuple[PressLightAgentManager, dict[str, Any]]:
    agent = PressLightAgentManager(pl_cfg, network_spec, device=pl_cfg.device)
    if trusted_checkpoint:
        payload = agent.load_checkpoint(checkpoint, load_optimizer=False)
    else:
        payload = agent.load_evaluation_checkpoint(checkpoint)
    agent.set_training(False)
    return agent, payload


def _run_eval_episode(
    *,
    episode: int,
    seed: int,
    eval_episodes: int,
    worker_id: int,
    port: int,
    use_gui: bool,
    output_dir: str,
    base_cfg: Any,
    pl_cfg: Any,
    net_info: Any,
    network_spec: Any,
    observation_builder: PressLightObservationBuilder,
    mechanism_builder: ObsRewardBuilder,
    agent: PressLightAgentManager,
    emission_logger: TrainingLogger,
) -> dict[str, Any]:
    started_at = time.time()
    env: Optional[SumoEnv] = None
    rewards: list[float] = []
    pressures: list[float] = []
    signed_pressures: list[float] = []
    sum_abs_pressures: list[float] = []
    switches: list[float] = []
    beta_valid_values: list[float] = []
    link_nox_values: list[float] = []
    state_rows: list[dict[str, Any]] = []
    pressure_rows: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    emission_result: Any = None
    traffic_stats: Any = None
    tripinfo_path = ""
    reward_builder = PressLightRewardBuilder(pl_cfg, network_spec)
    emission_logger.begin_episode(
        episode,
        record_lane_step=True,
        record_phase_step=False,
        record_q_step=False,
    )
    try:
        env = SumoEnv(base_cfg.env, net_info, port=int(port), use_gui=bool(use_gui))
        if getattr(env, "emission_recorder", None) is None:
            raise RuntimeError(
                "PressLight emission recorder was not initialized; "
                f"emission_factor_csv={base_cfg.env.emission_factor_csv!r}"
            )
        env.start(episode_id=episode, seed=int(seed), control_tls=True)
        observations = observation_builder.build_all(env)
        for step in range(int(base_cfg.env.steps_per_episode)):
            actions, action_infos = agent.act(
                observations, epsilon=0.0, deterministic=True
            )
            switches.extend(
                1.0 if int(actions[tl_id]) != int(obs.current_phase) else 0.0
                for tl_id, obs in observations.items()
            )
            next_raw = env.step(
                canonical_actions_to_env_actions(actions, network_spec),
                decision_step=step,
            )
            step_emission = getattr(env, "_last_step_emission", None)
            link_nox_values.extend(
                emission_logger.compute_step_link_emission_values(
                    step_emission, pollutant="NOx"
                )
            )
            if pl_cfg.evaluation.save_detail:
                emission_logger.log_step_emission(episode, step, step_emission)

            next_observations = observation_builder.build_all(env)
            mechanism_observations = mechanism_builder.build_observations(next_raw)
            emission_logger.log_step(
                episode,
                step,
                next_raw,
                mechanism_observations,
                actions,
                {},
                {},
            )
            reward_results = reward_builder.compute_all(env, next_observations)
            for result in reward_results.values():
                rewards.append(float(result.reward))
                pressures.append(float(result.intersection_pressure))
                signed_pressures.append(float(result.signed_pressure_sum))
                sum_abs_pressures.append(float(result.sum_abs_movement_pressure))
                beta_valid_values.append(1.0 if result.beta_valid else 0.0)
            if pl_cfg.evaluation.save_detail:
                state_rows.extend(
                    observation_step_rows(episode, step, next_observations, network_spec)
                )
                pressure_rows.extend(pressure_step_rows(episode, step, reward_results))
                q_rows.extend(q_step_rows(episode, step, action_infos))
            observations = next_observations

        tripinfo_path = str(getattr(env, "_tripinfo_path", ""))
        traffic_stats = env.close()
        recorder = getattr(env, "emission_recorder", None)
        if recorder is not None:
            emission_result = recorder.finalize_episode()
        env = None
    finally:
        if env is not None:
            try:
                env.close(parse_tripinfo=False)
            except Exception:
                pass

    if pl_cfg.evaluation.save_detail:
        detail_dir = os.path.join(output_dir, "physical", f"episode_{episode:04d}")
        _write_csv(os.path.join(detail_dir, "presslight_state_step.csv"), state_rows)
        _write_csv(os.path.join(detail_dir, "presslight_pressure_step.csv"), pressure_rows)
        _write_csv(os.path.join(detail_dir, "presslight_q_step.csv"), q_rows)
        emission_logger.flush_episode_emission(episode)
        emission_logger.flush_episode_physical(episode)
    emission_logger.write_episode_emission_result(
        episode,
        emission_result,
        save_step_df=True,
        save_edge_step_raw=False,
        save_lane_step=True,
        save_vehicle_step=bool(pl_cfg.evaluation.save_vehicle_emission_step),
    )
    vehicle_emission_path = os.path.join(
        output_dir,
        "emission",
        f"episode_{episode:04d}",
        "vehicle_emission.csv",
    )
    if not os.path.isfile(vehicle_emission_path):
        raise RuntimeError(
            "PressLight vehicle emission output was not created: "
            f"{vehicle_emission_path}"
        )

    row: dict[str, Any] = {
        "eval_episode": int(episode),
        "sumo_seed": int(seed),
        "reward_mean": _mean(rewards),
        "reward_sum": float(sum(rewards)),
        "pressure_mean": _mean(pressures),
        "pressure_max": max(pressures) if pressures else 0.0,
        "signed_pressure_mean": _mean(signed_pressures),
        "sum_abs_movement_pressure_mean": _mean(sum_abs_pressures),
        "phase_switch_rate": _mean(switches),
        "beta_valid_rate": _mean(beta_valid_values),
        "avg_delay_s": float(getattr(traffic_stats, "avg_delay_s", 0.0) or 0.0),
        "avg_travel_time_s": float(
            getattr(traffic_stats, "avg_travel_time_s", 0.0) or 0.0
        ),
        "total_arrived": int(getattr(traffic_stats, "total_arrived", 0) or 0),
        "completion_rate": float(
            getattr(traffic_stats, "completion_rate", 0.0) or 0.0
        ),
        "throughput_veh_per_h": float(
            int(getattr(traffic_stats, "total_arrived", 0) or 0)
            * 3600.0
            / max(float(base_cfg.env.episode_duration), 1.0)
        ),
        "link_nox_mean_mg_per_step": _mean(link_nox_values),
        "link_nox_p95_mg_per_step": float(np.percentile(link_nox_values, 95))
        if link_nox_values
        else 0.0,
        "wall_time_s": float(time.time() - started_at),
    }
    row.update(_tripinfo_extra(tripinfo_path))
    row.update(_vehicle_emission_metrics(emission_result))
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [eval] "
        f"worker={worker_id} port={port} episode={episode}/{eval_episodes} "
        f"seed={seed} delay={row['avg_delay_s']:.2f}s "
        f"travel={row['avg_travel_time_s']:.2f}s "
        f"completion={row['completion_rate']:.4f} "
        f"pressure={row['pressure_mean']:.4f} "
        f"NOx={row['vehicle_nox_total_mg']:.2f}mg "
        f"elapsed={row['wall_time_s']:.2f}s",
        flush=True,
    )
    return row


def _split_episode_batches(
    episodes: list[tuple[int, int]],
    max_workers: int,
) -> list[list[tuple[int, int]]]:
    worker_count = max(1, min(int(max_workers), len(episodes)))
    batches: list[list[tuple[int, int]]] = [[] for _ in range(worker_count)]
    for index, item in enumerate(episodes):
        batches[index % worker_count].append(item)
    return [batch for batch in batches if batch]


def _run_episode_batch_worker(payload: dict[str, Any]) -> dict[str, Any]:
    worker_id = int(payload["worker_id"])
    port = int(payload["port"])
    pl_cfg = payload["pl_cfg"]
    network_spec = payload["network_spec"]
    set_global_seed(int(pl_cfg.evaluation.seed) + worker_id)
    observation_builder = PressLightObservationBuilder(pl_cfg, network_spec)
    mechanism_builder = ObsRewardBuilder(payload["base_cfg"], payload["net_info"])
    agent, _checkpoint_payload = _load_agent(
        pl_cfg,
        network_spec,
        str(payload["checkpoint"]),
        bool(payload["trusted_checkpoint"]),
    )
    emission_logger = TrainingLogger(
        payload["base_cfg"],
        payload["net_info"],
        model_meta=agent.model_meta(),
        write_metadata=False,
    )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for episode, seed in payload["episodes"]:
        episode_started_at = time.time()
        try:
            rows.append(
                _run_eval_episode(
                    episode=int(episode),
                    seed=int(seed),
                    eval_episodes=int(payload["eval_episodes"]),
                    worker_id=worker_id,
                    port=port,
                    use_gui=bool(payload["use_gui"]),
                    output_dir=str(payload["output_dir"]),
                    base_cfg=payload["base_cfg"],
                    pl_cfg=pl_cfg,
                    net_info=payload["net_info"],
                    network_spec=network_spec,
                    observation_builder=observation_builder,
                    mechanism_builder=mechanism_builder,
                    agent=agent,
                    emission_logger=emission_logger,
                )
            )
        except Exception as error:
            elapsed_s = time.time() - episode_started_at
            print(
                f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [eval error] "
                f"worker={worker_id} port={port} "
                f"episode={episode}/{payload['eval_episodes']} seed={seed} "
                f"elapsed={elapsed_s:.2f}s error={error!r}",
                file=sys.stderr,
                flush=True,
            )
            errors.append(
                {
                    "worker_id": worker_id,
                    "port": port,
                    "eval_episode": int(episode),
                    "sumo_seed": int(seed),
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            )
    return {
        "worker_id": worker_id,
        "port": port,
        "rows": rows,
        "errors": errors,
    }


def main() -> None:
    mp.freeze_support()
    args = parse_args()
    if not 1 <= int(args.port) <= 65535:
        raise ValueError(f"port must be in [1, 65535]; actual={args.port!r}")
    if int(args.max_workers) <= 0:
        raise ValueError(f"max-workers must be > 0; actual={args.max_workers!r}")
    if int(args.port_stride) <= 0:
        raise ValueError(f"port-stride must be > 0; actual={args.port_stride!r}")
    last_port = int(args.port) + (int(args.max_workers) - 1) * int(args.port_stride)
    if last_port > 65535:
        raise ValueError(
            f"worker ports exceed 65535; base={args.port}, "
            f"workers={args.max_workers}, stride={args.port_stride}"
        )
    result_prefix = str(args.result_prefix).strip()
    if result_prefix and (
        Path(result_prefix).name != result_prefix
        or result_prefix in {".", ".."}
    ):
        raise ValueError(
            "result-prefix must be a filename prefix without directory separators; "
            f"actual={args.result_prefix!r}"
        )
    filename_prefix = f"{result_prefix}_" if result_prefix else ""
    network_spec_filename = (
        f"{result_prefix}_network_spec.json"
        if result_prefix
        else "presslight_network_spec.json"
    )
    resolved_config_filename = (
        f"{result_prefix}_config_resolved.json"
        if result_prefix
        else "presslight_config_resolved.json"
    )
    base_path = _resolve(args.base_config, required=True) if args.base_config else None
    base_cfg = get_config(base_path)
    pl_path = (
        _resolve(args.presslight_config, required=True) if args.presslight_config else None
    )
    pl_cfg = get_presslight_config(pl_path, base_cfg)
    if args.device:
        pl_cfg.device = str(args.device)
    if args.episodes is not None:
        pl_cfg.evaluation.episodes = int(args.episodes)
    if args.seed is not None:
        pl_cfg.evaluation.seed = int(args.seed)
    if args.no_detail:
        pl_cfg.evaluation.save_detail = False
    if args.no_fcd:
        pl_cfg.evaluation.save_fcd = False

    for attr, required in (
        ("sumo_cfg", True),
        ("net_xml", True),
        ("base_add_xml", True),
        ("intersection_groups_json", False),
        ("presslight_add_xml", True),
        ("detector_map_json", False),
    ):
        setattr(pl_cfg.paths, attr, _resolve(getattr(pl_cfg.paths, attr), required=required))
    pl_cfg.paths.output_root = _resolve(pl_cfg.paths.output_root, required=False)
    if pl_cfg.grouping.manual_groups_json:
        pl_cfg.grouping.manual_groups_json = _resolve(
            pl_cfg.grouping.manual_groups_json, required=True
        )
    checkpoint = _resolve(args.checkpoint, required=True)

    # Scenario thresholds take precedence over the shared base configuration.
    if pl_cfg.paths.emission_thresholds_json:
        threshold_path = Path(pl_cfg.paths.emission_thresholds_json).expanduser()
        if not threshold_path.is_absolute():
            threshold_path = PROJECT_ROOT / threshold_path
        threshold_path = threshold_path.resolve()
        if not threshold_path.is_file():
            raise FileNotFoundError(f"PressLight NOx thresholds not found: {threshold_path}")
        base_cfg.emission_risk.thresholds_json = str(threshold_path)
        print(f"[eval config] thresholds_json={threshold_path}", flush=True)

    # Validate CLI overrides before any output, network parsing, or SUMO setup.
    pl_cfg.validate(base_cfg)
    output_dir = _resolve(args.output_dir, required=False) if args.output_dir else pl_cfg.eval_run_dir
    os.makedirs(output_dir, exist_ok=True)

    set_global_seed(pl_cfg.evaluation.seed)
    net_info = parse_base_network(
        pl_cfg.paths.net_xml,
        pl_cfg.paths.base_add_xml,
        pl_cfg.paths.intersection_groups_json,
    )
    network_spec = build_presslight_network_spec(net_info, pl_cfg)
    save_network_spec(
        network_spec,
        os.path.join(output_dir, network_spec_filename),
    )
    base_cfg.env.sumo_cfg = prepare_presslight_runtime_sumo_cfg(
        pl_cfg.paths.sumo_cfg,
        pl_cfg.paths.presslight_add_xml,
        os.path.join(output_dir, "sumo_template"),
    )
    base_cfg.seed = int(pl_cfg.evaluation.seed)
    base_cfg.env.net_xml = pl_cfg.paths.net_xml
    base_cfg.env.add_xml = pl_cfg.paths.base_add_xml
    base_cfg.env.intersection_groups_json = pl_cfg.paths.intersection_groups_json
    base_cfg.env.tripinfo_dir = os.path.join(output_dir, "tripinfo")
    base_cfg.env.save_fcd_output = bool(pl_cfg.evaluation.save_fcd)
    base_cfg.env.save_sumo_aux_outputs = False
    if args.save_vehicle_state and consolidate_vehicle_state_outputs is None:
        raise RuntimeError(
            "vehicle_state output was requested, but the selected MGMQ_PROJECT_ROOT "
            f"does not provide vehicle_state.py: {PROJECT_ROOT}"
        )
    if args.save_vehicle_state and not hasattr(base_cfg.env, "save_vehicle_state_output"):
        raise RuntimeError(
            "vehicle_state output was requested, but the selected EnvConfig does not "
            f"support save_vehicle_state_output: {PROJECT_ROOT}"
        )
    if hasattr(base_cfg.env, "save_vehicle_state_output"):
        base_cfg.env.save_vehicle_state_output = bool(args.save_vehicle_state)
    if hasattr(base_cfg.env, "save_vehicle_second_output"):
        base_cfg.env.save_vehicle_second_output = False
    if args.save_vehicle_state:
        base_cfg.env.vehicle_state_row_group_rows = int(
            getattr(base_cfg.env, "vehicle_state_row_group_rows", 100_000)
        )
        base_cfg.env.vehicle_state_compression = str(
            getattr(base_cfg.env, "vehicle_state_compression", "zstd")
        )
        base_cfg.env.vehicle_state_compression_level = int(
            getattr(base_cfg.env, "vehicle_state_compression_level", 3)
        )
    # Mechanism analysis requires the actual second-by-second green service.
    # This diagnostic output does not alter PressLight actions or SUMO control.
    base_cfg.env.save_tls_phase_outputs = True
    base_cfg.env.emission_factor_csv = _resolve_emission_factor_csv(
        base_cfg.env.emission_factor_csv
    )
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [eval config] "
        f"emission_factor_csv={base_cfg.env.emission_factor_csv}",
        flush=True,
    )
    # Reuse the shared emission and lane-level physical logging utilities.
    base_cfg.log.log_root = os.path.dirname(output_dir)
    base_cfg.log.run_id = os.path.basename(output_dir)
    # Fig. 9 requires the unified lane_step.csv physical mechanism log.
    base_cfg.log.save_step_physical = True
    base_cfg.validate()
    pl_cfg.validate(base_cfg)
    pl_cfg.save_json(
        os.path.join(output_dir, resolved_config_filename)
    )

    if args.trusted_checkpoint:
        print(
            "[security warning] Loading a trusted training checkpoint with "
            "weights_only=False; only use files generated by this project from a trusted source.",
            flush=True,
        )
    configured_device = pl_cfg.device
    pl_cfg.device = "cpu"
    meta_agent, checkpoint_payload = _load_agent(
        pl_cfg, network_spec, checkpoint, bool(args.trusted_checkpoint)
    )
    pl_cfg.device = configured_device
    emission_logger = TrainingLogger(
        base_cfg,
        net_info,
        model_meta=meta_agent.model_meta(),
        write_metadata=False,
    )
    emission_logger.save_link_map()
    del meta_agent

    eval_episodes = int(pl_cfg.evaluation.episodes)
    seeds = build_sumo_seed_list(
        global_seed=int(pl_cfg.evaluation.seed),
        total_episodes=eval_episodes,
        mode=str(base_cfg.env.sumo_seed_mode),
    )
    episode_items = [
        (episode, int(seeds[episode - 1]))
        for episode in range(1, eval_episodes + 1)
    ]
    batches = _split_episode_batches(episode_items, int(args.max_workers))
    worker_count = len(batches)
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [eval start] "
        f"checkpoint_episode={checkpoint_payload.get('episode')} "
        f"episodes={eval_episodes} deterministic=True epsilon=0 "
        f"workers={worker_count}",
        flush=True,
    )

    payloads = [
        {
            "worker_id": worker_id,
            "port": int(args.port) + worker_id * int(args.port_stride),
            "episodes": batch,
            "eval_episodes": eval_episodes,
            "use_gui": bool(args.use_gui),
            "output_dir": output_dir,
            "base_cfg": base_cfg,
            "pl_cfg": pl_cfg,
            "net_info": net_info,
            "network_spec": network_spec,
            "checkpoint": checkpoint,
            "trusted_checkpoint": bool(args.trusted_checkpoint),
        }
        for worker_id, batch in enumerate(batches)
    ]
    episode_rows: list[dict[str, Any]] = []
    worker_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if worker_count == 1:
        results = [_run_episode_batch_worker(payloads[0])]
    else:
        results = []
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(_run_episode_batch_worker, payload)
                for payload in payloads
            ]
            for future in as_completed(futures):
                results.append(future.result())
    for result in results:
        rows = list(result["rows"])
        result_errors = list(result["errors"])
        episode_rows.extend(rows)
        errors.extend(result_errors)
        worker_rows.append(
            {
                "worker_id": int(result["worker_id"]),
                "port": int(result["port"]),
                "episodes_completed": len(rows),
                "errors": len(result_errors),
            }
        )
    episode_rows.sort(key=lambda row: int(row["eval_episode"]))
    worker_rows.sort(key=lambda row: int(row["worker_id"]))
    _write_csv(
        os.path.join(output_dir, f"{filename_prefix}eval_worker_log.csv"),
        worker_rows,
    )
    if errors:
        error_path = os.path.join(output_dir, f"{filename_prefix}eval_errors.json")
        _save_json(error_path, errors)
        raise RuntimeError(
            f"{len(errors)} evaluation episodes failed; see {error_path}"
        )

    episode_log_path = os.path.join(
        output_dir, f"{filename_prefix}eval_episode_log.csv"
    )
    summary_path = os.path.join(output_dir, f"{filename_prefix}eval_summary.json")
    _write_csv(episode_log_path, episode_rows)
    summary = _summary(episode_rows, checkpoint, [int(x) for x in seeds])
    summary.update(
        {
            "experiment_name": pl_cfg.experiment_name,
            "checkpoint_episode": int(checkpoint_payload.get("episode", -1)),
            "deterministic": True,
            "epsilon": 0.0,
            "vehicle_state_output_enabled": bool(args.save_vehicle_state),
            "network_spec_digest": network_spec.digest,
            "parallel": worker_count > 1,
            "max_workers": worker_count,
            "worker_ports": [int(row["port"]) for row in worker_rows],
        }
    )
    _save_json(summary_path, summary)
    if args.save_vehicle_state:
        assert consolidate_vehicle_state_outputs is not None
        consolidate_vehicle_state_outputs(output_dir)
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [eval done] "
        f"episode_log={episode_log_path} summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] "
            f"[eval error] {error!r}\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        raise
