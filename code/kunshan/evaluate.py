
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from agent import MGMQAgentManager, set_global_seed
from config import MasterConfig, build_sumo_seed_list, get_config
from env import SumoEnv, StepRawObs
from logger import TrainingLogger
from network_parser import parse_network
from obs_reward import ObsRewardBuilder
from profiler import StageProfiler
from vehicle_state import consolidate_vehicle_state_outputs

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

EVAL_EPISODES = 20
EVAL_SEED = 19
EVAL_LOG_ROOT = "results/evaluation/kunshan"
EVAL_PORT = 8813
EVAL_USE_GUI = False

EVAL_PARALLEL_EPISODES = True
EVAL_MAX_WORKERS = 3
EVAL_PORT_STRIDE = 10
EVAL_FORCE_TRACI = True
EVAL_PARALLEL_DEVICE = "cpu"

EVAL_PROFILE = False
EVAL_PROFILE_SYNC_CUDA = False

EVAL_SAVE_DETAIL = False
EVAL_SAVE_VEHICLE_EMISSION = True
EVAL_SAVE_LANE_NOX = True
EVAL_SAVE_VEHICLE_STATE = True

EVAL_SAVE_FCD = False
EVAL_FCD_ACCELERATION = True
EVAL_FCD_PERIOD_S = 1.0

EVAL_CASES = [
    {
        "name": "FLORE-TO_w100",
        "config": "configs/kunshan/spatial_scale/flore_to_w100.yaml",
        "checkpoint": "models/kunshan_spatial/kunshan_w100_to.pt",
        "run_id": "FLORE-TO_w100",
    },
    {
        "name": "FLORE_w100",
        "config": "configs/kunshan/spatial_scale/flore_w100.yaml",
        "checkpoint": "models/kunshan_spatial/kunshan_w100_mo.pt",
        "run_id": "FLORE_w100",
    },
    {
        "name": "FLORE-TO_w300",
        "config": "configs/kunshan/spatial_scale/flore_to_w300.yaml",
        "checkpoint": "models/kunshan_spatial/kunshan_w300_to.pt",
        "run_id": "FLORE-TO_w300",
    },
    {
        "name": "FLORE_w300",
        "config": "configs/kunshan/flore.yaml",
        "checkpoint": "models/kunshan_spatial/kunshan_w300_mo.pt",
        "run_id": "FLORE_w300",
    },
    {
        "name": "FLORE-TS_w300",
        "config": "configs/kunshan/evaluation/flore_ts.yaml",
        "checkpoint": "models/kunshan_spatial/kunshan_w300_m0.pt",
        "run_id": "FLORE-TS_w300",
    },
    {
        "name": "FLORE_NOx-only_w300",
        "config": "configs/kunshan/spatial_scale/flore_nox_only_w300.yaml",
        "checkpoint": "models/kunshan_spatial/kunshan_w300_nox_only.pt",
        "run_id": "FLORE_NOx-only_w300",
    },
    {
        "name": "FLORE_NOx-dominant_w300",
        "config": "configs/kunshan/spatial_scale/flore_nox_dominant_w300.yaml",
        "checkpoint": "models/kunshan_spatial/kunshan_w300_nox_dominant.pt",
        "run_id": "FLORE_NOx-dominant_w300",
    },
    {
        "name": "FLORE-TO_w500",
        "config": "configs/kunshan/spatial_scale/flore_to_w500.yaml",
        "checkpoint": "models/kunshan_spatial/kunshan_w500_to.pt",
        "run_id": "FLORE-TO_w500",
    },
    {
        "name": "FLORE_w500",
        "config": "configs/kunshan/spatial_scale/flore_w500.yaml",
        "checkpoint": "models/kunshan_spatial/kunshan_w500_mo.pt",
        "run_id": "FLORE_w500",
    },
]

# Canonical display names; legacy case tokens remain accepted for existing commands.
EVAL_CASE_ALIASES = {'kunshan_w100_to': 'FLORE-TO_w100', 'kunshan_w100_mo': 'FLORE_w100', 'kunshan_w300_to': 'FLORE-TO_w300', 'kunshan_w300_mo': 'FLORE_w300', 'kunshan_w300_m0': 'FLORE-TS_w300', 'kunshan_w500_to': 'FLORE-TO_w500', 'kunshan_w500_mo': 'FLORE_w500', 'kunshan_w300_nox_only': 'FLORE_NOx-only_w300', 'kunshan_w300_nox_dominant': 'FLORE_NOx-dominant_w300'}

EVAL_PARAMETER_SOURCES_JSON = "data/kunshan/intersection_parameter_sources.json"
STRUCTURE_PARAMETER_SOURCES = {
    "structure_01": "standard",
    "structure_02": "standard",
    "structure_03": "hetero_ew_shared_ls",
    "structure_04": "hetero_ew_shared_ls",
    "structure_05": "hetero_ew_shared_ls",
}

def resolve_path(path: Optional[str], *, must_exist: bool = False) -> Optional[str]:
    if path is None or str(path).strip() == "":
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        project_candidate = PROJECT_ROOT / p
        script_candidate = SCRIPT_DIR / p
        if project_candidate.exists():
            p = project_candidate
        elif script_candidate.exists():
            p = script_candidate
        else:
            p = project_candidate
    p = p.resolve()
    if must_exist and not p.exists():
        raise FileNotFoundError(str(p))
    return str(p)

def validate_eval_assets(cases: list[dict[str, str]]) -> None:
    missing: list[str] = []
    for case in cases:
        for key in ("config", "checkpoint"):
            path = Path(resolve_path(str(case[key])) or "")
            if not path.is_file():
                missing.append(f"{case['name']} {key}: {path}")
    parameter_sources = Path(resolve_path(EVAL_PARAMETER_SOURCES_JSON) or "")
    if not parameter_sources.is_file():
        missing.append(f"parameter sources: {parameter_sources}")
    if missing:
        raise FileNotFoundError("Missing evaluation assets:\n" + "\n".join(missing))


def _load_checkpoint_payload(path: str, map_location: Any = "cpu") -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload must be a dict: {path}")
    return payload


def _checkpoint_fingerprint(path: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _cfg_value(cfg_dict: dict[str, Any], section: str, key: str) -> Any:
    value = cfg_dict.get(section, {})
    return value.get(key) if isinstance(value, dict) else None


def _path_basename(value: Any) -> str:
    return Path(str(value or "").replace("\\", "/")).name


def _checkpoint_eval_mismatches(payload: Dict[str, Any], cfg: MasterConfig) -> list[str]:
    mismatches: list[str] = []
    checkpoint_cfg = payload.get("cfg", {})
    if not isinstance(checkpoint_cfg, dict):
        return ["checkpoint cfg is missing or is not a mapping"]

    current_cfg = cfg.to_dict()
    exact_fields = {
        "env": (
            "episode_duration",
            "decision_interval",
            "yellow_duration",
            "min_green",
            "max_green",
            "min_red",
            "use_state_based_tls_control",
            "vehicle_type_policy",
        ),
        "reward": ("gamma", "wait_time_scale_k", "objective_mode"),
        "state": ("lane_input_dim", "observation_scope", "lane_feature_names"),
        "emission_risk": (
            "use_truck_count_state",
            "use_nox_risk_state",
            "enabled_reward",
            "threshold_policy",
            "risk_mode",
            "pressure_beta",
            "base_weight",
            "tail_weight",
            "penalty_aggregate",
            "penalty_time_normalize",
            "rho",
            "kappa",
            "risk_clip",
            "reward_alpha",
            "lambda_e",
        ),
        "network": (
            "lane_input_dim",
            "node_hidden_dim",
            "gat_heads",
            "node_update_dim",
            "net_node_dim",
            "bigru_hidden_dim",
            "q_hidden_dim",
            "leaky_relu_slope",
            "directions",
            "use_group_parameter_sharing",
        ),
        "dual_head": ("enabled", "nox_loss_beta", "nox_q_softplus"),
    }
    for section, keys in exact_fields.items():
        for key in keys:
            checkpoint_value = _cfg_value(checkpoint_cfg, section, key)
            current_value = _cfg_value(current_cfg, section, key)
            if checkpoint_value != current_value:
                mismatches.append(
                    f"{section}.{key}: checkpoint={checkpoint_value!r}, eval={current_value!r}"
                )

    path_fields = {
        "env": ("sumo_cfg", "net_xml", "add_xml", "intersection_groups_json"),
        "emission_risk": ("thresholds_json",),
    }
    for section, keys in path_fields.items():
        for key in keys:
            checkpoint_value = _path_basename(_cfg_value(checkpoint_cfg, section, key))
            current_value = _path_basename(_cfg_value(current_cfg, section, key))
            if checkpoint_value != current_value:
                mismatches.append(
                    f"{section}.{key}: checkpoint={checkpoint_value!r}, eval={current_value!r}"
                )
    return mismatches

def load_mixed_control_sets(net_info: Any) -> tuple[list[str], list[str]]:
    path = resolve_path(EVAL_PARAMETER_SOURCES_JSON, must_exist=True)
    with open(path or EVAL_PARAMETER_SOURCES_JSON, "r", encoding="utf-8") as f:
        sources = json.load(f)
    all_ids = set(net_info.intersection_ids)
    if set(sources) != all_ids:
        raise ValueError(
            "Kunshan parameter-source mapping must cover every TLS exactly: "
            f"missing={sorted(all_ids - set(sources))}, extra={sorted(set(sources) - all_ids)}"
        )
    allowed = {"standard", "hetero_ew_shared_ls", "other"}
    invalid = {tl_id: source for tl_id, source in sources.items() if source not in allowed}
    if invalid:
        raise ValueError(f"Invalid Kunshan parameter sources: {invalid}")
    controlled = [tl_id for tl_id in net_info.intersection_ids if sources[tl_id] != "other"]
    actuated = [tl_id for tl_id in net_info.intersection_ids if sources[tl_id] == "other"]
    counts = {name: sum(value == name for value in sources.values()) for name in allowed}
    if counts != {"standard": 22, "hetero_ew_shared_ls": 3, "other": 2}:
        raise ValueError(f"Unexpected Kunshan control split: {counts}")
    if set(actuated) != {"nt21", "nt40"}:
        raise ValueError(f"Only nt21 and nt40 may remain Gap-actuated, got {actuated}")
    return controlled, actuated

def load_checkpoint_for_kunshan(agent: MGMQAgentManager, checkpoint: str) -> Dict[str, Any]:
    path = resolve_path(checkpoint, must_exist=True) or checkpoint
    payload = _load_checkpoint_payload(path, map_location=agent.device)
    agent._validate_checkpoint_compatibility(payload, path)
    agent.online_bank.load_state_dict(payload["online_state_dict"], strict=True)
    agent.target_bank.load_state_dict(payload["target_state_dict"], strict=True)
    agent.global_step = int(payload.get("global_step", 0))
    agent.update_count = int(payload.get("update_count", 0))
    agent.target_update_count = int(payload.get("target_update_count", 0))
    return payload

def parse_args() -> argparse.Namespace:
    case_names = [str(case["name"]) for case in EVAL_CASES]
    parser = argparse.ArgumentParser(
        description="Evaluate Kunshan spatial-scale FLORE checkpoints."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        type=lambda value: EVAL_CASE_ALIASES.get(value, value),
        choices=case_names,
        default=case_names,
        help="Evaluation cases to run; defaults to all configured cases.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-id", default="custom")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument("--log-root", default=EVAL_LOG_ROOT)
    parser.add_argument("--sumocfg", default=None)
    parser.add_argument("--net-xml", default=None)
    parser.add_argument("--add-xml", default=None)
    parser.add_argument("--groups-json", default=None)
    parser.add_argument("--emission-factor-csv", default=None)
    parser.add_argument(
        "--run-tag",
        default=time.strftime("%Y%m%d_%H%M%S"),
        help="Unique tag recorded in the evaluation batch metadata.",
    )
    parser.add_argument("--port", type=int, default=EVAL_PORT)
    parser.add_argument("--port-stride", type=int, default=EVAL_PORT_STRIDE)
    parser.add_argument("--max-workers", type=int, default=EVAL_MAX_WORKERS)
    parser.add_argument("--device", default=EVAL_PARALLEL_DEVICE)
    parallel = parser.add_mutually_exclusive_group()
    parallel.add_argument("--parallel", dest="parallel_episodes", action="store_true")
    parallel.add_argument("--serial", dest="parallel_episodes", action="store_false")
    parser.add_argument("--use-gui", action="store_true", default=EVAL_USE_GUI)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate assets, configs, network structure, and checkpoints without running SUMO.",
    )
    parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help="Allow evaluation when checkpoint training config differs from the evaluation config.",
    )
    parser.set_defaults(parallel_episodes=EVAL_PARALLEL_EPISODES)
    args = parser.parse_args()
    if int(args.eval_episodes) <= 0:
        parser.error("--eval-episodes must be > 0")
    if int(args.max_workers) <= 0:
        parser.error("--max-workers must be > 0")
    if int(args.port_stride) <= 0:
        parser.error("--port-stride must be > 0")
    if bool(args.parallel_episodes) and bool(args.use_gui):
        parser.error("--use-gui requires --serial")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(args.run_tag)):
        parser.error("--run-tag may contain only letters, digits, dot, underscore, and hyphen")
    return args

def apply_overrides(cfg: MasterConfig, args: argparse.Namespace) -> MasterConfig:
    cfg.seed = int(args.seed)
    if args.sumocfg:
        cfg.env.sumo_cfg = args.sumocfg
    if args.net_xml:
        cfg.env.net_xml = args.net_xml
    if args.add_xml:
        cfg.env.add_xml = args.add_xml
    if args.groups_json:
        cfg.env.intersection_groups_json = args.groups_json
    if args.emission_factor_csv:
        cfg.env.emission_factor_csv = args.emission_factor_csv
    if args.log_root:
        cfg.log.log_root = args.log_root
    cfg.log.run_id = str(args.run_id)
    cfg.env.save_sumo_aux_outputs = False
    cfg.env.keep_sumo_runtime_files = False
    cfg.env.save_fcd_output = bool(EVAL_SAVE_FCD)
    cfg.env.save_vehicle_state_output = bool(EVAL_SAVE_VEHICLE_STATE)
    cfg.env.record_vehicle_emission_steps = bool(EVAL_SAVE_DETAIL)
    cfg.env.enable_lane_mechanism_metrics = True
    cfg.env.fcd_output_acceleration = bool(EVAL_FCD_ACCELERATION)
    cfg.env.fcd_output_period = float(EVAL_FCD_PERIOD_S)
    cfg.env.sumo_cfg = resolve_path(cfg.env.sumo_cfg, must_exist=True) or cfg.env.sumo_cfg
    cfg.env.net_xml = resolve_path(cfg.env.net_xml, must_exist=True) or cfg.env.net_xml
    cfg.env.add_xml = resolve_path(cfg.env.add_xml, must_exist=True) or cfg.env.add_xml
    cfg.env.intersection_groups_json = (
        resolve_path(cfg.env.intersection_groups_json, must_exist=True)
        or cfg.env.intersection_groups_json
    )
    cfg.env.emission_factor_csv = (
        resolve_path(cfg.env.emission_factor_csv, must_exist=True)
        or cfg.env.emission_factor_csv
    )
    cfg.emission_risk.thresholds_json = (
        resolve_path(cfg.emission_risk.thresholds_json, must_exist=True)
        or cfg.emission_risk.thresholds_json
    )
    cfg.log.log_root = resolve_path(cfg.log.log_root, must_exist=False) or cfg.log.log_root
    cfg.env.tripinfo_dir = os.path.join(cfg.log.log_root, cfg.log.run_id, "tripinfo")
    cfg.validate()
    return cfg


def preflight_case(
    case: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = apply_overrides(get_config(args.config), args)
    checkpoint_path = resolve_path(args.checkpoint, must_exist=True) or str(args.checkpoint)
    payload = _load_checkpoint_payload(checkpoint_path, map_location="cpu")
    mismatches = _checkpoint_eval_mismatches(payload, cfg)
    if mismatches and not bool(args.allow_config_mismatch):
        details = "\n".join(f"  - {item}" for item in mismatches)
        raise ValueError(
            f"Checkpoint/evaluation config mismatch for {case['name']}:\n{details}\n"
            "Use --allow-config-mismatch only when this difference is intentional."
        )

    net_info = parse_network(
        cfg.env.net_xml,
        cfg.env.add_xml,
        cfg.env.intersection_groups_json,
    )
    load_mixed_control_sets(net_info)
    ObsRewardBuilder(cfg, net_info)
    agent = MGMQAgentManager(cfg, net_info, device="cpu")
    agent._validate_checkpoint_compatibility(payload, checkpoint_path)
    agent.online_bank.load_state_dict(payload["online_state_dict"], strict=True)
    agent.target_bank.load_state_dict(payload["target_state_dict"], strict=True)

    fingerprint = _checkpoint_fingerprint(checkpoint_path)
    result = {
        "case": str(case["name"]),
        "config": str(resolve_path(args.config, must_exist=True)),
        "checkpoint": fingerprint,
        "checkpoint_episode": int(payload.get("episode", 0)),
        "checkpoint_global_step": int(payload.get("global_step", 0)),
        "checkpoint_update_count": int(payload.get("update_count", 0)),
        "ablation_meta": payload.get("ablation_meta", {}),
        "config_mismatches": mismatches,
        "network_intersections": int(len(net_info.intersection_ids)),
        "status": "ok" if not mismatches else "allowed_mismatch",
    }
    del agent
    return result

def collect_current_raw_obs(env: SumoEnv, tl_ids: list[str]) -> Dict[str, StepRawObs]:
    return {tl_id: env._collect_obs(tl_id, None) for tl_id in tl_ids}  # type: ignore[attr-defined]

def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def _safe_mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0

def _safe_sum(xs: list[float]) -> float:
    return float(np.sum(xs)) if xs else 0.0

def _safe_percentile(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else 0.0

def _ratio_of_means(weighted: list[float], traffic: list[float]) -> float:
    if not weighted or not traffic:
        return 0.0
    return float(abs(np.mean(weighted)) / max(abs(float(np.mean(traffic))), 1e-6))

VEHICLE_EMISSION_METRIC_KEYS = (
    "vehicle_count_emission",
    "vehicle_total_distance_km",
    "vehicle_total_nox_mg",
    "vehicle_mean_nox_mg_per_km",
    "vehicle_p50_nox_mg_per_km",
    "vehicle_p90_nox_mg_per_km",
    "vehicle_p95_nox_mg_per_km",
    "vehicle_max_nox_mg_per_km",
    "truck_vehicle_count",
    "truck_total_nox_mg",
    "truck_nox_share",
    "sedan_vehicle_count",
    "sedan_total_nox_mg",
)

def _zero_vehicle_emission_metrics() -> dict[str, float]:
    return {key: 0.0 for key in VEHICLE_EMISSION_METRIC_KEYS}

def _frame_columns(frame: Any) -> set[str]:
    return {str(col) for col in getattr(frame, "columns", [])}

def _float_values(frame: Any, column: str) -> list[float]:
    if column not in _frame_columns(frame):
        return []
    try:
        raw_values = frame[column].tolist()
    except Exception:
        return []
    values: list[float] = []
    for value in raw_values:
        try:
            if value is None:
                continue
            number = float(value)
            if np.isnan(number):
                continue
            values.append(number)
        except Exception:
            continue
    return values

def _unique_vehicle_count(frame: Any) -> float:
    if "veh_id" in _frame_columns(frame):
        try:
            return float(frame["veh_id"].nunique(dropna=True))
        except Exception:
            pass
    try:
        return float(len(frame))
    except Exception:
        return 0.0

def _vehicle_emission_metrics_from_df(vehicle_df: Any) -> dict[str, float]:
    metrics = _zero_vehicle_emission_metrics()
    if vehicle_df is None or getattr(vehicle_df, "empty", True):
        return metrics

    columns = _frame_columns(vehicle_df)
    nox_df = vehicle_df
    if "pollutant" in columns:
        try:
            nox_df = vehicle_df[vehicle_df["pollutant"].astype(str).str.upper() == "NOX"]
        except Exception:
            return metrics
    if nox_df is None or getattr(nox_df, "empty", True):
        return metrics

    columns = _frame_columns(nox_df)
    total_col = "total_NOx_mg" if "total_NOx_mg" in columns else "total_emission_mg"
    per_km_col = "NOx_mg_per_km" if "NOx_mg_per_km" in columns else "emission_mg_per_km"
    type_col = "vehicle_type_used" if "vehicle_type_used" in columns else "vehicle_type"

    unique_df = nox_df
    if "veh_id" in columns and hasattr(nox_df, "drop_duplicates"):
        try:
            unique_df = nox_df.drop_duplicates(subset=["veh_id"])
        except Exception:
            unique_df = nox_df

    total_nox_values = _float_values(nox_df, total_col)
    per_km_values = _float_values(nox_df, per_km_col)
    total_nox = float(np.sum(total_nox_values)) if total_nox_values else 0.0

    metrics["vehicle_count_emission"] = _unique_vehicle_count(nox_df)
    metrics["vehicle_total_distance_km"] = float(np.sum(_float_values(unique_df, "distance_m")) / 1000.0)
    metrics["vehicle_total_nox_mg"] = total_nox
    metrics["vehicle_mean_nox_mg_per_km"] = _safe_mean(per_km_values)
    metrics["vehicle_p50_nox_mg_per_km"] = _safe_percentile(per_km_values, 50)
    metrics["vehicle_p90_nox_mg_per_km"] = _safe_percentile(per_km_values, 90)
    metrics["vehicle_p95_nox_mg_per_km"] = _safe_percentile(per_km_values, 95)
    metrics["vehicle_max_nox_mg_per_km"] = float(max(per_km_values)) if per_km_values else 0.0

    if type_col in columns:
        try:
            vehicle_types = nox_df[type_col].astype(str).str.lower()
            truck_df = nox_df[vehicle_types == "truck"]
            sedan_df = nox_df[vehicle_types == "sedan"]
            truck_total = float(np.sum(_float_values(truck_df, total_col)))
            sedan_total = float(np.sum(_float_values(sedan_df, total_col)))
            metrics["truck_vehicle_count"] = _unique_vehicle_count(truck_df)
            metrics["truck_total_nox_mg"] = truck_total
            metrics["truck_nox_share"] = float(truck_total / total_nox) if total_nox > 0.0 else 0.0
            metrics["sedan_vehicle_count"] = _unique_vehicle_count(sedan_df)
            metrics["sedan_total_nox_mg"] = sedan_total
        except Exception:
            pass

    return metrics

def _vehicle_emission_episode_metrics(emission_result: Any) -> dict[str, float]:
    if emission_result is None:
        return _zero_vehicle_emission_metrics()
    return _vehicle_emission_metrics_from_df(getattr(emission_result, "vehicle_df", None))

def _episode_from_vehicle_emission_path(path: Path) -> Optional[int]:
    name = path.parent.name
    if not name.startswith("episode_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return None

def merge_vehicle_emission_outputs(out_dir: str) -> None:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        print(f"[WARN] vehicle emission merge skipped: pandas unavailable: {exc}", flush=True)
        return

    out_path = Path(out_dir)
    emission_dir = out_path / "emission"
    paths = sorted(out_path.glob("emission/episode_*/vehicle_emission.csv"))
    if not paths:
        print(f"[WARN] vehicle emission merge skipped: no vehicle_emission.csv under {emission_dir}", flush=True)
        return

    frames: list[Any] = []
    summary_rows: list[dict[str, Any]] = []
    for path in paths:
        episode = _episode_from_vehicle_emission_path(path)
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] failed to read vehicle emission file {path}: {exc}", flush=True)
            continue
        if "episode" not in frame.columns and episode is not None:
            frame["episode"] = int(episode)
        frames.append(frame)
        row: dict[str, Any] = {"episode": int(episode) if episode is not None else 0}
        row.update(_vehicle_emission_metrics_from_df(frame))
        summary_rows.append(row)

    if not frames:
        print(f"[WARN] vehicle emission merge skipped: no readable vehicle_emission.csv under {emission_dir}", flush=True)
        return

    emission_dir.mkdir(parents=True, exist_ok=True)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(emission_dir / "vehicle_emission_all.csv", index=False)
    pd.DataFrame(summary_rows, columns=["episode", *VEHICLE_EMISSION_METRIC_KEYS]).to_csv(
        emission_dir / "vehicle_emission_summary_by_episode.csv",
        index=False,
    )

def _build_summary(rows: list[dict], args: argparse.Namespace, *, parallel: bool, max_workers: int) -> dict:
    if not rows:
        summary: dict[str, Any] = {}
    else:
        exclude_keys = {"eval_episode", "sumo_seed", "worker_id", "port"}
        summary = {}
        for key in rows[0].keys():
            if key in exclude_keys:
                continue
            values: list[float] = []
            for row in rows:
                value = row.get(key)
                if isinstance(value, bool):
                    values.append(float(value))
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    values.append(float(value))
            if values:
                summary[key] = float(np.mean(values))
    summary["eval_episodes"] = int(args.eval_episodes)
    summary["seed"] = int(args.seed)
    summary["checkpoint"] = str(args.checkpoint)
    summary["checkpoint_resolved"] = str(
        getattr(args, "checkpoint_info", {}).get("path", resolve_path(args.checkpoint))
    )
    summary["checkpoint_sha256"] = str(
        getattr(args, "checkpoint_info", {}).get("sha256", "")
    )
    summary["config"] = str(args.config)
    summary["config_resolved"] = str(resolve_path(args.config))
    summary["run_id"] = str(args.run_id)
    summary["run_tag"] = str(args.run_tag)
    summary["parallel_episodes"] = bool(parallel)
    summary["max_workers"] = int(max_workers)
    return summary

def run_one_eval_episode(
    *,
    args: argparse.Namespace,
    cfg: MasterConfig,
    net_info: Any,
    builder: ObsRewardBuilder,
    agent: MGMQAgentManager,
    logger: TrainingLogger,
    profiler: StageProfiler,
    ep: int,
    seed: int,
    eval_episodes: int,
    worker_id: int,
) -> dict:
    ep_start_time = time.time()
    episode_emission_result: Any = None
    episode_link_nox_values: list[float] = []
    logger.begin_episode(
        ep,
        record_lane_step=bool(EVAL_SAVE_DETAIL or EVAL_SAVE_LANE_NOX),
        record_phase_step=bool(EVAL_SAVE_DETAIL),
        record_q_step=bool(EVAL_SAVE_DETAIL),
    )
    env: Optional[SumoEnv] = None
    try:
        controlled_tl_ids, actuated_tl_ids = load_mixed_control_sets(net_info)
        env = SumoEnv(
            cfg.env,
            net_info,
            port=int(args.port),
            use_gui=bool(args.use_gui),
            controlled_tl_ids=controlled_tl_ids,
        )
        env.start(episode_id=ep, seed=seed, control_tls=True)
        with profiler.timeit("initial_collect_raw_obs", episode=ep, step=-1, global_step=None):
            raw_obs = collect_current_raw_obs(env, controlled_tl_ids)
        with profiler.timeit("initial_build_observations", episode=ep, step=-1, global_step=None):
            obs = builder.build_observations(raw_obs)

        rewards_all: list[float] = []
        switch_flags: list[float] = []
        traffic_rewards: list[float] = []
        emission_penalties: list[float] = []
        lambda_e_values: list[float] = []
        weighted_emission_penalties: list[float] = []
        emission_penalty_ratios: list[float] = []
        nox_risk_means: list[float] = []
        nox_risk_sums: list[float] = []
        nox_risk_maxs: list[float] = []
        nox_exceed_counts: list[float] = []
        network_nox_values: list[float] = []
        total_lane_nox_values: list[float] = []
        max_lane_nox_values: list[float] = []
        all_intersection_lanes = {
            lane_id
            for tl_id in net_info.intersection_ids
            for lane_id in net_info.get_intersection(tl_id).all_inc_lanes_flat()
        }

        for step in range(int(cfg.env.steps_per_episode)):
            with profiler.timeit("bookkeeping", episode=ep, step=step, global_step=None):
                prev_phases = {tl_id: int(raw.current_phase) for tl_id, raw in raw_obs.items()}
            with profiler.timeit("act", episode=ep, step=step, global_step=None):
                actions, action_infos = agent.act(
                    obs,
                    epsilon=0.0,
                    deterministic=True,
                    record_q_values=bool(EVAL_SAVE_DETAIL),
                )
            with profiler.timeit("env_step", episode=ep, step=step, global_step=None):
                next_raw = env.step(actions, decision_step=step)
            with profiler.timeit("link_emission_compute", episode=ep, step=step, global_step=None):
                step_emission = getattr(env, "_last_step_emission", None)
                if step_emission is not None:
                    network_nox_values.append(
                        float(step_emission.total_by_pollutant.get("NOx", 0.0))
                    )
                    lane_nox = [
                        float(step_emission.by_lane.get((lane_id, "NOx"), 0.0))
                        for lane_id in all_intersection_lanes
                    ]
                    total_lane_nox_values.append(_safe_sum(lane_nox))
                    if lane_nox:
                        max_lane_nox_values.append(float(max(lane_nox)))
                link_nox_values = logger.compute_step_link_emission_values(
                    step_emission=step_emission,
                    pollutant="NOx",
                )
                episode_link_nox_values.extend(link_nox_values)
            if bool(EVAL_SAVE_DETAIL):
                with profiler.timeit("log_step_emission", episode=ep, step=step, global_step=None):
                    logger.log_step_emission(episode=ep, step=step, step_emission=step_emission)
            with profiler.timeit("compute_rewards", episode=ep, step=step, global_step=None):
                rewards, comps = builder.compute_rewards(next_raw, actions, prev_phases)
            with profiler.timeit("build_next_observations", episode=ep, step=step, global_step=None):
                next_obs = builder.build_observations(next_raw)
            with profiler.timeit("bookkeeping", episode=ep, step=step, global_step=None):
                rewards_all.extend(float(v) for v in rewards.values())
                switch_flags.extend(1.0 if c.rp < 0 else 0.0 for c in comps.values())
                for c in comps.values():
                    traffic_reward = float(getattr(c, "traffic_reward", 0.0))
                    emission_penalty = float(getattr(c, "emission_penalty", 0.0))
                    lambda_e = float(getattr(c, "lambda_e", 0.0))
                    weighted = float(lambda_e * emission_penalty)
                    ratio = float(abs(weighted) / max(abs(traffic_reward), 1e-6))
                    traffic_rewards.append(traffic_reward)
                    emission_penalties.append(emission_penalty)
                    lambda_e_values.append(lambda_e)
                    weighted_emission_penalties.append(weighted)
                    emission_penalty_ratios.append(ratio)
                nox_risk_means.extend(float(getattr(c, "nox_risk_mean", 0.0)) for c in comps.values())
                nox_risk_sums.extend(float(getattr(c, "nox_risk_sum", 0.0)) for c in comps.values())
                nox_risk_maxs.extend(float(getattr(c, "nox_risk_max", 0.0)) for c in comps.values())
                nox_exceed_counts.extend(float(getattr(c, "nox_exceed_lane_count", 0.0)) for c in comps.values())
            if bool(EVAL_SAVE_DETAIL or EVAL_SAVE_LANE_NOX):
                with profiler.timeit("log_step", episode=ep, step=step, global_step=None):
                    logger.log_step(ep, step, next_raw, next_obs, actions, action_infos, comps)
            with profiler.timeit("bookkeeping", episode=ep, step=step, global_step=None):
                obs = next_obs
                raw_obs = next_raw
            profiler.add_step_meta(
                episode=ep,
                step=step,
                global_step=None,
                replay_size="",
                update_due=False,
                n_update_groups=0,
                n_agents=len(controlled_tl_ids),
            )

        with profiler.timeit("env_close", episode=ep, step=-1, global_step=None):
            stats = env.close()
        with profiler.timeit("finalize_episode_emission", episode=ep, step=-1, global_step=None):
            recorder = getattr(env, "emission_recorder", None)
            if recorder is not None:
                try:
                    episode_emission_result = recorder.finalize_episode()
                except Exception as exc:
                    print(f"[WARN] eval_episode={ep}: finalize vehicle emission failed: {exc}", flush=True)
                    episode_emission_result = None
        env = None
        if bool(EVAL_SAVE_DETAIL or EVAL_SAVE_LANE_NOX):
            with profiler.timeit("flush_episode_physical", episode=ep, step=-1, global_step=None):
                logger.flush_episode_physical(ep)
        if bool(EVAL_SAVE_DETAIL):
            with profiler.timeit("flush_episode_emission", episode=ep, step=-1, global_step=None):
                logger.flush_episode_emission(ep)
        if bool(EVAL_SAVE_VEHICLE_EMISSION):
            with profiler.timeit("write_episode_emission_result", episode=ep, step=-1, global_step=None):
                logger.write_episode_emission_result(
                    episode=ep,
                    emission_result=episode_emission_result,
                    save_step_df=False,
                    save_edge_step_raw=False,
                    save_lane_step=False,
                    save_vehicle_step=bool(EVAL_SAVE_DETAIL),
                )
    finally:
        if env is not None:
            try:
                env.close(parse_tripinfo=False)
            except Exception:
                pass

    ep_wall_time_s = time.time() - ep_start_time
    row = {
        "eval_episode": int(ep),
        "sumo_seed": int(seed),
        "rl_controlled_tls": int(len(controlled_tl_ids)),
        "sumo_actuated_tls": int(len(actuated_tl_ids)),
        "sumo_actuated_tl_ids": ",".join(actuated_tl_ids),
        "worker_id": int(worker_id),
        "port": int(args.port),
        "wall_time_s": float(ep_wall_time_s),
        "reward_mean": float(np.mean(rewards_all)) if rewards_all else 0.0,
        "reward_sum": float(np.sum(rewards_all)) if rewards_all else 0.0,
        "traffic_reward_mean": _safe_mean(traffic_rewards),
        "traffic_reward_abs_mean": _safe_mean([abs(x) for x in traffic_rewards]),
        "emission_penalty_mean": _safe_mean(emission_penalties),
        "emission_penalty_abs_mean": _safe_mean([abs(x) for x in emission_penalties]),
        "weighted_emission_penalty_mean": _safe_mean(weighted_emission_penalties),
        "weighted_emission_penalty_abs_mean": _safe_mean([abs(x) for x in weighted_emission_penalties]),
        "weighted_emission_penalty_sum": _safe_sum(weighted_emission_penalties),
        "final_reward_mean": float(np.mean(rewards_all)) if rewards_all else 0.0,
        "emission_penalty_ratio_mean": _safe_mean(emission_penalty_ratios),
        "emission_penalty_ratio_p50": _safe_percentile(emission_penalty_ratios, 50),
        "emission_penalty_ratio_p90": _safe_percentile(emission_penalty_ratios, 90),
        "emission_penalty_ratio_p95": _safe_percentile(emission_penalty_ratios, 95),
        "emission_penalty_ratio_max": float(max(emission_penalty_ratios)) if emission_penalty_ratios else 0.0,
        "emission_penalty_ratio_of_means": _ratio_of_means(weighted_emission_penalties, traffic_rewards),
        "lambda_e_mean": _safe_mean(lambda_e_values),
        "nox_risk_mean": float(np.mean(nox_risk_means)) if nox_risk_means else 0.0,
        "nox_risk_sum_mean": float(np.mean(nox_risk_sums)) if nox_risk_sums else 0.0,
        "nox_risk_max_mean": float(np.mean(nox_risk_maxs)) if nox_risk_maxs else 0.0,
        "link_nox_mean": float(np.mean(episode_link_nox_values)) if episode_link_nox_values else 0.0,
        "nox_exceed_lane_count_mean": float(np.mean(nox_exceed_counts)) if nox_exceed_counts else 0.0,
        "network_total_nox_mg": _safe_sum(network_nox_values),
        "network_mean_step_nox_mg": _safe_mean(network_nox_values),
        "all_intersection_incoming_lane_nox_mg": _safe_sum(total_lane_nox_values),
        "mean_step_all_intersection_incoming_lane_nox_mg": _safe_mean(total_lane_nox_values),
        "max_lane_nox_mg": float(np.max(max_lane_nox_values)) if max_lane_nox_values else 0.0,
        "avg_delay_s": float(getattr(stats, "avg_delay_s", 0.0) or 0.0),
        "avg_travel_time_s": float(getattr(stats, "avg_travel_time_s", 0.0) or 0.0),
        "total_arrived": int(getattr(stats, "total_arrived", 0) or 0),
        "completion_rate": float(getattr(stats, "completion_rate", 0.0) or 0.0),
        "phase_switch_rate": float(np.mean(switch_flags)) if switch_flags else 0.0,
    }
    row.update(_vehicle_emission_episode_metrics(episode_emission_result))
    if profiler.enabled:
        profiler.finish_episode(
            episode=ep,
            wall_time_s=time.time() - ep_start_time,
            steps=int(cfg.env.steps_per_episode),
            n_agents=len(controlled_tl_ids),
            replay_size_end="",
        )
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"worker={worker_id} port={int(args.port)} "
        f"eval_episode={ep}/{eval_episodes} seed={seed} "
        f"wall_time={ep_wall_time_s:.1f}s reward_mean={row['reward_mean']:.3f} "
        f"ratio={row['emission_penalty_ratio_mean']:.4f} "
        f"avg_delay={row['avg_delay_s']:.3f}s "
        f"completion={row['completion_rate']:.3f} arrived={row['total_arrived']}"
    )
    return row

def split_episode_batches(
    episodes: list[tuple[int, int]],
    max_workers: int,
) -> list[list[tuple[int, int]]]:
    n_workers = max(1, min(int(max_workers), len(episodes)))
    batches: list[list[tuple[int, int]]] = [[] for _ in range(n_workers)]
    for i, item in enumerate(episodes):
        batches[i % n_workers].append(item)
    return [batch for batch in batches if batch]

def run_episode_batch_worker(payload: dict) -> dict:
    worker_start = time.time()
    if bool(payload.get("force_traci", True)):
        os.environ["SUMO_FORCE_TRACI"] = "1"

    worker_id = int(payload["worker_id"])
    episodes = [(int(ep), int(seed)) for ep, seed in payload["episodes"]]
    args = argparse.Namespace(**payload["args"])
    args.port = int(payload["base_port"]) + worker_id * int(payload["port_stride"])
    args.run_id = str(payload["run_id"])

    cfg = apply_overrides(get_config(args.config), args)
    cfg.log.save_step_physical = bool(payload["save_detail"] or EVAL_SAVE_LANE_NOX)
    cfg.log.save_q_step = bool(payload["save_detail"])
    cfg.log.save_emission_step = bool(payload["save_detail"])
    cfg.log.emission_step_log_interval_train = 1 if bool(payload["save_detail"]) else 0

    set_global_seed(int(cfg.seed))
    net_info = parse_network(cfg.env.net_xml, cfg.env.add_xml, cfg.env.intersection_groups_json)
    builder = ObsRewardBuilder(cfg, net_info)
    agent = MGMQAgentManager(cfg, net_info, device=str(payload.get("device", "cpu")))
    load_checkpoint_for_kunshan(agent, args.checkpoint)
    agent.online_bank.eval()
    logger = TrainingLogger(
        cfg,
        net_info,
        model_meta=agent.online_bank.model_meta(),
        write_metadata=False,
        prepare_model_dirs=False,
    )
    profiler = StageProfiler(
        enabled=bool(payload.get("profile", False)),
        sync_cuda=bool(payload.get("profile_sync_cuda", False)),
    )

    rows: list[dict] = []
    errors: list[dict] = []
    eval_episodes = int(payload["eval_episodes"])
    for ep, seed in episodes:
        try:
            row = run_one_eval_episode(
                args=args,
                cfg=cfg,
                net_info=net_info,
                builder=builder,
                agent=agent,
                logger=logger,
                profiler=profiler,
                ep=ep,
                seed=seed,
                eval_episodes=eval_episodes,
                worker_id=worker_id,
            )
            rows.append(row)
        except Exception as exc:
            errors.append({
                "worker_id": int(worker_id),
                "episode": int(ep),
                "seed": int(seed),
                "port": int(args.port),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })

    if profiler.enabled:
        out_dir = cfg.log.run_dir
        profiler.flush_step_csv(os.path.join(out_dir, f"eval_profile_step_log_worker_{worker_id}.csv"))
        profiler.flush_episode_csv(os.path.join(out_dir, f"eval_profile_episode_log_worker_{worker_id}.csv"))
        profiler.save_summary_json(os.path.join(out_dir, f"eval_profile_summary_worker_{worker_id}.json"))

    return {
        "worker_id": int(worker_id),
        "port": int(args.port),
        "rows": rows,
        "errors": errors,
        "wall_time_s": float(time.time() - worker_start),
    }

def _episode_items(cfg: MasterConfig, eval_episodes: int) -> list[tuple[int, int]]:
    sumo_seeds = build_sumo_seed_list(
        global_seed=int(cfg.seed),
        total_episodes=eval_episodes,
        mode=str(cfg.env.sumo_seed_mode),
    )
    return [
        (ep, int(sumo_seeds[ep - 1]))
        for ep in range(1, eval_episodes + 1)
    ]


def _write_sumo_seed_json(
    out_dir: str,
    cfg: MasterConfig,
    episode_items: list[tuple[int, int]],
) -> None:
    with open(os.path.join(out_dir, "eval_sumo_seeds.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "global_seed": int(cfg.seed),
                "sumo_seed_mode": str(cfg.env.sumo_seed_mode),
                "eval_episodes": int(len(episode_items)),
                "sumo_seeds": [
                    {"eval_episode": int(ep), "sumo_seed": int(seed)}
                    for ep, seed in episode_items
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

def run_single_evaluation_serial(args: argparse.Namespace) -> Dict[str, Any]:
    if bool(EVAL_FORCE_TRACI):
        os.environ["SUMO_FORCE_TRACI"] = "1"
    cfg = apply_overrides(get_config(args.config), args)
    cfg.log.save_step_physical = bool(EVAL_SAVE_DETAIL or EVAL_SAVE_LANE_NOX)
    cfg.log.save_q_step = bool(EVAL_SAVE_DETAIL)
    cfg.log.save_emission_step = bool(EVAL_SAVE_DETAIL)
    cfg.log.emission_step_log_interval_train = 1 if bool(EVAL_SAVE_DETAIL) else 0
    set_global_seed(cfg.seed)
    net_info = parse_network(cfg.env.net_xml, cfg.env.add_xml, cfg.env.intersection_groups_json)
    builder = ObsRewardBuilder(cfg, net_info)
    agent = MGMQAgentManager(cfg, net_info, device=str(args.device))
    load_checkpoint_for_kunshan(agent, args.checkpoint)
    agent.online_bank.eval()
    logger = TrainingLogger(
        cfg, net_info, model_meta=agent.online_bank.model_meta(), prepare_model_dirs=False
    )
    profiler = StageProfiler(
        enabled=bool(EVAL_PROFILE),
        sync_cuda=bool(EVAL_PROFILE_SYNC_CUDA),
    )

    out_dir = cfg.log.run_dir
    os.makedirs(out_dir, exist_ok=True)
    eval_episodes = int(args.eval_episodes)
    episode_items = _episode_items(cfg, eval_episodes)
    _write_sumo_seed_json(out_dir, cfg, episode_items)
    rows = [
        run_one_eval_episode(
            args=args,
            cfg=cfg,
            net_info=net_info,
            builder=builder,
            agent=agent,
            logger=logger,
            profiler=profiler,
            ep=ep,
            seed=seed,
            eval_episodes=eval_episodes,
            worker_id=0,
        )
        for ep, seed in episode_items
    ]
    rows = sorted(rows, key=lambda r: int(r["eval_episode"]))
    with profiler.timeit("write_eval_outputs", episode=-1, step=-1, global_step=None):
        write_csv(os.path.join(out_dir, "eval_episode_log.csv"), rows)
        summary = _build_summary(rows, args, parallel=False, max_workers=1)
        with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        merge_vehicle_emission_outputs(out_dir)
        if EVAL_SAVE_VEHICLE_STATE:
            consolidate_vehicle_state_outputs(out_dir)
    if profiler.enabled:
        profiler.flush_step_csv(os.path.join(out_dir, "eval_profile_step_log.csv"))
        profiler.flush_episode_csv(os.path.join(out_dir, "eval_profile_episode_log.csv"))
        profiler.save_summary_json(os.path.join(out_dir, "eval_profile_summary.json"))
    print(f"evaluation saved to: {out_dir}")
    return summary

def run_single_evaluation_parallel(args: argparse.Namespace) -> Dict[str, Any]:
    base_cfg = apply_overrides(get_config(args.config), args)
    base_cfg.log.save_step_physical = bool(EVAL_SAVE_DETAIL or EVAL_SAVE_LANE_NOX)
    base_cfg.log.save_q_step = bool(EVAL_SAVE_DETAIL)
    base_cfg.log.save_emission_step = bool(EVAL_SAVE_DETAIL)
    base_cfg.log.emission_step_log_interval_train = 1 if bool(EVAL_SAVE_DETAIL) else 0
    base_out_dir = base_cfg.log.run_dir
    os.makedirs(base_out_dir, exist_ok=True)

    eval_episodes = int(args.eval_episodes)
    episode_items = _episode_items(base_cfg, eval_episodes)
    _write_sumo_seed_json(base_out_dir, base_cfg, episode_items)
    batches = split_episode_batches(episode_items, int(args.max_workers))

    net_info = parse_network(base_cfg.env.net_xml, base_cfg.env.add_xml, base_cfg.env.intersection_groups_json)
    meta_agent = MGMQAgentManager(base_cfg, net_info, device=str(args.device))
    load_checkpoint_for_kunshan(meta_agent, args.checkpoint)
    TrainingLogger(
        base_cfg,
        net_info,
        model_meta=meta_agent.online_bank.model_meta(),
        write_metadata=True,
        prepare_model_dirs=False,
    )

    payloads = []
    for worker_id, batch in enumerate(batches):
        payloads.append({
            "worker_id": int(worker_id),
            "episodes": batch,
            "args": vars(args),
            "run_id": str(args.run_id),
            "base_port": int(args.port),
            "port_stride": int(args.port_stride),
            "eval_episodes": int(eval_episodes),
            "save_detail": bool(EVAL_SAVE_DETAIL),
            "profile": bool(EVAL_PROFILE),
            "profile_sync_cuda": bool(EVAL_PROFILE_SYNC_CUDA),
            "force_traci": bool(EVAL_FORCE_TRACI),
            "device": str(args.device),
        })

    all_rows: list[dict] = []
    all_errors: list[dict] = []
    worker_infos: list[dict] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=ctx) as executor:
        futures = [executor.submit(run_episode_batch_worker, payload) for payload in payloads]
        for fut in as_completed(futures):
            result = fut.result()
            rows = list(result.get("rows", []))
            errors = list(result.get("errors", []))
            all_rows.extend(rows)
            all_errors.extend(errors)
            worker_infos.append({
                "worker_id": result.get("worker_id"),
                "port": result.get("port"),
                "wall_time_s": result.get("wall_time_s"),
                "n_rows": len(rows),
                "n_errors": len(errors),
            })

    all_rows = sorted(all_rows, key=lambda r: int(r["eval_episode"]))
    worker_infos = sorted(worker_infos, key=lambda r: int(r.get("worker_id", 0) or 0))
    write_csv(os.path.join(base_out_dir, "eval_worker_log.csv"), worker_infos)

    if all_errors:
        with open(os.path.join(base_out_dir, "eval_errors.json"), "w", encoding="utf-8") as f:
            json.dump(all_errors, f, indent=2, ensure_ascii=False)
        raise RuntimeError(f"{len(all_errors)} evaluation episodes failed. See eval_errors.json")

    write_csv(os.path.join(base_out_dir, "eval_episode_log.csv"), all_rows)
    summary = _build_summary(all_rows, args, parallel=True, max_workers=len(payloads))
    with open(os.path.join(base_out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    merge_vehicle_emission_outputs(base_out_dir)
    if EVAL_SAVE_VEHICLE_STATE:
        consolidate_vehicle_state_outputs(base_out_dir)
    print(f"parallel evaluation saved to: {base_out_dir}")
    return summary

def main() -> None:
    mp.freeze_support()
    base_args = parse_args()
    selected_names = {str(name) for name in base_args.cases}
    selected_cases = [
        case for case in EVAL_CASES
        if str(case["name"]) in selected_names
    ]

    base_log_root = Path(
        resolve_path(str(base_args.log_root), must_exist=False)
        or str(base_args.log_root)
    )
    batch_root = base_log_root
    if base_args.config or base_args.checkpoint:
        if not (base_args.config and base_args.checkpoint):
            raise ValueError('--config and --checkpoint must be supplied together')
        selected_cases = [{'name': base_args.run_id, 'config': base_args.config,
                           'checkpoint': base_args.checkpoint, 'run_id': base_args.run_id}]
    validate_eval_assets(selected_cases)
    prepared_cases: list[tuple[dict[str, str], argparse.Namespace, dict[str, Any]]] = []
    for case in selected_cases:
        args = argparse.Namespace(**vars(base_args))
        args.config = str(case["config"])
        args.checkpoint = str(case["checkpoint"])
        args.run_id = str(case["run_id"])
        args.log_root = str(batch_root)
        preflight = preflight_case(case, args)
        args.checkpoint_info = dict(preflight["checkpoint"])
        prepared_cases.append((case, args, preflight))
        print(
            f"[preflight] {case['name']}: ok, "
            f"checkpoint_episode={preflight['checkpoint_episode']}, "
            f"sha256={preflight['checkpoint']['sha256'][:12]}",
            flush=True,
        )

    if bool(base_args.preflight_only):
        print(json.dumps(
            {
                "status": "preflight_ok",
                "eval_episodes": int(base_args.eval_episodes),
                "seed": int(base_args.seed),
                "cases": [item[2] for item in prepared_cases],
            },
            indent=2,
            ensure_ascii=False,
        ))
        return

    existing_run_dirs = [
        batch_root / str(case["run_id"])
        for case in selected_cases
        if (batch_root / str(case["run_id"])).exists()
        and any((batch_root / str(case["run_id"])).iterdir())
    ]
    if existing_run_dirs:
        raise FileExistsError(
            "Evaluation case directories are not empty: "
            + ", ".join(str(path) for path in existing_run_dirs)
            + ". Remove or rename them before rerunning to avoid mixing results."
        )
    batch_root.mkdir(parents=True, exist_ok=True)

    batch_manifest: dict[str, Any] = {
        "created_at_unix": time.time(),
        "run_tag": str(base_args.run_tag),
        "batch_root": str(batch_root),
        "eval_episodes": int(base_args.eval_episodes),
        "seed": int(base_args.seed),
        "parallel_episodes": bool(base_args.parallel_episodes),
        "max_workers": int(base_args.max_workers),
        "base_port": int(base_args.port),
        "port_stride": int(base_args.port_stride),
        "cases": [item[2] for item in prepared_cases],
    }
    manifest_path = batch_root / "eval_batch_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(batch_manifest, f, indent=2, ensure_ascii=False)

    all_summaries: list[dict] = []

    for case, args, _preflight in prepared_cases:
        print("=" * 80)
        print(f"Start evaluation: {case['name']}")
        print(f"config     : {args.config}")
        print(f"checkpoint : {args.checkpoint}")
        print(f"run_id     : {args.run_id}")
        print(
            f"parallel   : {bool(args.parallel_episodes)}, "
            f"workers={int(args.max_workers)}"
        )
        print("=" * 80)

        if bool(args.parallel_episodes):
            summary = run_single_evaluation_parallel(args)
        else:
            summary = run_single_evaluation_serial(args)

        summary["case"] = str(case["name"])
        summary["config"] = str(case["config"])
        summary["checkpoint"] = str(case["checkpoint"])
        summary["run_id"] = str(case["run_id"])
        all_summaries.append(summary)

    if all_summaries:
        write_csv(str(batch_root / "eval_all_summary.csv"), all_summaries)
    batch_manifest["completed_at_unix"] = time.time()
    batch_manifest["status"] = "completed"
    batch_manifest["summaries"] = all_summaries
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(batch_manifest, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"all evaluations completed. summary saved to: {batch_root}")
    print("=" * 80)

if __name__ == "__main__":
    main()
