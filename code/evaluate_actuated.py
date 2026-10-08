# -*- coding: utf-8 -*-
"""
evaluate_actuated.py
====================
Evaluate a SUMO native actuated baseline without any RL agent or checkpoint.

Example (does not write FCD unless --save-fcd is supplied):
python code/evaluate_actuated.py ^
  --config configs/grid36/calibration/sumo_actuated_eval.yaml ^
  --eval-episodes 20 ^
  --seed 19 ^
  --log-root results/evaluation/grid36/grid36 ^
  --run-id sumo_actuated ^
  --port 8814 ^
  --parallel ^
  --max-workers 3 ^
  --profile ^
  --save-detail
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from config import MasterConfig, build_sumo_seed_list, get_config
from env import SumoEnv, StepRawObs
from logger import TrainingLogger
from network_parser import parse_network
from obs_reward import ObsRewardBuilder
from profiler import StageProfiler
from vehicle_state import consolidate_vehicle_state_outputs


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _console(message: str) -> None:
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}",
        flush=True,
    )

# ================================================================
# SUMO FCD output settings
# ================================================================
EVAL_SAVE_FCD = False
EVAL_SAVE_VEHICLE_STATE = False
EVAL_FCD_ACCELERATION = True
EVAL_FCD_PERIOD_S = 1.0


@dataclass
class PassiveActionInfo:
    tl_id: str
    group_id: str
    action: int
    epsilon: float = 0.0
    is_random_action: bool = False
    is_greedy_action: bool = False
    q_values: list[float] = field(default_factory=list)
    q_masked_values: list[float] = field(default_factory=list)
    q_max: float = 0.0
    q_selected: float = 0.0
    q_margin: float = 0.0
    valid_action_count: int = 0
    action_mask: list[bool] = field(default_factory=list)


def set_global_seed_local(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def resolve_path(path: Optional[str], *, must_exist: bool = False) -> Optional[str]:
    if path is None or str(path).strip() == "":
        return None
    p = Path(path).expanduser()
    candidates: list[Path]
    if p.is_absolute():
        candidates = [p]
    else:
        candidates = [
            PROJECT_ROOT / p,
            SCRIPT_DIR / p,
            Path.cwd() / p,
        ]
    chosen = candidates[0]
    for candidate in candidates:
        if candidate.exists():
            chosen = candidate
            break
    chosen = chosen.resolve()
    if must_exist and not chosen.exists():
        raise FileNotFoundError(str(chosen))
    return str(chosen)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SUMO native actuated control without loading RL checkpoints."
    )
    parser.add_argument(
        "--config",
        default="configs/grid36/calibration/sumo_actuated_eval.yaml",
    )
    parser.add_argument("--sumocfg", default="")
    parser.add_argument("--net-xml", default="")
    parser.add_argument("--add-xml", default="")
    parser.add_argument("--groups-json", default="")
    parser.add_argument("--emission-factor-csv", default="")
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--episode-duration", type=int, default=None)
    parser.add_argument("--calibration-all", action="store_true")
    parser.add_argument("--calibration-root", default="")
    parser.add_argument(
        "--calibration-output-root",
        default=str(Path(__file__).resolve().parents[1] / "results/evaluation/grid36" / "sumo_actuated_calibration"),
    )
    parser.add_argument("--start-episode", type=int, default=1, help="First episode to run (inclusive; seeds use the full --eval-episodes sequence).")
    parser.add_argument("--end-episode", type=int, default=None, help="Last episode to run (inclusive; defaults to --eval-episodes).")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--log-root", default="results/evaluation/grid36/grid36")
    parser.add_argument("--run-id", default="sumo_actuated")
    fcd = parser.add_mutually_exclusive_group()
    fcd.add_argument("--save-fcd", dest="save_fcd", action="store_true")
    fcd.add_argument("--no-save-fcd", dest="save_fcd", action="store_false")
    parser.add_argument("--fcd-period", type=float, default=EVAL_FCD_PERIOD_S)
    parser.add_argument("--port", type=int, default=8814)
    parallel = parser.add_mutually_exclusive_group()
    parallel.add_argument("--parallel", dest="parallel_episodes", action="store_true")
    parallel.add_argument("--serial", dest="parallel_episodes", action="store_false")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--port-stride", type=int, default=10)
    parser.add_argument("--use-gui", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-sync-cuda", action="store_true")
    detail = parser.add_mutually_exclusive_group()
    detail.add_argument("--save-detail", dest="save_detail", action="store_true")
    detail.add_argument("--no-save-detail", dest="save_detail", action="store_false")
    parser.set_defaults(
        save_detail=True,
        parallel_episodes=True,
        save_fcd=EVAL_SAVE_FCD,
    )
    return parser.parse_args()


def apply_overrides(cfg: MasterConfig, args: argparse.Namespace) -> MasterConfig:
    save_detail = bool(args.save_detail)
    cfg.seed = int(args.seed)
    if args.episode_duration is not None:
        if int(args.episode_duration) <= 0:
            raise ValueError("--episode-duration must be > 0")
        cfg.env.episode_duration = int(args.episode_duration)
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
    if args.log_root:
        cfg.log.log_root = str(args.log_root)
    cfg.log.run_id = str(args.run_id)
    cfg.env.baseline_control_mode = "sumo_actuated"
    cfg.env.save_detector_outputs = False
    cfg.env.save_sumo_aux_outputs = False
    cfg.env.save_tls_phase_outputs = True
    if float(args.fcd_period) <= 0.0:
        raise ValueError("--fcd-period must be > 0")
    cfg.env.save_fcd_output = bool(args.save_fcd)
    cfg.env.save_vehicle_state_output = bool(EVAL_SAVE_VEHICLE_STATE)
    cfg.env.save_vehicle_second_output = False
    cfg.env.fcd_output_acceleration = bool(EVAL_FCD_ACCELERATION)
    cfg.env.fcd_output_period = float(args.fcd_period)

    cfg.env.sumo_cfg = resolve_path(cfg.env.sumo_cfg, must_exist=True) or cfg.env.sumo_cfg
    cfg.env.net_xml = resolve_path(cfg.env.net_xml, must_exist=True) or cfg.env.net_xml
    cfg.env.add_xml = resolve_path(cfg.env.add_xml, must_exist=True) or cfg.env.add_xml
    cfg.env.intersection_groups_json = (
        resolve_path(cfg.env.intersection_groups_json, must_exist=False)
        or cfg.env.intersection_groups_json
    )
    cfg.env.emission_factor_csv = (
        resolve_path(cfg.env.emission_factor_csv, must_exist=False)
        or cfg.env.emission_factor_csv
    )
    cfg.log.log_root = resolve_path(cfg.log.log_root, must_exist=False) or cfg.log.log_root
    cfg.env.tripinfo_dir = os.path.join(cfg.log.log_root, cfg.log.run_id, "tripinfo")
    cfg.log.save_step_physical = save_detail
    cfg.log.save_emission_step = save_detail
    cfg.log.save_q_step = False
    cfg.log.emission_step_log_interval_train = 1 if save_detail else 0
    cfg.validate()
    return cfg


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


def validate_fcd_output(path: str, *, episode: int, enabled: bool) -> None:
    """Fail immediately when an enabled episode did not produce a usable FCD file."""
    if not enabled:
        return
    fcd_path = Path(path)
    if not path or not fcd_path.is_file():
        raise RuntimeError(
            f"Episode {int(episode):04d} did not produce the expected FCD file: {fcd_path}"
        )
    if fcd_path.stat().st_size <= 0:
        raise RuntimeError(
            f"Episode {int(episode):04d} produced an empty FCD file: {fcd_path}"
        )


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
        _console(f"WARN vehicle emission merge skipped: pandas unavailable: {exc}")
        return

    out_path = Path(out_dir)
    emission_dir = out_path / "emission"
    paths = sorted(out_path.glob("emission/episode_*/vehicle_emission.csv"))
    if not paths:
        _console(f"WARN vehicle emission merge skipped: no vehicle_emission.csv under {emission_dir}")
        return

    frames: list[Any] = []
    summary_rows: list[dict[str, Any]] = []
    for path in paths:
        episode = _episode_from_vehicle_emission_path(path)
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            _console(f"WARN failed to read vehicle emission file {path}: {exc}")
            continue
        if "episode" not in frame.columns and episode is not None:
            frame["episode"] = int(episode)
        frames.append(frame)
        row: dict[str, Any] = {"episode": int(episode) if episode is not None else 0}
        row.update(_vehicle_emission_metrics_from_df(frame))
        summary_rows.append(row)

    if not frames:
        _console(f"WARN vehicle emission merge skipped: no readable vehicle_emission.csv under {emission_dir}")
        return

    emission_dir.mkdir(parents=True, exist_ok=True)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(emission_dir / "vehicle_emission_all.csv", index=False)
    pd.DataFrame(summary_rows, columns=["episode", *VEHICLE_EMISSION_METRIC_KEYS]).to_csv(
        emission_dir / "vehicle_emission_summary_by_episode.csv",
        index=False,
    )


def _build_summary(rows: list[dict], args: argparse.Namespace, cfg: MasterConfig) -> dict:
    if not rows:
        summary: dict[str, Any] = {}
    else:
        exclude_keys = {"eval_episode", "sumo_seed"}
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

    summary.update({
        "controller": "sumo_actuated",
        "uses_rl_agent": False,
        "uses_checkpoint": False,
        "checkpoint": "",
        "config": str(args.config),
        "run_id": str(args.run_id),
        "eval_episodes": int(args.eval_episodes),
        "start_episode": int(args.start_episode),
        "end_episode": int(args.end_episode if args.end_episode is not None else args.eval_episodes),
        "executed_episode_count": len(rows),
        "seed": int(args.seed),
        "parallel_episodes": bool(getattr(args, "parallel_episodes", False)),
        "max_workers": int(getattr(args, "max_workers", 1) or 1),
        "sumocfg": str(cfg.env.sumo_cfg),
        "net_xml": str(cfg.env.net_xml),
        "add_xml": str(cfg.env.add_xml),
        "groups_json": str(cfg.env.intersection_groups_json),
        "emission_factor_csv": str(cfg.env.emission_factor_csv),
    })
    return summary


def _make_passive_action_infos(
    next_obs: Mapping[str, Any],
    actions: Mapping[str, int],
) -> dict[str, PassiveActionInfo]:
    action_infos: dict[str, PassiveActionInfo] = {}
    for tl_id, obs in next_obs.items():
        mask = np.asarray(obs.action_mask, dtype=bool)
        action_infos[tl_id] = PassiveActionInfo(
            tl_id=str(tl_id),
            group_id=str(obs.group_id),
            action=int(actions[tl_id]),
            valid_action_count=int(np.sum(mask)),
            action_mask=[bool(x) for x in mask.tolist()],
        )
    return action_infos


def _episode_row(
    *,
    ep: int,
    seed: int,
    wall_time_s: float,
    stats: Any,
    rewards_all: list[float],
    switch_flags: list[float],
    traffic_rewards: list[float],
    emission_penalties: list[float],
    lambda_e_values: list[float],
    weighted_emission_penalties: list[float],
    emission_penalty_ratios: list[float],
    nox_risk_means: list[float],
    nox_risk_sums: list[float],
    nox_risk_maxs: list[float],
    nox_pressure_means: list[float],
    nox_pressure_maxs: list[float],
    nox_exceed_counts: list[float],
    total_lane_nox_values: list[float],
    max_lane_nox_values: list[float],
    episode_link_nox_values: list[float],
) -> dict:
    return {
        "eval_episode": int(ep),
        "sumo_seed": int(seed),
        "wall_time_s": float(wall_time_s),
        "reward_mean": _safe_mean(rewards_all),
        "reward_sum": _safe_sum(rewards_all),
        "traffic_reward_mean": _safe_mean(traffic_rewards),
        "traffic_reward_abs_mean": _safe_mean([abs(x) for x in traffic_rewards]),
        "emission_penalty_mean": _safe_mean(emission_penalties),
        "emission_penalty_abs_mean": _safe_mean([abs(x) for x in emission_penalties]),
        "weighted_emission_penalty_mean": _safe_mean(weighted_emission_penalties),
        "weighted_emission_penalty_abs_mean": _safe_mean([abs(x) for x in weighted_emission_penalties]),
        "weighted_emission_penalty_sum": _safe_sum(weighted_emission_penalties),
        "final_reward_mean": _safe_mean(rewards_all),
        "emission_penalty_ratio_mean": _safe_mean(emission_penalty_ratios),
        "emission_penalty_ratio_p50": _safe_percentile(emission_penalty_ratios, 50),
        "emission_penalty_ratio_p90": _safe_percentile(emission_penalty_ratios, 90),
        "emission_penalty_ratio_p95": _safe_percentile(emission_penalty_ratios, 95),
        "emission_penalty_ratio_max": float(max(emission_penalty_ratios)) if emission_penalty_ratios else 0.0,
        "emission_penalty_ratio_of_means": _ratio_of_means(weighted_emission_penalties, traffic_rewards),
        "lambda_e_mean": _safe_mean(lambda_e_values),
        "nox_risk_mean": _safe_mean(nox_risk_means),
        "nox_risk_sum_mean": _safe_mean(nox_risk_sums),
        "nox_risk_max_mean": _safe_mean(nox_risk_maxs),
        "nox_pressure_mean": _safe_mean(nox_pressure_means),
        "nox_pressure_max_mean": _safe_mean(nox_pressure_maxs),
        "link_nox_mean": _safe_mean(episode_link_nox_values),
        "nox_exceed_lane_count_mean": _safe_mean(nox_exceed_counts),
        "total_lane_nox_mg": _safe_sum(total_lane_nox_values),
        "mean_step_intersection_lane_nox_mg": _safe_mean(total_lane_nox_values),
        "max_lane_nox_mg": float(np.max(max_lane_nox_values)) if max_lane_nox_values else 0.0,
        "avg_delay_s": float(getattr(stats, "avg_delay_s", 0.0) or 0.0),
        "avg_travel_time_s": float(getattr(stats, "avg_travel_time_s", 0.0) or 0.0),
        "total_arrived": int(getattr(stats, "total_arrived", 0) or 0),
        "completion_rate": float(getattr(stats, "completion_rate", 0.0) or 0.0),
        "phase_switch_rate": _safe_mean(switch_flags),
        "random_action_rate": 0.0,
        "greedy_action_rate": 0.0,
        "q_loss_mean": 0.0,
        "td_error_mean": 0.0,
        "q_pred_mean": 0.0,
        "q_target_mean": 0.0,
    }


def run_one_actuated_episode(
    *,
    args: argparse.Namespace,
    cfg: MasterConfig,
    net_info: Any,
    builder: ObsRewardBuilder,
    logger: TrainingLogger,
    profiler: StageProfiler,
    ep: int,
    seed: int,
) -> dict:
    ep_start_time = time.time()
    save_detail = bool(args.save_detail)
    episode_emission_result: Any = None
    episode_link_nox_values: list[float] = []
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
    nox_pressure_means: list[float] = []
    nox_pressure_maxs: list[float] = []
    nox_exceed_counts: list[float] = []
    total_lane_nox_values: list[float] = []
    max_lane_nox_values: list[float] = []
    stats: Any = None

    logger.begin_episode(
        ep,
        record_lane_step=True,
        record_phase_step=True,
        record_q_step=False,
    )
    env: Optional[SumoEnv] = None
    try:
        env = SumoEnv(cfg.env, net_info, port=int(args.port), use_gui=bool(args.use_gui))
        with profiler.timeit("env_start", episode=ep, step=-1, global_step=None):
            env.start(episode_id=ep, seed=seed, control_tls=False)
        with profiler.timeit("initial_collect_raw_obs", episode=ep, step=-1, global_step=None):
            raw_obs = collect_current_raw_obs(env, net_info.intersection_ids)

        for step in range(int(cfg.env.steps_per_episode)):
            with profiler.timeit("bookkeeping", episode=ep, step=step, global_step=None):
                prev_phases = {tl_id: int(raw.current_phase) for tl_id, raw in raw_obs.items()}

            with profiler.timeit("env_step", episode=ep, step=step, global_step=None):
                next_raw = env.step_passive(decision_step=step)

            with profiler.timeit("link_emission_compute", episode=ep, step=step, global_step=None):
                step_emission = getattr(env, "_last_step_emission", None)
                link_nox_values = logger.compute_step_link_emission_values(
                    step_emission=step_emission,
                    pollutant="NOx",
                )
                episode_link_nox_values.extend(link_nox_values)

            if save_detail:
                with profiler.timeit("log_step_emission", episode=ep, step=step, global_step=None):
                    logger.log_step_emission(episode=ep, step=step, step_emission=step_emission)

            with profiler.timeit("passive_action_snapshot", episode=ep, step=step, global_step=None):
                actions = {
                    tl_id: int(next_raw[tl_id].current_phase)
                    for tl_id in net_info.intersection_ids
                }

            with profiler.timeit("compute_rewards", episode=ep, step=step, global_step=None):
                rewards, comps = builder.compute_rewards(next_raw, actions, prev_phases)

            with profiler.timeit("build_next_observations", episode=ep, step=step, global_step=None):
                next_obs = builder.build_observations(next_raw)

            with profiler.timeit("bookkeeping", episode=ep, step=step, global_step=None):
                action_infos = _make_passive_action_infos(next_obs, actions)
                rewards_all.extend(float(v) for v in rewards.values())
                for tl_id, c in comps.items():
                    action = int(actions.get(tl_id, -1))
                    prev_phase = int(prev_phases.get(tl_id, -1))
                    switch_flags.append(1.0 if action != prev_phase else 0.0)
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
                    nox_risk_means.append(float(getattr(c, "nox_risk_mean", 0.0)))
                    nox_risk_sums.append(float(getattr(c, "nox_risk_sum", 0.0)))
                    nox_risk_maxs.append(float(getattr(c, "nox_risk_max", 0.0)))
                    nox_pressure_means.append(float(getattr(c, "nox_pressure_mean", 0.0)))
                    nox_pressure_maxs.append(float(getattr(c, "nox_pressure_max", 0.0)))
                    nox_exceed_counts.append(float(getattr(c, "nox_exceed_lane_count", 0.0)))
                for raw_item in next_raw.values():
                    lane_nox = np.asarray(getattr(raw_item, "NOx_mg", []), dtype=np.float64)
                    if lane_nox.size:
                        total_lane_nox_values.append(float(np.sum(lane_nox)))
                        max_lane_nox_values.append(float(np.max(lane_nox)))

            if save_detail:
                with profiler.timeit("log_step", episode=ep, step=step, global_step=None):
                    logger.log_step(ep, step, next_raw, next_obs, actions, action_infos, comps)

            with profiler.timeit("bookkeeping", episode=ep, step=step, global_step=None):
                raw_obs = next_raw
            profiler.add_step_meta(
                episode=ep,
                step=step,
                global_step=None,
                replay_size="",
                update_due=False,
                n_update_groups=0,
                n_agents=len(net_info.intersection_ids),
            )

        fcd_output_path = env.fcd_output_path
        with profiler.timeit("env_close", episode=ep, step=-1, global_step=None):
            stats = env.close()
        validate_fcd_output(
            fcd_output_path,
            episode=ep,
            enabled=bool(getattr(cfg.env, "save_fcd_output", False)),
        )
        with profiler.timeit("finalize_episode_emission", episode=ep, step=-1, global_step=None):
            recorder = getattr(env, "emission_recorder", None)
            if recorder is not None:
                try:
                    episode_emission_result = recorder.finalize_episode()
                except Exception as exc:
                    _console(f"WARN eval_episode={ep}: finalize vehicle emission failed: {exc}")
                    episode_emission_result = None
        env = None
        if save_detail:
            with profiler.timeit("flush_episode_physical", episode=ep, step=-1, global_step=None):
                logger.flush_episode_physical(ep)
            with profiler.timeit("flush_episode_emission", episode=ep, step=-1, global_step=None):
                logger.flush_episode_emission(ep)
            with profiler.timeit("write_episode_emission_result", episode=ep, step=-1, global_step=None):
                logger.write_episode_emission_result(
                    episode=ep,
                    emission_result=episode_emission_result,
                    save_step_df=True,
                    save_edge_step_raw=False,
                    save_lane_step=True,
                    save_vehicle_step=False,
                )
    finally:
        if env is not None:
            try:
                env.close(parse_tripinfo=False)
            except Exception:
                pass

    wall_time_s = time.time() - ep_start_time
    row = _episode_row(
        ep=ep,
        seed=seed,
        wall_time_s=wall_time_s,
        stats=stats,
        rewards_all=rewards_all,
        switch_flags=switch_flags,
        traffic_rewards=traffic_rewards,
        emission_penalties=emission_penalties,
        lambda_e_values=lambda_e_values,
        weighted_emission_penalties=weighted_emission_penalties,
        emission_penalty_ratios=emission_penalty_ratios,
        nox_risk_means=nox_risk_means,
        nox_risk_sums=nox_risk_sums,
        nox_risk_maxs=nox_risk_maxs,
        nox_pressure_means=nox_pressure_means,
        nox_pressure_maxs=nox_pressure_maxs,
        nox_exceed_counts=nox_exceed_counts,
        total_lane_nox_values=total_lane_nox_values,
        max_lane_nox_values=max_lane_nox_values,
        episode_link_nox_values=episode_link_nox_values,
    )
    row.update(_vehicle_emission_episode_metrics(episode_emission_result))
    if profiler.enabled:
        profiler.finish_episode(
            episode=ep,
            wall_time_s=wall_time_s,
            steps=int(cfg.env.steps_per_episode),
            n_agents=len(net_info.intersection_ids),
            replay_size_end="",
        )
    _console(
        f"controller=sumo_actuated eval_episode={ep}/{int(args.eval_episodes)} "
        f"seed={seed} wall_time={wall_time_s:.1f}s "
        f"reward_mean={row['reward_mean']:.3f} "
        f"avg_delay={row['avg_delay_s']:.3f}s "
        f"completion={row['completion_rate']:.3f} arrived={row['total_arrived']}"
    )
    return row


def _model_meta() -> dict[str, Any]:
    return {
        "controller": "sumo_actuated",
        "uses_rl_agent": False,
        "uses_checkpoint": False,
        "action_source": "SUMO native actuated program",
    }


def _episode_items(cfg: MasterConfig, eval_episodes: int, start_episode: int = 1, end_episode: Optional[int] = None) -> list[tuple[int, int]]:
    end_episode = int(eval_episodes) if end_episode is None else int(end_episode)
    if not 1 <= int(start_episode) <= end_episode <= int(eval_episodes):
        raise ValueError("Require 1 <= --start-episode <= --end-episode <= --eval-episodes")
    sumo_seeds = build_sumo_seed_list(
        global_seed=int(cfg.seed),
        total_episodes=int(eval_episodes),
        mode=str(cfg.env.sumo_seed_mode),
    )
    return [
        (ep, int(sumo_seeds[ep - 1]))
        for ep in range(int(start_episode), end_episode + 1)
    ]


def _write_sumo_seed_json(out_dir: str, cfg: MasterConfig, episode_items: list[tuple[int, int]]) -> None:
    with open(os.path.join(out_dir, "eval_sumo_seeds.json"), "w", encoding="utf-8") as f:
        json.dump({
            "controller": "sumo_actuated",
            "global_seed": int(cfg.seed),
            "sumo_seed_mode": str(cfg.env.sumo_seed_mode),
            "eval_episodes": int(len(episode_items)),
            "sumo_seeds": [
                {"eval_episode": int(ep), "sumo_seed": int(seed)}
                for ep, seed in episode_items
            ],
        }, f, indent=2, ensure_ascii=False)


def split_episode_batches(
    episodes: list[tuple[int, int]],
    max_workers: int,
) -> list[list[tuple[int, int]]]:
    n_workers = max(1, min(int(max_workers), len(episodes)))
    batches: list[list[tuple[int, int]]] = [[] for _ in range(n_workers)]
    for i, item in enumerate(episodes):
        batches[i % n_workers].append(item)
    return [batch for batch in batches if batch]


def _read_csv_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _merge_profile_csvs(out_dir: str, prefix: str, out_name: str) -> None:
    rows: list[dict] = []
    for path in sorted(Path(out_dir).glob(f"{prefix}*.csv")):
        rows.extend(_read_csv_rows(str(path)))
    if rows:
        write_csv(os.path.join(out_dir, out_name), rows)


def _write_parallel_profile_summary(out_dir: str) -> None:
    worker_summaries: list[dict] = []
    for path in sorted(Path(out_dir).glob("eval_profile_summary_worker_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            payload["worker_summary_file"] = str(path)
            worker_summaries.append(payload)
        except Exception:
            continue
    if worker_summaries:
        with open(os.path.join(out_dir, "eval_profile_summary.json"), "w", encoding="utf-8") as f:
            json.dump({
                "enabled": True,
                "parallel_episodes": True,
                "worker_summaries": worker_summaries,
            }, f, indent=2, ensure_ascii=False)


def run_episode_batch_worker(payload: dict) -> dict:
    if bool(payload.get("force_traci", True)):
        os.environ["SUMO_FORCE_TRACI"] = "1"

    worker_id = int(payload["worker_id"])
    args = argparse.Namespace(**payload["args"])
    args.port = int(payload["base_port"]) + worker_id * int(payload["port_stride"])
    cfg = apply_overrides(get_config(args.config), args)
    set_global_seed_local(int(cfg.seed))

    net_info = parse_network(cfg.env.net_xml, cfg.env.add_xml, cfg.env.intersection_groups_json)
    builder = ObsRewardBuilder(cfg, net_info)
    logger = TrainingLogger(
        cfg,
        net_info,
        model_meta=_model_meta(),
        write_metadata=False,
    )
    profiler = StageProfiler(
        enabled=bool(args.profile),
        sync_cuda=bool(args.profile_sync_cuda),
    )

    rows: list[dict] = []
    errors: list[dict] = []
    for ep, seed in [(int(ep), int(seed)) for ep, seed in payload["episodes"]]:
        try:
            row = run_one_actuated_episode(
                args=args,
                cfg=cfg,
                net_info=net_info,
                builder=builder,
                logger=logger,
                profiler=profiler,
                ep=ep,
                seed=seed,
            )
            row["worker_id"] = int(worker_id)
            row["port"] = int(args.port)
            rows.append(row)
        except Exception as exc:
            errors.append({
                "worker_id": int(worker_id),
                "eval_episode": int(ep),
                "sumo_seed": int(seed),
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
    }


def run_single_evaluation_serial(args: argparse.Namespace) -> Dict[str, Any]:
    os.environ["SUMO_FORCE_TRACI"] = "1"
    cfg = apply_overrides(get_config(args.config), args)
    set_global_seed_local(cfg.seed)
    net_info = parse_network(cfg.env.net_xml, cfg.env.add_xml, cfg.env.intersection_groups_json)
    builder = ObsRewardBuilder(cfg, net_info)
    logger = TrainingLogger(
        cfg,
        net_info,
        model_meta=_model_meta(),
    )
    profiler = StageProfiler(
        enabled=bool(args.profile),
        sync_cuda=bool(args.profile_sync_cuda),
    )

    out_dir = cfg.log.run_dir
    os.makedirs(out_dir, exist_ok=True)
    eval_episodes = int(args.eval_episodes)
    if eval_episodes <= 0:
        raise ValueError("--eval-episodes must be > 0")
    episode_items = _episode_items(cfg, eval_episodes, args.start_episode, args.end_episode)
    _write_sumo_seed_json(out_dir, cfg, episode_items)

    rows: list[dict] = []
    errors: list[dict] = []
    for ep, seed in episode_items:
        try:
            rows.append(run_one_actuated_episode(
                args=args,
                cfg=cfg,
                net_info=net_info,
                builder=builder,
                logger=logger,
                profiler=profiler,
                ep=ep,
                seed=seed,
            ))
        except Exception as exc:
            errors.append({
                "eval_episode": int(ep),
                "sumo_seed": int(seed),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
            with open(os.path.join(out_dir, "eval_errors.json"), "w", encoding="utf-8") as f:
                json.dump(errors, f, indent=2, ensure_ascii=False)
            raise

    rows = sorted(rows, key=lambda r: int(r["eval_episode"]))
    with profiler.timeit("write_eval_outputs", episode=-1, step=-1, global_step=None):
        write_csv(os.path.join(out_dir, "eval_episode_log.csv"), rows)
        summary = _build_summary(rows, args, cfg)
        with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        merge_vehicle_emission_outputs(out_dir)
        if EVAL_SAVE_VEHICLE_STATE:
            consolidate_vehicle_state_outputs(out_dir)
    if profiler.enabled:
        profiler.flush_step_csv(os.path.join(out_dir, "eval_profile_step_log.csv"))
        profiler.flush_episode_csv(os.path.join(out_dir, "eval_profile_episode_log.csv"))
        profiler.save_summary_json(os.path.join(out_dir, "eval_profile_summary.json"))
    _console(f"actuated evaluation saved to: {out_dir}")
    return summary


def run_single_evaluation_parallel(args: argparse.Namespace) -> Dict[str, Any]:
    if bool(args.use_gui):
        raise ValueError("--use-gui is not supported with --parallel; use --serial --use-gui")
    os.environ["SUMO_FORCE_TRACI"] = "1"
    cfg = apply_overrides(get_config(args.config), args)
    out_dir = cfg.log.run_dir
    os.makedirs(out_dir, exist_ok=True)
    eval_episodes = int(args.eval_episodes)
    if eval_episodes <= 0:
        raise ValueError("--eval-episodes must be > 0")

    episode_items = _episode_items(cfg, eval_episodes, args.start_episode, args.end_episode)
    _write_sumo_seed_json(out_dir, cfg, episode_items)

    net_info = parse_network(cfg.env.net_xml, cfg.env.add_xml, cfg.env.intersection_groups_json)
    TrainingLogger(
        cfg,
        net_info,
        model_meta=_model_meta(),
        write_metadata=True,
    )

    batches = split_episode_batches(episode_items, int(args.max_workers))
    payloads = []
    for worker_id, batch in enumerate(batches):
        payloads.append({
            "worker_id": int(worker_id),
            "episodes": batch,
            "args": vars(args),
            "base_port": int(args.port),
            "port_stride": int(args.port_stride),
            "force_traci": True,
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
                "worker_id": int(result.get("worker_id", 0)),
                "port": int(result.get("port", 0)),
                "n_rows": int(len(rows)),
                "n_errors": int(len(errors)),
            })

    all_rows = sorted(all_rows, key=lambda r: int(r["eval_episode"]))
    worker_infos = sorted(worker_infos, key=lambda r: int(r.get("worker_id", 0) or 0))
    write_csv(os.path.join(out_dir, "eval_worker_log.csv"), worker_infos)

    if all_errors:
        with open(os.path.join(out_dir, "eval_errors.json"), "w", encoding="utf-8") as f:
            json.dump(all_errors, f, indent=2, ensure_ascii=False)
        raise RuntimeError(f"{len(all_errors)} actuated evaluation episodes failed. See eval_errors.json")

    write_csv(os.path.join(out_dir, "eval_episode_log.csv"), all_rows)
    summary = _build_summary(all_rows, args, cfg)
    with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    merge_vehicle_emission_outputs(out_dir)
    if EVAL_SAVE_VEHICLE_STATE:
        consolidate_vehicle_state_outputs(out_dir)

    if bool(args.profile):
        _merge_profile_csvs(out_dir, "eval_profile_step_log_worker_", "eval_profile_step_log.csv")
        _merge_profile_csvs(out_dir, "eval_profile_episode_log_worker_", "eval_profile_episode_log.csv")
        _write_parallel_profile_summary(out_dir)

    _console(f"parallel actuated evaluation saved to: {out_dir}")
    return summary


def run_single_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    if bool(getattr(args, "parallel_episodes", True)):
        return run_single_evaluation_parallel(args)
    return run_single_evaluation_serial(args)


def run_calibration_suite(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run demand scenarios sequentially, with up to three episode workers."""
    config_path = Path(args.config).expanduser().resolve()
    network_root = Path(__file__).resolve().parent.parent / "data" / "grid36"
    calibration_root = (
        Path(args.calibration_root).expanduser().resolve()
        if args.calibration_root
        else network_root / "calibration_sumocfg"
    )
    output_root = Path(args.calibration_output_root).expanduser().resolve()
    if not calibration_root.is_dir():
        raise FileNotFoundError(f"Calibration SUMO config directory not found: {calibration_root}")
    cases = []
    for demand_dir in sorted(p for p in calibration_root.iterdir() if p.is_dir()):
        sumocfgs = sorted(demand_dir.glob("*.sumocfg"))
        if len(sumocfgs) != 1:
            raise ValueError(f"Expected one .sumocfg in {demand_dir}, found {len(sumocfgs)}")
        cases.append((demand_dir.name, sumocfgs[0]))
    if not cases:
        raise FileNotFoundError(f"No calibration demand directories under {calibration_root}")

    summaries = []
    for demand, sumocfg in cases:
        case_args = argparse.Namespace(**vars(args))
        case_args.config = str(config_path)
        case_args.sumocfg = str(sumocfg)
        case_args.net_xml = str(network_root / "truck_sensitive_grid36.net.xml")
        case_args.add_xml = str(network_root / "truck_sensitive_grid36.add.xml")
        case_args.groups_json = str(network_root / "intersection_groups.json")
        case_args.episode_duration = args.episode_duration or 4200
        case_args.eval_episodes = args.eval_episodes
        case_args.log_root = str(output_root)
        case_args.run_id = demand
        case_args.parallel_episodes = args.parallel_episodes
        case_args.max_workers = args.max_workers
        _console(
            f"[calibration] demand={demand} output={output_root / demand} "
            "duration=4200s episodes=12 workers=3"
        )
        summaries.append(run_single_evaluation_parallel(case_args))
    return summaries


def main() -> None:
    run_started = time.perf_counter()
    mp.freeze_support()
    args = parse_args()
    separator = "=" * 78
    _console(separator)
    _console("Start evaluation: SUMO actuated")
    _console(
        f"episodes     : {args.eval_episodes}, seed={args.seed}, "
        f"parallel={args.parallel_episodes}"
    )
    _console(f"output       : {Path(args.log_root).resolve() / args.run_id}")
    _console(separator)
    if args.calibration_all:
        run_calibration_suite(args)
    else:
        run_single_evaluation(args)
    total_wall_time_s = time.perf_counter() - run_started
    _console(separator)
    _console(
        f"Evaluation completed: SUMO actuated, "
        f"total_time={total_wall_time_s:.1f}s "
        f"({_format_elapsed(total_wall_time_s)})"
    )
    _console(separator)


if __name__ == "__main__":
    main()
