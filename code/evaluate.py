# -*- coding: utf-8 -*-
"""
evaluate.py
===========
Independent batch evaluation entry point for FLORE.

The evaluation logic stays deterministic: epsilon=0, masked argmax actions,
no replay writes, and no network updates.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from agent import MGMQAgentManager, set_global_seed
from config import MasterConfig, build_sumo_seed_list, get_config
from env import SumoEnv, StepRawObs
from logger import TrainingLogger
from network_parser import parse_network
from obs_reward import ObsRewardBuilder, _resolve_effective_risk_weights
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
# Fixed batch evaluation settings
# ================================================================
EVAL_EPISODES = 20
EVAL_SEED = 19
EVAL_LOG_ROOT = "results/evaluation/grid36"
EVAL_GROUP_OUTPUT_BY_CHECKPOINT_DIR = True
EVAL_PORT = 8813
EVAL_USE_GUI = False

# ================================================================
# Parallel episode evaluation settings
# ================================================================
EVAL_PARALLEL_EPISODES = True
EVAL_MAX_WORKERS = 3
EVAL_PORT_STRIDE = 10
EVAL_FORCE_TRACI = True
EVAL_PARALLEL_DEVICE = "cpu"

# Parallel evaluation defaults to no shared profiling files. If profiling is
# enabled later, each worker writes worker-specific profile files.
EVAL_PROFILE = False
EVAL_PROFILE_SYNC_CUDA = False
EVAL_ALLOW_THRESHOLD_MISMATCH = True

# True saves lane/phase/q/emission step details. Episode directories are unique,
# so this is safe in parallel; set False for faster eval_episode_log-only runs.
EVAL_SAVE_DETAIL = True

# ================================================================
# SUMO FCD output settings
# ================================================================
EVAL_SAVE_FCD = False
EVAL_SAVE_VEHICLE_SECOND = False
EVAL_SAVE_VEHICLE_STATE = True
EVAL_SAVE_TLS_PHASE = False
EVAL_FCD_ACCELERATION = True
EVAL_FCD_PERIOD_S = 1.0


def resolve_path(path: Optional[str], *, must_exist: bool = False) -> Optional[str]:
    if path is None or str(path).strip() == "":
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()
    if must_exist and not p.exists():
        raise FileNotFoundError(str(p))
    return str(p)


def build_checkpoint_cases(checkpoint_dir: str, config: str) -> list[dict[str, str]]:
    """Build one evaluation case per .pt file, using its stem as output folder."""
    resolved_dir = resolve_path(checkpoint_dir, must_exist=True)
    checkpoint_root = Path(resolved_dir or checkpoint_dir)
    if not checkpoint_root.is_dir():
        raise NotADirectoryError(str(checkpoint_root))

    checkpoint_paths = sorted(
        (path for path in checkpoint_root.glob("*.pt") if path.is_file()),
        key=lambda path: path.name.lower(),
    )
    if not checkpoint_paths:
        raise FileNotFoundError(f"No .pt checkpoints found in: {checkpoint_root}")

    cases: list[dict[str, str]] = []
    for checkpoint_path in checkpoint_paths:
        try:
            checkpoint = checkpoint_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            checkpoint = str(checkpoint_path)
        cases.append({
            "name": checkpoint_path.stem,
            "config": str(config),
            "checkpoint": checkpoint,
            "run_id": checkpoint_path.stem,
        })
    return cases


def parse_args() -> argparse.Namespace:
    global EVAL_PARALLEL_EPISODES, EVAL_MAX_WORKERS, EVAL_PORT, EVAL_PORT_STRIDE
    global EVAL_SAVE_DETAIL, EVAL_SAVE_VEHICLE_STATE, EVAL_PARALLEL_DEVICE
    global EVAL_ALLOW_THRESHOLD_MISMATCH
    p = argparse.ArgumentParser(description="Evaluate a FLORE checkpoint without learning")
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--eval-episodes', type=int, default=20)
    p.add_argument('--seed', type=int, default=19)
    p.add_argument('--log-root', default='results/evaluation/grid36')
    p.add_argument('--run-id', default='flore')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--port', type=int, default=8813)
    p.add_argument('--port-stride', type=int, default=10)
    p.add_argument('--device', default='cpu')
    p.add_argument('--use-gui', action='store_true')
    p.add_argument('--save-detail', action='store_true')
    p.add_argument('--save-vehicle-state', action='store_true')
    for option in ('sumocfg','net-xml','add-xml','groups-json','emission-factor-csv'):
        p.add_argument('--' + option, default=None)
    args = p.parse_args()
    if args.eval_episodes < 1 or args.eval_episodes > 2000 or args.workers < 1:
        p.error('episodes must be 1..2000 and workers must be positive')
    if args.use_gui and args.workers > 1:
        p.error('GUI evaluation requires --workers 1')
    args.case_metadata = {}
    EVAL_PARALLEL_EPISODES = args.workers > 1
    EVAL_MAX_WORKERS = args.workers
    EVAL_PORT, EVAL_PORT_STRIDE = args.port, args.port_stride
    EVAL_SAVE_DETAIL, EVAL_SAVE_VEHICLE_STATE = args.save_detail, args.save_vehicle_state
    EVAL_PARALLEL_DEVICE = args.device
    EVAL_ALLOW_THRESHOLD_MISMATCH = False
    return args


def apply_overrides(cfg: MasterConfig, args: argparse.Namespace) -> MasterConfig:
    cfg.seed = int(args.seed)
    cfg.device = str(args.device)
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
    cfg.env.save_detector_outputs = False
    cfg.env.save_sumo_aux_outputs = False
    cfg.env.save_fcd_output = bool(EVAL_SAVE_FCD)
    cfg.env.save_vehicle_second_output = bool(EVAL_SAVE_VEHICLE_SECOND)
    cfg.env.save_vehicle_state_output = bool(EVAL_SAVE_VEHICLE_STATE)
    cfg.env.save_tls_phase_outputs = bool(EVAL_SAVE_TLS_PHASE)
    cfg.env.fcd_output_acceleration = bool(EVAL_FCD_ACCELERATION)
    cfg.env.fcd_output_period = float(EVAL_FCD_PERIOD_S)
    cfg.env.sumo_cfg = resolve_path(cfg.env.sumo_cfg, must_exist=True) or cfg.env.sumo_cfg
    cfg.env.net_xml = resolve_path(cfg.env.net_xml, must_exist=True) or cfg.env.net_xml
    cfg.env.add_xml = resolve_path(cfg.env.add_xml, must_exist=True) or cfg.env.add_xml
    cfg.env.intersection_groups_json = resolve_path(cfg.env.intersection_groups_json, must_exist=False) or cfg.env.intersection_groups_json
    cfg.env.emission_factor_csv = resolve_path(cfg.env.emission_factor_csv, must_exist=False) or cfg.env.emission_factor_csv
    cfg.log.log_root = resolve_path(cfg.log.log_root, must_exist=False) or cfg.log.log_root
    cfg.env.tripinfo_dir = os.path.join(cfg.log.log_root, cfg.log.run_id, "tripinfo")
    cfg.validate()
    return cfg


def _risk_evaluation_metadata(
    cfg: MasterConfig,
    checkpoint_payload: Mapping[str, Any],
    evaluation_seed: int,
) -> dict[str, Any]:
    er = cfg.emission_risk
    risk_mode = str(er.risk_mode).strip().lower()
    if risk_mode == "hybrid_pressure_tail":
        effective_base_weight, effective_tail_weight = (
            _resolve_effective_risk_weights(er.base_weight, er.tail_weight)
        )
    else:
        effective_base_weight, effective_tail_weight = 0.0, 1.0

    training_seed: int | str = ""
    checkpoint_cfg = checkpoint_payload.get("cfg", {})
    if isinstance(checkpoint_cfg, Mapping):
        raw_seed = checkpoint_cfg.get("seed")
        if raw_seed is not None:
            training_seed = int(raw_seed)

    return {
        "risk_mode": risk_mode,
        "base_weight": float(er.base_weight),
        "tail_weight": float(er.tail_weight),
        "effective_base_weight": float(effective_base_weight),
        "effective_tail_weight": float(effective_tail_weight),
        "training_seed": training_seed,
        "evaluation_seed": int(evaluation_seed),
    }


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


def _build_summary(rows: list[dict], args: argparse.Namespace, *, parallel: bool, max_workers: int) -> dict:
    case_metadata = dict(getattr(args, "case_metadata", {}) or {})
    if not rows:
        summary: dict[str, Any] = {}
    else:
        exclude_keys = {
            "eval_episode",
            "sumo_seed",
            "worker_id",
            "port",
            *case_metadata.keys(),
        }
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
    summary["config"] = str(args.config)
    summary["run_id"] = str(args.run_id)
    summary["parallel_episodes"] = bool(parallel)
    summary["max_workers"] = int(max_workers)
    summary.update(case_metadata)
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
        record_lane_step=bool(EVAL_SAVE_DETAIL),
        record_phase_step=bool(EVAL_SAVE_DETAIL),
        record_q_step=bool(EVAL_SAVE_DETAIL),
    )
    env: Optional[SumoEnv] = None
    try:
        env = SumoEnv(cfg.env, net_info, port=int(args.port), use_gui=bool(args.use_gui))
        env.start(episode_id=ep, seed=seed, control_tls=True)
        with profiler.timeit("initial_collect_raw_obs", episode=ep, step=-1, global_step=None):
            raw_obs = collect_current_raw_obs(env, net_info.intersection_ids)
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
        reward_risk_sums: list[float] = []
        reward_risk_maxs: list[float] = []
        mean_lane_risks: list[float] = []
        p95_lane_risks: list[float] = []
        nox_exceed_counts: list[float] = []
        total_lane_nox_values: list[float] = []
        max_lane_nox_values: list[float] = []

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
                reward_risk_sums.extend(float(getattr(c, "reward_risk_sum", 0.0)) for c in comps.values())
                reward_risk_maxs.extend(float(getattr(c, "reward_risk_max", 0.0)) for c in comps.values())
                mean_lane_risks.extend(float(getattr(c, "mean_lane_risk", 0.0)) for c in comps.values())
                p95_lane_risks.extend(float(getattr(c, "p95_lane_risk", 0.0)) for c in comps.values())
                nox_exceed_counts.extend(float(getattr(c, "nox_exceed_lane_count", 0.0)) for c in comps.values())
                for raw_item in next_raw.values():
                    lane_nox = np.asarray(getattr(raw_item, "NOx_mg", []), dtype=np.float64)
                    if lane_nox.size:
                        total_lane_nox_values.append(float(np.sum(lane_nox)))
                        max_lane_nox_values.append(float(np.max(lane_nox)))
            if bool(EVAL_SAVE_DETAIL):
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
                n_agents=len(net_info.intersection_ids),
            )

        with profiler.timeit("env_close", episode=ep, step=-1, global_step=None):
            stats = env.close()
        with profiler.timeit("finalize_episode_emission", episode=ep, step=-1, global_step=None):
            recorder = getattr(env, "emission_recorder", None)
            if recorder is not None:
                try:
                    episode_emission_result = recorder.finalize_episode()
                except Exception as exc:
                    _console(f"WARN eval_episode={ep}: finalize vehicle emission failed: {exc}")
                    episode_emission_result = None
        env = None
        if bool(EVAL_SAVE_DETAIL):
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

    ep_wall_time_s = time.time() - ep_start_time
    row = {
        "eval_episode": int(ep),
        "sumo_seed": int(seed),
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
        "risk_reward_mode": str(cfg.emission_risk.reward_risk_mode),
        "reward_risk_sum_mean": _safe_mean(reward_risk_sums),
        "reward_risk_max_mean": _safe_mean(reward_risk_maxs),
        "mean_lane_risk": _safe_mean(mean_lane_risks),
        "p95_lane_risk": _safe_mean(p95_lane_risks),
        "link_nox_mean": float(np.mean(episode_link_nox_values)) if episode_link_nox_values else 0.0,
        "nox_exceed_lane_count_mean": float(np.mean(nox_exceed_counts)) if nox_exceed_counts else 0.0,
        "total_lane_nox_mg": float(np.sum(total_lane_nox_values)) if total_lane_nox_values else 0.0,
        "mean_step_intersection_lane_nox_mg": float(np.mean(total_lane_nox_values)) if total_lane_nox_values else 0.0,
        "max_lane_nox_mg": float(np.max(max_lane_nox_values)) if max_lane_nox_values else 0.0,
        "avg_delay_s": float(getattr(stats, "avg_delay_s", 0.0) or 0.0),
        "avg_travel_time_s": float(getattr(stats, "avg_travel_time_s", 0.0) or 0.0),
        "total_arrived": int(getattr(stats, "total_arrived", 0) or 0),
        "completion_rate": float(getattr(stats, "completion_rate", 0.0) or 0.0),
        "phase_switch_rate": float(np.mean(switch_flags)) if switch_flags else 0.0,
    }
    row.update(_vehicle_emission_episode_metrics(episode_emission_result))
    row.update(dict(getattr(args, "case_metadata", {}) or {}))
    if profiler.enabled:
        profiler.finish_episode(
            episode=ep,
            wall_time_s=time.time() - ep_start_time,
            steps=int(cfg.env.steps_per_episode),
            n_agents=len(net_info.intersection_ids),
            replay_size_end="",
        )
    _console(
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
    cfg.log.save_step_physical = bool(payload["save_detail"])
    cfg.log.save_q_step = bool(payload["save_detail"])
    cfg.log.save_emission_step = bool(payload["save_detail"])
    cfg.log.emission_step_log_interval_train = 1 if bool(payload["save_detail"]) else 0

    set_global_seed(int(cfg.seed))
    net_info = parse_network(cfg.env.net_xml, cfg.env.add_xml, cfg.env.intersection_groups_json)
    builder = ObsRewardBuilder(cfg, net_info)
    agent = MGMQAgentManager(cfg, net_info, device=str(payload.get("device", "cpu")))
    checkpoint_payload = agent.load_checkpoint(
        resolve_path(args.checkpoint, must_exist=True) or args.checkpoint,
        load_optimizer=False,
        allow_threshold_mismatch=bool(payload.get("allow_threshold_mismatch", False)),
    )
    args.case_metadata = _risk_evaluation_metadata(
        cfg,
        checkpoint_payload,
        int(args.seed),
    )
    agent.online_bank.eval()
    logger = TrainingLogger(
        cfg,
        net_info,
        model_meta=agent.online_bank.model_meta(),
        write_metadata=False,
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


def run_single_evaluation_serial(args: argparse.Namespace) -> Dict[str, Any]:
    if bool(EVAL_FORCE_TRACI):
        os.environ["SUMO_FORCE_TRACI"] = "1"
    cfg = apply_overrides(get_config(args.config), args)
    cfg.log.save_step_physical = bool(EVAL_SAVE_DETAIL)
    cfg.log.save_q_step = bool(EVAL_SAVE_DETAIL)
    cfg.log.save_emission_step = bool(EVAL_SAVE_DETAIL)
    cfg.log.emission_step_log_interval_train = 1 if bool(EVAL_SAVE_DETAIL) else 0
    set_global_seed(cfg.seed)
    net_info = parse_network(cfg.env.net_xml, cfg.env.add_xml, cfg.env.intersection_groups_json)
    builder = ObsRewardBuilder(cfg, net_info)
    agent = MGMQAgentManager(cfg, net_info)
    checkpoint_payload = agent.load_checkpoint(
        resolve_path(args.checkpoint, must_exist=True) or args.checkpoint,
        load_optimizer=False,
        allow_threshold_mismatch=bool(EVAL_ALLOW_THRESHOLD_MISMATCH),
    )
    args.case_metadata = _risk_evaluation_metadata(
        cfg,
        checkpoint_payload,
        int(args.seed),
    )
    agent.online_bank.eval()
    logger = TrainingLogger(cfg, net_info, model_meta=agent.online_bank.model_meta())
    profiler = StageProfiler(
        enabled=bool(EVAL_PROFILE),
        sync_cuda=bool(EVAL_PROFILE_SYNC_CUDA),
    )

    out_dir = cfg.log.run_dir
    os.makedirs(out_dir, exist_ok=True)
    eval_episodes = int(args.eval_episodes)
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
        for ep, seed in _episode_items(cfg, eval_episodes)
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
    _console(f"evaluation saved to: {out_dir}")
    return summary


def run_single_evaluation_parallel(args: argparse.Namespace) -> Dict[str, Any]:
    base_cfg = apply_overrides(get_config(args.config), args)
    base_cfg.log.save_step_physical = bool(EVAL_SAVE_DETAIL)
    base_cfg.log.save_q_step = bool(EVAL_SAVE_DETAIL)
    base_cfg.log.save_emission_step = bool(EVAL_SAVE_DETAIL)
    base_cfg.log.emission_step_log_interval_train = 1 if bool(EVAL_SAVE_DETAIL) else 0
    base_out_dir = base_cfg.log.run_dir
    os.makedirs(base_out_dir, exist_ok=True)

    eval_episodes = int(args.eval_episodes)
    episode_items = _episode_items(base_cfg, eval_episodes)
    batches = split_episode_batches(episode_items, EVAL_MAX_WORKERS)

    net_info = parse_network(base_cfg.env.net_xml, base_cfg.env.add_xml, base_cfg.env.intersection_groups_json)
    meta_agent = MGMQAgentManager(base_cfg, net_info, device=str(EVAL_PARALLEL_DEVICE))
    checkpoint_payload = meta_agent.load_checkpoint(
        resolve_path(args.checkpoint, must_exist=True) or args.checkpoint,
        load_optimizer=False,
        allow_threshold_mismatch=bool(EVAL_ALLOW_THRESHOLD_MISMATCH),
    )
    args.case_metadata = _risk_evaluation_metadata(
        base_cfg,
        checkpoint_payload,
        int(args.seed),
    )
    TrainingLogger(
        base_cfg,
        net_info,
        model_meta=meta_agent.online_bank.model_meta(),
        write_metadata=True,
    )

    payloads = []
    for worker_id, batch in enumerate(batches):
        payloads.append({
            "worker_id": int(worker_id),
            "episodes": batch,
            "args": vars(args),
            "run_id": str(args.run_id),
            "base_port": int(EVAL_PORT),
            "port_stride": int(EVAL_PORT_STRIDE),
            "eval_episodes": int(eval_episodes),
            "save_detail": bool(EVAL_SAVE_DETAIL),
            "profile": bool(EVAL_PROFILE),
            "profile_sync_cuda": bool(EVAL_PROFILE_SYNC_CUDA),
            "force_traci": bool(EVAL_FORCE_TRACI),
            "allow_threshold_mismatch": bool(EVAL_ALLOW_THRESHOLD_MISMATCH),
            "device": str(EVAL_PARALLEL_DEVICE),
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
    _console(f"parallel evaluation saved to: {base_out_dir}")
    return summary


def main() -> None:
    mp.freeze_support()
    args = parse_args()
    out = Path(resolve_path(args.log_root)) / args.run_id
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output is not empty: {out}; choose another --run-id")
    if EVAL_PARALLEL_EPISODES:
        run_single_evaluation_parallel(args)
    else:
        run_single_evaluation_serial(args)

if __name__ == '__main__':
    main()
