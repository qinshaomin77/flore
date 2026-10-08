
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from agent import MGMQAgentManager, set_global_seed
from buffer import GlobalTransition
from config import MasterConfig, build_sumo_seed_list, get_config, make_run_id
from env import SumoEnv, StepRawObs
from logger import TrainingLogger
from network_parser import parse_network
from obs_reward import ObsRewardBuilder
from profiler import StageProfiler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PARAMETER_SOURCES_JSON = PROJECT_ROOT / "data/kunshan/intersection_parameter_sources.json"

def load_control_sets(net_info: Any) -> tuple[list[str], list[str]]:
    with PARAMETER_SOURCES_JSON.open("r", encoding="utf-8") as f:
        sources = json.load(f)
    all_ids = set(net_info.intersection_ids)
    if set(sources) != all_ids:
        raise ValueError(
            f"Control mapping mismatch: missing={sorted(all_ids - set(sources))}, "
            f"extra={sorted(set(sources) - all_ids)}"
        )
    controlled = [tl_id for tl_id in net_info.intersection_ids if sources[tl_id] != "other"]
    actuated = [tl_id for tl_id in net_info.intersection_ids if sources[tl_id] == "other"]
    if len(controlled) != 25 or set(actuated) != {"nt21", "nt40"}:
        raise ValueError(f"Expected 25 RL TLS and nt21/nt40 actuated, got {controlled}, {actuated}")
    return controlled, actuated

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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train strict-reproduction MGMQ-DDQN traffic signal control model.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--sumocfg", type=str, default=None)
    p.add_argument("--net-xml", type=str, default=None)
    p.add_argument("--add-xml", type=str, default=None)
    p.add_argument("--groups-json", type=str, default=None)
    p.add_argument("--emission-factor-csv", type=str, default=None)
    p.add_argument(
        "--reward-risk-mode",
        "--risk_reward_mode",
        dest="reward_risk_mode",
        choices=("full", "no_tail", "sum_only"),
        default=None,
    )
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-root", type=str, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--port", type=int, default=8813)
    p.add_argument("--use-gui", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--load-optimizer", action="store_true")
    p.add_argument("--deterministic", action="store_true", help="rollout 使用 masked argmax，仅调试用")
    p.add_argument("--profile", action="store_true", help="enable lightweight stage profiling")
    p.add_argument("--profile-sync-cuda", action="store_true", help="synchronize CUDA before/after timed stages")
    return p.parse_args()

def apply_overrides(cfg: MasterConfig, args: argparse.Namespace) -> MasterConfig:
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.episodes is not None:
        cfg.ddqn.total_episodes = int(args.episodes)
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
    if args.reward_risk_mode is not None:
        cfg.emission_risk.reward_risk_mode = str(args.reward_risk_mode)
    if args.log_root:
        cfg.log.log_root = str(args.log_root)
    if args.run_id:
        cfg.log.run_id = make_run_id(str(args.run_id))
    else:
        cfg.log.run_id = make_run_id(str(cfg.log.run_id))

    cfg.env.sumo_cfg = resolve_path(cfg.env.sumo_cfg, must_exist=True) or cfg.env.sumo_cfg
    cfg.env.net_xml = resolve_path(cfg.env.net_xml, must_exist=True) or cfg.env.net_xml
    cfg.env.add_xml = resolve_path(cfg.env.add_xml, must_exist=True) or cfg.env.add_xml
    cfg.env.intersection_groups_json = resolve_path(cfg.env.intersection_groups_json, must_exist=False) or cfg.env.intersection_groups_json
    cfg.env.emission_factor_csv = resolve_path(cfg.env.emission_factor_csv, must_exist=False) or cfg.env.emission_factor_csv
    cfg.log.log_root = resolve_path(cfg.log.log_root, must_exist=False) or cfg.log.log_root
    cfg.env.tripinfo_dir = os.path.join(cfg.log.log_root, cfg.log.run_id, "tripinfo")
    cfg.validate()
    return cfg

def collect_current_raw_obs(env: SumoEnv, tl_ids: list[str]) -> Dict[str, StepRawObs]:
    if not hasattr(env, "_collect_obs"):
        raise AttributeError("SumoEnv missing _collect_obs")
    return {tl_id: env._collect_obs(tl_id, None) for tl_id in tl_ids}  # type: ignore[attr-defined]

def _format_elapsed_hms(elapsed_seconds: float) -> str:
    total_seconds = max(0, int(elapsed_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def summarize_episode(
    episode: int,
    wall_time_s: float,
    sumo_seed: Optional[int],
    eps_values: list[float],
    reward_values: list[float],
    reward_components: list[Any],
    action_infos: list[Any],
    update_rows: list[Any],
    traffic_stats: Any,
    replay_size: int,
    n_agents: int,
    best_score: float,
    is_best: bool,
    best_metric: str,
) -> Dict[str, Any]:
    rp = [float(c.rp) for c in reward_components]
    rwn = [float(c.rwn) for c in reward_components]
    rpn = [float(c.rpn) for c in reward_components]
    rwt = [float(c.rwt) for c in reward_components]
    traffic_reward = [float(getattr(c, "traffic_reward", c.reward)) for c in reward_components]
    emission_penalty = [float(getattr(c, "emission_penalty", 0.0)) for c in reward_components]
    lambda_e_values = [float(getattr(c, "lambda_e", 0.0)) for c in reward_components]
    weighted_emission_penalty = [
        float(le * ep)
        for le, ep in zip(lambda_e_values, emission_penalty)
    ]
    traffic_reward_abs = [abs(x) for x in traffic_reward]
    emission_penalty_abs = [abs(x) for x in emission_penalty]
    weighted_emission_penalty_abs = [abs(x) for x in weighted_emission_penalty]
    emission_penalty_ratio = [
        abs(w) / max(abs(t), 1e-6)
        for w, t in zip(weighted_emission_penalty, traffic_reward)
    ]
    nox_risk_mean = [float(getattr(c, "nox_risk_mean", 0.0)) for c in reward_components]
    nox_risk_sum = [float(getattr(c, "nox_risk_sum", 0.0)) for c in reward_components]
    nox_risk_max = [float(getattr(c, "nox_risk_max", 0.0)) for c in reward_components]
    reward_risk_sum = [float(getattr(c, "reward_risk_sum", 0.0)) for c in reward_components]
    reward_risk_max = [float(getattr(c, "reward_risk_max", 0.0)) for c in reward_components]
    mean_lane_risk = [float(getattr(c, "mean_lane_risk", 0.0)) for c in reward_components]
    p95_lane_risk = [float(getattr(c, "p95_lane_risk", 0.0)) for c in reward_components]
    reward_risk_sum = [float(getattr(c, "reward_risk_sum", 0.0)) for c in reward_components]
    reward_risk_max = [float(getattr(c, "reward_risk_max", 0.0)) for c in reward_components]
    mean_lane_risk = [float(getattr(c, "mean_lane_risk", 0.0)) for c in reward_components]
    p95_lane_risk = [float(getattr(c, "p95_lane_risk", 0.0)) for c in reward_components]
    nox_pressure_mean = [float(getattr(c, "nox_pressure_mean", 0.0)) for c in reward_components]
    nox_pressure_max = [float(getattr(c, "nox_pressure_max", 0.0)) for c in reward_components]
    nox_warning_lane_count = [float(getattr(c, "nox_warning_lane_count", 0.0)) for c in reward_components]
    nox_exceed_lane_count = [float(getattr(c, "nox_exceed_lane_count", 0.0)) for c in reward_components]
    random_flags = [1.0 if a.is_random_action else 0.0 for a in action_infos]
    greedy_flags = [1.0 if a.is_greedy_action else 0.0 for a in action_infos]
    mask_flags = [1.0 if a.valid_action_count < len(a.action_mask) else 0.0 for a in action_infos]
    switch_flags = [1.0 if c.rp < 0 else 0.0 for c in reward_components]
    q_loss = [float(u.q_loss) for u in update_rows]
    td = [float(u.td_error_mean) for u in update_rows]
    q_pred = [float(u.q_pred_mean) for u in update_rows]
    q_target = [float(u.q_target_mean) for u in update_rows]

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

    return {
        "episode": int(episode),
        "wall_time_s": float(wall_time_s),
        "sumo_seed": "" if sumo_seed is None else int(sumo_seed),
        "epsilon_start": float(eps_values[0]) if eps_values else 0.0,
        "epsilon_end": float(eps_values[-1]) if eps_values else 0.0,
        "epsilon_mean": float(np.mean(eps_values)) if eps_values else 0.0,
        "replay_size_end": int(replay_size),
        "reward_mean": float(np.mean(reward_values)) if reward_values else 0.0,
        "reward_sum": float(np.sum(reward_values)) if reward_values else 0.0,
        "reward_std": float(np.std(reward_values)) if reward_values else 0.0,
        "traffic_reward_mean": _safe_mean(traffic_reward),
        "traffic_reward_abs_mean": _safe_mean(traffic_reward_abs),
        "emission_penalty_mean": _safe_mean(emission_penalty),
        "emission_penalty_abs_mean": _safe_mean(emission_penalty_abs),
        "weighted_emission_penalty_mean": _safe_mean(weighted_emission_penalty),
        "weighted_emission_penalty_abs_mean": _safe_mean(weighted_emission_penalty_abs),
        "weighted_emission_penalty_sum": _safe_sum(weighted_emission_penalty),
        "final_reward_mean": float(np.mean(reward_values)) if reward_values else 0.0,
        "emission_penalty_ratio_mean": _safe_mean(emission_penalty_ratio),
        "emission_penalty_ratio_p50": _safe_percentile(emission_penalty_ratio, 50),
        "emission_penalty_ratio_p90": _safe_percentile(emission_penalty_ratio, 90),
        "emission_penalty_ratio_p95": _safe_percentile(emission_penalty_ratio, 95),
        "emission_penalty_ratio_max": float(max(emission_penalty_ratio)) if emission_penalty_ratio else 0.0,
        "emission_penalty_ratio_of_means": _ratio_of_means(weighted_emission_penalty, traffic_reward),
        "lambda_e_mean": _safe_mean(lambda_e_values),
        "nox_risk_mean": float(np.mean(nox_risk_mean)) if nox_risk_mean else 0.0,
        "nox_risk_sum_mean": float(np.mean(nox_risk_sum)) if nox_risk_sum else 0.0,
        "nox_risk_max_mean": float(np.mean(nox_risk_max)) if nox_risk_max else 0.0,
        "risk_reward_mode": str(getattr(reward_components[0], "risk_reward_mode", "full")) if reward_components else "full",
        "reward_risk_sum_mean": _safe_mean(reward_risk_sum),
        "reward_risk_max_mean": _safe_mean(reward_risk_max),
        "mean_lane_risk": _safe_mean(mean_lane_risk),
        "p95_lane_risk": _safe_mean(p95_lane_risk),
        "risk_reward_mode": str(getattr(reward_components[0], "risk_reward_mode", "full")) if reward_components else "full",
        "reward_risk_sum_mean": _safe_mean(reward_risk_sum),
        "reward_risk_max_mean": _safe_mean(reward_risk_max),
        "mean_lane_risk": _safe_mean(mean_lane_risk),
        "p95_lane_risk": _safe_mean(p95_lane_risk),
        "nox_pressure_mean": float(np.mean(nox_pressure_mean)) if nox_pressure_mean else 0.0,
        "nox_pressure_max_mean": float(np.mean(nox_pressure_max)) if nox_pressure_max else 0.0,
        "nox_warning_lane_count_mean": float(np.mean(nox_warning_lane_count)) if nox_warning_lane_count else 0.0,
        "nox_exceed_lane_count_mean": float(np.mean(nox_exceed_lane_count)) if nox_exceed_lane_count else 0.0,
        "rp_mean": float(np.mean(rp)) if rp else 0.0,
        "rwn_mean": float(np.mean(rwn)) if rwn else 0.0,
        "rpn_mean": float(np.mean(rpn)) if rpn else 0.0,
        "rwt_mean": float(np.mean(rwt)) if rwt else 0.0,
        "phase_switch_rate": float(np.mean(switch_flags)) if switch_flags else 0.0,
        "mask_active_rate": float(np.mean(mask_flags)) if mask_flags else 0.0,
        "random_action_rate": float(np.mean(random_flags)) if random_flags else 0.0,
        "greedy_action_rate": float(np.mean(greedy_flags)) if greedy_flags else 0.0,
        "avg_delay_s": float(getattr(traffic_stats, "avg_delay_s", 0.0) or 0.0),
        "avg_travel_time_s": float(getattr(traffic_stats, "avg_travel_time_s", 0.0) or 0.0),
        "total_arrived": int(getattr(traffic_stats, "total_arrived", 0) or 0),
        "completion_rate": float(getattr(traffic_stats, "completion_rate", 0.0) or 0.0),
        "total_throughput": int(getattr(traffic_stats, "total_arrived", 0) or 0),
        "q_loss_mean": float(np.mean(q_loss)) if q_loss else 0.0,
        "td_error_mean": float(np.mean(td)) if td else 0.0,
        "q_pred_mean": float(np.mean(q_pred)) if q_pred else 0.0,
        "q_target_mean": float(np.mean(q_target)) if q_target else 0.0,
        "is_best": bool(is_best),
        "best_metric": best_metric,
        "best_score": float(best_score),
        "steps": int(len(eps_values)),
        "n_agents": int(n_agents),
    }

def main() -> None:
    args = parse_args()
    cfg = apply_overrides(get_config(args.config), args)
    set_global_seed(int(cfg.seed))

    net_info = parse_network(cfg.env.net_xml, cfg.env.add_xml, cfg.env.intersection_groups_json)
    controlled_tl_ids, actuated_tl_ids = load_control_sets(net_info)
    builder = ObsRewardBuilder(cfg, net_info)
    agent = MGMQAgentManager(cfg, net_info)
    resume_path = None
    if args.resume:
        resume_path = resolve_path(args.resume, must_exist=True) or args.resume
        agent.load_checkpoint(resume_path, load_optimizer=bool(args.load_optimizer))

    logger = TrainingLogger(
        cfg,
        net_info,
        model_meta=agent.online_bank.model_meta(),
        save_reward_step=True,
    )
    profiler = StageProfiler(
        enabled=bool(args.profile),
        sync_cuda=bool(args.profile_sync_cuda),
    )
    current_env: Dict[str, Optional[SumoEnv]] = {"env": None}

    def _close_current_env(parse_tripinfo: bool = False) -> None:
        env_obj = current_env.get("env")
        if env_obj is None:
            return
        try:
            env_obj.close(parse_tripinfo=parse_tripinfo)
        except Exception:
            pass
        finally:
            current_env["env"] = None

    def _log_uncaught_exception(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        try:
            _close_current_env(parse_tripinfo=False)
            logger.log_event(
                "exception",
                global_step=agent.global_step,
                message=f"{exc_type.__name__}: {exc}",
                detail={"exception_type": exc_type.__name__, "exception": str(exc)},
            )
        finally:
            sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _log_uncaught_exception

    total_episodes = int(cfg.ddqn.total_episodes)
    sumo_seeds = build_sumo_seed_list(
        global_seed=int(cfg.seed),
        total_episodes=total_episodes,
        mode=str(cfg.env.sumo_seed_mode),
    )

    logger.log_event(
        "run_start",
        message="training started",
        detail={
            "run_dir": cfg.log.run_dir,
            "total_episodes": int(cfg.ddqn.total_episodes),
            "seed": int(cfg.seed),
            "sumo_seed_mode": str(cfg.env.sumo_seed_mode),
            "sumo_seed_count": len(sumo_seeds),
            "sumo_seed_preview": sumo_seeds[: min(5, len(sumo_seeds))],
            "device": str(agent.device),
            "deterministic": bool(args.deterministic),
            "decision_interval": int(cfg.env.decision_interval),
            "emission_factor_csv": str(cfg.env.emission_factor_csv),
            "emission_pollutants": list(cfg.env.emission_pollutants),
            "lane_input_dim": int(cfg.network.lane_input_dim),
            "lane_feature_names": list(cfg.state.lane_feature_names),
            "use_truck_count_state": bool(cfg.emission_risk.use_truck_count_state),
            "use_nox_risk_state": bool(cfg.emission_risk.use_nox_risk_state),
            "enabled_reward": bool(cfg.emission_risk.enabled_reward),
            "objective_mode": str(cfg.reward.objective_mode),
            "log_lane_risk": bool(cfg.emission_risk.log_lane_risk),
            "lambda_e": float(cfg.emission_risk.lambda_e),
            "risk_mode": str(cfg.emission_risk.risk_mode),
            "reward_risk_mode": str(cfg.emission_risk.reward_risk_mode),
            "reward_risk_mode": str(cfg.emission_risk.reward_risk_mode),
            "base_weight": float(cfg.emission_risk.base_weight),
            "tail_weight": float(cfg.emission_risk.tail_weight),
            "pressure_beta": float(cfg.emission_risk.pressure_beta),
            "penalty_time_normalize": bool(
                cfg.emission_risk.penalty_time_normalize
            ),
            "threshold_policy": str(cfg.emission_risk.threshold_policy),
            "dual_head_enabled": bool(getattr(cfg.dual_head, "enabled", False)),
            "dual_head_nox_loss_beta": float(getattr(cfg.dual_head, "nox_loss_beta", 1.0)),
            "dual_head_nox_q_softplus": bool(getattr(cfg.dual_head, "nox_q_softplus", False)),
            "emission_risk_thresholds_json": str(cfg.emission_risk.thresholds_json),
            "profile_enabled": bool(args.profile),
            "profile_sync_cuda": bool(args.profile_sync_cuda),
        },
    )
    logger.log_event(
        "network_loaded",
        message="network parsed",
        detail={
            "n_intersections": len(net_info.intersection_ids),
            "n_rl_intersections": len(controlled_tl_ids),
            "sumo_actuated_intersections": actuated_tl_ids,
            "n_lanes_total": len(net_info.lane_info),
            "n_edges_total": len(net_info.edge_info),
        },
    )
    logger.log_event(
        "agent_initialized",
        message="DDQN agent ready",
        detail={
            "replay_capacity": int(cfg.ddqn.replay_memory_size),
            "min_replay_size": int(cfg.ddqn.min_replay_size),
            "batch_size": int(cfg.ddqn.batch_size),
            "target_update_interval": int(cfg.ddqn.target_update_interval),
            "dual_head_enabled": bool(getattr(cfg.dual_head, "enabled", False)),
            "nox_loss_beta": float(getattr(cfg.dual_head, "nox_loss_beta", 1.0)),
            "replay_sampling_mode": str(cfg.ddqn.replay_sampling_mode),
            "usage_penalty_alpha": float(cfg.ddqn.usage_penalty_alpha),
            "usage_weight_min": float(cfg.ddqn.usage_weight_min),
        },
    )
    logger.log_event(
        "target_network_synced_init",
        message="initial target network sync completed",
        detail={"target_update_count": int(agent.target_update_count)},
    )
    if resume_path:
        logger.log_event(
            "checkpoint_loaded",
            global_step=agent.global_step,
            message="checkpoint loaded",
            detail={"checkpoint_path": resume_path, "load_optimizer": bool(args.load_optimizer)},
        )
    best_score = -float("inf") if cfg.log.best_mode == "max" else float("inf")

    steps_per_episode = int(cfg.env.steps_per_episode)
    step_log_interval = max(1, int(cfg.log.step_progress_event_interval))

    def _should_keep_by_interval(ep: int, interval: int) -> bool:
        return int(interval) > 0 and int(ep) % int(interval) == 0

    def _detail_log_policy(ep: int, replay_size_at_start: int) -> Dict[str, bool]:
        replay_ready = int(replay_size_at_start) >= int(cfg.ddqn.min_replay_size)
        if bool(cfg.log.detail_log_until_replay_ready) and not replay_ready:
            return {
                "record_phase_step": True,
                "record_lane_step": True,
                "record_q_step": True,
                "keep_tripinfo": True,
                "replay_ready": False,
            }
        return {
            "record_phase_step": _should_keep_by_interval(
                ep, int(cfg.log.phase_step_log_interval_after_replay)
            ),
            "record_lane_step": _should_keep_by_interval(
                ep, int(cfg.log.lane_step_log_interval_after_replay)
            ),
            "record_q_step": _should_keep_by_interval(
                ep, int(cfg.log.q_step_log_interval_after_replay)
            ),
            "keep_tripinfo": _should_keep_by_interval(
                ep, int(cfg.log.tripinfo_keep_interval_after_replay)
            ),
            "replay_ready": True,
        }

    training_start_time = time.perf_counter()
    for episode in range(1, total_episodes + 1):
        ep_start = time.time()
        replay_size_at_episode_start = len(agent.memory)
        detail_policy = _detail_log_policy(episode, replay_size_at_episode_start)
        logger.begin_episode(
            episode,
            record_lane_step=bool(detail_policy["record_lane_step"]),
            record_phase_step=bool(detail_policy["record_phase_step"]),
            record_q_step=bool(detail_policy["record_q_step"]),
        )
        emission_step_interval_train = int(getattr(cfg.log, "emission_step_log_interval_train", 20))
        record_emission_step = (
            bool(getattr(cfg.log, "save_emission_step", True))
            and emission_step_interval_train > 0
            and episode % emission_step_interval_train == 0
        )
        record_q_values = bool(detail_policy["record_q_step"]) and bool(cfg.log.save_q_step)
        episode_link_nox_values: list[float] = []
        logger.clear_episode_emission()
        sumo_seed = int(sumo_seeds[episode - 1])
        logger.log_event(
            "episode_start",
            episode=episode,
            global_step=agent.global_step,
            message=f"episode {episode}/{total_episodes} started",
            detail={
                "sumo_seed": sumo_seed,
                "epsilon": float(agent.epsilon()),
                "replay_size_at_episode_start": int(replay_size_at_episode_start),
                "detail_replay_ready": bool(detail_policy["replay_ready"]),
                "record_phase_step": bool(detail_policy["record_phase_step"]),
                "record_lane_step": bool(detail_policy["record_lane_step"]),
                "record_q_step": bool(detail_policy["record_q_step"]),
                "record_emission_step": bool(record_emission_step),
                "keep_tripinfo": bool(detail_policy["keep_tripinfo"]),
            },
        )
        env = SumoEnv(
            cfg.env,
            net_info,
            port=int(args.port),
            use_gui=bool(args.use_gui),
            controlled_tl_ids=controlled_tl_ids,
        )
        current_env["env"] = env
        env.start(episode_id=episode, seed=sumo_seed, control_tls=True)
        logger.log_event(
            "sumo_started",
            episode=episode,
            global_step=agent.global_step,
            message="SUMO TraCI session started",
            detail={
                "port": int(args.port),
                "use_gui": bool(args.use_gui),
                "tripinfo_path": getattr(env, "_tripinfo_path", ""),
                "emission_recorder_enabled": bool(getattr(env, "emission_recorder", None) is not None),
                "emission_factor_csv": str(cfg.env.emission_factor_csv),
            },
            to_console=False,
        )
        if episode == 1 and getattr(env, "emission_recorder", None) is None:
            logger.log_event(
                "emission_recorder_disabled",
                episode=episode,
                global_step=agent.global_step,
                message="emission recorder is disabled; NOx fields will be zero",
                detail={"emission_factor_csv": str(cfg.env.emission_factor_csv)},
            )
        with profiler.timeit("initial_collect_raw_obs", episode=episode, step=-1, global_step=agent.global_step):
            raw_obs = collect_current_raw_obs(env, controlled_tl_ids)
        with profiler.timeit("initial_build_observations", episode=episode, step=-1, global_step=agent.global_step):
            obs = builder.build_observations(raw_obs)
        logger.log_event(
            "initial_observation_collected",
            episode=episode,
            global_step=agent.global_step,
            message="initial observations ready",
            detail={"raw_obs": len(raw_obs), "observations": len(obs)},
            to_console=False,
        )

        eps_values: list[float] = []
        reward_values: list[float] = []
        all_comps: list[Any] = []
        all_action_infos: list[Any] = []
        all_updates: list[Any] = []

        for step in range(steps_per_episode):
            with profiler.timeit("bookkeeping", episode=episode, step=step, global_step=agent.global_step):
                epsilon = 0.0 if args.deterministic else agent.epsilon()
                previous_phases = {tl_id: int(raw.current_phase) for tl_id, raw in raw_obs.items()}
            with profiler.timeit("act", episode=episode, step=step, global_step=agent.global_step):
                actions, action_infos = agent.act(
                    obs,
                    epsilon=epsilon,
                    deterministic=bool(args.deterministic),
                    record_q_values=record_q_values,
                )
            with profiler.timeit("env_step", episode=episode, step=step, global_step=agent.global_step):
                next_raw_obs = env.step(actions, decision_step=step)
            with profiler.timeit("link_emission_compute", episode=episode, step=step, global_step=agent.global_step):
                step_emission = getattr(env, "_last_step_emission", None)
                link_nox_values = logger.compute_step_link_emission_values(step_emission=step_emission, pollutant="NOx")
                episode_link_nox_values.extend(link_nox_values)
            if record_emission_step:
                with profiler.timeit("log_step_emission", episode=episode, step=step, global_step=agent.global_step):
                    logger.log_step_emission(episode=episode, step=step, step_emission=step_emission)
            with profiler.timeit("compute_rewards", episode=episode, step=step, global_step=agent.global_step):
                rewards, comps = builder.compute_rewards(next_raw_obs, actions, previous_phases)
            with profiler.timeit("build_next_observations", episode=episode, step=step, global_step=agent.global_step):
                next_obs = builder.build_observations(next_raw_obs)
            with profiler.timeit("bookkeeping", episode=episode, step=step, global_step=agent.global_step):
                done = bool(step == steps_per_episode - 1)
            with profiler.timeit("store_transition", episode=episode, step=step, global_step=agent.global_step):
                traffic_rewards = {
                    tl_id: float(getattr(comps[tl_id], "traffic_reward", rewards[tl_id]))
                    for tl_id in rewards.keys()
                }
                nox_penalties = {
                    tl_id: float(getattr(comps[tl_id], "emission_penalty", 0.0))
                    for tl_id in rewards.keys()
                }
                agent.store_transition(GlobalTransition(
                    obs=obs,
                    actions={k: int(v) for k, v in actions.items()},
                    rewards={k: float(v) for k, v in rewards.items()},
                    next_obs=next_obs,
                    done=done,
                    traffic_rewards=traffic_rewards,
                    nox_penalties=nox_penalties,
                ))

            update_due = agent.global_step % int(cfg.ddqn.online_update_interval) == 0
            update_stats = []
            with profiler.timeit("update_from_replay", episode=episode, step=step, global_step=agent.global_step):
                if update_due:
                    update_stats = agent.update_from_replay()
            with profiler.timeit("maybe_sync_target", episode=episode, step=step, global_step=agent.global_step):
                target_synced = agent.maybe_sync_target()
            with profiler.timeit("bookkeeping", episode=episode, step=step, global_step=agent.global_step):
                if target_synced and update_stats:
                    for u in update_stats:
                        u.target_synced = True
            if update_stats:
                with profiler.timeit("log_update", episode=episode, step=step, global_step=agent.global_step):
                    logger.log_update(episode, agent.global_step, update_stats, agent.update_count, agent.target_update_count)
                with profiler.timeit("bookkeeping", episode=episode, step=step, global_step=agent.global_step):
                    all_updates.extend(update_stats)
                    should_log_update_event = (
                        int(cfg.log.update_event_interval) > 0
                        and int(agent.update_count) % int(cfg.log.update_event_interval) == 0
                    )
                    if should_log_update_event:
                        replay_stats = agent.memory.stats()
                        logger.log_event(
                            "online_network_updated",
                            episode=episode,
                            step=step,
                            global_step=agent.global_step,
                            message="online Q network updated from replay",
                            detail={
                                "groups": len(update_stats),
                                "replay_size": len(agent.memory),
                                "q_loss_mean": float(np.mean([s.q_loss for s in update_stats])),
                                "td_error_abs_mean": float(np.mean([s.td_error_abs_mean for s in update_stats])),
                                "replay_sampling_mode": str(replay_stats.get("sampling_mode", "")),
                                "sample_times_mean": float(replay_stats.get("sample_times_mean", 0.0)),
                                "sample_times_max": int(replay_stats.get("sample_times_max", 0)),
                                "sample_times_zero_ratio": float(replay_stats.get("sample_times_zero_ratio", 0.0)),
                            },
                            to_console=False,
                        )
            elif update_due and len(agent.memory) < int(cfg.ddqn.min_replay_size):
                with profiler.timeit("bookkeeping", episode=episode, step=step, global_step=agent.global_step):
                    if step == 0 or (step + 1) % step_log_interval == 0 or done:
                        logger.log_event(
                            "replay_update_skipped",
                            episode=episode,
                            step=step,
                            global_step=agent.global_step,
                            message="replay memory below min_replay_size",
                            detail={
                                "replay_size": len(agent.memory),
                                "min_replay_size": int(cfg.ddqn.min_replay_size),
                            },
                            to_console=False,
                        )
            if target_synced:
                with profiler.timeit("bookkeeping", episode=episode, step=step, global_step=agent.global_step):
                    logger.log_event(
                        "target_network_synced",
                        episode=episode,
                        step=step,
                        global_step=agent.global_step,
                        message="target Q network synced from online network",
                        detail={"target_update_count": int(agent.target_update_count)},
                        to_console=False,
                    )

            with profiler.timeit("log_step", episode=episode, step=step, global_step=agent.global_step):
                logger.log_step(episode, step, next_raw_obs, next_obs, actions, action_infos, comps)
            with profiler.timeit("bookkeeping", episode=episode, step=step, global_step=agent.global_step):
                if step == 0 or (step + 1) % step_log_interval == 0 or done:
                    sim_time = None
                    if next_raw_obs:
                        sim_time = float(next(iter(next_raw_obs.values())).sim_time)
                    logger.log_event(
                        "step_progress",
                        episode=episode,
                        step=step + 1,
                        global_step=agent.global_step,
                        sim_time=sim_time,
                        message=f"step {step + 1}/{steps_per_episode}",
                        detail={
                            "epsilon": float(epsilon),
                            "replay_size": len(agent.memory),
                            "reward_mean": float(np.mean(list(rewards.values()))) if rewards else 0.0,
                            "emission_penalty_mean": float(np.mean([getattr(c, "emission_penalty", 0.0) for c in comps.values()])) if comps else 0.0,
                            "nox_risk_max_mean": float(np.mean([getattr(c, "nox_risk_max", 0.0) for c in comps.values()])) if comps else 0.0,
                            "random_action_rate": float(np.mean([1.0 if a.is_random_action else 0.0 for a in action_infos.values()])) if action_infos else 0.0,
                        },
                        to_console=False,
                    )
                eps_values.append(float(epsilon))
                reward_values.extend([float(v) for v in rewards.values()])
                all_comps.extend(list(comps.values()))
                all_action_infos.extend(list(action_infos.values()))

                obs = next_obs
                raw_obs = next_raw_obs
                agent.increment_step()
            profiler.add_step_meta(
                episode=episode,
                step=step,
                global_step=agent.global_step,
                replay_size=len(agent.memory),
                update_due=bool(update_due),
                n_update_groups=len(update_stats),
                n_agents=len(controlled_tl_ids),
            )

        with profiler.timeit("env_close", episode=episode, step=-1, global_step=agent.global_step):
            traffic_stats = env.close()
        current_env["env"] = None
        tripinfo_path = getattr(env, "_tripinfo_path", "")
        wall_time = time.time() - ep_start
        logger.log_event(
            "episode_end",
            episode=episode,
            global_step=agent.global_step,
            message="episode simulation closed",
            detail={
                "wall_time_s": round(float(wall_time), 3),
                "total_arrived": int(getattr(traffic_stats, "total_arrived", 0) or 0),
                "avg_delay_s": float(getattr(traffic_stats, "avg_delay_s", 0.0) or 0.0),
                "completion_rate": float(getattr(traffic_stats, "completion_rate", 0.0) or 0.0),
            },
        )
        ep_row = summarize_episode(
            episode=episode,
            wall_time_s=wall_time,
            sumo_seed=sumo_seed,
            eps_values=eps_values,
            reward_values=reward_values,
            reward_components=all_comps,
            action_infos=all_action_infos,
            update_rows=all_updates,
            traffic_stats=traffic_stats,
            replay_size=len(agent.memory),
            n_agents=len(controlled_tl_ids),
            best_score=best_score,
            is_best=False,
            best_metric=cfg.log.best_metric,
        )
        ep_row["link_nox_mean"] = float(np.mean(episode_link_nox_values)) if episode_link_nox_values else 0.0

        metric_value = float(ep_row.get(cfg.log.best_metric, ep_row["reward_mean"]))
        improved = metric_value > best_score if cfg.log.best_mode == "max" else metric_value < best_score
        old_best = best_score
        with profiler.timeit("save_checkpoint", episode=episode, step=-1, global_step=agent.global_step):
            if improved:
                best_score = metric_value
                ep_row["is_best"] = True
                ep_row["best_score"] = best_score
                best_path = os.path.join(logger.checkpoint_dir_path, "checkpoint_best.pt")
                agent.save_checkpoint(best_path, episode, extra={"best_metric": cfg.log.best_metric, "best_score": best_score})
                best_info = dict(ep_row)
                best_info.update({
                    "best_metric": cfg.log.best_metric,
                    "best_score": best_score,
                    "old_best_score": old_best,
                    "global_step": agent.global_step,
                    "epsilon": eps_values[-1] if eps_values else 0.0,
                    "replay_size": len(agent.memory),
                })
                logger.log_best(episode, agent.global_step, best_info, agent.online_bank, best_path)
                logger.log_event(
                    "checkpoint_best_saved",
                    episode=episode,
                    global_step=agent.global_step,
                    message="best checkpoint saved",
                    detail={
                        "checkpoint_path": best_path,
                        "best_metric": cfg.log.best_metric,
                        "best_score": float(best_score),
                    },
                )

            latest_path = os.path.join(logger.checkpoint_dir_path, "checkpoint_latest.pt")
            agent.save_checkpoint(latest_path, episode, extra={"latest_metric": metric_value})
            logger.log_event(
                "checkpoint_latest_saved",
                episode=episode,
                global_step=agent.global_step,
                message="latest checkpoint saved",
                detail={"checkpoint_path": latest_path, "metric_value": float(metric_value)},
                to_console=False,
            )
        with profiler.timeit("log_episode", episode=episode, step=-1, global_step=agent.global_step):
            logger.log_episode(episode, ep_row)
        with profiler.timeit("flush_episode_physical", episode=episode, step=-1, global_step=agent.global_step):
            logger.flush_episode_physical(episode)
        with profiler.timeit("flush_episode_emission", episode=episode, step=-1, global_step=agent.global_step):
            if record_emission_step:
                logger.flush_episode_emission(episode)
            else:
                logger.clear_episode_emission()
        logger.log_event(
            "episode_summary_saved",
            episode=episode,
            global_step=agent.global_step,
            message=(
                f"reward_mean={ep_row['reward_mean']:.3f} "
                f"epsilon={ep_row['epsilon_end']:.3f} "
                f"replay={len(agent.memory)} best={best_score:.3f}"
            ),
            detail={
                "reward_mean": float(ep_row["reward_mean"]),
                "traffic_reward_mean": float(ep_row.get("traffic_reward_mean", 0.0)),
                "weighted_emission_penalty_mean": float(ep_row.get("weighted_emission_penalty_mean", 0.0)),
                "emission_penalty_ratio_mean": float(ep_row.get("emission_penalty_ratio_mean", 0.0)),
                "epsilon_end": float(ep_row["epsilon_end"]),
                "replay_size": int(len(agent.memory)),
                "best_score": float(best_score),
                "is_best": bool(ep_row.get("is_best", False)),
                "wall_time_s": round(float(wall_time), 3),
                "total_arrived": int(getattr(traffic_stats, "total_arrived", 0) or 0),
                "avg_delay_s": float(getattr(traffic_stats, "avg_delay_s", 0.0) or 0.0),
                "completion_rate": float(getattr(traffic_stats, "completion_rate", 0.0) or 0.0),
                "link_nox_mean": float(ep_row.get("link_nox_mean", 0.0)),
                "episode_log": logger.episode_csv,
                "physical_dir": os.path.join(logger.physical_root, f"episode_{episode:04d}"),
                "tripinfo_kept": bool(detail_policy["keep_tripinfo"]),
                "tripinfo_path": tripinfo_path if bool(detail_policy["keep_tripinfo"]) else "",
            },
        )
        if tripinfo_path and not bool(detail_policy["keep_tripinfo"]):
            try:
                if os.path.exists(tripinfo_path):
                    os.remove(tripinfo_path)
            except Exception as exc:
                logger.log_event(
                    "tripinfo_cleanup_failed",
                    episode=episode,
                    global_step=agent.global_step,
                    message=str(exc),
                    detail={"tripinfo_path": tripinfo_path},
                    to_console=False,
                )

        if cfg.log.save_parameter_stats and episode % int(cfg.log.parameter_log_interval) == 0:
            param_path = logger.log_parameter_stats(episode, agent.global_step, agent.online_bank, agent.target_bank)
            logger.log_event(
                "parameter_stats_saved",
                episode=episode,
                global_step=agent.global_step,
                message="parameter statistics saved",
                detail={"parameter_stats_path": param_path},
            )

        if profiler.enabled:
            profile_wall_time = time.time() - ep_start
            profiler.finish_episode(
                episode=episode,
                wall_time_s=profile_wall_time,
                steps=steps_per_episode,
                n_agents=len(controlled_tl_ids),
                replay_size_end=len(agent.memory),
            )
            profiler.flush_step_csv(os.path.join(cfg.log.run_dir, "profile_step_log.csv"))
            profiler.flush_episode_csv(os.path.join(cfg.log.run_dir, "profile_episode_log.csv"))
            profiler.save_summary_json(os.path.join(cfg.log.run_dir, "profile_summary.json"))

        if episode % 50 == 0:
            training_elapsed_s = time.perf_counter() - training_start_time
            training_elapsed_hms = _format_elapsed_hms(training_elapsed_s)
            logger.log_event(
                "training_time_progress",
                episode=episode,
                global_step=agent.global_step,
                message=f"completed {episode}/{total_episodes} episodes; cumulative training time {training_elapsed_hms}",
                detail={
                    "completed_episodes": episode,
                    "total_episodes": total_episodes,
                    "training_elapsed_s": round(float(training_elapsed_s), 3),
                    "training_elapsed_hms": training_elapsed_hms,
                },
                to_console=False,
            )
            print(
                f"[TRAINING PROGRESS] completed {episode}/{total_episodes} episodes; "
                f"elapsed time: {training_elapsed_hms}",
                flush=True,
            )

    training_elapsed_s = time.perf_counter() - training_start_time
    training_elapsed_hms = _format_elapsed_hms(training_elapsed_s)
    logger.log_event(
        "run_end",
        global_step=agent.global_step,
        message=f"training finished; cumulative training time {training_elapsed_hms}",
        detail={
            "total_episodes": total_episodes,
            "best_score": float(best_score),
            "training_elapsed_s": round(float(training_elapsed_s), 3),
            "training_elapsed_hms": training_elapsed_hms,
            "run_dir": cfg.log.run_dir,
        },
    )
    print(
        f"[TRAINING COMPLETE] total elapsed time: {training_elapsed_hms}",
        flush=True,
    )

if __name__ == "__main__":
    main()
