# -*- coding: utf-8 -*-
"""Train PressLight and save an unconditional checkpoint every 10 episodes."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from presslight_config import (  # noqa: E402
    PressLightConfig,
    discover_project_root,
    get_presslight_config,
)

sys.path.insert(0, str(SCRIPT_DIR.parent))
PROJECT_ROOT = discover_project_root(SCRIPT_DIR)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import build_sumo_seed_list, get_config  # noqa: E402
from env import SumoEnv  # noqa: E402
from presslight_agent import (  # noqa: E402
    PressLightAgentManager,
    PressLightTransition,
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
from presslight_reward import (  # noqa: E402
    PressLightRewardBuilder,
    pressure_step_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PressLight baseline")
    parser.add_argument("--base-config", default=None, help="Existing MGMQ YAML config")
    parser.add_argument("--presslight-config", default=None, help="PressLight YAML/JSON override")
    parser.add_argument("--resume", default=None, help="Checkpoint used to resume training")
    parser.add_argument("--device", default=None)
    parser.add_argument("--port", type=int, default=8813)
    parser.add_argument("--use-gui", action="store_true")
    parser.add_argument(
        "--disable-fixed-validation",
        action="store_true",
        help="Still save periodic checkpoints, but do not run fixed validation",
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

    # Accept CLI paths from the working directory and repository-relative
    # paths used by the distributed FLORE configurations.
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


def _write_csv(path: str, rows: Iterable[dict[str, Any]], append: bool = False) -> None:
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    if append and os.path.exists(path):
        with open(path, "r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            fieldnames = next(reader, [])
            seen.update(fieldnames)
    existing_fieldnames = list(fieldnames)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    if append and existing_fieldnames and fieldnames == existing_fieldnames:
        with open(path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writerows(rows)
        return
    # Episode/update schemas are stable.  Rewriting is only needed when a new
    # field appears after resuming from an older file.
    existing_rows: list[dict[str, Any]] = []
    if append and os.path.exists(path):
        with open(path, "r", newline="", encoding="utf-8") as file:
            existing_rows = list(csv.DictReader(file))
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(rows)


def _save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _mean(values: Iterable[float]) -> float:
    data = [float(x) for x in values]
    return float(np.mean(data)) if data else 0.0


def _log(message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    print(f"[{timestamp}] {message}", flush=True)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _build_configs(args: argparse.Namespace) -> tuple[Any, PressLightConfig]:
    base_config_path = _resolve(args.base_config, required=True) if args.base_config else None
    base_cfg = get_config(base_config_path)
    pl_config_path = (
        _resolve(args.presslight_config, required=True) if args.presslight_config else None
    )
    pl_cfg = get_presslight_config(pl_config_path, base_cfg)
    if args.device:
        pl_cfg.device = str(args.device)
    if args.resume:
        pl_cfg.training.resume_checkpoint = _resolve(args.resume, required=True)
    if args.disable_fixed_validation:
        pl_cfg.training.fixed_validation_enabled = False

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
    if pl_cfg.paths.train_output_dir:
        # Training output paths are relative to the PressLight project directory.
        train_output_dir = Path(pl_cfg.paths.train_output_dir).expanduser()
        if not train_output_dir.is_absolute():
            train_output_dir = PROJECT_ROOT / train_output_dir
        pl_cfg.paths.train_output_dir = str(train_output_dir.resolve())
    if pl_cfg.grouping.manual_groups_json:
        pl_cfg.grouping.manual_groups_json = _resolve(
            pl_cfg.grouping.manual_groups_json, required=True
        )
    pl_cfg.validate(base_cfg)

    base_cfg.seed = int(pl_cfg.seed)
    base_cfg.env.net_xml = pl_cfg.paths.net_xml
    base_cfg.env.add_xml = pl_cfg.paths.base_add_xml
    base_cfg.env.intersection_groups_json = pl_cfg.paths.intersection_groups_json
    base_cfg.env.tripinfo_dir = os.path.join(pl_cfg.train_run_dir, "tripinfo")
    base_cfg.env.save_fcd_output = False
    base_cfg.env.save_sumo_aux_outputs = False
    if not pl_cfg.training.record_emissions:
        base_cfg.env.emission_factor_csv = ""
    base_cfg.validate()
    return base_cfg, pl_cfg


def _validation_episode(
    *,
    base_cfg: Any,
    pl_cfg: PressLightConfig,
    network_spec: Any,
    agent: PressLightAgentManager,
    seed: int,
    episode_id: int,
    port: int,
    use_gui: bool,
    validation_dir: str,
) -> dict[str, Any]:
    env_cfg = copy.deepcopy(base_cfg.env)
    env_cfg.tripinfo_dir = os.path.join(validation_dir, "tripinfo")
    env_cfg.save_fcd_output = False
    env_cfg.save_sumo_aux_outputs = False
    env_cfg.emission_factor_csv = ""
    env: Optional[SumoEnv] = None
    started_at = time.time()
    pressures: list[float] = []
    rewards: list[float] = []
    switches: list[float] = []
    try:
        env = SumoEnv(env_cfg, network_spec._base_net_info, port=port, use_gui=use_gui)
        env.start(episode_id=episode_id, seed=int(seed), control_tls=True)
        observation_builder = PressLightObservationBuilder(pl_cfg, network_spec)
        # A separate builder prevents validation turn ratios from changing the
        # PressLight-TR priors used by training.
        reward_builder = PressLightRewardBuilder(pl_cfg, network_spec)
        observations = observation_builder.build_all(env)
        for step in range(int(env_cfg.steps_per_episode)):
            actions, _infos = agent.act(observations, epsilon=0.0, deterministic=True)
            switches.extend(
                1.0 if int(actions[tl_id]) != int(obs.current_phase) else 0.0
                for tl_id, obs in observations.items()
            )
            env.step(
                canonical_actions_to_env_actions(actions, network_spec),
                decision_step=step,
            )
            next_observations = observation_builder.build_all(env)
            results = reward_builder.compute_all(env, next_observations)
            pressures.extend(result.intersection_pressure for result in results.values())
            rewards.extend(result.reward for result in results.values())
            observations = next_observations
        traffic_stats = env.close()
        env = None
        return {
            "seed": int(seed),
            "avg_delay_s": float(getattr(traffic_stats, "avg_delay_s", 0.0) or 0.0),
            "avg_travel_time_s": float(
                getattr(traffic_stats, "avg_travel_time_s", 0.0) or 0.0
            ),
            "total_arrived": int(getattr(traffic_stats, "total_arrived", 0) or 0),
            "completion_rate": float(
                getattr(traffic_stats, "completion_rate", 0.0) or 0.0
            ),
            "pressure_mean": _mean(pressures),
            "reward_mean": _mean(rewards),
            "phase_switch_rate": _mean(switches),
            "wall_time_s": float(time.time() - started_at),
        }
    finally:
        if env is not None:
            try:
                env.close(parse_tripinfo=False)
            except Exception:
                pass


def _run_fixed_validation(
    *,
    checkpoint_episode: int,
    base_cfg: Any,
    pl_cfg: PressLightConfig,
    network_spec: Any,
    agent: PressLightAgentManager,
    port: int,
    use_gui: bool,
    validation_dir: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    was_training = any(network.training for network in agent.online_networks.values())
    agent.set_training(False)
    rows: list[dict[str, Any]] = []
    try:
        for index, seed in enumerate(pl_cfg.training.validation_seeds):
            row = _validation_episode(
                base_cfg=base_cfg,
                pl_cfg=pl_cfg,
                network_spec=network_spec,
                agent=agent,
                seed=int(seed),
                episode_id=800000 + checkpoint_episode * 10 + index,
                port=port,
                use_gui=use_gui,
                validation_dir=validation_dir,
            )
            row["checkpoint_episode"] = int(checkpoint_episode)
            rows.append(row)
    finally:
        agent.set_training(was_training)

    completion_values = [float(row["completion_rate"]) for row in rows]
    summary = {
        "checkpoint_episode": int(checkpoint_episode),
        "n_validation_seeds": len(rows),
        "validation_seeds": [int(x) for x in pl_cfg.training.validation_seeds],
        "avg_delay_s": _mean(row["avg_delay_s"] for row in rows),
        "avg_travel_time_s": _mean(row["avg_travel_time_s"] for row in rows),
        "total_arrived": _mean(row["total_arrived"] for row in rows),
        "completion_rate": _mean(completion_values),
        "completion_rate_min": min(completion_values) if completion_values else 0.0,
        "pressure_mean": _mean(row["pressure_mean"] for row in rows),
        "reward_mean": _mean(row["reward_mean"] for row in rows),
        "phase_switch_rate": _mean(row["phase_switch_rate"] for row in rows),
    }
    summary["completion_qualified"] = bool(
        summary["completion_rate"] >= pl_cfg.training.min_completion_rate
        and summary["completion_rate_min"] >= pl_cfg.training.min_completion_rate
    )
    return rows, summary


def _is_better(value: float, best_value: Optional[float], mode: str) -> bool:
    if not math.isfinite(float(value)):
        return False
    if best_value is None:
        return True
    return float(value) < float(best_value) if mode == "min" else float(value) > float(best_value)


def main() -> None:
    training_started_at = time.time()
    args = parse_args()
    if not 1 <= int(args.port) <= 65535:
        raise ValueError(f"port must be in [1, 65535]; actual={args.port!r}")
    base_cfg, pl_cfg = _build_configs(args)
    set_global_seed(pl_cfg.seed)
    run_dir = pl_cfg.train_run_dir
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    validation_dir = os.path.join(run_dir, "validation")
    detail_dir = os.path.join(run_dir, "step_detail")
    reward_step_dir = os.path.join(run_dir, "reward_step")
    directories = [run_dir, checkpoint_dir, validation_dir, detail_dir]
    if bool(pl_cfg.log.save_reward_step):
        directories.append(reward_step_dir)
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Build base NetworkInfo before attaching three-segment E2 detectors.  This
    # preserves the existing one-lane-one-E2 MGMQ map.
    net_info = parse_base_network(
        pl_cfg.paths.net_xml,
        pl_cfg.paths.base_add_xml,
        pl_cfg.paths.intersection_groups_json,
    )
    network_spec = build_presslight_network_spec(net_info, pl_cfg)
    # Internal convenience used only by this entry point; it is never included
    # in the serialized network summary/digest.
    network_spec._base_net_info = net_info
    save_network_spec(network_spec, os.path.join(run_dir, "presslight_network_spec.json"))

    runtime_sumo_cfg = prepare_presslight_runtime_sumo_cfg(
        pl_cfg.paths.sumo_cfg,
        pl_cfg.paths.presslight_add_xml,
        os.path.join(run_dir, "sumo_template"),
    )
    base_cfg.env.sumo_cfg = runtime_sumo_cfg
    pl_cfg.save_json(os.path.join(run_dir, "presslight_config_resolved.json"))
    _save_json(
        os.path.join(run_dir, "base_config_resolved.json"),
        base_cfg.to_dict() if hasattr(base_cfg, "to_dict") else {},
    )

    observation_builder = PressLightObservationBuilder(pl_cfg, network_spec)
    reward_builder = PressLightRewardBuilder(pl_cfg, network_spec)
    agent = PressLightAgentManager(pl_cfg, network_spec, device=pl_cfg.device)
    agent.set_training(True)

    start_episode = 1
    resume_path = str(pl_cfg.training.resume_checkpoint or "")
    if resume_path:
        resume_path = _resolve(resume_path, required=True)
        _log(
            "[security warning] Loading a trusted training checkpoint with "
            "weights_only=False; only use files generated by this project from a trusted source."
        )
        payload = agent.load_checkpoint(
            resume_path,
            load_optimizer=True,
            load_replay=bool(pl_cfg.dqn.checkpoint_include_replay),
            restore_rng=True,
        )
        start_episode = int(payload.get("episode", 0)) + 1
        _log(f"[resume] {resume_path}; next episode={start_episode}")

    total_episodes = int(pl_cfg.training.total_episodes)
    training_seeds = build_sumo_seed_list(
        global_seed=int(pl_cfg.seed),
        total_episodes=total_episodes,
        mode=str(base_cfg.env.sumo_seed_mode),
    )
    overlap = set(training_seeds).intersection(pl_cfg.training.validation_seeds)
    if overlap:
        raise ValueError(f"Training and validation SUMO seeds overlap: {sorted(overlap)}")

    best_value: Optional[float] = None
    best_summary_path = os.path.join(run_dir, "best_validation_summary.json")
    if os.path.exists(best_summary_path):
        try:
            with open(best_summary_path, "r", encoding="utf-8") as file:
                old_best = json.load(file)
            best_value = float(old_best[pl_cfg.training.best_metric])
        except Exception:
            best_value = None

    episode_log_path = os.path.join(run_dir, "episode_log.csv")
    update_log_path = os.path.join(run_dir, "update_log.csv")
    validation_episode_path = os.path.join(validation_dir, "validation_episode_log.csv")
    validation_summary_path = os.path.join(validation_dir, "validation_summary_log.csv")
    _log(
        f"[start] experiment={pl_cfg.experiment_name} episodes={total_episodes} "
        f"groups={len(network_spec.group_ids)} checkpoints_every="
        f"{pl_cfg.training.checkpoint_interval} validation="
        f"{pl_cfg.training.fixed_validation_enabled}"
    )

    try:
        for episode in range(start_episode, total_episodes + 1):
            episode_started = time.time()
            seed = int(training_seeds[episode - 1])
            epsilon_start = float(agent.epsilon)
            env: Optional[SumoEnv] = None
            rewards_all: list[float] = []
            pressures_all: list[float] = []
            pressure_max_values: list[float] = []
            q_losses: list[float] = []
            switch_flags: list[float] = []
            state_rows: list[dict[str, Any]] = []
            pressure_rows: list[dict[str, Any]] = []
            q_rows: list[dict[str, Any]] = []
            reward_step_rows: list[dict[str, Any]] = []
            traffic_stats: Any = None
            try:
                env = SumoEnv(
                    base_cfg.env,
                    net_info,
                    port=int(args.port),
                    use_gui=bool(args.use_gui),
                )
                env.start(episode_id=episode, seed=seed, control_tls=True)
                observations = observation_builder.build_all(env)

                for step in range(int(base_cfg.env.steps_per_episode)):
                    actions, action_infos = agent.act(observations)
                    switch_flags.extend(
                        1.0 if int(actions[tl_id]) != int(obs.current_phase) else 0.0
                        for tl_id, obs in observations.items()
                    )
                    env.step(
                        canonical_actions_to_env_actions(actions, network_spec),
                        decision_step=step,
                    )
                    next_observations = observation_builder.build_all(env)
                    reward_results = reward_builder.compute_all(env, next_observations)
                    if bool(pl_cfg.log.save_reward_step):
                        step_rewards = [
                            float(result.reward)
                            for result in reward_results.values()
                        ]
                        sim_time_s = (
                            float(
                                next(iter(next_observations.values())).debug.get(
                                    "sim_time", 0.0
                                )
                            )
                            if next_observations
                            else float(
                                (step + 1) * base_cfg.env.decision_interval
                            )
                        )
                        reward_step_rows.append(
                            {
                                "episode": int(episode),
                                "decision_step": int(step),
                                "sim_time_s": sim_time_s,
                                "intersection_count": int(len(step_rewards)),
                                "reward_mean": _mean(step_rewards),
                            }
                        )
                    done = step == int(base_cfg.env.steps_per_episode) - 1

                    for tl_id, obs in observations.items():
                        result = reward_results[tl_id]
                        agent.store_transition(
                            PressLightTransition(
                                tl_id=tl_id,
                                group_id=obs.group_id,
                                state=obs.state,
                                action=int(actions[tl_id]),
                                reward=float(result.reward),
                                next_state=next_observations[tl_id].state,
                                next_action_mask=next_observations[tl_id].action_mask,
                                done=done,
                            )
                        )
                        rewards_all.append(float(result.reward))
                        pressures_all.append(float(result.intersection_pressure))
                        pressure_max_values.append(float(result.intersection_pressure))

                    update_stats = agent.update_from_replay()
                    agent.increment_step()
                    if update_stats:
                        q_losses.extend(float(item.q_loss) for item in update_stats)
                        _write_csv(
                            update_log_path,
                            [
                                {
                                    "episode": episode,
                                    "step": step,
                                    "global_decision_step": agent.global_decision_step,
                                    **asdict(item),
                                }
                                for item in update_stats
                            ],
                            append=True,
                        )

                    detail_interval = int(pl_cfg.log.train_step_detail_interval)
                    record_detail = bool(pl_cfg.log.save_train_step_detail) and (
                        detail_interval <= 1 or step % detail_interval == 0
                    )
                    if record_detail:
                        state_rows.extend(
                            observation_step_rows(episode, step, next_observations, network_spec)
                        )
                        pressure_rows.extend(pressure_step_rows(episode, step, reward_results))
                        q_rows.extend(q_step_rows(episode, step, action_infos))

                    observations = next_observations
                traffic_stats = env.close()
                env = None
            finally:
                if env is not None:
                    try:
                        env.close(parse_tripinfo=False)
                    except Exception:
                        pass

            if state_rows:
                episode_detail_dir = os.path.join(detail_dir, f"episode_{episode:04d}")
                _write_csv(os.path.join(episode_detail_dir, "presslight_state_step.csv"), state_rows)
                _write_csv(
                    os.path.join(episode_detail_dir, "presslight_pressure_step.csv"),
                    pressure_rows,
                )
                _write_csv(os.path.join(episode_detail_dir, "presslight_q_step.csv"), q_rows)

            if bool(pl_cfg.log.save_reward_step):
                _write_csv(
                    os.path.join(
                        reward_step_dir,
                        f"episode_{episode:04d}.csv",
                    ),
                    reward_step_rows,
                )

            episode_row = {
                "episode": int(episode),
                "sumo_seed": seed,
                "epsilon_start": epsilon_start,
                "epsilon_end": float(agent.epsilon),
                "reward_mean": _mean(rewards_all),
                "reward_sum": float(sum(rewards_all)),
                "pressure_mean": _mean(pressures_all),
                "pressure_max": max(pressure_max_values) if pressure_max_values else 0.0,
                "q_loss_mean": _mean(q_losses),
                "phase_switch_rate": _mean(switch_flags),
                "avg_delay_s": float(getattr(traffic_stats, "avg_delay_s", 0.0) or 0.0),
                "avg_travel_time_s": float(
                    getattr(traffic_stats, "avg_travel_time_s", 0.0) or 0.0
                ),
                "total_arrived": int(getattr(traffic_stats, "total_arrived", 0) or 0),
                "completion_rate": float(
                    getattr(traffic_stats, "completion_rate", 0.0) or 0.0
                ),
                "global_decision_step": int(agent.global_decision_step),
                "update_count": int(agent.update_count),
                "target_update_count": int(agent.target_update_count),
                "wall_time_s": float(time.time() - episode_started),
            }
            _write_csv(episode_log_path, [episode_row], append=True)

            periodic = episode % int(pl_cfg.training.checkpoint_interval) == 0
            if periodic:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_ep_{episode:04d}.pt"
                )
                agent.save_checkpoint(checkpoint_path, episode, extra={"train": episode_row})
                eval_checkpoint_path = os.path.join(
                    checkpoint_dir, f"eval_checkpoint_ep_{episode:04d}.pt"
                )
                agent.save_evaluation_checkpoint(eval_checkpoint_path, episode)
                _log(f"[checkpoint] saved {checkpoint_path}")
                _log(f"[checkpoint] saved {eval_checkpoint_path}")

                if pl_cfg.training.fixed_validation_enabled:
                    validation_rows, validation_summary = _run_fixed_validation(
                        checkpoint_episode=episode,
                        base_cfg=base_cfg,
                        pl_cfg=pl_cfg,
                        network_spec=network_spec,
                        agent=agent,
                        port=int(args.port),
                        use_gui=bool(args.use_gui),
                        validation_dir=validation_dir,
                    )
                    _write_csv(validation_episode_path, validation_rows, append=True)
                    _write_csv(validation_summary_path, [validation_summary], append=True)
                    metric_name = pl_cfg.training.best_metric
                    metric_value = float(validation_summary[metric_name])
                    qualified = bool(validation_summary["completion_qualified"])
                    if qualified and _is_better(
                        metric_value, best_value, pl_cfg.training.best_mode
                    ):
                        best_value = metric_value
                        best_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
                        agent.save_checkpoint(
                            best_path,
                            episode,
                            extra={
                                "train": episode_row,
                                "validation": validation_summary,
                                "source_periodic_checkpoint": checkpoint_path,
                            },
                        )
                        agent.save_evaluation_checkpoint(
                            os.path.join(checkpoint_dir, "best_eval_checkpoint.pt"),
                            episode,
                        )
                        _save_json(best_summary_path, validation_summary)
                        _log(
                            f"[best] episode={episode} {metric_name}={metric_value:.4f} "
                            f"completion={validation_summary['completion_rate']:.4f}"
                        )
                    else:
                        _log(
                            f"[validation] episode={episode} {metric_name}={metric_value:.4f} "
                            f"completion={validation_summary['completion_rate']:.4f} "
                            f"qualified={qualified}"
                        )

            if pl_cfg.training.save_latest_each_episode:
                agent.save_checkpoint(
                    os.path.join(checkpoint_dir, "latest_checkpoint.pt"),
                    episode,
                    extra={"train": episode_row},
                )
            _log(
                f"[episode] {episode}/{total_episodes} completed "
                f"episode_time={_format_duration(episode_row['wall_time_s'])} "
                f"total_elapsed={_format_duration(time.time() - training_started_at)}"
            )

        if total_episodes % int(pl_cfg.training.checkpoint_interval) != 0:
            final_path = os.path.join(
                checkpoint_dir, f"checkpoint_ep_{total_episodes:04d}.pt"
            )
            agent.save_checkpoint(final_path, total_episodes)
            _log(f"[checkpoint] saved final {final_path}")
        _log(
            f"[done] episodes={total_episodes} "
            f"total_time={_format_duration(time.time() - training_started_at)} "
            f"outputs={run_dir}"
        )
    except Exception as exc:
        _save_json(
            os.path.join(run_dir, "training_error.json"),
            {"error": repr(exc), "traceback": traceback.format_exc()},
        )
        _log(
            f"[error] total_time={_format_duration(time.time() - training_started_at)} "
            f"error={exc!r}"
        )
        raise


if __name__ == "__main__":
    main()
