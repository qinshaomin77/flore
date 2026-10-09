"""Evaluate the no-learning E2 movement Max-Pressure baseline."""
from __future__ import annotations

import argparse
import csv
import json
import inspect
import multiprocessing
import os
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from config import build_sumo_seed_list, get_config
from emission_lookup import EmissionEpisodeRecorder, build_emission_lookup
from env import SumoEnv
from logger import TrainingLogger
from max_pressure_controller import MaxPressureController
from network_parser import parse_network
try:
    from vehicle_state import consolidate_vehicle_state_outputs
except ImportError:
    consolidate_vehicle_state_outputs = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKERS = 3
DEFAULT_OUTPUT_ROOT = "results/evaluation/grid36/max_pressure"
DEFAULT_SAVE_PHYSICAL_LANE_STEP = False
CONTROLLER_SPECS = {
    "max_pressure": {
        "name": "e2_movement_max_pressure",
        "label": "Max-Pressure",
        "diagnostics_dir": "maxpressure_diagnostics",
        "sedan_priority_weight": 1.0,
        "truck_priority_weight": 1.0,
        "spatial_scale_invariant": False,
    },
    "truck_weighted_max_pressure": {
        "name": "e2_movement_truck_weighted_max_pressure",
        "label": "MaxPressure-TW",
        "diagnostics_dir": "truck_weighted_maxpressure_diagnostics",
        "sedan_priority_weight": 1.0,
        "truck_priority_weight": 2.0,
        "spatial_scale_invariant": True,
    },
}
DEFAULTS = {
    "sumocfg": (
        "data/grid36/"
        "truck_sensitive_grid36_maxpressure.sumocfg"
    ),
    "net": (
        "data/grid36/"
        "truck_sensitive_grid36.net.xml"
    ),
    "add": (
        "data/grid36/"
        "truck_sensitive_grid36.add.xml"
    ),
    "groups": (
        "data/grid36/"
        "intersection_groups.json"
    ),
    "map": (
        "data/grid36/"
        "maxpressure_detector_map.json"
    ),
}


def resolve_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def abspath(path: str | Path) -> Path:
    """Retain the original public helper for callers that import this module."""
    return resolve_path(ROOT, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prepare_detector_geometry(
    *,
    root: Path,
    defaults: dict[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], int]:
    """Create an output-local E2 file with geometry clamped to its net.

    Some real-world assets were generated from a different geometric revision
    of the same network. Keeping the correction output-local avoids modifying
    the source dataset and gives every parallel worker one validated input.
    """
    prepared = dict(defaults)
    if not bool(prepared.get("clamp_detector_geometry", False)):
        return prepared, 0

    net_path = resolve_path(root, prepared["net"])
    add_path = resolve_path(root, prepared["add"])
    net_root = ET.parse(net_path).getroot()
    lane_lengths = {
        lane.get("id"): float(lane.get("length"))
        for lane in net_root.findall("edge/lane")
        if lane.get("id") and lane.get("length")
    }
    add_tree = ET.parse(add_path)
    adjustments = 0
    for detector in add_tree.getroot().findall("laneAreaDetector"):
        lane_id = detector.get("lane")
        if lane_id not in lane_lengths:
            raise ValueError(
                f"Detector {detector.get('id')} references unknown lane "
                f"{lane_id}"
            )
        lane_length = lane_lengths[lane_id]
        detector_length = min(
            max(float(detector.get("length", "0")), 0.0),
            lane_length,
        )
        position = float(detector.get("pos", "0"))
        corrected_position = min(
            max(position, 0.0),
            max(0.0, lane_length - detector_length),
        )
        if (
            abs(detector_length - float(detector.get("length", "0"))) > 1e-9
            or abs(corrected_position - position) > 1e-9
        ):
            detector.set("length", f"{detector_length:.6f}")
            detector.set("pos", f"{corrected_position:.6f}")
            detector.set("friendlyPos", "true")
            adjustments += 1

    asset_dir = output_root / "runtime_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    runtime_add = asset_dir / add_path.name
    add_tree.write(runtime_add, encoding="utf-8", xml_declaration=True)
    prepared["add"] = str(runtime_add.resolve())
    return prepared, adjustments


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument(
        "--controller",
        choices=sorted(CONTROLLER_SPECS),
        default="max_pressure",
    )
    parser.add_argument(
        "--network",
        choices=["synthetic", "kunshan"],
        default="synthetic",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--seed",
        type=int,
        default=19,
        help="全局母种子，需与 evaluate.py 的 EVAL_SEED 保持一致",
    )
    parser.add_argument("--output-root", default="")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="并行评估进程数（默认 3）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8813,
        help="第一个工作进程的 TraCI 端口；后续进程依次加 1",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--duration", type=int)
    physical_group = parser.add_mutually_exclusive_group()
    physical_group.add_argument(
        "--save-physical-lane-step",
        dest="save_physical_lane_step",
        action="store_true",
        help="write physical/episode_XXXX/lane_step.csv",
    )
    physical_group.add_argument(
        "--no-save-physical-lane-step",
        dest="save_physical_lane_step",
        action="store_false",
        help="disable physical lane-step output",
    )
    parser.set_defaults(
        save_physical_lane_step=DEFAULT_SAVE_PHYSICAL_LANE_STEP
    )
    return parser.parse_args()


def _prepare_config(
    *,
    root: Path,
    defaults: dict[str, str],
    config_path: str | None,
    seed: int,
    duration: int | None,
    output_root: Path,
    save_physical_lane_step: bool = False,
):
    cfg = get_config(config_path)
    cfg.seed = int(seed)
    cfg.env.sumo_cfg = str(resolve_path(root, defaults["sumocfg"]))
    cfg.env.net_xml = str(resolve_path(root, defaults["net"]))
    cfg.env.add_xml = str(resolve_path(root, defaults["add"]))
    cfg.env.intersection_groups_json = str(resolve_path(root, defaults["groups"]))
    cfg.env.emission_factor_csv = str(
        resolve_path(root, cfg.env.emission_factor_csv)
    )
    if duration is not None:
        cfg.env.episode_duration = int(duration)

    # Optional outputs use episode-specific paths and are parallel-safe.
    cfg.env.save_detector_outputs = False
    cfg.env.save_sumo_aux_outputs = False
    cfg.env.save_fcd_output = False
    if hasattr(cfg.env, "save_vehicle_state_output"):
        cfg.env.save_vehicle_state_output = consolidate_vehicle_state_outputs is not None
    cfg.env.save_tls_phase_outputs = True
    cfg.env.tripinfo_dir = str(output_root / "tripinfo")
    cfg.log.log_root = str(output_root.parent)
    cfg.log.run_id = output_root.name
    cfg.log.save_step_physical = bool(save_physical_lane_step)
    if hasattr(cfg.env, "enable_lane_mechanism_metrics"):
        cfg.env.enable_lane_mechanism_metrics = bool(
            save_physical_lane_step
        )
    return cfg


def _run_single_episode(
    *,
    episode: int,
    seed: int,
    worker_id: int,
    port: int,
    use_gui: bool,
    cfg,
    net,
    detector_map: dict[str, Any],
    controller: MaxPressureController,
    emission_lookup,
    emission_recorder: EmissionEpisodeRecorder,
    logger: TrainingLogger,
    output_root: Path,
    controller_key: str,
    observation_builder=None,
    save_physical_lane_step: bool = False,
) -> dict[str, Any]:
    controller_spec = CONTROLLER_SPECS[controller_key]
    controller.reset()
    if save_physical_lane_step:
        if observation_builder is None:
            raise RuntimeError(
                "physical lane-step output requires ObsRewardBuilder"
            )
        logger.begin_episode(
            episode,
            record_lane_step=True,
            record_phase_step=False,
            record_q_step=False,
        )
    # Preserve the original E1 inputs alongside the movement E2 detector file.
    # Replacing all additional files would remove detectors parsed into net.
    cfg.env.replace_sumocfg_additional_files = False
    env = SumoEnv(
        cfg.env,
        net,
        port=port,
        use_gui=use_gui,
        emission_lookup=emission_lookup,
        emission_recorder=emission_recorder,
    )
    started = time.time()
    pressure_rows: list[dict[str, Any]] = []
    movement_rows: list[dict[str, Any]] = []
    shared_rows: list[dict[str, Any]] = []
    selected: list[float] = []
    ties = 0
    unknown = 0
    classified = 0
    total_nox: list[float] = []
    link_nox: list[float] = []
    stats = None
    episode_emission_result = None

    try:
        env.start(episode_id=episode, seed=seed, control_tls=True)
        if env.emission_recorder is None:
            raise RuntimeError(
                f"emission recorder is None at episode {episode}; "
                f"csv={cfg.env.emission_factor_csv}"
            )

        raw = {
            tl_id: env._collect_obs(tl_id, None)
            for tl_id in net.intersection_ids
        }
        for step in range(int(cfg.env.steps_per_episode)):
            previous_phases = {
                tl_id: int(obs.current_phase) for tl_id, obs in raw.items()
            }
            actions, infos, diagnostics = controller.act(
                env,
                raw,
                episode,
                step,
            )
            sim_time = float(
                getattr(
                    env,
                    "_sim_time",
                    step * cfg.env.decision_interval,
                )
            )

            for tl_id, info in infos.items():
                pressure_rows.append(
                    {
                        "episode": episode,
                        "step": step,
                        "sim_time": sim_time,
                        "tl_id": tl_id,
                        "current_phase": info.current_phase,
                        "selected_phase": info.selected_phase,
                        "phase_switched": int(
                            info.current_phase != info.selected_phase
                        ),
                        "action_mask_json": json.dumps(info.action_mask),
                        "phase_pressures_json": json.dumps(
                            info.phase_pressures
                        ),
                        "masked_phase_pressures_json": json.dumps(
                            info.masked_phase_pressures
                        ),
                        "selected_pressure": info.selected_pressure,
                        "tie_count": info.tie_count,
                        "tie_kept_current": int(info.tie_kept_current),
                    }
                )
                selected.append(info.selected_pressure)
                ties += int(info.tie_count > 1)

            for movement_id, movement_pressure in diagnostics[
                "movement_pressures"
            ].items():
                movement = controller.movements[movement_id]
                movement_row = {
                    "episode": episode,
                    "step": step,
                    "sim_time": sim_time,
                    "tl_id": movement.tl_id,
                    "movement_id": movement_id,
                    "from_lane": movement.from_lane,
                    "direction": movement.direction,
                    "to_edge": movement.to_edge,
                    "to_lanes_json": json.dumps(movement.to_lanes),
                    "upstream_detector_id": detector_map["upstream"][
                        movement.tl_id
                    ][movement.from_lane],
                    "downstream_detector_ids_json": json.dumps(
                        [
                            detector_map["downstream"][movement.tl_id][lane]
                            for lane in movement.to_lanes
                        ]
                    ),
                    "upstream_halting": movement_pressure.upstream_halting,
                    "downstream_halting_json": json.dumps(
                        movement_pressure.downstream_halting_by_lane
                    ),
                    "downstream_weights_json": json.dumps(
                        movement_pressure.downstream_weights
                    ),
                    "effective_downstream_halting": (
                        movement_pressure.effective_downstream_halting
                    ),
                    "movement_pressure": movement_pressure.pressure,
                    "service_weight": 1.0,
                }
                if controller_key == "truck_weighted_max_pressure":
                    movement_row.update(
                        {
                        "upstream_halting_raw": getattr(
                            movement_pressure,
                            "upstream_halting_raw",
                            movement_pressure.upstream_halting,
                        ),
                        "upstream_truck_halting": getattr(
                            movement_pressure,
                            "upstream_truck_halting",
                            0.0,
                        ),
                        "downstream_halting_json": json.dumps(
                            movement_pressure.downstream_halting_by_lane
                        ),
                        "downstream_halting_raw_json": json.dumps(
                            getattr(
                                movement_pressure,
                                "downstream_halting_raw_by_lane",
                                movement_pressure.downstream_halting_by_lane,
                            )
                        ),
                        "downstream_truck_halting_json": json.dumps(
                            getattr(
                                movement_pressure,
                                "downstream_truck_halting_by_lane",
                                dict.fromkeys(
                                    movement_pressure.downstream_halting_by_lane,
                                    0.0,
                                ),
                            )
                        ),
                        }
                    )
                movement_rows.append(movement_row)

            for classification in diagnostics[
                "shared_lane_classifications"
            ]:
                row = {
                    "episode": episode,
                    "step": step,
                    "sim_time": sim_time,
                    **classification,
                }
                shared_rows.append(row)
                unknown += row["unknown_halting"]
                classified += row["e2_halting_count"]

            raw = env.step(actions, decision_step=step)
            if save_physical_lane_step:
                observations = observation_builder.build_observations(raw)
                _, reward_components = observation_builder.compute_rewards(
                    raw,
                    actions,
                    previous_phases,
                )
                logger.log_step(
                    episode,
                    step,
                    raw,
                    observations,
                    actions,
                    infos,
                    reward_components,
                )
            emission = getattr(env, "_last_step_emission", None)
            if emission is not None:
                total_nox.append(
                    float(
                        getattr(
                            emission,
                            "total_by_pollutant",
                            {},
                        ).get("NOx", 0.0)
                    )
                )
                link_nox.extend(
                    logger.compute_step_link_emission_values(
                        step_emission=emission,
                        pollutant="NOx",
                    )
                )

        stats = env.close()
        recorder = getattr(env, "emission_recorder", None)
        if recorder is not None:
            try:
                episode_emission_result = recorder.finalize_episode()
            except Exception as exc:
                print(
                    f"[WARN] episode {episode}: finalize vehicle emission "
                    f"failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        env = None
    finally:
        if env is not None:
            env.close(parse_tripinfo=False)

    logger.write_episode_emission_result(
        episode=episode,
        emission_result=episode_emission_result,
        save_step_df=True,
        save_edge_step_raw=False,
        save_lane_step=True,
        save_vehicle_step=False,
    )
    if save_physical_lane_step:
        logger.flush_episode_physical(episode)
        lane_step_path = (
            output_root
            / "physical"
            / f"episode_{episode:04d}"
            / "lane_step.csv"
        )
        if not lane_step_path.is_file():
            raise RuntimeError(
                "Physical lane-step output was not created: "
                f"{lane_step_path}"
            )
    vehicle_emission_path = (
        output_root
        / "emission"
        / f"episode_{episode:04d}"
        / "vehicle_emission.csv"
    )
    if not vehicle_emission_path.is_file():
        raise RuntimeError(
            "Max-Pressure vehicle emission output was not created: "
            f"{vehicle_emission_path}"
        )

    if float(np.sum(total_nox)) == 0.0:
        print(
            f"[WARN] episode {episode}: total NOx is 0 mg; "
            "verify that this episode truly had no vehicle emissions.",
            file=sys.stderr,
            flush=True,
        )

    episode_dir = (
        output_root
        / controller_spec["diagnostics_dir"]
        / f"episode_{episode:04d}"
    )
    write_csv(episode_dir / "pressure_step.csv", pressure_rows)
    write_csv(
        episode_dir / "movement_pressure_step.csv",
        movement_rows,
    )
    write_csv(
        episode_dir / "shared_lane_classification_step.csv",
        shared_rows,
    )

    all_phase_pressures = [
        value
        for row in pressure_rows
        for value in json.loads(row["phase_pressures_json"])
    ]
    result = {
        "episode": episode,
        "controller_name": controller_spec["name"],
        "sumo_seed": seed,
        "worker_id": worker_id,
        "worker_pid": os.getpid(),
        "sumo_port": port,
        "avg_delay_s": getattr(stats, "avg_delay_s", 0.0),
        "avg_travel_time_s": getattr(
            stats,
            "avg_travel_time_s",
            0.0,
        ),
        "avg_waiting_time_s": getattr(
            stats,
            "avg_waiting_time_s",
            0.0,
        ),
        "arrived": getattr(stats, "total_arrived", 0),
        "throughput": getattr(stats, "total_arrived", 0),
        "total_nox_mg": float(np.sum(total_nox)),
        "vehicle_level_nox_mg": float(np.sum(total_nox)),
        "link_nox_mean": (
            float(np.mean(link_nox)) if link_nox else 0.0
        ),
        "link_nox_p95": (
            float(np.percentile(link_nox, 95)) if link_nox else 0.0
        ),
        "link_nox_source": (
            "logger.compute_step_link_emission_values(NOx)"
        ),
        "lane_nox_diagnostic": (
            "emission recorder retained by SumoEnv"
        ),
        "phase_pressure_mean": float(np.mean(all_phase_pressures)),
        "phase_pressure_std": float(np.std(all_phase_pressures)),
        "selected_pressure_mean": float(np.mean(selected)),
        "negative_selected_pressure_rate": float(
            np.mean(np.asarray(selected) < 0)
        ),
        "tie_rate": ties / max(1, len(selected)),
        "unknown_turn_classification_rate": (
            unknown / max(1, classified)
        ),
        "episode_wall_time_s": time.time() - started,
    }
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] "
        f"[worker {worker_id}/{os.getpid()} port={port}] "
        f"episode {episode} completed; "
        f"elapsed={result['episode_wall_time_s']:.2f}s",
        flush=True,
    )
    return result


def _run_episode_batch(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Run one worker's episode partition on its dedicated TraCI port."""
    root = Path(task["root"])
    output_root = Path(task["output_root"])
    defaults = dict(task["defaults"])
    save_physical_lane_step = bool(
        task.get("save_physical_lane_step", False)
    )
    cfg = _prepare_config(
        root=root,
        defaults=defaults,
        config_path=task["config_path"],
        seed=task["global_seed"],
        duration=task["duration"],
        output_root=output_root,
        save_physical_lane_step=save_physical_lane_step,
    )
    parse_add_xml = str(
        resolve_path(
            root,
            defaults.get("parse_add", cfg.env.add_xml),
        )
    )
    net = parse_network(
        cfg.env.net_xml,
        parse_add_xml,
        cfg.env.intersection_groups_json,
    )
    detector_map = json.loads(
        resolve_path(root, defaults["map"]).read_text(encoding="utf-8")
    )
    controller_key = str(task.get("controller", "max_pressure"))
    if controller_key == "truck_weighted_max_pressure":
        from max_pressure_truck_weighted import (
            TruckWeightedMaxPressureController,
        )

        controller_class = TruckWeightedMaxPressureController
    else:
        controller_class = MaxPressureController
    controller_spec = CONTROLLER_SPECS[controller_key]
    controller = controller_class(
        net,
        detector_map,
        cfg.max_pressure,
    )
    emission_lookup = build_emission_lookup(
        cfg.env.emission_factor_csv,
        default_pollutant=cfg.env.emission_pollutants[0],
    )
    recorder_kwargs = {
        "pollutants": cfg.env.emission_pollutants,
        "output_unit": "mg",
    }
    if "record_vehicle_step" in inspect.signature(
        EmissionEpisodeRecorder.__init__
    ).parameters:
        recorder_kwargs["record_vehicle_step"] = False
    recorder_parameters = inspect.signature(
        EmissionEpisodeRecorder.__init__
    ).parameters
    if "enable_lane_mechanism_metrics" in recorder_parameters:
        recorder_kwargs.update(
            {
                "enable_lane_mechanism_metrics": save_physical_lane_step,
                "signal_sensitive_halting_speed_mps": (
                    cfg.env.signal_sensitive_halting_speed_mps
                ),
                "signal_sensitive_low_speed_mps": (
                    cfg.env.signal_sensitive_low_speed_mps
                ),
                "signal_sensitive_restart_accel_ms2": (
                    cfg.env.signal_sensitive_restart_accel_ms2
                ),
            }
        )
    emission_recorder = EmissionEpisodeRecorder(emission_lookup, **recorder_kwargs)
    logger = TrainingLogger(
        cfg,
        net,
        model_meta={
            "controller": controller_spec["name"],
            "uses_rl_agent": False,
            "uses_checkpoint": False,
        },
        write_metadata=False,
    )
    observation_builder = None
    if save_physical_lane_step:
        from obs_reward import ObsRewardBuilder

        observation_builder = ObsRewardBuilder(cfg, net)

    results = []
    for episode, seed in task["episodes"]:
        results.append(
            _run_single_episode(
                episode=int(episode),
                seed=int(seed),
                worker_id=int(task["worker_id"]),
                port=int(task["port"]),
                use_gui=bool(task["use_gui"]),
                cfg=cfg,
                net=net,
                detector_map=detector_map,
                controller=controller,
                emission_lookup=emission_lookup,
                emission_recorder=emission_recorder,
                logger=logger,
                output_root=output_root,
                controller_key=controller_key,
                observation_builder=observation_builder,
                save_physical_lane_step=save_physical_lane_step,
            )
        )
    return results


def _validate_args(parsed: argparse.Namespace) -> None:
    if parsed.episodes <= 0:
        raise ValueError("--episodes must be greater than 0")
    if parsed.workers <= 0:
        raise ValueError("--workers must be greater than 0")
    effective_workers = min(parsed.workers, parsed.episodes)
    if parsed.port <= 0 or parsed.port + effective_workers - 1 > 65535:
        raise ValueError(
            "TraCI port range is invalid for the requested worker count"
        )


def main() -> None:
    parsed = args()
    _validate_args(parsed)

    root = Path(ROOT).resolve()
    defaults = dict(DEFAULTS)
    controller_spec = CONTROLLER_SPECS[parsed.controller]
    output_path = parsed.output_root
    if not output_path:
        if parsed.controller == "max_pressure":
            output_path = DEFAULT_OUTPUT_ROOT
        else:
            default_path = Path(DEFAULT_OUTPUT_ROOT)
            default_name = default_path.name
            if default_name.endswith("_maxpressure"):
                default_name = (
                    default_name[: -len("_maxpressure")]
                    + "_truck_weighted_max_pressure"
                )
            elif default_name == "max_pressure":
                default_name = "truck_weighted_max_pressure"
            else:
                default_name += "_truck_weighted_max_pressure"
            output_path = str(default_path.with_name(default_name))
    output_root = resolve_path(root, output_path).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    defaults, detector_geometry_adjustments = _prepare_detector_geometry(
        root=root,
        defaults=defaults,
        output_root=output_root,
    )
    config_path = (
        str(resolve_path(root, parsed.config).resolve())
        if parsed.config
        else None
    )

    # Build the deterministic seed list once in the parent process.
    cfg = _prepare_config(
        root=root,
        defaults=defaults,
        config_path=config_path,
        seed=parsed.seed,
        duration=parsed.duration,
        output_root=output_root,
        save_physical_lane_step=parsed.save_physical_lane_step,
    )
    sumo_seeds = build_sumo_seed_list(
        global_seed=int(cfg.seed),
        total_episodes=int(parsed.episodes),
        mode=str(cfg.env.sumo_seed_mode),
    )
    effective_workers = min(int(parsed.workers), int(parsed.episodes))
    ports = [int(parsed.port) + index for index in range(effective_workers)]

    seed_metadata = {
        "global_seed": int(cfg.seed),
        "sumo_seed_mode": str(cfg.env.sumo_seed_mode),
        "episodes": int(parsed.episodes),
        "sumo_seeds": [int(seed) for seed in sumo_seeds],
        "configured_workers": int(parsed.workers),
        "effective_workers": effective_workers,
        "sumo_ports": ports,
        "detector_geometry_adjustments": detector_geometry_adjustments,
        "physical_lane_step_enabled": bool(
            parsed.save_physical_lane_step
        ),
    }
    if parsed.controller == "truck_weighted_max_pressure":
        seed_metadata["controller"] = parsed.controller
    (output_root / "sumo_seeds.json").write_text(
        json.dumps(seed_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if str(cfg.env.sumo_seed_mode) == "fixed":
        print(
            "[WARN] sumo_seed_mode=fixed：所有 episode 将使用同一种子，"
            "跨 episode 方差会退化为 0。",
            file=sys.stderr,
        )

    # The parent is the sole writer of the shared link map.
    parse_add_xml = str(
        resolve_path(
            root,
            defaults.get("parse_add", cfg.env.add_xml),
        )
    )
    net = parse_network(
        cfg.env.net_xml,
        parse_add_xml,
        cfg.env.intersection_groups_json,
    )
    logger = TrainingLogger(
        cfg,
        net,
        model_meta={
            "controller": controller_spec["name"],
            "uses_rl_agent": False,
            "uses_checkpoint": False,
        },
        write_metadata=False,
    )
    logger.save_link_map()

    partitions: list[list[tuple[int, int]]] = [
        [] for _ in range(effective_workers)
    ]
    for index, seed in enumerate(sumo_seeds):
        partitions[index % effective_workers].append(
            (index + 1, int(seed))
        )

    tasks = [
        {
            "root": str(root),
            "defaults": defaults,
            "config_path": config_path,
            "global_seed": int(parsed.seed),
            "duration": parsed.duration,
            "output_root": str(output_root),
            "worker_id": worker_index + 1,
            "port": ports[worker_index],
            "use_gui": bool(parsed.gui),
            "controller": parsed.controller,
            "save_physical_lane_step": bool(
                parsed.save_physical_lane_step
            ),
            "episodes": partition,
        }
        for worker_index, partition in enumerate(partitions)
    ]

    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] "
        f"Starting {parsed.episodes} {controller_spec['label']} episodes with "
        f"{effective_workers} worker processes; ports={ports}",
        flush=True,
    )
    started = time.time()
    summary: list[dict[str, Any]] = []
    if effective_workers == 1:
        summary.extend(_run_episode_batch(tasks[0]))
    else:
        spawn_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=spawn_context,
        ) as executor:
            futures = [
                executor.submit(_run_episode_batch, task)
                for task in tasks
            ]
            for future in as_completed(futures):
                summary.extend(future.result())

    summary.sort(key=lambda row: int(row["episode"]))
    write_csv(output_root / "episode_summary.csv", summary)
    if parsed.save_physical_lane_step:
        missing_lane_step_files = [
            output_root
            / "physical"
            / f"episode_{episode:04d}"
            / "lane_step.csv"
            for episode in range(1, int(parsed.episodes) + 1)
            if not (
                output_root
                / "physical"
                / f"episode_{episode:04d}"
                / "lane_step.csv"
            ).is_file()
        ]
        if missing_lane_step_files:
            raise RuntimeError(
                "Physical lane-step outputs are missing: "
                + ", ".join(str(path) for path in missing_lane_step_files)
            )
    if consolidate_vehicle_state_outputs is not None:
        consolidate_vehicle_state_outputs(output_root)
    metadata = {
        "controller_name": controller_spec["name"],
        "note": (
            "Reward and NOx components are diagnostics only and do not affect "
            f"{controller_spec['label']} action selection."
        ),
        "parallel_execution": {
            "enabled": effective_workers > 1,
            "configured_workers": int(parsed.workers),
            "effective_workers": effective_workers,
            "sumo_ports": ports,
            "wall_time_s": time.time() - started,
        },
        "detector_geometry_adjustments": detector_geometry_adjustments,
        "physical_lane_step": {
            "enabled": bool(parsed.save_physical_lane_step),
            "path_pattern": "physical/episode_XXXX/lane_step.csv",
            "interval_s": int(cfg.env.decision_interval),
        },
        "episodes": summary,
    }
    if parsed.controller == "truck_weighted_max_pressure":
        metadata.update(
            {
                "controller_key": parsed.controller,
                "sedan_priority_weight": controller_spec[
                    "sedan_priority_weight"
                ],
                "truck_priority_weight": controller_spec[
                    "truck_priority_weight"
                ],
                "control_detector_length_m": 300,
                "spatial_scale_invariant": True,
                "uses_rl_agent": False,
                "uses_checkpoint": False,
                "uses_reward_for_control": False,
                "uses_emissions_for_control": False,
            }
        )
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] "
        f"All episodes completed in {time.time() - started:.2f}s. "
        f"Output: {output_root}",
        flush=True,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
