# -*- coding: utf-8 -*-
"""
logger.py
=========
MGMQ-DDQN 复现版日志模块。

边界
----
logger 只负责记录，不计算 reward、不计算 Q loss、不选择动作、不调用模型 forward、
不采样 buffer、不调用 SUMO/TraCI。
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from config import MasterConfig
from env import EpisodeTrafficStats, StepRawObs
from network_parser import NetworkInfo, build_group_meta, build_phase_lane_mask
from obs_reward import AgentObservation, RewardComponents


REWARD_RATIO_EPS = 1e-6


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if (
        obj.__class__.__module__.startswith("torch")
        and hasattr(obj, "detach")
        and hasattr(obj, "cpu")
    ):
        return obj.detach().cpu().tolist()
    return obj


def _append_csv(path: str, row: Mapping[str, Any], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_csv(path: str, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _append_text(path: str, line: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return default
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return default
    return float(np.nanmean(arr))


class TrainingLogger:
    """MGMQ-DDQN 训练日志器。"""

    REWARD_STEP_FIELDS = [
        "episode",
        "decision_step",
        "sim_time_s",
        "intersection_count",
        "reward_mean",
        "risk_reward_mode",
        "risk_sum",
        "risk_max",
        "mean_lane_risk",
        "p95_lane_risk",
    ]

    EPISODE_FIELDS = [
        "episode", "wall_time_s", "sumo_seed", "scenario", "scenario_round",
        "scenario_sample_index", "scenario_sumocfg", "route_file",
        "epsilon_start", "epsilon_end", "epsilon_mean", "replay_size_end",
        "reward_mean", "reward_sum", "reward_std",
        "traffic_reward_mean", "traffic_reward_abs_mean",
        "emission_penalty_mean", "emission_penalty_abs_mean",
        "weighted_emission_penalty_mean", "weighted_emission_penalty_abs_mean",
        "weighted_emission_penalty_sum", "final_reward_mean",
        "emission_penalty_ratio_mean", "emission_penalty_ratio_p50",
        "emission_penalty_ratio_p90", "emission_penalty_ratio_p95",
        "emission_penalty_ratio_max", "emission_penalty_ratio_of_means",
        "lambda_e_mean",
        "nox_risk_mean", "nox_risk_sum_mean", "nox_risk_max_mean",
        "risk_reward_mode", "reward_risk_sum_mean", "reward_risk_max_mean",
        "mean_lane_risk", "p95_lane_risk",
        "nox_pressure_mean", "nox_pressure_max_mean",
        "link_nox_mean",
        "nox_warning_lane_count_mean", "nox_exceed_lane_count_mean",
        "rp_mean", "rwn_mean", "rpn_mean", "rwt_mean",
        "phase_switch_rate", "mask_active_rate", "random_action_rate", "greedy_action_rate",
        "avg_delay_s", "avg_travel_time_s", "total_arrived", "completion_rate", "total_throughput",
        "q_loss_mean", "td_error_mean", "q_pred_mean", "q_target_mean",
        "is_best", "best_metric", "best_score", "steps", "n_agents",
    ]

    UPDATE_FIELDS = [
        "global_step", "episode", "update_index", "group_id",
        "batch_size", "replay_size", "q_loss", "td_error_mean", "td_error_abs_mean",
        "q_pred_mean", "q_target_mean", "reward_batch_mean", "grad_norm",
        "traffic_loss", "nox_loss", "td_traffic_mean", "td_nox_mean",
        "q_traffic_pred_mean", "q_nox_pred_mean", "target_traffic_mean", "target_nox_mean",
        "traffic_reward_batch_mean", "nox_penalty_batch_mean",
        "learning_rate", "online_update_count", "target_update_count", "target_synced",
    ]

    LANE_STEP_FIELDS = [
        "episode", "step", "sim_time", "tl_id", "group_id", "lane_index", "lane_id",
        "demand", "queue", "wait_vwt", "pass_veh", "green_service_s", "lane_phase",
        "speed_mps", "truck_count", "truck_count_is_model_input", "nox_risk_is_model_input", "total_count", "lane_nox_mg",
        "nox_tau_mg", "nox_pressure",
        "nox_base_risk", "nox_tail_risk", "nox_risk", "nox_risk_mode",
        "risk_reward_mode", "reward_lane_risk",
        "nox_base_weight_effective", "nox_tail_weight_effective",
        "nox_warning", "nox_exceed",
        "current_phase", "action_selected",
    ]

    PHASE_STEP_FIELDS = [
        "episode", "step", "sim_time", "tl_id", "group_id",
        "current_phase", "action_selected", "phase_switched", "elapsed_green", "in_yellow", "pending_phase",
        "mask_active", "valid_action_count", "action_mask_json",
        "rp", "rwn", "rpn", "rwt", "traffic_reward", "emission_penalty", "lambda_e",
        "weighted_emission_penalty", "abs_traffic_reward", "abs_emission_penalty",
        "abs_weighted_emission_penalty", "emission_penalty_ratio", "reward",
        "nox_risk_mean", "nox_risk_sum", "nox_risk_max", "nox_pressure_mean", "nox_pressure_max",
        "risk_reward_mode", "risk_sum", "risk_max", "mean_lane_risk", "p95_lane_risk",
        "nox_warning_lane_count", "nox_exceed_lane_count",
        "sum_queue", "sum_wait_vwt", "sum_pass_veh", "sum_demand",
        "sum_lane_nox_mg", "sum_edge_nox_mg", "edge_nox_json",
    ]

    Q_STEP_FIELDS = [
        "episode", "step", "sim_time", "tl_id", "group_id",
        "epsilon", "action_selected", "is_random_action", "is_greedy_action",
        "q_values_json", "q_masked_values_json", "q_max", "q_selected", "q_margin",
        "q_total_values_json", "q_traffic_values_json", "q_nox_values_json",
        "q_total_selected", "q_traffic_selected", "q_nox_selected",
        "valid_action_count", "action_mask_json",
    ]

    EDGE_EMISSION_FIELDS = [
        "episode", "step", "sim_time_start", "sim_time_end",
        "edge_id", "from_node", "to_node", "link_id", "is_controlled_link",
        "lane_ids_json", "n_lanes",
        "pollutant", "edge_emission_mg",
        "step_n_records", "step_n_vehicles",
    ]

    LINK_EMISSION_FIELDS = [
        "episode", "step", "sim_time_start", "sim_time_end",
        "link_id", "node_u", "node_v",
        "edge_ids_json", "n_edges",
        "lane_ids_json", "n_lanes",
        "pollutant", "link_emission_mg", "mean_edge_emission_mg",
        "step_n_records", "step_n_vehicles",
    ]

    PARAM_FIELDS = [
        "episode", "global_step", "network_type", "group_id", "module_name", "parameter_name",
        "shape", "numel", "requires_grad", "mean", "std", "min", "max", "abs_mean", "l2_norm",
        "grad_l2_norm", "grad_abs_mean",
    ]

    BEST_FIELDS = [
        "episode", "global_step", "best_metric", "old_best_score", "new_best_score",
        "improved", "checkpoint_path", "epsilon", "replay_size", "reward_mean",
        "avg_delay_s", "completion_rate", "q_loss_mean", "td_error_mean",
    ]

    def __init__(
        self,
        cfg: MasterConfig,
        net_info: NetworkInfo,
        model_meta: Optional[Mapping[str, Any]] = None,
        write_metadata: bool = True,
    ) -> None:
        self.cfg = cfg
        self.net_info = net_info
        self.run_dir = cfg.log.run_dir
        self.physical_root = os.path.join(self.run_dir, "physical")
        self.reward_step_root = os.path.join(self.run_dir, "reward_step")
        self.emission_root = os.path.join(self.run_dir, "emission")
        self.params_dir = os.path.join(self.run_dir, "params")
        self.best_dir = os.path.join(self.run_dir, "best")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.event_txt = os.path.join(self.run_dir, "event_log.txt")
        self.episode_csv = os.path.join(self.run_dir, "episode_log.csv")
        self.update_csv = os.path.join(self.run_dir, "update_log.csv")
        self.best_log_csv = os.path.join(self.run_dir, "best_log.csv")
        self._start_time = time.time()
        self._lane_rows: list[dict] = []
        self._phase_rows: list[dict] = []
        self._q_rows: list[dict] = []
        self._reward_step_rows: list[dict] = []
        self._edge_emission_rows: list[dict] = []
        self._link_emission_rows: list[dict] = []
        self._edge_to_link: dict[str, str] = {}
        self._link_to_edges: dict[str, list[str]] = {}
        self._link_nodes: dict[str, tuple[str, str]] = {}
        self._edge_nodes: dict[str, tuple[str, str]] = {}
        self._edge_to_lanes: dict[str, list[str]] = {}
        self._link_to_lanes: dict[str, list[str]] = {}
        self._lane_to_link: dict[str, str] = {}
        self._record_lane_step: bool = True
        self._record_phase_step: bool = True
        self._record_q_step: bool = True
        self._build_link_maps()
        self._prepare_dirs()
        if bool(write_metadata):
            self.save_link_map()
            self.save_run_metadata(model_meta=model_meta or {})

    def _prepare_dirs(self) -> None:
        directories = [
            self.run_dir,
            self.physical_root,
            self.emission_root,
            self.params_dir,
            self.best_dir,
            self.checkpoint_dir,
        ]
        if bool(getattr(self.cfg.log, "save_reward_step", True)):
            directories.append(self.reward_step_root)
        for d in directories:
            os.makedirs(d, exist_ok=True)

    def _build_link_maps(self) -> None:
        self._edge_to_link.clear()
        self._link_to_edges.clear()
        self._link_nodes.clear()
        self._edge_nodes.clear()
        self._edge_to_lanes.clear()
        self._link_to_lanes.clear()
        self._lane_to_link.clear()

        tl_set = set(str(x) for x in self.net_info.intersection_ids)
        for edge_id_raw, einfo in self.net_info.edge_info.items():
            edge_id = str(edge_id_raw)
            u = str(getattr(einfo, "from_node", "") or "")
            v = str(getattr(einfo, "to_node", "") or "")
            lane_ids = [str(x) for x in (getattr(einfo, "lane_ids", []) or [])]
            self._edge_nodes[edge_id] = (u, v)
            self._edge_to_lanes[edge_id] = lane_ids

            if u in tl_set and v in tl_set:
                node_u, node_v = sorted([u, v])
                link_id = f"{node_u}__{node_v}"
                self._edge_to_link[edge_id] = link_id
                self._link_to_edges.setdefault(link_id, []).append(edge_id)
                self._link_nodes[link_id] = (node_u, node_v)
                self._link_to_lanes.setdefault(link_id, []).extend(lane_ids)
                for lane_id in lane_ids:
                    self._lane_to_link[lane_id] = link_id
            else:
                self._edge_to_link[edge_id] = ""

        for link_id in list(self._link_to_edges.keys()):
            self._link_to_edges[link_id] = sorted(set(self._link_to_edges[link_id]))
            self._link_to_lanes[link_id] = sorted(set(self._link_to_lanes.get(link_id, [])))

    def save_link_map(self) -> None:
        links: Dict[str, Any] = {}
        for link_id in sorted(self._link_to_edges.keys()):
            edge_ids = list(self._link_to_edges.get(link_id, []))
            lane_ids = list(self._link_to_lanes.get(link_id, []))
            node_u, node_v = self._link_nodes.get(link_id, ("", ""))
            links[link_id] = {
                "node_u": node_u,
                "node_v": node_v,
                "edge_ids": edge_ids,
                "lane_ids": lane_ids,
                "n_edges": int(len(edge_ids)),
                "n_lanes": int(len(lane_ids)),
            }
        self.save_json(os.path.join(self.emission_root, "link_map.json"), {
            "n_links": int(len(links)),
            "links": links,
        })

    def save_json(self, path: str, payload: Any) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(payload), f, indent=2, ensure_ascii=False)

    def save_run_metadata(self, model_meta: Mapping[str, Any]) -> None:
        self.save_json(os.path.join(self.run_dir, "run_config.json"), {
            "created_at_unix": time.time(),
            "run_dir": self.run_dir,
            "cfg": self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else _to_jsonable(self.cfg),
        })
        net_summary: Dict[str, Any] = {
            "n_intersections": len(self.net_info.intersection_ids),
            "n_lanes_total": len(self.net_info.lane_info),
            "n_edges_total": len(self.net_info.edge_info),
            "intersections": {},
        }
        for tl_id in self.net_info.intersection_ids:
            iinfo = self.net_info.get_intersection(tl_id)
            net_summary["intersections"][tl_id] = {
                "group_id": self.net_info.get_group(tl_id),
                "n_lanes": iinfo.n_lanes,
                "n_green_phases": iinfo.n_green_phases,
                "neighbors_nesw": self.net_info.get_neighbors(tl_id),
                "green_phase_states": [p.state for p in iinfo.green_phases],
                "phase_lane_mask_shape": list(build_phase_lane_mask(iinfo).shape),
            }
        self.save_json(os.path.join(self.run_dir, "network_summary.json"), net_summary)
        group_meta = build_group_meta(self.net_info, node_update_dim=int(self.cfg.network.node_update_dim))
        self.save_json(os.path.join(self.run_dir, "group_summary.json"), group_meta)
        self.save_json(os.path.join(self.run_dir, "model_structure.json"), model_meta)

    @staticmethod
    def _fmt_compact_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if abs(value) >= 1000:
                return f"{value:.1f}"
            return f"{value:.4g}"
        return str(value)

    @staticmethod
    def _compact_detail(detail: Mapping[str, Any], keys: Sequence[str]) -> str:
        parts = []
        for key in keys:
            if key in detail:
                parts.append(f"{key}={TrainingLogger._fmt_compact_value(detail[key])}")
        return ", ".join(parts)

    def _format_compact_event(
        self,
        event: str,
        time_iso: str,
        *,
        episode: Optional[int],
        global_step: Optional[int],
        message: str,
        detail_payload: Any,
    ) -> str:
        detail = detail_payload if isinstance(detail_payload, Mapping) else {}
        prefix = f"[{time_iso}]"

        if event == "run_start":
            suffix = self._compact_detail(detail, ("total_episodes", "seed", "device", "deterministic"))
            return f"{prefix} run start | {suffix}" if suffix else f"{prefix} run start"

        if event == "episode_start":
            title = message or f"episode {episode} started"
            suffix = self._compact_detail(detail, ("scenario", "sumo_seed", "epsilon"))
            return f"{prefix} {title} | {suffix}" if suffix else f"{prefix} {title}"

        if event == "episode_summary_saved":
            title = f"episode {episode} done" if episode is not None else "episode done"
            parts = [message] if message else []
            suffix = self._compact_detail(
                detail,
                ("wall_time_s", "total_arrived", "avg_delay_s", "completion_rate", "link_nox_mean", "is_best"),
            )
            if suffix:
                parts.append(suffix)
            return f"{prefix} {title} | " + " | ".join(parts) if parts else f"{prefix} {title}"

        if event == "checkpoint_best_saved":
            suffix = self._compact_detail(detail, ("best_metric", "best_score"))
            title = f"episode {episode} best checkpoint" if episode is not None else "best checkpoint"
            return f"{prefix} {title} | {suffix}" if suffix else f"{prefix} {title}"

        if event == "run_end":
            suffix = self._compact_detail(detail, ("total_episodes", "best_score", "training_elapsed_s"))
            return f"{prefix} run end | {suffix}" if suffix else f"{prefix} run end"

        if event == "exception":
            return f"{prefix} exception | {message}" if message else f"{prefix} exception"

        ctx = []
        if episode is not None:
            ctx.append(f"ep={int(episode)}")
        if global_step is not None:
            ctx.append(f"global={int(global_step)}")
        ctx_text = " ".join(ctx)
        head = f"{prefix} {event}"
        if ctx_text:
            head += f" {ctx_text}"
        return f"{head} | {message}" if message else head

    def log_event(
        self,
        event: str,
        *,
        episode: Optional[int] = None,
        step: Optional[int] = None,
        global_step: Optional[int] = None,
        sim_time: Optional[float] = None,
        message: str = "",
        detail: Optional[Mapping[str, Any]] = None,
        to_console: bool = True,
    ) -> None:
        detail_payload = _to_jsonable(detail or {})
        time_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        wall_time_s = round(float(time.time() - self._start_time), 3)
        detail_json = json.dumps(detail_payload, ensure_ascii=False, sort_keys=True)

        ctx = []
        if episode is not None:
            ctx.append(f"ep={int(episode)}")
        if step is not None:
            ctx.append(f"step={int(step)}")
        if global_step is not None:
            ctx.append(f"global={int(global_step)}")
        if sim_time is not None:
            ctx.append(f"sim={float(sim_time):.1f}s")

        line = f"[{time_iso}] +{wall_time_s:.3f}s {event}"
        if ctx:
            line += " " + " ".join(ctx)
        if message:
            line += f" | {message}"

        detail_items = []
        if isinstance(detail_payload, Mapping):
            for key, value in detail_payload.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    detail_items.append(f"{key}={value}")
        if detail_items:
            line += " | " + ", ".join(detail_items)

        file_line = line
        if detail_payload:
            file_line += f" | detail_json={detail_json}"
        _append_text(self.event_txt, file_line)

        console_enabled = bool(getattr(self.cfg.log, "console_event_log", False))
        is_exception = event == "exception"
        if (to_console or is_exception) and console_enabled:
            mode = str(getattr(self.cfg.log, "console_mode", "full")).strip().lower()
            if mode == "silent" and not is_exception:
                return
            if mode == "compact" or (mode == "silent" and is_exception):
                whitelist = set(getattr(self.cfg.log, "console_event_whitelist", ()))
                if event in whitelist or is_exception:
                    print(
                        self._format_compact_event(
                            event,
                            time_iso,
                            episode=episode,
                            global_step=global_step,
                            message=message,
                            detail_payload=detail_payload,
                        ),
                        flush=True,
                    )
                return
            print(line, flush=True)

    def begin_episode(
        self,
        episode: int,
        *,
        record_lane_step: bool = True,
        record_phase_step: bool = True,
        record_q_step: bool = True,
    ) -> None:
        self._lane_rows.clear()
        self._phase_rows.clear()
        self._q_rows.clear()
        self._reward_step_rows.clear()
        self.clear_episode_emission()
        self._record_lane_step = bool(record_lane_step) and bool(self.cfg.log.save_step_physical)
        self._record_phase_step = bool(record_phase_step) and bool(self.cfg.log.save_step_physical)
        self._record_q_step = bool(record_q_step) and bool(self.cfg.log.save_q_step)

    def log_step_reward(
        self,
        *,
        episode: int,
        decision_step: int,
        sim_time_s: float,
        rewards: Mapping[str, float],
        reward_components: Mapping[str, RewardComponents] | None = None,
    ) -> None:
        """Record the mean reward across all intersections for one decision."""
        if not bool(getattr(self.cfg.log, "save_reward_step", True)):
            return
        values = [float(value) for value in rewards.values()]
        components = list((reward_components or {}).values())
        modes = [str(getattr(comp, "risk_reward_mode", "full")) for comp in components]
        risk_sums = [float(getattr(comp, "reward_risk_sum", 0.0)) for comp in components]
        risk_maxes = [float(getattr(comp, "reward_risk_max", 0.0)) for comp in components]
        lane_means = [float(getattr(comp, "mean_lane_risk", 0.0)) for comp in components]
        lane_p95s = [float(getattr(comp, "p95_lane_risk", 0.0)) for comp in components]
        self._reward_step_rows.append(
            {
                "episode": int(episode),
                "decision_step": int(decision_step),
                "sim_time_s": float(sim_time_s),
                "intersection_count": int(len(values)),
                "reward_mean": float(np.mean(values)) if values else 0.0,
                "risk_reward_mode": modes[0] if modes else "",
                "risk_sum": float(np.mean(risk_sums)) if risk_sums else 0.0,
                "risk_max": float(np.mean(risk_maxes)) if risk_maxes else 0.0,
                "mean_lane_risk": float(np.mean(lane_means)) if lane_means else 0.0,
                "p95_lane_risk": float(np.mean(lane_p95s)) if lane_p95s else 0.0,
            }
        )

    def log_step(
        self,
        episode: int,
        step: int,
        raw_obs: Mapping[str, StepRawObs],
        observations: Mapping[str, AgentObservation],
        actions: Mapping[str, int],
        action_infos: Mapping[str, Any],
        reward_components: Mapping[str, RewardComponents],
    ) -> None:
        need_lane = bool(self._record_lane_step)
        need_phase = bool(self._record_phase_step)
        need_q = bool(self._record_q_step)
        if not (need_lane or need_phase or need_q):
            return

        def _safe_float(values: Any, idx: int, default: float = 0.0) -> float:
            try:
                return float(values[idx])
            except Exception:
                return float(default)

        def _safe_value(values: Any, idx: int, default: Any = "") -> Any:
            try:
                return values[idx]
            except Exception:
                return default

        for tl_id, obs in observations.items():
            raw = raw_obs[tl_id]
            sim_time = float(raw.sim_time)
            action = int(actions.get(tl_id, -1))
            info = action_infos.get(tl_id) if (need_phase or need_q) else None

            if need_lane:
                lanes = self.net_info.get_intersection(tl_id).all_inc_lanes_flat()
                features = obs.lane_features
                feature_names = tuple(getattr(obs, "lane_feature_names", ()))
                feature_idx = {name: i for i, name in enumerate(feature_names)}
                truck_count_is_model_input = "truck_count" in feature_idx
                nox_risk_is_model_input = "nox_risk" in feature_idx
                pass_veh = getattr(raw, "pass_veh", [0.0] * len(lanes))
                green_service_s = getattr(raw, "green_service_s", [])
                speed_mps = getattr(raw, "speed_mps", [])
                if truck_count_is_model_input:
                    truck_count = features[:, feature_idx["truck_count"]]
                else:
                    truck_count = getattr(raw, "truck_count", [])
                total_count = getattr(raw, "total_count", [])
                lane_nox_mg = getattr(raw, "NOx_mg", [])
                lane_debug = obs.debug.get("lane", {}) if isinstance(obs.debug, Mapping) else {}
                nox_tau_mg = lane_debug.get("nox_tau_mg", [])
                nox_pressure = lane_debug.get("nox_pressure", [])
                nox_base_risk = lane_debug.get("nox_base_risk", [])
                nox_tail_risk = lane_debug.get("nox_tail_risk", [])
                nox_risk = lane_debug.get("nox_risk", [])
                nox_risk_mode = lane_debug.get("nox_risk_mode", [])
                lane_comp = reward_components.get(tl_id)
                risk_reward_mode = str(
                    getattr(lane_comp, "risk_reward_mode", "full")
                )
                if risk_reward_mode == "no_tail":
                    reward_lane_risk = np.clip(
                        np.asarray(nox_base_risk, dtype=np.float32),
                        0.0,
                        float(self.cfg.emission_risk.risk_clip),
                    )
                else:
                    reward_lane_risk = nox_risk
                nox_base_weight_effective = lane_debug.get(
                    "nox_base_weight_effective", []
                )
                nox_tail_weight_effective = lane_debug.get(
                    "nox_tail_weight_effective", []
                )
                nox_warning = lane_debug.get("nox_warning", [])
                nox_exceed = lane_debug.get("nox_exceed", [])
                for idx, lane_id in enumerate(lanes):
                    self._lane_rows.append({
                        "episode": episode,
                        "step": step,
                        "sim_time": sim_time,
                        "tl_id": tl_id,
                        "group_id": obs.group_id,
                        "lane_index": idx,
                        "lane_id": lane_id,
                        "demand": float(features[idx, 0]),
                        "queue": float(features[idx, 1]),
                        "wait_vwt": float(features[idx, 2]),
                        "pass_veh": _safe_float(pass_veh, idx),
                        "green_service_s": _safe_float(green_service_s, idx),
                        "lane_phase": float(features[idx, 3]),
                        "speed_mps": _safe_float(speed_mps, idx),
                        "truck_count": _safe_float(truck_count, idx),
                        "truck_count_is_model_input": bool(truck_count_is_model_input),
                        "nox_risk_is_model_input": bool(nox_risk_is_model_input),
                        "total_count": _safe_float(total_count, idx),
                        "lane_nox_mg": _safe_float(lane_nox_mg, idx),
                        "nox_tau_mg": _safe_float(nox_tau_mg, idx),
                        "nox_pressure": _safe_float(nox_pressure, idx),
                        "nox_base_risk": _safe_float(nox_base_risk, idx),
                        "nox_tail_risk": _safe_float(nox_tail_risk, idx),
                        "nox_risk": _safe_float(nox_risk, idx),
                        "nox_risk_mode": str(
                            _safe_value(nox_risk_mode, idx, "")
                        ),
                        "risk_reward_mode": risk_reward_mode,
                        "reward_lane_risk": _safe_float(reward_lane_risk, idx),
                        "nox_base_weight_effective": _safe_float(
                            nox_base_weight_effective, idx
                        ),
                        "nox_tail_weight_effective": _safe_float(
                            nox_tail_weight_effective, idx
                        ),
                        "nox_warning": int(_safe_float(nox_warning, idx)),
                        "nox_exceed": int(_safe_float(nox_exceed, idx)),
                        "current_phase": int(raw.current_phase),
                        "action_selected": action,
                    })

            if need_phase:
                comp = reward_components.get(tl_id)
                pass_veh = getattr(raw, "pass_veh", [])
                lane_nox_mg = getattr(raw, "NOx_mg", [])
                edge_nox_mg = getattr(raw, "edge_NOx_mg", [])
                mask_active = bool(info.valid_action_count < len(info.action_mask)) if info is not None else False
                if comp is not None:
                    traffic_reward = float(getattr(comp, "traffic_reward", comp.reward))
                    emission_penalty = float(getattr(comp, "emission_penalty", 0.0))
                    lambda_e = float(getattr(comp, "lambda_e", 0.0))
                    weighted_emission_penalty = float(lambda_e * emission_penalty)
                    abs_traffic_reward = float(abs(traffic_reward))
                    abs_emission_penalty = float(abs(emission_penalty))
                    abs_weighted_emission_penalty = float(abs(weighted_emission_penalty))
                    emission_penalty_ratio = float(
                        abs_weighted_emission_penalty / max(abs_traffic_reward, REWARD_RATIO_EPS)
                    )
                    final_reward = float(comp.reward)
                else:
                    traffic_reward = ""
                    emission_penalty = ""
                    lambda_e = ""
                    weighted_emission_penalty = ""
                    abs_traffic_reward = ""
                    abs_emission_penalty = ""
                    abs_weighted_emission_penalty = ""
                    emission_penalty_ratio = ""
                    final_reward = ""
                self._phase_rows.append({
                    "episode": episode,
                    "step": step,
                    "sim_time": sim_time,
                    "tl_id": tl_id,
                    "group_id": obs.group_id,
                    "current_phase": int(raw.current_phase),
                    "action_selected": action,
                    "phase_switched": int(action != int(raw.current_phase)),
                    "elapsed_green": float(raw.elapsed_green),
                    "in_yellow": bool(raw.in_yellow),
                    "pending_phase": int(getattr(raw, "pending_phase", -1)),
                    "mask_active": mask_active,
                    "valid_action_count": int(info.valid_action_count) if info is not None else int(np.sum(obs.action_mask)),
                    "action_mask_json": json.dumps(obs.action_mask.astype(int).tolist(), ensure_ascii=False),
                    "rp": float(comp.rp) if comp is not None else "",
                    "rwn": float(comp.rwn) if comp is not None else "",
                    "rpn": float(comp.rpn) if comp is not None else "",
                    "rwt": float(comp.rwt) if comp is not None else "",
                    "traffic_reward": traffic_reward,
                    "emission_penalty": emission_penalty,
                    "lambda_e": lambda_e,
                    "weighted_emission_penalty": weighted_emission_penalty,
                    "abs_traffic_reward": abs_traffic_reward,
                    "abs_emission_penalty": abs_emission_penalty,
                    "abs_weighted_emission_penalty": abs_weighted_emission_penalty,
                    "emission_penalty_ratio": emission_penalty_ratio,
                    "reward": final_reward,
                    "nox_risk_mean": float(getattr(comp, "nox_risk_mean", 0.0)) if comp is not None else "",
                    "nox_risk_sum": float(getattr(comp, "nox_risk_sum", 0.0)) if comp is not None else "",
                    "nox_risk_max": float(getattr(comp, "nox_risk_max", 0.0)) if comp is not None else "",
                    "risk_reward_mode": str(getattr(comp, "risk_reward_mode", "")) if comp is not None else "",
                    "risk_sum": float(getattr(comp, "reward_risk_sum", 0.0)) if comp is not None else "",
                    "risk_max": float(getattr(comp, "reward_risk_max", 0.0)) if comp is not None else "",
                    "mean_lane_risk": float(getattr(comp, "mean_lane_risk", 0.0)) if comp is not None else "",
                    "p95_lane_risk": float(getattr(comp, "p95_lane_risk", 0.0)) if comp is not None else "",
                    "nox_pressure_mean": float(getattr(comp, "nox_pressure_mean", 0.0)) if comp is not None else "",
                    "nox_pressure_max": float(getattr(comp, "nox_pressure_max", 0.0)) if comp is not None else "",
                    "nox_warning_lane_count": int(getattr(comp, "nox_warning_lane_count", 0)) if comp is not None else "",
                    "nox_exceed_lane_count": int(getattr(comp, "nox_exceed_lane_count", 0)) if comp is not None else "",
                    "sum_queue": float(np.sum(raw.queue_veh)),
                    "sum_wait_vwt": float(np.sum(raw.wait_vwt)),
                    "sum_pass_veh": float(np.sum(pass_veh)),
                    "sum_demand": float(np.sum(raw.demand_veh)),
                    "sum_lane_nox_mg": float(np.sum(lane_nox_mg)),
                    "sum_edge_nox_mg": float(np.sum(edge_nox_mg)),
                    "edge_nox_json": json.dumps([float(x) for x in edge_nox_mg], ensure_ascii=False),
                })

            if need_q and info is not None:
                self._q_rows.append({
                    "episode": episode,
                    "step": step,
                    "sim_time": sim_time,
                    "tl_id": tl_id,
                    "group_id": obs.group_id,
                    "epsilon": float(info.epsilon),
                    "action_selected": int(info.action),
                    "is_random_action": bool(info.is_random_action),
                    "is_greedy_action": bool(info.is_greedy_action),
                    "q_values_json": json.dumps(info.q_values, ensure_ascii=False),
                    "q_masked_values_json": json.dumps(info.q_masked_values, ensure_ascii=False),
                    "q_max": float(info.q_max),
                    "q_selected": float(info.q_selected),
                    "q_margin": float(info.q_margin),
                    "q_total_values_json": json.dumps(
                        getattr(info, "q_total_values", getattr(info, "q_values", [])),
                        ensure_ascii=False,
                    ),
                    "q_traffic_values_json": json.dumps(
                        getattr(info, "q_traffic_values", []),
                        ensure_ascii=False,
                    ),
                    "q_nox_values_json": json.dumps(
                        getattr(info, "q_nox_values", []),
                        ensure_ascii=False,
                    ),
                    "q_total_selected": float(getattr(info, "q_total_selected", getattr(info, "q_selected", 0.0))),
                    "q_traffic_selected": float(getattr(info, "q_traffic_selected", 0.0)),
                    "q_nox_selected": float(getattr(info, "q_nox_selected", 0.0)),
                    "valid_action_count": int(info.valid_action_count),
                    "action_mask_json": json.dumps([int(x) for x in info.action_mask], ensure_ascii=False),
                })

    def compute_step_link_emission_values(self, step_emission: Any, pollutant: str = "NOx") -> list[float]:
        if step_emission is None:
            return []
        by_edge = getattr(step_emission, "by_edge", {}) or {}
        values: list[float] = []
        pollutant = str(pollutant)
        for link_id in sorted(self._link_to_edges.keys()):
            total = 0.0
            for edge_id in self._link_to_edges.get(link_id, []):
                total += float(by_edge.get((edge_id, pollutant), 0.0) or 0.0)
            values.append(float(total))
        return values

    def log_step_emission(self, episode: int, step: int, step_emission: Any) -> None:
        if step_emission is None:
            return

        by_edge_raw = getattr(step_emission, "by_edge", {}) or {}
        if not isinstance(by_edge_raw, Mapping):
            return

        edge_values: dict[tuple[str, str], float] = {}
        for key, value in by_edge_raw.items():
            try:
                edge_id_raw, pollutant_raw = key
            except Exception:
                continue
            edge_id = str(edge_id_raw)
            pollutant = str(pollutant_raw)
            edge_values[(edge_id, pollutant)] = float(value or 0.0)

        sim_time_start = getattr(step_emission, "sim_time_start", "")
        sim_time_end = getattr(step_emission, "sim_time_end", "")
        step_n_records = getattr(step_emission, "step_n_records", getattr(step_emission, "n_records", ""))
        step_n_vehicles = getattr(step_emission, "step_n_vehicles", getattr(step_emission, "n_vehicles", ""))

        for edge_id, pollutant in sorted(edge_values.keys()):
            edge_value = edge_values[(edge_id, pollutant)]
            from_node, to_node = self._edge_nodes.get(edge_id, ("", ""))
            lane_ids = list(self._edge_to_lanes.get(edge_id, []))
            link_id = self._edge_to_link.get(edge_id, "")
            self._edge_emission_rows.append({
                "episode": int(episode),
                "step": int(step),
                "sim_time_start": sim_time_start,
                "sim_time_end": sim_time_end,
                "edge_id": edge_id,
                "from_node": from_node,
                "to_node": to_node,
                "link_id": link_id,
                "is_controlled_link": bool(link_id),
                "lane_ids_json": json.dumps(lane_ids, ensure_ascii=False),
                "n_lanes": int(len(lane_ids)),
                "pollutant": pollutant,
                "edge_emission_mg": float(edge_value),
                "step_n_records": step_n_records,
                "step_n_vehicles": step_n_vehicles,
            })

        pollutants = sorted({pollutant for (_edge_id, pollutant) in edge_values.keys()})
        for link_id in sorted(self._link_to_edges.keys()):
            edge_ids = list(self._link_to_edges.get(link_id, []))
            lane_ids = list(self._link_to_lanes.get(link_id, []))
            node_u, node_v = self._link_nodes.get(link_id, ("", ""))
            n_edges = int(len(edge_ids))
            for pollutant in pollutants:
                link_emission = 0.0
                for edge_id in edge_ids:
                    link_emission += float(edge_values.get((edge_id, pollutant), 0.0) or 0.0)
                self._link_emission_rows.append({
                    "episode": int(episode),
                    "step": int(step),
                    "sim_time_start": sim_time_start,
                    "sim_time_end": sim_time_end,
                    "link_id": link_id,
                    "node_u": node_u,
                    "node_v": node_v,
                    "edge_ids_json": json.dumps(edge_ids, ensure_ascii=False),
                    "n_edges": n_edges,
                    "lane_ids_json": json.dumps(lane_ids, ensure_ascii=False),
                    "n_lanes": int(len(lane_ids)),
                    "pollutant": pollutant,
                    "link_emission_mg": float(link_emission),
                    "mean_edge_emission_mg": float(link_emission / max(n_edges, 1)),
                    "step_n_records": step_n_records,
                    "step_n_vehicles": step_n_vehicles,
                })

    def clear_episode_emission(self) -> None:
        self._edge_emission_rows.clear()
        self._link_emission_rows.clear()

    def flush_episode_emission(self, episode: int) -> None:
        ep_dir = os.path.join(self.emission_root, f"episode_{episode:04d}")
        os.makedirs(ep_dir, exist_ok=True)
        _write_csv(os.path.join(ep_dir, "edge_emission_step.csv"), self._edge_emission_rows, self.EDGE_EMISSION_FIELDS)
        _write_csv(os.path.join(ep_dir, "link_emission_step.csv"), self._link_emission_rows, self.LINK_EMISSION_FIELDS)
        self.clear_episode_emission()

    def write_episode_emission_result(
        self,
        episode: int,
        emission_result: Any,
        *,
        save_step_df: bool = True,
        save_edge_step_raw: bool = False,
        save_lane_step: bool = True,
        save_vehicle_step: bool = False,
    ) -> None:
        if emission_result is None:
            return

        ep_dir = os.path.join(self.emission_root, f"episode_{episode:04d}")
        os.makedirs(ep_dir, exist_ok=True)

        def _write_df_attr(attr_name: str, filename: str, enabled: bool = True) -> None:
            if not enabled:
                return
            df = getattr(emission_result, attr_name, None)
            if df is None or not hasattr(df, "to_csv"):
                return
            df.to_csv(os.path.join(ep_dir, filename), index=False)

        _write_df_attr("vehicle_df", "vehicle_emission.csv", True)
        _write_df_attr("step_df", "decision_step_emission.csv", save_step_df)
        _write_df_attr("lane_step_df", "lane_emission_step.csv", save_lane_step)
        _write_df_attr("edge_step_df", "edge_emission_step_raw.csv", save_edge_step_raw)
        _write_df_attr("vehicle_step_df", "vehicle_emission_step.csv", save_vehicle_step)

    def flush_episode_physical(self, episode: int) -> None:
        has_lane = bool(self._lane_rows)
        has_phase = bool(self._phase_rows)
        has_q = bool(self._q_rows)
        if not (has_lane or has_phase or has_q):
            return

        ep_dir = os.path.join(self.physical_root, f"episode_{episode:04d}")
        os.makedirs(ep_dir, exist_ok=True)
        if has_lane:
            _write_csv(os.path.join(ep_dir, "lane_step.csv"), self._lane_rows, self.LANE_STEP_FIELDS)
        if has_phase:
            _write_csv(os.path.join(ep_dir, "phase_step.csv"), self._phase_rows, self.PHASE_STEP_FIELDS)
        if has_q:
            _write_csv(os.path.join(ep_dir, "q_step.csv"), self._q_rows, self.Q_STEP_FIELDS)

    def flush_episode_reward_step(self, episode: int) -> None:
        """Write one compact decision-reward CSV for the completed episode."""
        if not bool(getattr(self.cfg.log, "save_reward_step", True)):
            return
        _write_csv(
            os.path.join(self.reward_step_root, f"episode_{episode:04d}.csv"),
            self._reward_step_rows,
            self.REWARD_STEP_FIELDS,
        )

    def log_update(self, episode: int, global_step: int, update_stats: Sequence[Any], online_update_count: int, target_update_count: int) -> None:
        for s in update_stats:
            row = s.as_dict() if hasattr(s, "as_dict") else dict(s)
            row.update({
                "episode": int(episode),
                "global_step": int(global_step),
                "online_update_count": int(online_update_count),
                "target_update_count": int(target_update_count),
            })
            _append_csv(self.update_csv, row, self.UPDATE_FIELDS)

    def log_episode(self, episode: int, row: Mapping[str, Any]) -> None:
        payload = dict(row)
        payload.setdefault("episode", int(episode))
        _append_csv(self.episode_csv, payload, self.EPISODE_FIELDS)

    def _iter_parameter_rows(self, episode: int, global_step: int, bank: Any, network_type: str) -> list[dict]:
        rows: list[dict] = []
        named_params = list(bank.named_parameters())
        for name, p in named_params:
            parts = name.split(".")
            group_id = parts[1] if len(parts) > 1 and parts[0] == "networks" else ""
            module_name = ".".join(parts[2:-1]) if len(parts) > 3 else ""
            data = p.detach().float().cpu()
            grad = p.grad.detach().float().cpu() if p.grad is not None else None
            rows.append({
                "episode": int(episode),
                "global_step": int(global_step),
                "network_type": network_type,
                "group_id": group_id,
                "module_name": module_name,
                "parameter_name": name,
                "shape": str(tuple(data.shape)),
                "numel": int(data.numel()),
                "requires_grad": bool(p.requires_grad),
                "mean": float(data.mean().item()) if data.numel() else 0.0,
                "std": float(data.std(unbiased=False).item()) if data.numel() else 0.0,
                "min": float(data.min().item()) if data.numel() else 0.0,
                "max": float(data.max().item()) if data.numel() else 0.0,
                "abs_mean": float(data.abs().mean().item()) if data.numel() else 0.0,
                "l2_norm": float(data.norm().item()) if data.numel() else 0.0,
                "grad_l2_norm": float(grad.norm().item()) if grad is not None and grad.numel() else 0.0,
                "grad_abs_mean": float(grad.abs().mean().item()) if grad is not None and grad.numel() else 0.0,
            })
        return rows

    def log_parameter_stats(self, episode: int, global_step: int, online_bank: Any, target_bank: Optional[Any] = None) -> str:
        rows = self._iter_parameter_rows(episode, global_step, online_bank, "online")
        if target_bank is not None:
            rows.extend(self._iter_parameter_rows(episode, global_step, target_bank, "target"))
        path = os.path.join(self.params_dir, f"parameter_stats_ep{episode:04d}.csv")
        _write_csv(path, rows, self.PARAM_FIELDS)
        return path

    def log_best(
        self,
        episode: int,
        global_step: int,
        best_info: Mapping[str, Any],
        online_bank: Any,
        checkpoint_path: str,
    ) -> None:
        self.save_json(os.path.join(self.best_dir, "best_summary.json"), dict(best_info, checkpoint_path=checkpoint_path))
        rows = self._iter_parameter_rows(episode, global_step, online_bank, "online")
        _write_csv(os.path.join(self.best_dir, "best_parameter_stats.csv"), rows, self.PARAM_FIELDS)
        _append_csv(self.best_log_csv, {
            "episode": int(episode),
            "global_step": int(global_step),
            "best_metric": best_info.get("best_metric", self.cfg.log.best_metric),
            "old_best_score": best_info.get("old_best_score", ""),
            "new_best_score": best_info.get("best_score", ""),
            "improved": True,
            "checkpoint_path": checkpoint_path,
            "epsilon": best_info.get("epsilon", ""),
            "replay_size": best_info.get("replay_size", ""),
            "reward_mean": best_info.get("reward_mean", ""),
            "avg_delay_s": best_info.get("avg_delay_s", ""),
            "completion_rate": best_info.get("completion_rate", ""),
            "q_loss_mean": best_info.get("q_loss_mean", ""),
            "td_error_mean": best_info.get("td_error_mean", ""),
        }, self.BEST_FIELDS)

    @property
    def checkpoint_dir_path(self) -> str:
        return self.checkpoint_dir
