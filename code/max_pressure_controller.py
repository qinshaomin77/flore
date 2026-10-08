"""E2-based movement Max-Pressure controller (no learning dependencies)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, Mapping
import math
import numpy as np

from network_parser import (Movement, build_maxpressure_movements,
                            build_phase_movement_map, build_turn_lookup)


@dataclass
class MovementPressure:
    movement_id: str
    upstream_halting: float
    downstream_halting_by_lane: Dict[str, float]
    downstream_weights: Dict[str, float]
    effective_downstream_halting: float
    pressure: float


@dataclass
class MaxPressureActionInfo:
    tl_id: str
    current_phase: int
    selected_phase: int
    phase_pressures: list
    masked_phase_pressures: list
    selected_pressure: float
    action_mask: list
    tie_count: int
    tie_kept_current: bool


class MaxPressureController:
    """Select legal protected-green phases using instantaneous E2 halt counts."""
    def __init__(self, net_info, detector_map: Mapping, config):
        self.net_info, self.detector_map, self.config = net_info, detector_map, config
        self.movements: Dict[str, Movement] = build_maxpressure_movements(net_info)
        self.phase_movements = build_phase_movement_map(net_info, self.movements)
        self.turn_lookup = build_turn_lookup(net_info, self.movements)
        self.by_tls, self.by_lane = defaultdict(list), defaultdict(list)
        for movement in self.movements.values():
            self.by_tls[movement.tl_id].append(movement)
            self.by_lane[movement.from_lane].append(movement)
        self.reset()

    def reset(self):
        self.last_diagnostics = {}

    def collect_e2_snapshot(self, env):
        ids = set()
        for role in ("upstream", "downstream"):
            for lane_map in self.detector_map[role].values(): ids.update(lane_map.values())
        return env.get_lanearea_snapshot(sorted(ids))

    def classify_shared_lane_queues(self, env, snapshots):
        queues, rows = defaultdict(float), []
        threshold = float(self.config.halting_speed_threshold_mps)
        for lane, movements in self.by_lane.items():
            if len(movements) <= 1: continue
            tl_id = movements[0].tl_id
            detector_id = self.detector_map["upstream"][tl_id][lane]
            unknown = 0
            direction_counts = defaultdict(int)
            for veh_id in snapshots[detector_id]["vehicle_ids"]:
                ctx = env.get_vehicle_route_context(veh_id)
                if ctx["speed_mps"] >= threshold: continue
                route, idx = ctx["route"], ctx["route_index"]
                if (ctx["lane_id"] != lane or idx < 0 or idx >= len(route) - 1
                        or route[idx] != ctx["road_id"]):
                    unknown += 1; continue
                movement_id = self.turn_lookup.get((lane, route[idx + 1]))
                if movement_id is None:
                    unknown += 1; continue
                queues[movement_id] += 1.0
                direction_counts[self.movements[movement_id].direction] += 1
            reported = int(snapshots[detector_id]["halting_number"])
            classified = int(sum(direction_counts.values()))
            rows.append({"tl_id": tl_id, "from_lane": lane, "detector_id": detector_id,
                         "e2_vehicle_count": snapshots[detector_id]["vehicle_number"],
                         "e2_halting_count": reported,
                         "straight_halting": direction_counts["s"], "left_halting": direction_counts["l"],
                         "right_halting": direction_counts["r"], "uturn_halting": direction_counts["t"],
                         "unknown_halting": unknown,
                         "classification_difference": reported - classified - unknown})
        return queues, rows

    def build_movement_queues(self, env, snapshots):
        shared, rows = self.classify_shared_lane_queues(env, snapshots)
        queues = {}
        for movement in self.movements.values():
            if len(self.by_lane[movement.from_lane]) == 1:
                det = self.detector_map["upstream"][movement.tl_id][movement.from_lane]
                queues[movement.movement_id] = float(snapshots[det]["halting_number"])
            else:
                queues[movement.movement_id] = float(shared[movement.movement_id])
        return queues, rows

    def compute_movement_pressures(self, queues, snapshots):
        result = {}
        for movement in self.movements.values():
            downstream = {lane: float(snapshots[self.detector_map["downstream"][movement.tl_id][lane]]["halting_number"])
                          for lane in movement.to_lanes}
            weight = 1.0 / len(downstream)
            weights = {lane: weight for lane in downstream}
            effective = sum(weights[lane] * value for lane, value in downstream.items())
            upstream = float(queues[movement.movement_id])
            result[movement.movement_id] = MovementPressure(
                movement.movement_id, upstream, downstream, weights, effective, upstream - effective)
        return result

    def compute_phase_pressures(self, movement_pressures):
        return {tl_id: [sum(movement_pressures[mid].pressure for mid in mids)
                        for mids in self.phase_movements[tl_id]]
                for tl_id in self.net_info.intersection_ids}

    def select_actions(self, env, raw_obs, phase_pressures):
        actions, infos = {}, {}
        for tl_id, values in phase_pressures.items():
            mask = np.asarray(env.compute_action_mask(tl_id), dtype=bool)
            if not mask.any(): raise RuntimeError(f"No legal Max-Pressure phase for {tl_id}")
            masked = np.asarray(values, dtype=float); masked[~mask] = -np.inf
            best = float(np.max(masked)); candidates = np.flatnonzero(masked == best).tolist()
            current = int(raw_obs[tl_id].current_phase)
            selected = current if current in candidates else int(min(candidates))
            actions[tl_id] = selected
            infos[tl_id] = MaxPressureActionInfo(tl_id, current, selected, list(map(float, values)),
                list(map(float, masked)), best, mask.tolist(), len(candidates), current in candidates)
        return actions, infos

    def act(self, env, raw_obs, episode=0, step=0):
        snapshots = self.collect_e2_snapshot(env)
        queues, classifications = self.build_movement_queues(env, snapshots)
        movement_pressures = self.compute_movement_pressures(queues, snapshots)
        phase_pressures = self.compute_phase_pressures(movement_pressures)
        actions, infos = self.select_actions(env, raw_obs, phase_pressures)
        diagnostics = {"snapshots": snapshots, "movement_pressures": movement_pressures,
                       "shared_lane_classifications": classifications,
                       "metadata_note": "Reward and NOx components are diagnostics only and do not affect Max-Pressure action selection."}
        self.last_diagnostics = diagnostics
        return actions, infos, diagnostics
