"""MaxPressure-TW (truck-weighted E2 movement pressure) controller.

MaxPressure-TW is a fixed, rule-based baseline. Its only difference from ordinary
Max-Pressure is that a halted truck contributes twice the queue pressure of a
halted sedan. Reward, emissions, NOx pressure, and RL state are never read for
action selection.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Dict, Mapping, MutableMapping, Optional

from max_pressure_controller import MaxPressureController


SEDAN_PRIORITY_WEIGHT = 1.0
TRUCK_PRIORITY_WEIGHT = 2.0


@dataclass
class TruckWeightedMovementPressure:
    movement_id: str
    upstream_halting_raw: float
    upstream_truck_halting: float
    upstream_halting_weighted: float
    downstream_halting_raw_by_lane: Dict[str, float]
    downstream_truck_halting_by_lane: Dict[str, float]
    downstream_halting_weighted_by_lane: Dict[str, float]
    downstream_weights: Dict[str, float]
    effective_downstream_halting_weighted: float
    pressure: float

    # Compatibility aliases allow the existing MP logger to consume weighted
    # values while the raw and truck-only values remain available for audit.
    @property
    def upstream_halting(self) -> float:
        return self.upstream_halting_weighted

    @property
    def downstream_halting_by_lane(self) -> Dict[str, float]:
        return self.downstream_halting_weighted_by_lane

    @property
    def effective_downstream_halting(self) -> float:
        return self.effective_downstream_halting_weighted


class TruckWeightedMaxPressureController(MaxPressureController):
    """Max-Pressure with fixed sedan=1 and truck=2 queue contributions."""

    def __init__(
        self,
        net_info,
        detector_map: Mapping,
        config,
        truck_priority_weight: float = TRUCK_PRIORITY_WEIGHT,
    ):
        super().__init__(net_info, detector_map, config)
        weight = float(truck_priority_weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("truck_priority_weight must be finite and > 0")
        self.sedan_priority_weight = SEDAN_PRIORITY_WEIGHT
        self.truck_priority_weight = weight

    def reset(self):
        super().reset()
        self._detector_queue_details: Dict[str, Dict[str, float]] = {}
        self._upstream_queue_details: Dict[str, Dict[str, float]] = {}

    def _vehicle_context(
        self,
        env,
        veh_id: str,
        cache: MutableMapping[str, dict],
    ) -> dict:
        cached = cache.get(veh_id)
        if cached is not None:
            return cached
        context = dict(env.get_vehicle_route_context(veh_id))
        context["vehicle_type"] = env.get_vehicle_type(veh_id)
        cache[veh_id] = context
        return context

    def count_halting_trucks(
        self,
        env,
        snapshot: Mapping,
        expected_lane: Optional[str] = None,
        vehicle_context_cache: Optional[MutableMapping[str, dict]] = None,
    ) -> int:
        """Count halted trucks represented by one E2 detector snapshot."""
        cache = vehicle_context_cache if vehicle_context_cache is not None else {}
        threshold = float(self.config.halting_speed_threshold_mps)
        truck_halting = 0
        for veh_id in snapshot["vehicle_ids"]:
            context = self._vehicle_context(env, str(veh_id), cache)
            if float(context["speed_mps"]) >= threshold:
                continue
            if expected_lane is not None and context["lane_id"] != expected_lane:
                continue
            if context["vehicle_type"] == "truck":
                truck_halting += 1
        return truck_halting

    def _prepare_detector_queue_details(
        self,
        env,
        snapshots: Mapping[str, Mapping],
        vehicle_context_cache: MutableMapping[str, dict],
    ) -> None:
        detector_lanes: Dict[str, str] = {}
        for role in ("upstream", "downstream"):
            for lane_map in self.detector_map[role].values():
                for lane, detector_id in lane_map.items():
                    detector_lanes[str(detector_id)] = str(lane)

        details: Dict[str, Dict[str, float]] = {}
        for detector_id, snapshot in snapshots.items():
            raw = float(snapshot["halting_number"])
            trucks = float(
                self.count_halting_trucks(
                    env,
                    snapshot,
                    expected_lane=detector_lanes.get(str(detector_id)),
                    vehicle_context_cache=vehicle_context_cache,
                )
            )
            details[str(detector_id)] = {
                "raw": raw,
                "trucks": trucks,
                "weighted": raw
                + (self.truck_priority_weight - self.sedan_priority_weight)
                * trucks,
            }
        self._detector_queue_details = details

    def classify_shared_lane_queues(
        self,
        env,
        snapshots,
        vehicle_context_cache: Optional[MutableMapping[str, dict]] = None,
    ):
        queues, rows = defaultdict(float), []
        cache = vehicle_context_cache if vehicle_context_cache is not None else {}
        threshold = float(self.config.halting_speed_threshold_mps)
        upstream_details: Dict[str, Dict[str, float]] = {}

        for lane, movements in self.by_lane.items():
            if len(movements) <= 1:
                continue
            tl_id = movements[0].tl_id
            detector_id = self.detector_map["upstream"][tl_id][lane]
            unknown = 0
            unknown_trucks = 0
            raw_by_direction = defaultdict(int)
            trucks_by_direction = defaultdict(int)
            weighted_by_direction = defaultdict(float)
            raw_by_movement = defaultdict(int)
            trucks_by_movement = defaultdict(int)
            weighted_by_movement = defaultdict(float)

            for veh_id in snapshots[detector_id]["vehicle_ids"]:
                context = self._vehicle_context(env, str(veh_id), cache)
                if float(context["speed_mps"]) >= threshold:
                    continue
                is_truck = context["vehicle_type"] == "truck"
                route, idx = context["route"], int(context["route_index"])
                if (
                    context["lane_id"] != lane
                    or idx < 0
                    or idx >= len(route) - 1
                    or route[idx] != context["road_id"]
                ):
                    unknown += 1
                    unknown_trucks += int(is_truck)
                    continue
                movement_id = self.turn_lookup.get((lane, route[idx + 1]))
                if movement_id is None:
                    unknown += 1
                    unknown_trucks += int(is_truck)
                    continue

                direction = self.movements[movement_id].direction
                vehicle_weight = (
                    self.truck_priority_weight
                    if is_truck
                    else self.sedan_priority_weight
                )
                queues[movement_id] += vehicle_weight
                raw_by_direction[direction] += 1
                trucks_by_direction[direction] += int(is_truck)
                weighted_by_direction[direction] += vehicle_weight
                raw_by_movement[movement_id] += 1
                trucks_by_movement[movement_id] += int(is_truck)
                weighted_by_movement[movement_id] += vehicle_weight
                upstream_details[movement_id] = {
                    "raw": float(raw_by_movement[movement_id]),
                    "trucks": float(trucks_by_movement[movement_id]),
                    "weighted": float(weighted_by_movement[movement_id]),
                }

            reported = int(snapshots[detector_id]["halting_number"])
            classified = int(sum(raw_by_direction.values()))
            row = {
                "tl_id": tl_id,
                "from_lane": lane,
                "detector_id": detector_id,
                "e2_vehicle_count": snapshots[detector_id]["vehicle_number"],
                "e2_halting_count": reported,
                "unknown_halting": unknown,
                "unknown_truck_halting": unknown_trucks,
                "classification_difference": reported - classified - unknown,
            }
            for direction, name in (
                ("s", "straight"),
                ("l", "left"),
                ("r", "right"),
                ("t", "uturn"),
            ):
                row[f"{name}_halting_raw"] = raw_by_direction[direction]
                row[f"{name}_truck_halting"] = trucks_by_direction[direction]
                row[f"{name}_halting_weighted"] = weighted_by_direction[direction]
            rows.append(row)

        self._upstream_queue_details.update(upstream_details)
        return queues, rows

    def build_movement_queues(
        self,
        env,
        snapshots,
        vehicle_context_cache: Optional[MutableMapping[str, dict]] = None,
    ):
        cache = vehicle_context_cache if vehicle_context_cache is not None else {}
        self._prepare_detector_queue_details(env, snapshots, cache)
        self._upstream_queue_details = {}
        shared, rows = self.classify_shared_lane_queues(
            env,
            snapshots,
            vehicle_context_cache=cache,
        )
        queues = {}
        for movement in self.movements.values():
            movement_id = movement.movement_id
            if len(self.by_lane[movement.from_lane]) == 1:
                detector_id = self.detector_map["upstream"][movement.tl_id][
                    movement.from_lane
                ]
                detail = dict(self._detector_queue_details[str(detector_id)])
                queues[movement_id] = detail["weighted"]
                self._upstream_queue_details[movement_id] = detail
            else:
                queues[movement_id] = float(shared[movement_id])
                self._upstream_queue_details.setdefault(
                    movement_id,
                    {"raw": 0.0, "trucks": 0.0, "weighted": 0.0},
                )
        return queues, rows

    def compute_movement_pressures(self, queues, snapshots):
        result = {}
        for movement in self.movements.values():
            raw_downstream = {}
            truck_downstream = {}
            weighted_downstream = {}
            for lane in movement.to_lanes:
                detector_id = self.detector_map["downstream"][movement.tl_id][lane]
                detail = self._detector_queue_details[str(detector_id)]
                raw_downstream[lane] = detail["raw"]
                truck_downstream[lane] = detail["trucks"]
                weighted_downstream[lane] = detail["weighted"]

            downstream_weight = 1.0 / len(weighted_downstream)
            weights = {lane: downstream_weight for lane in weighted_downstream}
            effective = sum(
                weights[lane] * value
                for lane, value in weighted_downstream.items()
            )
            upstream = self._upstream_queue_details[movement.movement_id]
            result[movement.movement_id] = TruckWeightedMovementPressure(
                movement_id=movement.movement_id,
                upstream_halting_raw=upstream["raw"],
                upstream_truck_halting=upstream["trucks"],
                upstream_halting_weighted=upstream["weighted"],
                downstream_halting_raw_by_lane=raw_downstream,
                downstream_truck_halting_by_lane=truck_downstream,
                downstream_halting_weighted_by_lane=weighted_downstream,
                downstream_weights=weights,
                effective_downstream_halting_weighted=effective,
                pressure=upstream["weighted"] - effective,
            )
        return result

    def act(self, env, raw_obs, episode=0, step=0):
        snapshots = self.collect_e2_snapshot(env)
        vehicle_context_cache: Dict[str, dict] = {}
        queues, classifications = self.build_movement_queues(
            env,
            snapshots,
            vehicle_context_cache=vehicle_context_cache,
        )
        movement_pressures = self.compute_movement_pressures(queues, snapshots)
        phase_pressures = self.compute_phase_pressures(movement_pressures)
        actions, infos = self.select_actions(env, raw_obs, phase_pressures)
        diagnostics = {
            "snapshots": snapshots,
            "movement_pressures": movement_pressures,
            "shared_lane_classifications": classifications,
            "truck_priority_weight": self.truck_priority_weight,
            "sedan_priority_weight": self.sedan_priority_weight,
            "metadata_note": (
                "MaxPressure-TW uses only weighted traffic queue "
                "pressure for action selection. NOx, NOx pressure, risk states, "
                "and reward components are diagnostics/evaluation only."
            ),
        }
        self.last_diagnostics = diagnostics
        return actions, infos, diagnostics
