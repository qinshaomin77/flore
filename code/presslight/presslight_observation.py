# -*- coding: utf-8 -*-
"""Build group-specific PressLight observations in canonical structure order."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from presslight_config import PressLightConfig, SEGMENT_NAMES
from presslight_network import PressLightIntersectionSpec, PressLightNetworkSpec


@dataclass
class PressLightObservation:
    tl_id: str
    group_id: str
    state: np.ndarray
    action_mask: np.ndarray
    current_phase: int
    local_current_phase: int
    incoming_segments: np.ndarray
    incoming_total: np.ndarray
    outgoing_count: np.ndarray
    segment_mask: np.ndarray
    incoming_vehicle_ids: dict[str, list[str]] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


class PressLightObservationBuilder:
    """Read E2/lane snapshots at one decision boundary.

    Values are instantaneous counts.  They are never summed across the ten
    one-second SUMO steps in a decision interval.
    """

    def __init__(self, cfg: PressLightConfig, network_spec: PressLightNetworkSpec) -> None:
        self.cfg = cfg
        self.network_spec = network_spec

    @staticmethod
    def _vehicle_number(traci: Any, detector_id: str) -> float:
        try:
            return float(traci.lanearea.getLastStepVehicleNumber(detector_id))
        except Exception as exc:
            raise RuntimeError(f"Unable to read E2 detector {detector_id!r}") from exc

    @staticmethod
    def _vehicle_ids(traci: Any, detector_id: str) -> list[str]:
        try:
            return [str(x) for x in traci.lanearea.getLastStepVehicleIDs(detector_id)]
        except Exception as exc:
            raise RuntimeError(f"Unable to read vehicle IDs from E2 detector {detector_id!r}") from exc

    def build_one(self, env: Any, tl_id: str) -> PressLightObservation:
        spec: PressLightIntersectionSpec = self.network_spec.intersections[tl_id]
        traci = getattr(env, "_traci", None)
        if traci is None:
            raise RuntimeError("SUMO environment has no active TraCI connection")

        group_spec = self.network_spec.group_specs[spec.group_id]
        max_in = int(group_spec.n_incoming_lanes)
        max_out = int(group_spec.n_outgoing_lanes)
        max_phases = int(group_spec.n_phases)
        n_segments = len(SEGMENT_NAMES)
        incoming_segments = np.zeros((max_in, n_segments), dtype=np.float32)
        segment_mask = np.zeros((max_in, n_segments), dtype=bool)
        incoming_vehicle_ids: dict[str, list[str]] = {}

        shared_lanes = set(spec.shared_incoming_lanes)
        for lane_slot, lane_id in enumerate(spec.incoming_lanes):
            detector_ids = spec.incoming_segment_detectors.get(lane_id, {})
            lane_vehicle_ids: set[str] = set()
            for segment_slot, segment_name in enumerate(SEGMENT_NAMES):
                detector_id = str(detector_ids.get(segment_name, ""))
                if not detector_id:
                    continue
                incoming_segments[lane_slot, segment_slot] = self._vehicle_number(
                    traci, detector_id
                )
                segment_mask[lane_slot, segment_slot] = True
                if self.cfg.pressure.variant == "turn_ratio" and lane_id in shared_lanes:
                    lane_vehicle_ids.update(self._vehicle_ids(traci, detector_id))
            if lane_id in shared_lanes:
                incoming_vehicle_ids[lane_id] = sorted(lane_vehicle_ids)

        incoming_total = incoming_segments.sum(axis=1, dtype=np.float32)
        outgoing_count = np.zeros(max_out, dtype=np.float32)
        for lane_slot, lane_id in enumerate(spec.outgoing_lanes):
            try:
                outgoing_count[lane_slot] = float(
                    traci.lane.getLastStepVehicleNumber(lane_id)
                )
            except Exception as exc:
                raise RuntimeError(f"Unable to read outgoing lane {lane_id!r}") from exc

        timing = env.get_phase_timing_info(tl_id)
        local_current_phase = int(timing["current_phase"])
        expected_phases = int(spec.local_phase_count)
        dynamic_mask = np.asarray(env.compute_action_mask(tl_id), dtype=bool)
        mask_context = (
            f"tl_id={tl_id!r}, expected_phases={expected_phases}, "
            f"dynamic_mask.shape={dynamic_mask.shape}, max_phases={max_phases}, "
            f"local_current_phase={local_current_phase}"
        )
        if dynamic_mask.ndim != 1 or dynamic_mask.shape != (expected_phases,):
            raise RuntimeError(f"Invalid dynamic action-mask shape; {mask_context}")
        if not 0 <= local_current_phase < expected_phases:
            raise RuntimeError(
                f"Local current phase is outside the actual phase range; {mask_context}"
            )
        if sorted(spec.canonical_to_local_phase) != list(range(expected_phases)) or sorted(spec.local_to_canonical_phase) != list(range(expected_phases)):
            raise RuntimeError(f"Invalid canonical/local phase permutation; {mask_context}")
        current_phase = int(spec.local_to_canonical_phase[local_current_phase])
        phase_one_hot = np.zeros(max_phases, dtype=np.float32)
        phase_one_hot[current_phase] = 1.0

        action_mask = np.asarray(
            [dynamic_mask[spec.canonical_to_local_phase[c]] for c in range(max_phases)],
            dtype=bool,
        )
        if not action_mask.any():
            action_mask[current_phase] = True

        state = np.concatenate(
            [
                phase_one_hot,
                incoming_segments.reshape(-1),
                outgoing_count,
            ]
        ).astype(np.float32, copy=False)
        if state.shape != (group_spec.input_dim,):
            raise AssertionError(
                f"{tl_id}: expected state shape ({group_spec.input_dim},), got {state.shape}"
            )

        return PressLightObservation(
            tl_id=tl_id,
            group_id=spec.group_id,
            state=state,
            action_mask=action_mask,
            current_phase=current_phase,
            local_current_phase=local_current_phase,
            incoming_segments=incoming_segments,
            incoming_total=incoming_total,
            outgoing_count=outgoing_count,
            segment_mask=segment_mask,
            incoming_vehicle_ids=incoming_vehicle_ids,
            debug={
                "sim_time": float(timing.get("sim_time", 0.0)),
                "elapsed_green": float(timing.get("elapsed_green", 0.0)),
                "valid_action_count": int(action_mask.sum()),
            },
        )

    def build_all(self, env: Any) -> dict[str, PressLightObservation]:
        return {
            tl_id: self.build_one(env, tl_id)
            for tl_id in self.network_spec.intersection_ids
        }


def observation_step_rows(
    episode: int,
    step: int,
    observations: dict[str, PressLightObservation],
    network_spec: PressLightNetworkSpec,
) -> list[dict[str, Any]]:
    """Flatten PressLight state for optional diagnostic CSV output."""

    rows: list[dict[str, Any]] = []
    for tl_id, obs in observations.items():
        spec = network_spec.intersections[tl_id]
        group_spec = network_spec.group_specs[obs.group_id]
        for lane_slot, lane_id in enumerate(spec.incoming_lanes):
            rows.append(
                {
                    "episode": int(episode),
                    "step": int(step),
                    "sim_time": float(obs.debug.get("sim_time", 0.0)),
                    "tl_id": tl_id,
                    "group_id": obs.group_id,
                    "group_input_dim": int(group_spec.input_dim),
                    "group_action_dim": int(group_spec.action_dim),
                    "lane_slot": int(lane_slot),
                    "lane_id": lane_id,
                    "near_count": float(obs.incoming_segments[lane_slot, 0]),
                    "middle_count": float(obs.incoming_segments[lane_slot, 1]),
                    "far_count": float(obs.incoming_segments[lane_slot, 2]),
                    "incoming_total": float(obs.incoming_total[lane_slot]),
                    "current_phase": int(obs.current_phase),
                    "canonical_current_phase": int(obs.current_phase),
                    "local_current_phase": int(obs.local_current_phase),
                    "action_mask": "|".join("1" if x else "0" for x in obs.action_mask),
                }
            )
        for lane_slot, lane_id in enumerate(spec.outgoing_lanes):
            rows.append(
                {
                    "episode": int(episode),
                    "step": int(step),
                    "sim_time": float(obs.debug.get("sim_time", 0.0)),
                    "tl_id": tl_id,
                    "group_id": obs.group_id,
                    "group_input_dim": int(group_spec.input_dim),
                    "group_action_dim": int(group_spec.action_dim),
                    "lane_slot": int(lane_slot),
                    "lane_id": lane_id,
                    "outgoing_count": float(obs.outgoing_count[lane_slot]),
                    "current_phase": int(obs.current_phase),
                    "canonical_current_phase": int(obs.current_phase),
                    "local_current_phase": int(obs.local_current_phase),
                    "action_mask": "|".join("1" if x else "0" for x in obs.action_mask),
                }
            )
    return rows
