# -*- coding: utf-8 -*-
"""
evaluation_window_logger.py
===========================

Windowed evaluation collector for the FLORE SUMO project.

The collector records:

* network-level 10-second increments;
* E2-detector-level 10-second increments;
* 300-second aggregates;
* optional vehicle-level 10-second increments;
* E2-detector/vehicle 300-second aggregates;
* conservation and additivity quality-control rows.

The class intentionally does not change state, reward, actions, or model
parameters.  ``SumoEnv`` calls it through an explicit post-simulation-step
hook shared by RL, Max-Pressure, and Gap-actuated control.
"""

from __future__ import annotations

import csv
import math
import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional


EPS = 1.0e-9
METRIC_NAMES = ("time_loss_s", "distance_m", "waiting_time_s", "waiting_event_count")
TYPE_NAMES = ("truck", "sedan", "unknown")


def _new_type_float_dict() -> dict[str, float]:
    return {name: 0.0 for name in TYPE_NAMES}


def _new_type_set_dict() -> dict[str, set[str]]:
    return {name: set() for name in TYPE_NAMES}


def _new_metric_dict() -> dict[str, float]:
    return {name: 0.0 for name in METRIC_NAMES}


def _new_nested_metric_dict() -> dict[str, dict[str, float]]:
    return {name: _new_type_float_dict() for name in METRIC_NAMES}


@dataclass(frozen=True)
class WindowEvaluationConfig:
    """Evaluation-only time windows and output switches."""

    decision_interval_s: int = 10
    aggregate_window_s: int = 300
    analysis_start_s: int = 300
    analysis_end_s: int = 3300
    save_network_10s: bool = True
    save_e2_detector_10s: bool = True
    save_network_300s: bool = True
    save_e2_detector_300s: bool = True
    save_e2_vehicle_300s: bool = True
    save_vehicle_10s: bool = False
    save_decision_actions: bool = False
    exit_e1_ids: tuple[str, ...] = ()
    e2_id_prefix: str = "e2_all_"
    exit_e1_id_prefix: str = "e1_exit_"
    expected_e2_count: int = 504
    expected_exit_e1_count: int = 72
    pollutant: str = "NOx"
    output_unit: str = "mg"

    def validate(self) -> None:
        if self.decision_interval_s <= 0:
            raise ValueError("decision_interval_s must be > 0")
        if self.aggregate_window_s <= 0:
            raise ValueError("aggregate_window_s must be > 0")
        if self.aggregate_window_s % self.decision_interval_s != 0:
            raise ValueError(
                "aggregate_window_s must be divisible by decision_interval_s"
            )
        if self.analysis_start_s < 0:
            raise ValueError("analysis_start_s must be >= 0")
        if self.analysis_end_s <= self.analysis_start_s:
            raise ValueError("analysis_end_s must be greater than analysis_start_s")
        if self.analysis_start_s % self.aggregate_window_s != 0:
            raise ValueError(
                "analysis_start_s must align with aggregate_window_s"
            )
        if self.analysis_end_s % self.aggregate_window_s != 0:
            raise ValueError(
                "analysis_end_s must align with aggregate_window_s"
            )
        if str(self.output_unit).strip().lower() != "mg":
            raise ValueError("This logger currently writes NOx mass in mg")


@dataclass(frozen=True)
class E2DetectorMeta:
    detector_id: str
    lane_id: str
    start_pos_m: float
    length_m: float
    end_pos_m: float


@dataclass
class NetworkAccumulator:
    start_active_ids: set[str] = field(default_factory=set)
    end_active_ids: set[str] = field(default_factory=set)
    present_ids: set[str] = field(default_factory=set)
    entered_ids: set[str] = field(default_factory=set)
    arrived_ids: set[str] = field(default_factory=set)
    exit_e1_ids: set[str] = field(default_factory=set)
    active_vehicle_seconds: float = 0.0
    active_seconds_by_type: dict[str, float] = field(default_factory=_new_type_float_dict)
    metrics: dict[str, float] = field(default_factory=_new_metric_dict)
    metrics_by_type: dict[str, dict[str, float]] = field(
        default_factory=_new_nested_metric_dict
    )
    network_nox_mg: float = 0.0
    network_nox_available: bool = False
    overlap_e2_vehicle_seconds: float = 0.0
    negative_time_loss_delta_count: int = 0
    negative_distance_delta_count: int = 0
    teleport_start_count: int = 0
    teleport_end_count: int = 0
    terminal_time_loss_correction_s: float = 0.0
    terminal_distance_correction_m: float = 0.0
    terminal_waiting_time_correction_s: float = 0.0
    terminal_waiting_event_correction_count: float = 0.0


@dataclass
class E2Accumulator:
    unique_ids: set[str] = field(default_factory=set)
    unique_ids_by_type: dict[str, set[str]] = field(default_factory=_new_type_set_dict)
    vehicle_seconds: float = 0.0
    vehicle_seconds_by_type: dict[str, float] = field(
        default_factory=_new_type_float_dict
    )
    nox_mg: float = 0.0
    nox_by_type: dict[str, float] = field(default_factory=_new_type_float_dict)
    metrics: dict[str, float] = field(default_factory=_new_metric_dict)
    metrics_by_type: dict[str, dict[str, float]] = field(
        default_factory=_new_nested_metric_dict
    )


@dataclass
class VehicleAccumulator:
    vehicle_id: str
    vehicle_type: str = "unknown"
    present: bool = False
    entered: bool = False
    exited: bool = False
    metrics: dict[str, float] = field(default_factory=_new_metric_dict)
    e2_metrics: dict[str, float] = field(default_factory=_new_metric_dict)
    e2_nox_mg: float = 0.0


class WindowEvaluationLogger:
    """Collect and write one-second, 10-second, and 300-second metrics."""

    OUTPUT_FILES = (
        "network_10s.csv",
        "e2_detector_10s.csv",
        "network_300s.csv",
        "e2_detector_300s.csv",
        "e2_vehicle_300s.csv",
        "vehicle_10s.csv",
        "quality_control.csv",
    )

    def __init__(
        self,
        *,
        config: WindowEvaluationConfig,
        output_root: str | os.PathLike[str],
        emission_lookup: Any,
        controller: str,
        case_name: str,
        reward_mode: str = "",
        nominal_by_interval: Optional[Mapping[int, Mapping[str, float]]] = None,
    ) -> None:
        config.validate()
        if emission_lookup is None:
            raise ValueError("emission_lookup is required for E2 NOx accounting")

        self.config = config
        self.output_root = Path(output_root).expanduser().resolve()
        self.emission_lookup = emission_lookup
        self.controller = str(controller)
        self.case_name = str(case_name)
        self.reward_mode = str(reward_mode)
        self.nominal_by_interval = {
            int(key): dict(value)
            for key, value in (nominal_by_interval or {}).items()
        }

        self.episode = -1
        self.sumo_seed = -1
        self.episode_dir = self.output_root
        self.traci = None
        self.tc = None
        self.e2_meta: dict[str, E2DetectorMeta] = {}
        self.exit_e1_ids: tuple[str, ...] = ()
        self._attached_env = None
        self._original_record_second = None

        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        self.last_sim_time_s: float = 0.0
        self.vehicle_type_registry: dict[str, str] = {}
        self.vehicle_raw_type_registry: dict[str, str] = {}
        self.previous_time_loss: dict[str, float] = {}
        self.previous_distance: dict[str, float] = {}
        self.previous_is_waiting: dict[str, bool] = {}
        self.previous_active_ids: set[str] = set()
        self.exit_seen_vehicle_ids: set[str] = set()
        self.arrival_sample_time: dict[str, float] = {}
        self.vehicle_observed: dict[str, dict[str, float]] = defaultdict(
            _new_metric_dict
        )

        self.network_intervals: dict[int, NetworkAccumulator] = {}
        self.e2_intervals: dict[tuple[int, str], E2Accumulator] = {}
        self.vehicle_intervals: dict[tuple[int, str], VehicleAccumulator] = {}
        self.e2_vehicle_windows: dict[
            tuple[int, str, str], E2Accumulator
        ] = {}

        self.network_rows_10s: list[dict[str, Any]] = []
        self.e2_rows_10s: list[dict[str, Any]] = []
        self.network_rows_300s: list[dict[str, Any]] = []
        self.e2_rows_300s: list[dict[str, Any]] = []
        self.e2_vehicle_rows_300s: list[dict[str, Any]] = []
        self.vehicle_rows_10s: list[dict[str, Any]] = []
        self.quality_control_rows: list[dict[str, Any]] = []
        self.decision_action_rows: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}

    def record_decision_actions(
        self,
        *,
        decision_step: int,
        raw_observations: Mapping[str, Any],
        actions: Mapping[str, int],
    ) -> None:
        """Record pre-action signal state without affecting evaluation."""
        if not self.config.save_decision_actions:
            return
        for tl_id, action in sorted(actions.items()):
            raw = raw_observations[tl_id]
            current_phase = int(raw.current_phase)
            target_phase = int(action)
            self.decision_action_rows.append(
                {
                    "episode": self.episode,
                    "sumo_seed": self.sumo_seed,
                    "case_name": self.case_name,
                    "controller": self.controller,
                    "reward_mode": self.reward_mode,
                    "decision_step": int(decision_step),
                    "simulation_time_s": float(raw.sim_time),
                    "tls_id": str(tl_id),
                    "current_green_phase": current_phase,
                    "selected_action": target_phase,
                    "target_green_phase": target_phase,
                    "is_phase_switch": int(target_phase != current_phase),
                    "in_yellow": int(bool(raw.in_yellow)),
                    "pending_green_phase": int(raw.pending_phase),
                    "sumo_green_phase": int(raw.sumo_green_phase),
                    "sumo_raw_phase": int(raw.sumo_raw_phase),
                    "sumo_state": str(raw.sumo_state),
                }
            )

    # ------------------------------------------------------------------
    # Episode and environment integration
    # ------------------------------------------------------------------

    def start_episode(
        self,
        *,
        traci: Any,
        tc: Any,
        episode: int,
        sumo_seed: int,
    ) -> None:
        self._reset_episode_state()
        self.traci = traci
        self.tc = tc
        self.episode = int(episode)
        self.sumo_seed = int(sumo_seed)
        self.episode_dir = (
            self.output_root
            / "windows_episode"
            / f"episode_{self.episode:04d}"
        )
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        all_e2_ids = sorted(str(x) for x in traci.lanearea.getIDList())
        selected_e2_ids = [
            detector_id
            for detector_id in all_e2_ids
            if detector_id.startswith(self.config.e2_id_prefix)
        ]
        if (
            self.config.expected_e2_count > 0
            and len(selected_e2_ids) != self.config.expected_e2_count
        ):
            raise RuntimeError(
                "Evaluation E2 detector mismatch: "
                f"prefix={self.config.e2_id_prefix!r}, "
                f"expected={self.config.expected_e2_count}, "
                f"loaded={len(selected_e2_ids)}. "
                "Check the runtime SUMO additional-files configuration."
            )

        self.e2_meta = {}
        for detector_id in selected_e2_ids:
            lane_id = str(traci.lanearea.getLaneID(detector_id))
            start_pos = float(traci.lanearea.getPosition(detector_id))
            length = float(traci.lanearea.getLength(detector_id))
            self.e2_meta[detector_id] = E2DetectorMeta(
                detector_id=detector_id,
                lane_id=lane_id,
                start_pos_m=start_pos,
                length_m=length,
                end_pos_m=start_pos + length,
            )
            try:
                traci.lanearea.subscribe(
                    detector_id,
                    [tc.LAST_STEP_VEHICLE_ID_LIST],
                )
            except Exception:
                pass

        all_e1_ids = sorted(str(x) for x in traci.inductionloop.getIDList())
        if self.config.exit_e1_ids:
            selected_exit_ids = list(dict.fromkeys(self.config.exit_e1_ids))
        else:
            selected_exit_ids = [
                detector_id
                for detector_id in all_e1_ids
                if detector_id.startswith(self.config.exit_e1_id_prefix)
            ]
        missing_exit_ids = sorted(set(selected_exit_ids) - set(all_e1_ids))
        if missing_exit_ids:
            raise RuntimeError(
                "Configured exit E1 detectors are not loaded: "
                + ", ".join(missing_exit_ids[:10])
            )
        if (
            self.config.expected_exit_e1_count > 0
            and len(selected_exit_ids) != self.config.expected_exit_e1_count
        ):
            raise RuntimeError(
                "Evaluation exit E1 detector mismatch: "
                f"prefix={self.config.exit_e1_id_prefix!r}, "
                f"expected={self.config.expected_exit_e1_count}, "
                f"loaded={len(selected_exit_ids)}."
            )
        self.exit_e1_ids = tuple(selected_exit_ids)
        for detector_id in self.exit_e1_ids:
            try:
                traci.inductionloop.subscribe(
                    detector_id,
                    [tc.LAST_STEP_VEHICLE_ID_LIST],
                )
            except Exception:
                pass

    def install_on_env(self, env: Any) -> None:
        """Attach to the existing per-second method of a started ``SumoEnv``."""
        if self.traci is None or self.tc is None:
            raise RuntimeError("Call start_episode() after env.start() first")
        if self._attached_env is not None:
            raise RuntimeError("WindowEvaluationLogger is already attached")
        if not hasattr(env, "_record_emissions_for_second"):
            raise AttributeError(
                "SumoEnv lacks _record_emissions_for_second; add an explicit "
                "collect_second hook after each simulationStep instead"
            )

        self._attached_env = env
        self._original_record_second = env._record_emissions_for_second

        # Replace the existing context subscription with a superset that
        # includes timeLoss and current waiting time.
        anchor = getattr(env, "_context_anchor", None)
        if anchor:
            var_ids = [
                self.tc.VAR_TYPE,
                self.tc.VAR_SPEED,
                self.tc.VAR_ACCELERATION,
                self.tc.VAR_LANE_ID,
                self.tc.VAR_ROAD_ID,
                self.tc.VAR_DISTANCE,
                self.tc.VAR_TIMELOSS,
                self.tc.VAR_WAITING_TIME,
            ]
            self.traci.lane.subscribeContext(
                anchor,
                self.tc.CMD_GET_VEHICLE_VARIABLE,
                1_000_000.0,
                var_ids,
            )

        original = self._original_record_second

        def _record_and_collect(decision_step: int) -> None:
            original(int(decision_step))
            context: Mapping[str, Mapping[int, Any]] = {}
            context_anchor = getattr(env, "_context_anchor", None)
            if context_anchor:
                try:
                    context = (
                        self.traci.lane.getContextSubscriptionResults(
                            context_anchor
                        )
                        or {}
                    )
                except Exception:
                    context = {}
            self.collect_second(
                sim_time=float(getattr(env, "_sim_time", 0.0)),
                decision_step=int(decision_step),
                vehicle_context=context,
            )

        env._record_emissions_for_second = _record_and_collect

    def detach_from_env(self) -> None:
        if self._attached_env is not None and self._original_record_second is not None:
            self._attached_env._record_emissions_for_second = (
                self._original_record_second
            )
        self._attached_env = None
        self._original_record_second = None

    # ------------------------------------------------------------------
    # Per-second collection
    # ------------------------------------------------------------------

    def _interval_index(self, sample_time: float) -> int:
        return max(
            0,
            int(math.floor((float(sample_time) + EPS) / self.config.decision_interval_s)),
        )

    def _window_index_from_interval(self, interval_index: int) -> int:
        intervals_per_window = (
            self.config.aggregate_window_s // self.config.decision_interval_s
        )
        return int(interval_index // intervals_per_window)

    def _network_acc(self, interval_index: int) -> NetworkAccumulator:
        if interval_index not in self.network_intervals:
            self.network_intervals[interval_index] = NetworkAccumulator(
                start_active_ids=set(self.previous_active_ids)
            )
        return self.network_intervals[interval_index]

    def _e2_acc(self, interval_index: int, detector_id: str) -> E2Accumulator:
        key = (int(interval_index), str(detector_id))
        if key not in self.e2_intervals:
            self.e2_intervals[key] = E2Accumulator()
        return self.e2_intervals[key]

    def _vehicle_acc(
        self,
        interval_index: int,
        vehicle_id: str,
    ) -> VehicleAccumulator:
        key = (int(interval_index), str(vehicle_id))
        if key not in self.vehicle_intervals:
            self.vehicle_intervals[key] = VehicleAccumulator(
                vehicle_id=str(vehicle_id),
                vehicle_type=self.vehicle_type_registry.get(
                    str(vehicle_id), "unknown"
                ),
            )
        return self.vehicle_intervals[key]

    def _e2_vehicle_window_acc(
        self,
        window_index: int,
        detector_id: str,
        vehicle_id: str,
    ) -> E2Accumulator:
        key = (int(window_index), str(detector_id), str(vehicle_id))
        if key not in self.e2_vehicle_windows:
            self.e2_vehicle_windows[key] = E2Accumulator()
        return self.e2_vehicle_windows[key]

    def _classify_vehicle_type(self, raw_type: Any) -> str:
        raw = "" if raw_type is None else str(raw_type).strip()
        low = raw.lower()
        if not low:
            return "unknown"

        truck_aliases = set(
            getattr(self.emission_lookup, "TRUCK_ALIASES", {"truck"})
        )
        sedan_aliases = set(
            getattr(self.emission_lookup, "SEDAN_ALIASES", {"sedan"})
        )
        if (
            low in truck_aliases
            or "truck" in low
            or "lorry" in low
            or "hdv" in low
            or "freight" in low
        ):
            return "truck"
        if (
            low in sedan_aliases
            or "sedan" in low
            or "passenger" in low
            or "car" in low
            or low == "vehicle"
        ):
            return "sedan"
        return "unknown"

    def _register_vehicle(
        self,
        vehicle_id: str,
        raw_type: Any = None,
    ) -> str:
        vehicle_id = str(vehicle_id)
        existing = self.vehicle_type_registry.get(vehicle_id)
        if existing in {"truck", "sedan"}:
            return existing
        if raw_type is None or str(raw_type).strip() == "":
            try:
                raw_type = self.traci.vehicle.getTypeID(vehicle_id)
            except Exception:
                raw_type = self.vehicle_raw_type_registry.get(vehicle_id, "")

        if str(raw_type).strip():
            self.vehicle_raw_type_registry[vehicle_id] = str(raw_type)
            classified = self._classify_vehicle_type(raw_type)
            if classified == "unknown":
                try:
                    vehicle_class = self.traci.vehicle.getVehicleClass(
                        vehicle_id
                    )
                except Exception:
                    vehicle_class = ""
                classified = self._classify_vehicle_type(vehicle_class)
            self.vehicle_type_registry[vehicle_id] = classified
        return self.vehicle_type_registry.get(vehicle_id, "unknown")

    def _typed_ids(self, ids: Iterable[str]) -> dict[str, set[str]]:
        result = _new_type_set_dict()
        for vehicle_id in ids:
            result[self.vehicle_type_registry.get(vehicle_id, "unknown")].add(
                vehicle_id
            )
        return result

    def _get_e2_membership(self) -> dict[str, set[str]]:
        membership: dict[str, set[str]] = defaultdict(set)
        try:
            all_results = self.traci.lanearea.getAllSubscriptionResults() or {}
        except Exception:
            all_results = {}
        for detector_id in self.e2_meta:
            ids: Iterable[str]
            try:
                result = all_results.get(detector_id, {})
                if not result and not all_results:
                    result = (
                        self.traci.lanearea.getSubscriptionResults(detector_id)
                        or {}
                    )
                ids = result.get(self.tc.LAST_STEP_VEHICLE_ID_LIST, ())
            except Exception:
                ids = ()
            if not ids:
                try:
                    ids = self.traci.lanearea.getLastStepVehicleIDs(detector_id)
                except Exception:
                    ids = ()
            for vehicle_id in ids:
                membership[str(vehicle_id)].add(detector_id)
        return membership

    def _get_exit_e1_ids(self) -> set[str]:
        current: set[str] = set()
        try:
            all_results = (
                self.traci.inductionloop.getAllSubscriptionResults() or {}
            )
        except Exception:
            all_results = {}
        for detector_id in self.exit_e1_ids:
            ids: Iterable[str]
            try:
                result = all_results.get(detector_id, {})
                if not result and not all_results:
                    result = (
                        self.traci.inductionloop.getSubscriptionResults(
                            detector_id
                        )
                        or {}
                    )
                ids = result.get(self.tc.LAST_STEP_VEHICLE_ID_LIST, ())
            except Exception:
                ids = ()
            if not ids:
                try:
                    ids = self.traci.inductionloop.getLastStepVehicleIDs(
                        detector_id
                    )
                except Exception:
                    ids = ()
            current.update(str(x) for x in ids)
        new_ids = current - self.exit_seen_vehicle_ids
        self.exit_seen_vehicle_ids.update(new_ids)
        return new_ids

    def _vehicle_state(
        self,
        vehicle_id: str,
        vehicle_context: Mapping[str, Mapping[int, Any]],
    ) -> dict[str, Any]:
        values = vehicle_context.get(vehicle_id, {})

        def context_or_call(var_id: int, method_name: str, default: Any) -> Any:
            if var_id in values:
                return values[var_id]
            try:
                return getattr(self.traci.vehicle, method_name)(vehicle_id)
            except Exception:
                return default

        return {
            "raw_type": context_or_call(self.tc.VAR_TYPE, "getTypeID", ""),
            "speed_mps": float(
                context_or_call(self.tc.VAR_SPEED, "getSpeed", 0.0)
            ),
            "accel_ms2": float(
                context_or_call(
                    self.tc.VAR_ACCELERATION, "getAcceleration", 0.0
                )
            ),
            "distance_m": float(
                context_or_call(self.tc.VAR_DISTANCE, "getDistance", 0.0)
            ),
            "time_loss_s": float(
                context_or_call(self.tc.VAR_TIMELOSS, "getTimeLoss", 0.0)
            ),
            "waiting_time_now_s": float(
                context_or_call(
                    self.tc.VAR_WAITING_TIME, "getWaitingTime", 0.0
                )
            ),
        }

    def collect_second(
        self,
        *,
        sim_time: float,
        decision_step: int,
        vehicle_context: Mapping[str, Mapping[int, Any]],
    ) -> None:
        """Collect the one-second state after one ``simulationStep``."""
        if self.traci is None or self.tc is None:
            raise RuntimeError("start_episode() has not been called")
        self.last_sim_time_s = float(sim_time)

        try:
            dt = float(self.traci.simulation.getDeltaT())
        except Exception:
            dt = 1.0
        if dt <= 0.0:
            dt = 1.0
        sample_time = max(0.0, float(sim_time) - dt)
        interval_index = self._interval_index(sample_time)
        window_index = self._window_index_from_interval(interval_index)
        network = self._network_acc(interval_index)

        departed_ids = {
            str(x) for x in self.traci.simulation.getDepartedIDList()
        }
        arrived_ids = {
            str(x) for x in self.traci.simulation.getArrivedIDList()
        }
        active_ids = {str(x) for x in self.traci.vehicle.getIDList()}

        for vehicle_id in departed_ids | active_ids:
            raw_type = None
            values = vehicle_context.get(vehicle_id, {})
            if self.tc.VAR_TYPE in values:
                raw_type = values[self.tc.VAR_TYPE]
            self._register_vehicle(vehicle_id, raw_type)

        network.entered_ids.update(departed_ids)
        network.arrived_ids.update(arrived_ids)
        for vehicle_id in arrived_ids:
            self.arrival_sample_time[vehicle_id] = sample_time

        new_exit_ids = self._get_exit_e1_ids()
        network.exit_e1_ids.update(new_exit_ids)

        try:
            network.teleport_start_count += len(
                self.traci.simulation.getStartingTeleportIDList()
            )
            network.teleport_end_count += len(
                self.traci.simulation.getEndingTeleportIDList()
            )
        except Exception:
            pass

        e2_membership = self._get_e2_membership()
        network.overlap_e2_vehicle_seconds += sum(
            max(0, len(detectors) - 1) * dt
            for detectors in e2_membership.values()
        )

        network.present_ids.update(active_ids)
        network.end_active_ids = set(active_ids)
        network.active_vehicle_seconds += len(active_ids) * dt

        for vehicle_id in active_ids:
            state = self._vehicle_state(vehicle_id, vehicle_context)
            vehicle_type = self._register_vehicle(
                vehicle_id, state["raw_type"]
            )
            network.active_seconds_by_type[vehicle_type] += dt

            time_loss_now = state["time_loss_s"]
            distance_now = state["distance_m"]
            first_observation = vehicle_id not in self.previous_time_loss
            previous_tl = (
                0.0
                if first_observation and vehicle_id in departed_ids
                else self.previous_time_loss.get(vehicle_id, time_loss_now)
            )
            previous_distance = (
                0.0
                if first_observation and vehicle_id in departed_ids
                else self.previous_distance.get(vehicle_id, distance_now)
            )
            raw_delta_tl = time_loss_now - previous_tl
            raw_delta_distance = distance_now - previous_distance
            if raw_delta_tl < -EPS:
                network.negative_time_loss_delta_count += 1
            if raw_delta_distance < -EPS:
                network.negative_distance_delta_count += 1
            delta_tl = max(0.0, raw_delta_tl)
            delta_distance = max(0.0, raw_delta_distance)

            is_waiting = state["waiting_time_now_s"] > 0.0
            waiting_seconds = dt if is_waiting else 0.0
            waiting_event = (
                1.0
                if is_waiting
                and not self.previous_is_waiting.get(vehicle_id, False)
                else 0.0
            )

            increments = {
                "time_loss_s": delta_tl,
                "distance_m": delta_distance,
                "waiting_time_s": waiting_seconds,
                "waiting_event_count": waiting_event,
            }
            for metric, value in increments.items():
                network.metrics[metric] += value
                network.metrics_by_type[metric][vehicle_type] += value
                self.vehicle_observed[vehicle_id][metric] += value

            nox_mg = float(
                self.emission_lookup.get_emission(
                    vehicle_type=(
                        vehicle_type
                        if vehicle_type in {"truck", "sedan"}
                        else state["raw_type"]
                    ),
                    speed_ms=state["speed_mps"],
                    accel_ms2=state["accel_ms2"],
                    sim_step=dt,
                    pollutant=self.config.pollutant,
                    output_unit="mg",
                )
            )
            network.network_nox_mg += nox_mg
            network.network_nox_available = True

            detectors = e2_membership.get(vehicle_id, set())

            for detector_id in detectors:
                e2 = self._e2_acc(interval_index, detector_id)
                e2.unique_ids.add(vehicle_id)
                e2.unique_ids_by_type[vehicle_type].add(vehicle_id)
                e2.vehicle_seconds += dt
                e2.vehicle_seconds_by_type[vehicle_type] += dt
                e2.nox_mg += nox_mg
                e2.nox_by_type[vehicle_type] += nox_mg
                for metric, value in increments.items():
                    e2.metrics[metric] += value
                    e2.metrics_by_type[metric][vehicle_type] += value

                e2_vehicle = self._e2_vehicle_window_acc(
                    window_index, detector_id, vehicle_id
                )
                e2_vehicle.unique_ids.add(vehicle_id)
                e2_vehicle.unique_ids_by_type[vehicle_type].add(vehicle_id)
                e2_vehicle.vehicle_seconds += dt
                e2_vehicle.vehicle_seconds_by_type[vehicle_type] += dt
                e2_vehicle.nox_mg += nox_mg
                e2_vehicle.nox_by_type[vehicle_type] += nox_mg
                for metric, value in increments.items():
                    e2_vehicle.metrics[metric] += value
                    e2_vehicle.metrics_by_type[metric][vehicle_type] += value

            if self.config.save_vehicle_10s:
                vehicle_acc = self._vehicle_acc(interval_index, vehicle_id)
                vehicle_acc.vehicle_type = vehicle_type
                vehicle_acc.present = True
                vehicle_acc.entered = vehicle_id in departed_ids
                vehicle_acc.exited = (
                    vehicle_id in arrived_ids or vehicle_id in new_exit_ids
                )
                for metric, value in increments.items():
                    vehicle_acc.metrics[metric] += value
                    if detectors:
                        # Deduplicate overlapping E2 detectors for this
                        # network-wide per-vehicle E2 summary.
                        vehicle_acc.e2_metrics[metric] += value
                if detectors:
                    vehicle_acc.e2_nox_mg += nox_mg

            self.previous_time_loss[vehicle_id] = time_loss_now
            self.previous_distance[vehicle_id] = distance_now
            self.previous_is_waiting[vehicle_id] = is_waiting

        if self.config.save_vehicle_10s:
            for vehicle_id in departed_ids:
                acc = self._vehicle_acc(interval_index, vehicle_id)
                acc.entered = True
                acc.vehicle_type = self.vehicle_type_registry.get(
                    vehicle_id, "unknown"
                )
            for vehicle_id in arrived_ids | new_exit_ids:
                acc = self._vehicle_acc(interval_index, vehicle_id)
                acc.exited = True
                acc.vehicle_type = self.vehicle_type_registry.get(
                    vehicle_id, "unknown"
                )

        for vehicle_id in arrived_ids:
            self.previous_time_loss.pop(vehicle_id, None)
            self.previous_distance.pop(vehicle_id, None)
            self.previous_is_waiting.pop(vehicle_id, None)
        self.previous_active_ids = set(active_ids)

    def end_decision_step(
        self,
        *,
        decision_step: int,
        step_emission: Any,
    ) -> None:
        """Compatibility no-op; NOx is accumulated from the same 1-s snapshot."""
        _ = (decision_step, step_emission)

    # ------------------------------------------------------------------
    # Terminal correction and row construction
    # ------------------------------------------------------------------

    @staticmethod
    def _float_attr(node: ET.Element, name: str, default: float = 0.0) -> float:
        try:
            return float(node.get(name, default))
        except (TypeError, ValueError):
            return float(default)

    def _apply_tripinfo_corrections(
        self,
        tripinfo_path: str | os.PathLike[str],
    ) -> dict[str, float]:
        totals = {
            "completed_trip_count": 0.0,
            "unfinished_trip_count": 0.0,
            "tripinfo_time_loss_s": 0.0,
            "tripinfo_route_length_m": 0.0,
            "tripinfo_waiting_time_s": 0.0,
            "tripinfo_waiting_count": 0.0,
            "raw_time_loss_residual_s": 0.0,
            "raw_distance_residual_m": 0.0,
            "raw_waiting_time_residual_s": 0.0,
            "raw_waiting_event_residual_count": 0.0,
        }
        path = Path(tripinfo_path)
        if not path.is_file():
            totals["tripinfo_missing"] = 1.0
            return totals

        root = ET.parse(path).getroot()
        for trip in root.findall("tripinfo"):
            vehicle_id = str(trip.get("id", ""))
            arrival = self._float_attr(trip, "arrival", -1.0)
            if arrival < 0.0:
                totals["unfinished_trip_count"] += 1.0
                continue
            totals["completed_trip_count"] += 1.0

            trip_values = {
                "time_loss_s": self._float_attr(trip, "timeLoss"),
                "distance_m": self._float_attr(trip, "routeLength"),
                "waiting_time_s": self._float_attr(trip, "waitingTime"),
                "waiting_event_count": self._float_attr(trip, "waitingCount"),
            }
            totals["tripinfo_time_loss_s"] += trip_values["time_loss_s"]
            totals["tripinfo_route_length_m"] += trip_values["distance_m"]
            totals["tripinfo_waiting_time_s"] += trip_values["waiting_time_s"]
            totals["tripinfo_waiting_count"] += trip_values[
                "waiting_event_count"
            ]

            observed = self.vehicle_observed.get(vehicle_id, _new_metric_dict())
            residuals = {
                name: trip_values[name] - observed.get(name, 0.0)
                for name in METRIC_NAMES
            }
            totals["raw_time_loss_residual_s"] += residuals["time_loss_s"]
            totals["raw_distance_residual_m"] += residuals["distance_m"]
            totals["raw_waiting_time_residual_s"] += residuals[
                "waiting_time_s"
            ]
            totals["raw_waiting_event_residual_count"] += residuals[
                "waiting_event_count"
            ]

            # A trip arriving exactly at t=10 belongs to the simulation step
            # that closed [9, 10), hence to the preceding 10-second interval.
            sample_time = max(0.0, arrival - 1.0e-6)
            interval_index = self._interval_index(sample_time)
            acc = self._network_acc(interval_index)
            vehicle_type = self.vehicle_type_registry.get(
                vehicle_id, "unknown"
            )

            positive_corrections = {
                name: max(0.0, value) for name, value in residuals.items()
            }
            for metric, correction in positive_corrections.items():
                acc.metrics[metric] += correction
                acc.metrics_by_type[metric][vehicle_type] += correction

            acc.terminal_time_loss_correction_s += positive_corrections[
                "time_loss_s"
            ]
            acc.terminal_distance_correction_m += positive_corrections[
                "distance_m"
            ]
            acc.terminal_waiting_time_correction_s += positive_corrections[
                "waiting_time_s"
            ]
            acc.terminal_waiting_event_correction_count += (
                positive_corrections["waiting_event_count"]
            )
        return totals

    def _base_metadata(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "sumo_seed": self.sumo_seed,
            "controller": self.controller,
            "reward_mode": self.reward_mode,
            "case_name": self.case_name,
        }

    def _nominal_values(self, interval_index: int) -> dict[str, float]:
        item = self.nominal_by_interval.get(interval_index, {})
        return {
            "nominal_scheduled_vehicle_count": float(
                item.get("vehicle_count", 0.0)
            ),
            "nominal_scheduled_truck_count": float(
                item.get("truck_count", 0.0)
            ),
            "nominal_scheduled_sedan_count": float(
                item.get("sedan_count", 0.0)
            ),
            "nominal_scheduled_unknown_count": float(
                item.get("unknown_count", 0.0)
            ),
            "nominal_schedule_contains_expectation": int(
                bool(item.get("contains_expectation", False))
            ),
        }

    def _network_10s_row(
        self,
        interval_index: int,
        acc: NetworkAccumulator,
    ) -> dict[str, Any]:
        start_s = interval_index * self.config.decision_interval_s
        end_s = start_s + self.config.decision_interval_s
        entered = self._typed_ids(acc.entered_ids)
        arrived = self._typed_ids(acc.arrived_ids)
        exits = self._typed_ids(acc.exit_e1_ids)
        present = self._typed_ids(acc.present_ids)
        end_active = self._typed_ids(acc.end_active_ids)

        row: dict[str, Any] = {
            **self._base_metadata(),
            "decision_step": interval_index,
            "interval_start_s": start_s,
            "interval_end_s": end_s,
            **self._nominal_values(interval_index),
            "entered_vehicle_count": len(acc.entered_ids),
            "entered_truck_count": len(entered["truck"]),
            "entered_sedan_count": len(entered["sedan"]),
            "entered_unknown_count": len(entered["unknown"]),
            "entered_truck_share": (
                len(entered["truck"]) / len(acc.entered_ids)
                if acc.entered_ids
                else 0.0
            ),
            "present_unique_vehicle_count": len(acc.present_ids),
            "present_unique_truck_count": len(present["truck"]),
            "present_unique_sedan_count": len(present["sedan"]),
            "present_unique_unknown_count": len(present["unknown"]),
            "present_unique_truck_share": (
                len(present["truck"]) / len(acc.present_ids)
                if acc.present_ids
                else 0.0
            ),
            "active_vehicle_seconds": acc.active_vehicle_seconds,
            "active_truck_seconds": acc.active_seconds_by_type["truck"],
            "active_sedan_seconds": acc.active_seconds_by_type["sedan"],
            "active_unknown_seconds": acc.active_seconds_by_type["unknown"],
            "mean_active_vehicle_count": (
                acc.active_vehicle_seconds / self.config.decision_interval_s
            ),
            "mean_active_truck_count": (
                acc.active_seconds_by_type["truck"]
                / self.config.decision_interval_s
            ),
            "active_truck_share": (
                acc.active_seconds_by_type["truck"]
                / acc.active_vehicle_seconds
                if acc.active_vehicle_seconds > EPS
                else 0.0
            ),
            "end_active_vehicle_count": len(acc.end_active_ids),
            "end_active_truck_count": len(end_active["truck"]),
            "end_active_sedan_count": len(end_active["sedan"]),
            "end_active_unknown_count": len(end_active["unknown"]),
            "exit_e1_configured": int(bool(self.exit_e1_ids)),
            "exit_e1_vehicle_count": len(acc.exit_e1_ids),
            "exit_e1_truck_count": len(exits["truck"]),
            "exit_e1_sedan_count": len(exits["sedan"]),
            "exit_e1_unknown_count": len(exits["unknown"]),
            "arrived_vehicle_count": len(acc.arrived_ids),
            "arrived_truck_count": len(arrived["truck"]),
            "arrived_sedan_count": len(arrived["sedan"]),
            "arrived_unknown_count": len(arrived["unknown"]),
            "throughput_source": (
                "exit_e1" if self.exit_e1_ids else "sumo_arrived"
            ),
            "throughput_vehicle_count": (
                len(acc.exit_e1_ids)
                if self.exit_e1_ids
                else len(acc.arrived_ids)
            ),
            "exit_vehicle_count": (
                len(acc.exit_e1_ids)
                if self.exit_e1_ids
                else len(acc.arrived_ids)
            ),
            "network_nox_mg": acc.network_nox_mg,
            "network_nox_available": int(acc.network_nox_available),
            "e2_overlap_vehicle_seconds": acc.overlap_e2_vehicle_seconds,
            "negative_time_loss_delta_count": (
                acc.negative_time_loss_delta_count
            ),
            "negative_distance_delta_count": (
                acc.negative_distance_delta_count
            ),
            "teleport_start_count": acc.teleport_start_count,
            "teleport_end_count": acc.teleport_end_count,
            "terminal_time_loss_correction_s": (
                acc.terminal_time_loss_correction_s
            ),
            "terminal_distance_correction_m": (
                acc.terminal_distance_correction_m
            ),
            "terminal_waiting_time_correction_s": (
                acc.terminal_waiting_time_correction_s
            ),
            "terminal_waiting_event_correction_count": (
                acc.terminal_waiting_event_correction_count
            ),
        }
        for metric in METRIC_NAMES:
            row[f"network_{metric}"] = acc.metrics[metric]
            row[f"truck_{metric}"] = acc.metrics_by_type[metric]["truck"]
            row[f"sedan_{metric}"] = acc.metrics_by_type[metric]["sedan"]
            row[f"unknown_{metric}"] = acc.metrics_by_type[metric]["unknown"]
        row["waiting_event_count"] = row["network_waiting_event_count"]
        return row

    def _e2_10s_row(
        self,
        interval_index: int,
        detector_id: str,
        acc: E2Accumulator,
    ) -> dict[str, Any]:
        meta = self.e2_meta[detector_id]
        start_s = interval_index * self.config.decision_interval_s
        row: dict[str, Any] = {
            **self._base_metadata(),
            "decision_step": interval_index,
            "interval_start_s": start_s,
            "interval_end_s": start_s + self.config.decision_interval_s,
            "e2_detector_id": detector_id,
            "lane_id": meta.lane_id,
            "e2_start_pos_m": meta.start_pos_m,
            "e2_length_m": meta.length_m,
            "e2_end_pos_m": meta.end_pos_m,
            "unique_vehicle_count": len(acc.unique_ids),
            "unique_truck_count": len(acc.unique_ids_by_type["truck"]),
            "unique_sedan_count": len(acc.unique_ids_by_type["sedan"]),
            "unique_unknown_count": len(acc.unique_ids_by_type["unknown"]),
            "vehicle_seconds": acc.vehicle_seconds,
            "truck_vehicle_seconds": acc.vehicle_seconds_by_type["truck"],
            "sedan_vehicle_seconds": acc.vehicle_seconds_by_type["sedan"],
            "unknown_vehicle_seconds": acc.vehicle_seconds_by_type["unknown"],
            "nox_mg": acc.nox_mg,
            "truck_nox_mg": acc.nox_by_type["truck"],
            "sedan_nox_mg": acc.nox_by_type["sedan"],
            "unknown_nox_mg": acc.nox_by_type["unknown"],
            "truck_vehicle_second_share": (
                acc.vehicle_seconds_by_type["truck"] / acc.vehicle_seconds
                if acc.vehicle_seconds > EPS
                else 0.0
            ),
            "truck_nox_share": (
                acc.nox_by_type["truck"] / acc.nox_mg
                if acc.nox_mg > EPS
                else 0.0
            ),
        }
        for metric in METRIC_NAMES:
            row[metric] = acc.metrics[metric]
            row[f"truck_{metric}"] = acc.metrics_by_type[metric]["truck"]
            row[f"sedan_{metric}"] = acc.metrics_by_type[metric]["sedan"]
            row[f"unknown_{metric}"] = acc.metrics_by_type[metric]["unknown"]
        return row

    @staticmethod
    def _sum_rows(rows: Iterable[Mapping[str, Any]], key: str) -> float:
        return float(sum(float(row.get(key, 0.0) or 0.0) for row in rows))

    def _aggregate_network_300s(self) -> list[dict[str, Any]]:
        interval_rows = {
            int(row["decision_step"]): row for row in self.network_rows_10s
        }
        rows: list[dict[str, Any]] = []
        total_duration_s = max(
            (
                (idx + 1) * self.config.decision_interval_s
                for idx in self.network_intervals
            ),
            default=0,
        )
        n_windows = int(
            math.ceil(total_duration_s / self.config.aggregate_window_s)
        )
        intervals_per_window = (
            self.config.aggregate_window_s // self.config.decision_interval_s
        )

        for window_index in range(n_windows):
            start_interval = window_index * intervals_per_window
            selected = [
                interval_rows[idx]
                for idx in range(
                    start_interval, start_interval + intervals_per_window
                )
                if idx in interval_rows
            ]
            if not selected:
                continue

            start_s = window_index * self.config.aggregate_window_s
            end_s = start_s + self.config.aggregate_window_s
            accs = [
                self.network_intervals[idx]
                for idx in range(
                    start_interval, start_interval + intervals_per_window
                )
                if idx in self.network_intervals
            ]
            present_ids = set().union(*(acc.present_ids for acc in accs))
            entered_ids = set().union(*(acc.entered_ids for acc in accs))
            arrived_ids = set().union(*(acc.arrived_ids for acc in accs))
            exit_ids = set().union(*(acc.exit_e1_ids for acc in accs))
            start_active_ids = set(accs[0].start_active_ids)
            end_active_ids = set(accs[-1].end_active_ids)

            present = self._typed_ids(present_ids)
            entered = self._typed_ids(entered_ids)
            arrived = self._typed_ids(arrived_ids)
            exits = self._typed_ids(exit_ids)
            end_active = self._typed_ids(end_active_ids)

            throughput_count = (
                len(exit_ids) if self.exit_e1_ids else len(arrived_ids)
            )
            duration_h = self.config.aggregate_window_s / 3600.0
            distance_m = self._sum_rows(selected, "network_distance_m")
            nox_mg = self._sum_rows(selected, "network_nox_mg")
            row: dict[str, Any] = {
                **self._base_metadata(),
                "window_index": window_index,
                "window_start_s": start_s,
                "window_end_s": end_s,
                "is_analysis_window": int(
                    start_s >= self.config.analysis_start_s
                    and end_s <= self.config.analysis_end_s
                ),
                "nominal_scheduled_vehicle_count": self._sum_rows(
                    selected, "nominal_scheduled_vehicle_count"
                ),
                "nominal_scheduled_truck_count": self._sum_rows(
                    selected, "nominal_scheduled_truck_count"
                ),
                "nominal_scheduled_sedan_count": self._sum_rows(
                    selected, "nominal_scheduled_sedan_count"
                ),
                "nominal_scheduled_unknown_count": self._sum_rows(
                    selected, "nominal_scheduled_unknown_count"
                ),
                "nominal_schedule_contains_expectation": int(
                    any(
                        int(row_i.get(
                            "nominal_schedule_contains_expectation", 0
                        ))
                        for row_i in selected
                    )
                ),
                "entered_vehicle_count": len(entered_ids),
                "entered_truck_count": len(entered["truck"]),
                "entered_sedan_count": len(entered["sedan"]),
                "entered_unknown_count": len(entered["unknown"]),
                "entered_demand_veh_h": len(entered_ids) / duration_h,
                "entered_truck_share": (
                    len(entered["truck"]) / len(entered_ids)
                    if entered_ids
                    else 0.0
                ),
                "present_unique_vehicle_count": len(present_ids),
                "present_unique_truck_count": len(present["truck"]),
                "present_unique_sedan_count": len(present["sedan"]),
                "present_unique_unknown_count": len(present["unknown"]),
                "present_unique_truck_share": (
                    len(present["truck"]) / len(present_ids)
                    if present_ids
                    else 0.0
                ),
                "active_vehicle_seconds": self._sum_rows(
                    selected, "active_vehicle_seconds"
                ),
                "active_truck_seconds": self._sum_rows(
                    selected, "active_truck_seconds"
                ),
                "active_sedan_seconds": self._sum_rows(
                    selected, "active_sedan_seconds"
                ),
                "active_unknown_seconds": self._sum_rows(
                    selected, "active_unknown_seconds"
                ),
                "mean_active_vehicle_count": self._sum_rows(
                    selected, "active_vehicle_seconds"
                )
                / self.config.aggregate_window_s,
                "mean_active_truck_count": self._sum_rows(
                    selected, "active_truck_seconds"
                )
                / self.config.aggregate_window_s,
                "active_truck_share": (
                    self._sum_rows(selected, "active_truck_seconds")
                    / self._sum_rows(selected, "active_vehicle_seconds")
                    if self._sum_rows(
                        selected, "active_vehicle_seconds"
                    )
                    > EPS
                    else 0.0
                ),
                "start_active_vehicle_count": len(start_active_ids),
                "end_active_vehicle_count": len(end_active_ids),
                "end_active_truck_count": len(end_active["truck"]),
                "end_active_sedan_count": len(end_active["sedan"]),
                "end_active_unknown_count": len(end_active["unknown"]),
                "exit_e1_configured": int(bool(self.exit_e1_ids)),
                "exit_e1_vehicle_count": len(exit_ids),
                "exit_e1_truck_count": len(exits["truck"]),
                "exit_e1_sedan_count": len(exits["sedan"]),
                "exit_e1_unknown_count": len(exits["unknown"]),
                "exit_e1_throughput_veh_h": (
                    len(exit_ids) / duration_h
                    if self.exit_e1_ids
                    else ""
                ),
                "arrived_vehicle_count": len(arrived_ids),
                "arrived_truck_count": len(arrived["truck"]),
                "arrived_sedan_count": len(arrived["sedan"]),
                "arrived_unknown_count": len(arrived["unknown"]),
                "arrived_throughput_veh_h": len(arrived_ids) / duration_h,
                "throughput_source": (
                    "exit_e1" if self.exit_e1_ids else "sumo_arrived"
                ),
                "throughput_vehicle_count": throughput_count,
                "exit_vehicle_count": throughput_count,
                "throughput_veh_h": throughput_count / duration_h,
                "throughput_truck_share": (
                    (
                        len(exits["truck"])
                        if self.exit_e1_ids
                        else len(arrived["truck"])
                    )
                    / throughput_count
                    if throughput_count
                    else 0.0
                ),
                "network_nox_mg": nox_mg,
                "network_nox_g": nox_mg / 1000.0,
                "network_nox_mg_per_km": (
                    nox_mg / (distance_m / 1000.0)
                    if distance_m > EPS
                    else 0.0
                ),
                "network_nox_available_interval_count": int(
                    self._sum_rows(selected, "network_nox_available")
                ),
                "e2_overlap_vehicle_seconds": self._sum_rows(
                    selected, "e2_overlap_vehicle_seconds"
                ),
                "negative_time_loss_delta_count": int(
                    self._sum_rows(
                        selected, "negative_time_loss_delta_count"
                    )
                ),
                "negative_distance_delta_count": int(
                    self._sum_rows(
                        selected, "negative_distance_delta_count"
                    )
                ),
                "teleport_start_count": int(
                    self._sum_rows(selected, "teleport_start_count")
                ),
                "teleport_end_count": int(
                    self._sum_rows(selected, "teleport_end_count")
                ),
                "terminal_time_loss_correction_s": self._sum_rows(
                    selected, "terminal_time_loss_correction_s"
                ),
                "terminal_distance_correction_m": self._sum_rows(
                    selected, "terminal_distance_correction_m"
                ),
                "terminal_waiting_time_correction_s": self._sum_rows(
                    selected, "terminal_waiting_time_correction_s"
                ),
                "terminal_waiting_event_correction_count": self._sum_rows(
                    selected, "terminal_waiting_event_correction_count"
                ),
            }
            for metric in METRIC_NAMES:
                row[f"network_{metric}"] = self._sum_rows(
                    selected, f"network_{metric}"
                )
                row[f"truck_{metric}"] = self._sum_rows(
                    selected, f"truck_{metric}"
                )
                row[f"sedan_{metric}"] = self._sum_rows(
                    selected, f"sedan_{metric}"
                )
                row[f"unknown_{metric}"] = self._sum_rows(
                    selected, f"unknown_{metric}"
                )
            row["mean_time_loss_s_per_present_vehicle"] = (
                row["network_time_loss_s"] / len(present_ids)
                if present_ids
                else 0.0
            )
            row["waiting_event_count"] = row[
                "network_waiting_event_count"
            ]
            rows.append(row)
        return rows

    def _aggregate_e2_300s(self) -> list[dict[str, Any]]:
        intervals_per_window = (
            self.config.aggregate_window_s // self.config.decision_interval_s
        )
        grouped: dict[tuple[int, str], list[E2Accumulator]] = defaultdict(list)
        for (interval_index, detector_id), acc in self.e2_intervals.items():
            grouped[
                (
                    interval_index // intervals_per_window,
                    detector_id,
                )
            ].append(acc)

        rows: list[dict[str, Any]] = []
        for (window_index, detector_id), accs in sorted(grouped.items()):
            meta = self.e2_meta[detector_id]
            start_s = window_index * self.config.aggregate_window_s
            end_s = start_s + self.config.aggregate_window_s
            ids = set().union(*(acc.unique_ids for acc in accs))
            ids_by_type = {
                name: set().union(
                    *(acc.unique_ids_by_type[name] for acc in accs)
                )
                for name in TYPE_NAMES
            }
            row: dict[str, Any] = {
                **self._base_metadata(),
                "window_index": window_index,
                "window_start_s": start_s,
                "window_end_s": end_s,
                "is_analysis_window": int(
                    start_s >= self.config.analysis_start_s
                    and end_s <= self.config.analysis_end_s
                ),
                "e2_detector_id": detector_id,
                "lane_id": meta.lane_id,
                "e2_start_pos_m": meta.start_pos_m,
                "e2_length_m": meta.length_m,
                "e2_end_pos_m": meta.end_pos_m,
                "unique_vehicle_count": len(ids),
                "unique_truck_count": len(ids_by_type["truck"]),
                "unique_sedan_count": len(ids_by_type["sedan"]),
                "unique_unknown_count": len(ids_by_type["unknown"]),
                "vehicle_seconds": sum(acc.vehicle_seconds for acc in accs),
                "truck_vehicle_seconds": sum(
                    acc.vehicle_seconds_by_type["truck"] for acc in accs
                ),
                "sedan_vehicle_seconds": sum(
                    acc.vehicle_seconds_by_type["sedan"] for acc in accs
                ),
                "unknown_vehicle_seconds": sum(
                    acc.vehicle_seconds_by_type["unknown"] for acc in accs
                ),
                "nox_mg": sum(acc.nox_mg for acc in accs),
                "truck_nox_mg": sum(
                    acc.nox_by_type["truck"] for acc in accs
                ),
                "sedan_nox_mg": sum(
                    acc.nox_by_type["sedan"] for acc in accs
                ),
                "unknown_nox_mg": sum(
                    acc.nox_by_type["unknown"] for acc in accs
                ),
            }
            row["truck_vehicle_second_share"] = (
                row["truck_vehicle_seconds"] / row["vehicle_seconds"]
                if row["vehicle_seconds"] > EPS
                else 0.0
            )
            row["truck_nox_share"] = (
                row["truck_nox_mg"] / row["nox_mg"]
                if row["nox_mg"] > EPS
                else 0.0
            )
            for metric in METRIC_NAMES:
                row[metric] = sum(acc.metrics[metric] for acc in accs)
                for name in TYPE_NAMES:
                    row[f"{name}_{metric}"] = sum(
                        acc.metrics_by_type[metric][name] for acc in accs
                    )
            rows.append(row)
        return rows

    def _build_e2_vehicle_300s(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (
            window_index,
            detector_id,
            vehicle_id,
        ), acc in sorted(self.e2_vehicle_windows.items()):
            start_s = window_index * self.config.aggregate_window_s
            end_s = start_s + self.config.aggregate_window_s
            vehicle_type = self.vehicle_type_registry.get(
                vehicle_id, "unknown"
            )
            row: dict[str, Any] = {
                **self._base_metadata(),
                "window_index": window_index,
                "window_start_s": start_s,
                "window_end_s": end_s,
                "is_analysis_window": int(
                    start_s >= self.config.analysis_start_s
                    and end_s <= self.config.analysis_end_s
                ),
                "e2_detector_id": detector_id,
                "lane_id": self.e2_meta[detector_id].lane_id,
                "vehicle_id": vehicle_id,
                "vehicle_type": vehicle_type,
                "vehicle_seconds": acc.vehicle_seconds,
                "nox_mg": acc.nox_mg,
                **acc.metrics,
            }
            rows.append(row)
        return rows

    def _build_vehicle_10s(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (interval_index, vehicle_id), acc in sorted(
            self.vehicle_intervals.items()
        ):
            start_s = interval_index * self.config.decision_interval_s
            rows.append(
                {
                    **self._base_metadata(),
                    "decision_step": interval_index,
                    "interval_start_s": start_s,
                    "interval_end_s": (
                        start_s + self.config.decision_interval_s
                    ),
                    "vehicle_id": vehicle_id,
                    "vehicle_type": self.vehicle_type_registry.get(
                        vehicle_id, acc.vehicle_type
                    ),
                    "present_in_interval": int(acc.present),
                    "entered_in_interval": int(acc.entered),
                    "exited_in_interval": int(acc.exited),
                    **acc.metrics,
                    "e2_time_loss_s": acc.e2_metrics["time_loss_s"],
                    "e2_distance_m": acc.e2_metrics["distance_m"],
                    "e2_waiting_time_s": acc.e2_metrics[
                        "waiting_time_s"
                    ],
                    "e2_waiting_event_count": acc.e2_metrics[
                        "waiting_event_count"
                    ],
                    "e2_nox_mg": acc.e2_nox_mg,
                }
            )
        return rows

    def _build_quality_control(
        self,
        tripinfo_totals: Mapping[str, float],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for network_row in self.network_rows_300s:
            window_index = int(network_row["window_index"])
            start_s = int(network_row["window_start_s"])
            end_s = int(network_row["window_end_s"])
            network_10s_rows = [
                row
                for row in self.network_rows_10s
                if start_s
                <= int(row["interval_start_s"])
                < end_s
            ]
            e2_rows = [
                row
                for row in self.e2_rows_300s
                if int(row["window_index"]) == window_index
            ]
            e2_10s_rows = [
                row
                for row in self.e2_rows_10s
                if start_s
                <= int(row["interval_start_s"])
                < end_s
            ]
            e2_nox_type_diff = sum(
                float(row["nox_mg"])
                - float(row["truck_nox_mg"])
                - float(row["sedan_nox_mg"])
                - float(row["unknown_nox_mg"])
                for row in e2_rows
            )
            active_balance = (
                int(network_row["start_active_vehicle_count"])
                + int(network_row["entered_vehicle_count"])
                - int(network_row["arrived_vehicle_count"])
            )
            rows.append(
                {
                    **self._base_metadata(),
                    "scope": "window_300s",
                    "window_index": window_index,
                    "window_start_s": start_s,
                    "window_end_s": end_s,
                    "is_analysis_window": int(
                        network_row["is_analysis_window"]
                    ),
                    "active_vehicle_balance_difference": (
                        int(network_row["end_active_vehicle_count"])
                        - active_balance
                    ),
                    "entered_type_balance_difference": (
                        int(network_row["entered_vehicle_count"])
                        - int(network_row["entered_truck_count"])
                        - int(network_row["entered_sedan_count"])
                        - int(network_row["entered_unknown_count"])
                    ),
                    "present_type_balance_difference": (
                        int(network_row["present_unique_vehicle_count"])
                        - int(network_row["present_unique_truck_count"])
                        - int(network_row["present_unique_sedan_count"])
                        - int(network_row["present_unique_unknown_count"])
                    ),
                    "e2_nox_type_balance_difference_mg": (
                        e2_nox_type_diff
                    ),
                    "network_nox_10s_minus_300s_mg": (
                        self._sum_rows(
                            network_10s_rows, "network_nox_mg"
                        )
                        - float(network_row["network_nox_mg"])
                    ),
                    "network_time_loss_10s_minus_300s_s": (
                        self._sum_rows(
                            network_10s_rows, "network_time_loss_s"
                        )
                        - float(network_row["network_time_loss_s"])
                    ),
                    "network_distance_10s_minus_300s_m": (
                        self._sum_rows(
                            network_10s_rows, "network_distance_m"
                        )
                        - float(network_row["network_distance_m"])
                    ),
                    "network_waiting_time_10s_minus_300s_s": (
                        self._sum_rows(
                            network_10s_rows, "network_waiting_time_s"
                        )
                        - float(network_row["network_waiting_time_s"])
                    ),
                    "e2_nox_10s_minus_300s_mg": (
                        self._sum_rows(e2_10s_rows, "nox_mg")
                        - self._sum_rows(e2_rows, "nox_mg")
                    ),
                    "e2_time_loss_10s_minus_300s_s": (
                        self._sum_rows(e2_10s_rows, "time_loss_s")
                        - self._sum_rows(e2_rows, "time_loss_s")
                    ),
                    "unknown_present_vehicle_count": int(
                        network_row["present_unique_unknown_count"]
                    ),
                    "negative_time_loss_delta_count": int(
                        network_row["negative_time_loss_delta_count"]
                    ),
                    "negative_distance_delta_count": int(
                        network_row["negative_distance_delta_count"]
                    ),
                    "teleport_start_count": int(
                        network_row["teleport_start_count"]
                    ),
                    "teleport_end_count": int(
                        network_row["teleport_end_count"]
                    ),
                    "network_nox_missing_interval_count": (
                        self.config.aggregate_window_s
                        // self.config.decision_interval_s
                        - int(
                            network_row[
                                "network_nox_available_interval_count"
                            ]
                        )
                    ),
                    "e2_overlap_vehicle_seconds": float(
                        network_row["e2_overlap_vehicle_seconds"]
                    ),
                }
            )

        rows.append(
            {
                **self._base_metadata(),
                "scope": "episode_terminal",
                "window_index": "",
                "window_start_s": 0,
                "window_end_s": max(
                    (
                        int(row["interval_end_s"])
                        for row in self.network_rows_10s
                    ),
                    default=0,
                ),
                "is_analysis_window": "",
                **dict(tripinfo_totals),
                "registered_vehicle_count": len(
                    self.vehicle_type_registry
                ),
                "registered_unknown_vehicle_count": sum(
                    vehicle_type == "unknown"
                    for vehicle_type in self.vehicle_type_registry.values()
                ),
            }
        )
        return rows

    # ------------------------------------------------------------------
    # Finalization and output
    # ------------------------------------------------------------------

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with path.open("w", newline="", encoding="utf-8-sig") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

    def print_episode_completed(
        self,
        *,
        wall_time_s: float,
        total_episodes: int,
    ) -> None:
        """Print one concise progress line after an episode is finalized."""
        print(
            f"[eval] scenario={self.case_name} "
            f"episode={self.episode}/{int(total_episodes)} "
            f"wall={float(wall_time_s):.1f}s",
            flush=True,
        )

    def finalize_episode(
        self,
        *,
        tripinfo_path: str | os.PathLike[str],
        traffic_stats: Any = None,
    ) -> dict[str, Any]:
        self.detach_from_env()
        tripinfo_totals = self._apply_tripinfo_corrections(tripinfo_path)

        max_interval = max(self.network_intervals, default=-1)
        for interval_index in range(max_interval + 1):
            self._network_acc(interval_index)
            for detector_id in self.e2_meta:
                self._e2_acc(interval_index, detector_id)

        self.network_rows_10s = [
            self._network_10s_row(index, self.network_intervals[index])
            for index in sorted(self.network_intervals)
        ]
        self.e2_rows_10s = [
            self._e2_10s_row(index, detector_id, acc)
            for (index, detector_id), acc in sorted(
                self.e2_intervals.items()
            )
        ]
        self.network_rows_300s = self._aggregate_network_300s()
        self.e2_rows_300s = self._aggregate_e2_300s()
        self.e2_vehicle_rows_300s = self._build_e2_vehicle_300s()
        self.vehicle_rows_10s = (
            self._build_vehicle_10s()
            if self.config.save_vehicle_10s
            else []
        )
        self.quality_control_rows = self._build_quality_control(
            tripinfo_totals
        )

        if self.config.save_network_10s:
            self._write_csv(
                self.episode_dir / "network_10s.csv",
                self.network_rows_10s,
            )
        if self.config.save_e2_detector_10s:
            self._write_csv(
                self.episode_dir / "e2_detector_10s.csv",
                self.e2_rows_10s,
            )
        if self.config.save_network_300s:
            self._write_csv(
                self.episode_dir / "network_300s.csv",
                self.network_rows_300s,
            )
        if self.config.save_e2_detector_300s:
            self._write_csv(
                self.episode_dir / "e2_detector_300s.csv",
                self.e2_rows_300s,
            )
        if self.config.save_e2_vehicle_300s:
            self._write_csv(
                self.episode_dir / "e2_vehicle_300s.csv",
                self.e2_vehicle_rows_300s,
            )
        if self.config.save_vehicle_10s:
            self._write_csv(
                self.episode_dir / "vehicle_10s.csv",
                self.vehicle_rows_10s,
            )
        if self.config.save_decision_actions:
            self._write_csv(
                self.episode_dir / "decision_action.csv",
                self.decision_action_rows,
            )
        self._write_csv(
            self.episode_dir / "quality_control.csv",
            self.quality_control_rows,
        )

        analysis_rows = [
            row
            for row in self.network_rows_300s
            if int(row["is_analysis_window"]) == 1
        ]
        self.summary = {
            **self._base_metadata(),
            "episode_dir": str(self.episode_dir),
            "final_sim_time_s": self.last_sim_time_s,
            "network_10s_row_count": len(self.network_rows_10s),
            "network_300s_row_count": len(self.network_rows_300s),
            "analysis_window_count": len(analysis_rows),
            "analysis_network_nox_mg": sum(
                float(row["network_nox_mg"]) for row in analysis_rows
            ),
            "analysis_network_time_loss_s": sum(
                float(row["network_time_loss_s"]) for row in analysis_rows
            ),
            "analysis_network_distance_m": sum(
                float(row["network_distance_m"]) for row in analysis_rows
            ),
            "analysis_arrived_vehicle_count": sum(
                int(row["arrived_vehicle_count"]) for row in analysis_rows
            ),
            "analysis_throughput_vehicle_count": sum(
                int(row["throughput_vehicle_count"]) for row in analysis_rows
            ),
            "tripinfo_completed_vehicle_count": int(
                tripinfo_totals.get("completed_trip_count", 0.0)
            ),
            "tripinfo_unfinished_vehicle_count": int(
                tripinfo_totals.get("unfinished_trip_count", 0.0)
            ),
            "avg_delay_s": float(
                getattr(traffic_stats, "avg_delay_s", 0.0) or 0.0
            ),
            "avg_travel_time_s": float(
                getattr(traffic_stats, "avg_travel_time_s", 0.0) or 0.0
            ),
            "total_arrived": int(
                getattr(traffic_stats, "total_arrived", 0) or 0
            ),
            "completion_rate": float(
                getattr(traffic_stats, "completion_rate", 0.0) or 0.0
            ),
        }
        return dict(self.summary)
