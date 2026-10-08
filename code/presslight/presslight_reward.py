# -*- coding: utf-8 -*-
"""Original PressLight pressure reward and optional turn-ratio adaptation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from presslight_config import PressLightConfig
from presslight_network import (
    PressLightIntersectionSpec,
    PressLightMovement,
    PressLightNetworkSpec,
)
from presslight_observation import PressLightObservation


@dataclass
class PressLightRewardResult:
    tl_id: str
    reward: float
    intersection_pressure: float
    signed_pressure_sum: float
    sum_abs_movement_pressure: float
    movement_pressures: dict[str, float]
    turn_ratios: dict[str, float]
    beta_valid: bool
    movement_details: dict[str, dict[str, Any]] = field(default_factory=dict)


class PressLightRewardBuilder:
    """Compute ``r_t = -P(s_{t+1})``.

    The caller is responsible for passing the observation collected *after*
    SUMO has executed the selected action for one decision interval.
    """

    def __init__(self, cfg: PressLightConfig, network_spec: PressLightNetworkSpec) -> None:
        self.cfg = cfg
        self.network_spec = network_spec
        self._turn_ratio_prior: dict[tuple[str, str], dict[str, float]] = {}
        self._initialize_priors()

    def _initialize_priors(self) -> None:
        for tl_id, spec in self.network_spec.intersections.items():
            by_from: dict[str, list[PressLightMovement]] = defaultdict(list)
            for movement in spec.movements:
                by_from[movement.from_lane].append(movement)
            for from_lane, movements in by_from.items():
                uniform = 1.0 / max(len(movements), 1)
                self._turn_ratio_prior[(tl_id, from_lane)] = {
                    movement.movement_id: uniform for movement in movements
                }

    @staticmethod
    def _next_route_edge(traci: Any, vehicle_id: str) -> str:
        try:
            route = [str(x) for x in traci.vehicle.getRoute(vehicle_id)]
            route_index = int(traci.vehicle.getRouteIndex(vehicle_id))
        except Exception:
            return ""
        if route_index < 0 or route_index + 1 >= len(route):
            return ""
        return route[route_index + 1]

    def _turn_ratios_for_lane(
        self,
        env: Any,
        tl_id: str,
        obs: PressLightObservation,
        from_lane: str,
        movements: list[PressLightMovement],
    ) -> tuple[dict[str, float], bool]:
        if len(movements) == 1:
            return {movements[0].movement_id: 1.0}, True

        prior_key = (tl_id, from_lane)
        prior = dict(self._turn_ratio_prior[prior_key])
        vehicle_ids = obs.incoming_vehicle_ids.get(from_lane, [])
        traci = getattr(env, "_traci", None)

        # Route information identifies the next edge, not the exact target
        # lane.  If multiple movement connections enter the same edge, divide
        # that edge's count evenly so that all beta values still sum to one.
        by_edge: dict[str, list[PressLightMovement]] = defaultdict(list)
        for movement in movements:
            by_edge[movement.to_edge].append(movement)
        edge_counts: dict[str, float] = defaultdict(float)
        if traci is not None:
            for vehicle_id in vehicle_ids:
                next_edge = self._next_route_edge(traci, vehicle_id)
                if next_edge in by_edge:
                    edge_counts[next_edge] += 1.0

        movement_counts = {movement.movement_id: 0.0 for movement in movements}
        for edge_id, count in edge_counts.items():
            edge_movements = by_edge[edge_id]
            share = float(count) / max(len(edge_movements), 1)
            for movement in edge_movements:
                movement_counts[movement.movement_id] += share

        observed_total = float(sum(movement_counts.values()))
        alpha = float(self.cfg.pressure.beta_pseudocount)
        denominator = observed_total + alpha
        if denominator <= 0.0:
            beta = prior
        else:
            beta = {
                movement.movement_id: (
                    float(movement_counts[movement.movement_id])
                    + alpha * float(prior[movement.movement_id])
                )
                / denominator
                for movement in movements
            }

        beta_sum = float(sum(beta.values()))
        tolerance = float(self.cfg.pressure.beta_tolerance)
        valid = bool(abs(beta_sum - 1.0) <= tolerance and all(v >= 0.0 for v in beta.values()))
        if not valid and beta_sum > 0.0:
            beta = {key: float(value / beta_sum) for key, value in beta.items()}
            valid = abs(sum(beta.values()) - 1.0) <= tolerance

        if observed_total > 0.0:
            rate = float(self.cfg.pressure.prior_update_rate)
            updated = {
                key: (1.0 - rate) * float(prior[key]) + rate * float(beta[key])
                for key in beta
            }
            updated_sum = sum(updated.values())
            if updated_sum > 0.0:
                updated = {key: value / updated_sum for key, value in updated.items()}
            self._turn_ratio_prior[prior_key] = updated
        return beta, valid

    def compute_one(
        self,
        env: Any,
        obs: PressLightObservation,
    ) -> PressLightRewardResult:
        tl_id = obs.tl_id
        spec: PressLightIntersectionSpec = self.network_spec.intersections[tl_id]

        incoming_density = {
            lane_id: float(obs.incoming_total[slot])
            / float(spec.incoming_observation_capacity[lane_id])
            for lane_id, slot in spec.incoming_lane_to_slot.items()
        }
        outgoing_density = {
            lane_id: float(obs.outgoing_count[slot])
            / float(spec.outgoing_capacity[lane_id])
            for lane_id, slot in spec.outgoing_lane_to_slot.items()
        }

        movements_by_from: dict[str, list[PressLightMovement]] = defaultdict(list)
        for movement in spec.movements:
            movements_by_from[movement.from_lane].append(movement)

        turn_ratios: dict[str, float] = {}
        beta_valid = True
        if self.cfg.pressure.variant == "turn_ratio":
            for from_lane, movements in movements_by_from.items():
                lane_beta, lane_valid = self._turn_ratios_for_lane(
                    env, tl_id, obs, from_lane, movements
                )
                turn_ratios.update(lane_beta)
                beta_valid = beta_valid and lane_valid
        else:
            turn_ratios = {movement.movement_id: 1.0 for movement in spec.movements}

        movement_pressures: dict[str, float] = {}
        details: dict[str, dict[str, Any]] = {}
        signed_pressure_sum = 0.0
        sum_abs_movement_pressure = 0.0
        for movement in spec.movements:
            rho_in = float(incoming_density[movement.from_lane])
            rho_out = float(outgoing_density[movement.to_lane])
            beta = float(turn_ratios[movement.movement_id])
            unweighted = rho_in - rho_out
            pressure = (
                beta * unweighted
                if self.cfg.pressure.variant == "turn_ratio"
                else unweighted
            )
            movement_pressures[movement.movement_id] = float(pressure)
            signed_pressure_sum += float(pressure)
            sum_abs_movement_pressure += abs(float(pressure))
            details[movement.movement_id] = {
                "from_lane": movement.from_lane,
                "to_lane": movement.to_lane,
                "to_edge": movement.to_edge,
                "link_index": movement.link_index,
                "direction": movement.direction,
                "phase_indices": list(movement.phase_indices),
                "canonical_phase_indices": list(movement.phase_indices),
                "local_phase_indices": list(movement.local_phase_indices),
                "incoming_density": rho_in,
                "outgoing_density": rho_out,
                "beta": beta,
                "unweighted_pressure": unweighted,
                "movement_pressure": float(pressure),
            }

        # Original PressLight Eq. (2): absolute value after summing movements.
        intersection_pressure = abs(float(signed_pressure_sum))
        reward = -float(self.cfg.pressure.reward_scale) * intersection_pressure
        return PressLightRewardResult(
            tl_id=tl_id,
            reward=reward,
            intersection_pressure=intersection_pressure,
            signed_pressure_sum=float(signed_pressure_sum),
            sum_abs_movement_pressure=float(sum_abs_movement_pressure),
            movement_pressures=movement_pressures,
            turn_ratios=turn_ratios,
            beta_valid=bool(beta_valid),
            movement_details=details,
        )

    def compute_all(
        self,
        env: Any,
        next_observations: dict[str, PressLightObservation],
    ) -> dict[str, PressLightRewardResult]:
        return {
            tl_id: self.compute_one(env, obs)
            for tl_id, obs in next_observations.items()
        }


def pressure_step_rows(
    episode: int,
    step: int,
    results: dict[str, PressLightRewardResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tl_id, result in results.items():
        for movement_id, detail in result.movement_details.items():
            rows.append(
                {
                    "episode": int(episode),
                    "step": int(step),
                    "tl_id": tl_id,
                    "movement_id": movement_id,
                    **detail,
                    "intersection_pressure": result.intersection_pressure,
                    "signed_pressure_sum": result.signed_pressure_sum,
                    "sum_abs_movement_pressure": result.sum_abs_movement_pressure,
                    "reward": result.reward,
                    "beta_valid": result.beta_valid,
                }
            )
    return rows
