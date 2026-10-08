# -*- coding: utf-8 -*-
"""Static SUMO-to-PressLight mapping.

This file contains road-network relations only.  Neural networks and replay
memory live in :mod:`presslight_agent`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from network_parser import NetworkInfo, NetworkParser
from presslight_config import PressLightConfig, SEGMENT_NAMES


@dataclass(frozen=True)
class PressLightMovement:
    movement_id: str
    from_lane: str
    to_lane: str
    to_edge: str
    link_index: int
    direction: str
    phase_indices: tuple[int, ...]
    local_phase_indices: tuple[int, ...] = ()


@dataclass
class PressLightIntersectionSpec:
    tl_id: str
    group_id: str
    structure_signature: str
    canonical_incoming_lanes: list[str]
    canonical_outgoing_lanes: list[str]
    local_incoming_lanes: list[str]
    local_outgoing_lanes: list[str]
    canonical_to_local_phase: list[int]
    local_to_canonical_phase: list[int]
    local_phase_count: int
    canonical_phase_count: int
    canonical_rotation: int
    movements: list[PressLightMovement]
    phase_to_movements: dict[int, list[str]]
    incoming_segment_detectors: dict[str, dict[str, str]]
    incoming_observation_capacity: dict[str, int]
    outgoing_capacity: dict[str, int]
    shared_incoming_lanes: list[str]
    incoming_lane_to_slot: dict[str, int]
    outgoing_lane_to_slot: dict[str, int]

    @property
    def incoming_lanes(self) -> list[str]:
        return self.canonical_incoming_lanes

    @property
    def outgoing_lanes(self) -> list[str]:
        return self.canonical_outgoing_lanes


@dataclass
class PressLightGroupSpec:
    group_id: str
    structure_signature: str
    tl_ids: list[str]
    n_incoming_lanes: int
    n_outgoing_lanes: int
    n_phases: int
    input_dim: int
    action_dim: int


@dataclass
class PressLightNetworkSpec:
    intersections: dict[str, PressLightIntersectionSpec]
    group_to_tls: dict[str, list[str]]
    tl_to_group: dict[str, str]
    group_specs: dict[str, PressLightGroupSpec]
    max_incoming_lanes: int
    max_outgoing_lanes: int
    max_phases: int
    source_net_xml: str
    source_detector_map: str
    digest: str = ""

    @property
    def intersection_ids(self) -> list[str]:
        return sorted(self.intersections)

    @property
    def group_ids(self) -> list[str]:
        return sorted(self.group_to_tls)

    def summary(self) -> dict[str, Any]:
        intersections: dict[str, Any] = {}
        for tl_id in self.intersection_ids:
            item = self.intersections[tl_id]
            intersections[tl_id] = {
                "group_id": item.group_id,
                "structure_signature": item.structure_signature,
                "canonical_incoming_lanes": item.canonical_incoming_lanes,
                "canonical_outgoing_lanes": item.canonical_outgoing_lanes,
                "local_incoming_lanes": item.local_incoming_lanes,
                "local_outgoing_lanes": item.local_outgoing_lanes,
                "canonical_to_local_phase": item.canonical_to_local_phase,
                "local_to_canonical_phase": item.local_to_canonical_phase,
                "canonical_rotation": item.canonical_rotation,
                "n_phases": item.canonical_phase_count,
                "n_movements": len(item.movements),
                "shared_incoming_lanes": item.shared_incoming_lanes,
                "incoming_segment_detectors": item.incoming_segment_detectors,
                "incoming_observation_capacity": item.incoming_observation_capacity,
                "outgoing_capacity": item.outgoing_capacity,
            }
        return {
            "digest": self.digest,
            "source_net_xml": self.source_net_xml,
            "source_detector_map": self.source_detector_map,
            "max_incoming_lanes": self.max_incoming_lanes,
            "max_outgoing_lanes": self.max_outgoing_lanes,
            "max_phases": self.max_phases,
            "group_to_tls": self.group_to_tls,
            "group_specs": {
                key: {
                    "group_id": value.group_id,
                    "structure_signature": value.structure_signature,
                    "tl_ids": value.tl_ids,
                    "n_incoming_lanes": value.n_incoming_lanes,
                    "n_outgoing_lanes": value.n_outgoing_lanes,
                    "n_phases": value.n_phases,
                    "input_dim": value.input_dim,
                    "action_dim": value.action_dim,
                }
                for key, value in sorted(self.group_specs.items())
            },
            "intersections": intersections,
        }


class _LenientNetworkParser(NetworkParser):
    """Allow PressLight to rebuild groups from its own structure signature."""

    def _validate_group_structures(self, net_info: NetworkInfo) -> None:  # noqa: D401
        return None


def parse_base_network(
    net_xml: str,
    base_add_xml: str,
    groups_json: str = "",
) -> NetworkInfo:
    """Parse the common network without forcing MGMQ group compatibility."""

    parser = _LenientNetworkParser(
        net_xml=net_xml,
        add_xml=base_add_xml,
        groups_json=groups_json or None,
    )
    return parser.parse()


def _local_name(tag: str) -> str:
    return str(tag).split("}", 1)[-1]


def _direction(cx: float, cy: float, ox: float, oy: float) -> str:
    dx, dy = ox - cx, oy - cy
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else "W"
    return "N" if dy > 0 else "S"


def _junction_coordinates(net_xml: str) -> dict[str, tuple[float, float]]:
    root = ET.parse(net_xml).getroot()
    result: dict[str, tuple[float, float]] = {}
    for elem in root.iter():
        if _local_name(elem.tag) != "junction":
            continue
        node_id = str(elem.get("id", ""))
        result[node_id] = (float(elem.get("x", 0.0)), float(elem.get("y", 0.0)))
    return result


def _normalize_segment_name(value: str) -> str:
    raw = str(value).strip().lower().replace("-", "_")
    aliases = {
        "near": "near", "nearest": "near", "close": "near", "0": "near", "1": "near",
        "middle": "middle", "mid": "middle", "1_middle": "middle", "2": "middle",
        "far": "far", "farthest": "far", "2_far": "far", "3": "far",
    }
    if raw in aliases:
        return aliases[raw]
    for name in SEGMENT_NAMES:
        if name in raw:
            return name
    return ""


def _extract_map_candidate(payload: Any) -> dict[str, dict[str, str]]:
    """Accept common detector-map JSON layouts."""

    if not isinstance(payload, Mapping):
        return {}
    for key in ("lane_segments", "lane_to_detectors", "detectors_by_lane", "lanes"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            payload = candidate
            break

    result: dict[str, dict[str, str]] = {}
    for lane_key, value in payload.items():
        lane_id = str(lane_key)
        if isinstance(value, Mapping):
            # Layout A: lane -> {near: detector_id, ...}
            segment_map: dict[str, str] = {}
            for segment_key, detector_value in value.items():
                segment = _normalize_segment_name(str(segment_key))
                if isinstance(detector_value, Mapping):
                    detector_id = detector_value.get("id", detector_value.get("detector_id", ""))
                    lane_id = str(detector_value.get("lane", lane_id))
                else:
                    detector_id = detector_value
                if segment and detector_id:
                    segment_map[segment] = str(detector_id)
            if segment_map:
                result.setdefault(lane_id, {}).update(segment_map)
                continue

            # Layout B: detector_id -> {lane, segment}
            detector_id = str(lane_key)
            actual_lane = str(value.get("lane", value.get("lane_id", "")))
            segment = _normalize_segment_name(str(value.get("segment", value.get("name", ""))))
            if actual_lane and segment:
                result.setdefault(actual_lane, {})[segment] = detector_id
        elif isinstance(value, (list, tuple)) and len(value) == 3:
            result[lane_id] = {
                segment: str(detector_id)
                for segment, detector_id in zip(SEGMENT_NAMES, value)
            }
    return result


def _detectors_from_add_xml(add_xml: str) -> dict[str, dict[str, str]]:
    if not add_xml or not os.path.exists(add_xml):
        return {}
    root = ET.parse(add_xml).getroot()
    by_lane: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    for elem in root.iter():
        if _local_name(elem.tag).lower() not in {"laneareadetector", "e2detector"}:
            continue
        detector_id = str(elem.get("id", ""))
        lane_id = str(elem.get("lane", ""))
        if not detector_id or not lane_id:
            continue
        segment = _normalize_segment_name(detector_id)
        pos = float(elem.get("pos", 0.0))
        by_lane[lane_id].append((detector_id, pos, segment))

    result: dict[str, dict[str, str]] = {}
    for lane_id, records in by_lane.items():
        explicit = {segment: det for det, _pos, segment in records if segment}
        if len(explicit) == 3:
            result[lane_id] = explicit
            continue
        # For both negative positions measured from lane end and ordinary
        # positive positions, the detector with the greatest start position is
        # closest to the stop line.
        ordered = sorted(records, key=lambda item: item[1], reverse=True)
        if len(ordered) >= 3:
            result[lane_id] = {
                segment: ordered[index][0]
                for index, segment in enumerate(SEGMENT_NAMES)
            }
    return result


def load_segment_detector_map(
    detector_map_json: str,
    presslight_add_xml: str,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    if detector_map_json and os.path.exists(detector_map_json):
        with open(detector_map_json, "r", encoding="utf-8") as file:
            result.update(_extract_map_candidate(json.load(file)))
    xml_map = _detectors_from_add_xml(presslight_add_xml)
    for lane_id, segments in xml_map.items():
        result.setdefault(lane_id, {}).update(
            {key: value for key, value in segments.items() if key not in result.get(lane_id, {})}
        )
    return result


def _lane_index(net_info: NetworkInfo, lane_id: str) -> int:
    lane = net_info.lane_info.get(lane_id)
    if lane is not None and hasattr(lane, "index"):
        return int(lane.index)
    try:
        return int(str(lane_id).rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _edge_angle(
    tl_id: str,
    edge_id: str,
    net_info: NetworkInfo,
    coordinates: Mapping[str, tuple[float, float]],
) -> float:
    edge = net_info.edge_info.get(edge_id)
    iinfo = net_info.get_intersection(tl_id)
    cx, cy = coordinates.get(tl_id, (float(iinfo.x), float(iinfo.y)))
    if edge is None:
        return 0.0
    other_node = edge.from_node if edge.to_node == tl_id else edge.to_node
    ox, oy = coordinates.get(str(other_node), (cx, cy))
    # Clockwise bearing starting at north.  IDs are never used for structure.
    return float(math.atan2(ox - cx, oy - cy) % (2.0 * math.pi))


def _controlled_approaches(
    tl_id: str,
    iinfo: Any,
    net_info: NetworkInfo,
    coordinates: Mapping[str, tuple[float, float]],
    incoming: bool,
) -> list[list[str]]:
    by_edge: dict[str, set[str]] = defaultdict(set)
    for conn in iinfo.connections:
        if incoming:
            edge_id = str(conn.from_edge)
            lane_id = str(conn.from_lane)
        else:
            edge_id = str(conn.to_edge)
            lane_id = f"{edge_id}_{int(conn.to_lane_idx)}"
        by_edge[edge_id].add(lane_id)
    ordered_edges = sorted(
        by_edge,
        key=lambda edge_id: (_edge_angle(tl_id, edge_id, net_info, coordinates), edge_id),
    )
    return [
        sorted(by_edge[edge_id], key=lambda lane: (_lane_index(net_info, lane), lane))
        for edge_id in ordered_edges
    ]


def _rotate(items: list[list[str]], shift: int, reflected: bool = False) -> list[list[str]]:
    ordered = list(reversed(items)) if reflected else list(items)
    if not ordered:
        return []
    shift %= len(ordered)
    return ordered[shift:] + ordered[:shift]


def _canonicalize_intersection(
    iinfo: Any,
    incoming_approaches: list[list[str]],
    outgoing_approaches: list[list[str]],
    raw_movements: list[PressLightMovement],
    cfg: PressLightConfig,
) -> dict[str, Any]:
    n_phases = len(iinfo.green_phases)
    rotations = range(max(len(incoming_approaches), 1)) if cfg.grouping.canonicalize_rotation else range(1)
    reflections = (False, True) if cfg.grouping.allow_reflection else (False,)
    candidates: list[tuple[str, dict[str, Any]]] = []
    for reflected in reflections:
        for shift in rotations:
            in_apps = _rotate(incoming_approaches, shift, reflected)
            out_shift = shift if len(outgoing_approaches) == len(incoming_approaches) else 0
            out_apps = _rotate(outgoing_approaches, out_shift, reflected)
            incoming = [lane for approach in in_apps for lane in approach]
            outgoing = [lane for approach in out_apps for lane in approach]
            in_slot = {lane: index for index, lane in enumerate(incoming)}
            out_slot = {lane: index for index, lane in enumerate(outgoing)}
            phase_patterns: list[tuple[tuple[Any, ...], int]] = []
            for local_phase in range(n_phases):
                phase_state = str(getattr(iinfo.green_phases[local_phase], "state", ""))
                pattern = tuple(sorted(
                    (
                        in_slot[m.from_lane], out_slot[m.to_lane], str(m.direction).lower(),
                        phase_state[m.link_index] if 0 <= m.link_index < len(phase_state) else "",
                    )
                    for m in raw_movements if local_phase in m.local_phase_indices
                ))
                phase_patterns.append((pattern, local_phase))
            phase_patterns.sort(key=lambda item: (item[0], item[1]))
            canonical_to_local = [local for _pattern, local in phase_patterns]
            local_to_canonical = [0] * n_phases
            for canonical, local in enumerate(canonical_to_local):
                local_to_canonical[local] = canonical
            movement_pattern = sorted(
                (
                    in_slot[m.from_lane], out_slot[m.to_lane], str(m.direction).lower(),
                    tuple(sorted(local_to_canonical[p] for p in m.local_phase_indices)),
                )
                for m in raw_movements
            )
            lane_turn_sets = [
                sorted({str(m.direction).lower() for m in raw_movements if m.from_lane == lane})
                for lane in incoming
            ]
            phase_movement_matrix = [
                sorted(index for index, movement in enumerate(movement_pattern) if phase in movement[3])
                for phase in range(n_phases)
            ]
            representation = {
                "incoming_approach_lane_counts": [len(x) for x in in_apps],
                "outgoing_approach_lane_counts": [len(x) for x in out_apps],
                "n_incoming_lanes": len(incoming),
                "n_outgoing_lanes": len(outgoing),
                "n_phases": n_phases,
                "lane_turn_sets": lane_turn_sets,
                "shared_lane_slots": [i for i, turns in enumerate(lane_turn_sets) if len(turns) > 1],
                "movements": movement_pattern,
                "phase_movement_matrix": phase_movement_matrix,
                "phase_signal_patterns": [pattern for pattern, _local in phase_patterns],
            }
            signature = json.dumps(representation, sort_keys=True, separators=(",", ":"))
            candidates.append((signature, {
                "incoming": incoming, "outgoing": outgoing,
                "canonical_to_local": canonical_to_local,
                "local_to_canonical": local_to_canonical,
                "rotation": int(shift), "reflected": reflected,
            }))
    signature, selected = min(candidates, key=lambda item: item[0])
    selected["signature"] = signature
    return selected


def _manual_group_map(path: str, intersection_ids: set[str]) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, Mapping) and isinstance(payload.get("groups"), Mapping):
        payload = payload["groups"]
    result: dict[str, str] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if isinstance(value, (list, tuple)):
                for tl_id in value:
                    if str(tl_id) in result:
                        raise ValueError(f"Manual grouping repeats intersection {tl_id!r}")
                    result[str(tl_id)] = str(key)
            else:
                result[str(key)] = str(value)
    unknown = set(result) - intersection_ids
    missing = intersection_ids - set(result)
    if unknown:
        raise ValueError(f"Manual grouping contains unknown intersections: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Manual grouping does not cover intersections: {sorted(missing)}")
    return result


def build_presslight_network_spec(
    net_info: NetworkInfo,
    cfg: PressLightConfig,
) -> PressLightNetworkSpec:
    coordinates = _junction_coordinates(cfg.paths.net_xml)
    detector_map = load_segment_detector_map(
        cfg.paths.detector_map_json,
        cfg.paths.presslight_add_xml,
    )

    raw_items: dict[str, dict[str, Any]] = {}
    for tl_id in sorted(net_info.intersection_ids):
        iinfo = net_info.get_intersection(tl_id)
        incoming_approaches = _controlled_approaches(tl_id, iinfo, net_info, coordinates, True)
        outgoing_approaches = _controlled_approaches(tl_id, iinfo, net_info, coordinates, False)
        local_incoming = [lane for approach in incoming_approaches for lane in approach]
        local_outgoing = [lane for approach in outgoing_approaches for lane in approach]
        for name, actual, limit in (
            ("incoming lanes", len(local_incoming), cfg.state.max_allowed_incoming_lanes),
            ("outgoing lanes", len(local_outgoing), cfg.state.max_allowed_outgoing_lanes),
            ("phases", len(iinfo.green_phases), cfg.state.max_allowed_phases),
        ):
            if limit is not None and actual > limit:
                raise ValueError(f"{tl_id}: {actual} {name} exceed configured safety limit {limit}")

        movements: list[PressLightMovement] = []
        for conn in iinfo.connections:
            from_lane = str(conn.from_lane)
            to_lane = f"{conn.to_edge}_{int(conn.to_lane_idx)}"
            if from_lane not in local_incoming or to_lane not in local_outgoing:
                continue
            local_phase_indices = tuple(
                phase_index
                for phase_index, phase in enumerate(iinfo.green_phases)
                if int(conn.link_index) in set(phase.green_link_indices)
            )
            movements.append(
                PressLightMovement(
                    movement_id=f"{tl_id}:{int(conn.link_index)}:{from_lane}->{to_lane}",
                    from_lane=from_lane,
                    to_lane=to_lane,
                    to_edge=str(conn.to_edge),
                    link_index=int(conn.link_index),
                    direction=str(conn.direction),
                    phase_indices=(),
                    local_phase_indices=local_phase_indices,
                )
            )
        canonical = _canonicalize_intersection(
            iinfo, incoming_approaches, outgoing_approaches, movements, cfg
        )
        movements = [PressLightMovement(
            movement_id=m.movement_id, from_lane=m.from_lane, to_lane=m.to_lane,
            to_edge=m.to_edge, link_index=m.link_index, direction=m.direction,
            phase_indices=tuple(sorted(canonical["local_to_canonical"][p] for p in m.local_phase_indices)),
            local_phase_indices=m.local_phase_indices,
        ) for m in movements]
        raw_items[tl_id] = {
            "iinfo": iinfo,
            "local_incoming": local_incoming,
            "local_outgoing": local_outgoing,
            "incoming_lanes": canonical["incoming"],
            "outgoing_lanes": canonical["outgoing"],
            "movements": movements,
            "signature": canonical["signature"],
            "canonical": canonical,
        }
    manual_map: dict[str, str] = {}
    if cfg.grouping.mode == "manual":
        manual_map = _manual_group_map(cfg.grouping.manual_groups_json, set(raw_items))
    intersections: dict[str, PressLightIntersectionSpec] = {}
    group_to_tls: dict[str, list[str]] = defaultdict(list)
    tl_to_group: dict[str, str] = {}

    for tl_id, item in raw_items.items():
        signature = str(item["signature"])
        if cfg.grouping.mode == "independent":
            group_id = tl_id
        elif cfg.grouping.mode == "manual":
            group_id = str(manual_map[tl_id])
        else:
            group_id = f"pl_struct_{hashlib.sha256(signature.encode()).hexdigest()[:12]}"

        incoming_lanes = item["incoming_lanes"]
        outgoing_lanes = item["outgoing_lanes"]
        movements = item["movements"]
        missing_detectors = [
            lane_id
            for lane_id in incoming_lanes
            if set(detector_map.get(lane_id, {})) != set(SEGMENT_NAMES)
        ]
        if missing_detectors and cfg.state.require_all_segment_detectors:
            preview = ", ".join(missing_detectors[:8])
            raise ValueError(
                f"{tl_id}: missing near/middle/far E2 mappings for "
                f"{len(missing_detectors)} incoming lanes: {preview}"
            )

        phase_to_movements = {
            phase_index: [m.movement_id for m in movements if phase_index in m.phase_indices]
            for phase_index in range(len(item["canonical"]["canonical_to_local"]))
        }
        movements_by_lane: dict[str, list[PressLightMovement]] = defaultdict(list)
        for movement in movements:
            movements_by_lane[movement.from_lane].append(movement)
        shared_lanes = sorted(
            lane_id
            for lane_id, lane_movements in movements_by_lane.items()
            if len(lane_movements) > 1
        )

        in_capacity: dict[str, int] = {}
        for lane_id in incoming_lanes:
            lane = net_info.lane_info[lane_id]
            observed_length = min(
                float(lane.length),
                float(cfg.pressure.incoming_observation_length_m),
            )
            in_capacity[lane_id] = max(
                1,
                int(math.floor(observed_length / cfg.pressure.effective_vehicle_length_m)),
            )
        out_capacity = {
            lane_id: max(
                1,
                int(
                    math.floor(
                        float(net_info.lane_info[lane_id].length)
                        / cfg.pressure.effective_vehicle_length_m
                    )
                ),
            )
            for lane_id in outgoing_lanes
        }

        spec = PressLightIntersectionSpec(
            tl_id=tl_id,
            group_id=group_id,
            structure_signature=signature,
            canonical_incoming_lanes=incoming_lanes,
            canonical_outgoing_lanes=outgoing_lanes,
            local_incoming_lanes=item["local_incoming"],
            local_outgoing_lanes=item["local_outgoing"],
            canonical_to_local_phase=list(item["canonical"]["canonical_to_local"]),
            local_to_canonical_phase=list(item["canonical"]["local_to_canonical"]),
            local_phase_count=len(item["canonical"]["local_to_canonical"]),
            canonical_phase_count=len(item["canonical"]["canonical_to_local"]),
            canonical_rotation=int(item["canonical"]["rotation"]),
            movements=movements,
            phase_to_movements=phase_to_movements,
            incoming_segment_detectors={
                lane_id: dict(detector_map.get(lane_id, {}))
                for lane_id in incoming_lanes
            },
            incoming_observation_capacity=in_capacity,
            outgoing_capacity=out_capacity,
            shared_incoming_lanes=shared_lanes,
            incoming_lane_to_slot={lane_id: i for i, lane_id in enumerate(incoming_lanes)},
            outgoing_lane_to_slot={lane_id: i for i, lane_id in enumerate(outgoing_lanes)},
        )
        intersections[tl_id] = spec
        group_to_tls[group_id].append(tl_id)
        tl_to_group[tl_id] = group_id

    for group_id, members in group_to_tls.items():
        signatures = {intersections[tl_id].structure_signature for tl_id in members}
        if len(signatures) != 1:
            raise ValueError(f"Group {group_id!r} contains non-isomorphic intersections: {members}")

    group_specs = {}
    for group_id, members in sorted(group_to_tls.items()):
        first = intersections[members[0]]
        n_in = len(first.canonical_incoming_lanes)
        n_out = len(first.canonical_outgoing_lanes)
        n_phases = first.canonical_phase_count
        group_specs[group_id] = PressLightGroupSpec(
            group_id=group_id, structure_signature=first.structure_signature,
            tl_ids=sorted(members), n_incoming_lanes=n_in, n_outgoing_lanes=n_out,
            n_phases=n_phases, input_dim=n_phases + 3 * n_in + n_out,
            action_dim=n_phases,
        )

    network_spec = PressLightNetworkSpec(
        intersections=intersections,
        group_to_tls={key: sorted(value) for key, value in sorted(group_to_tls.items())},
        tl_to_group=tl_to_group,
        group_specs=group_specs,
        max_incoming_lanes=max((len(x.canonical_incoming_lanes) for x in intersections.values()), default=0),
        max_outgoing_lanes=max((len(x.canonical_outgoing_lanes) for x in intersections.values()), default=0),
        max_phases=max((x.canonical_phase_count for x in intersections.values()), default=0),
        source_net_xml=cfg.paths.net_xml,
        source_detector_map=cfg.paths.detector_map_json,
    )
    digest_payload = network_spec.summary()
    digest_payload.pop("digest", None)
    network_spec.digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return network_spec


def save_network_spec(spec: PressLightNetworkSpec, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(spec.summary(), file, indent=2, ensure_ascii=False)


def canonical_actions_to_env_actions(
    canonical_actions: Mapping[str, int],
    network_spec: PressLightNetworkSpec,
) -> dict[str, int]:
    env_actions: dict[str, int] = {}
    for tl_id, canonical_action in canonical_actions.items():
        spec = network_spec.intersections[tl_id]
        action = int(canonical_action)
        if not 0 <= action < spec.canonical_phase_count:
            raise ValueError(
                f"{tl_id}: canonical action must be in [0, {spec.canonical_phase_count}); actual={action}"
            )
        env_actions[tl_id] = int(spec.canonical_to_local_phase[action])
    return env_actions


def prepare_presslight_runtime_sumo_cfg(
    sumo_cfg: str,
    presslight_add_xml: str,
    output_dir: str,
) -> str:
    """Attach a sanitized PressLight additional file to a SUMO config.

    Detector XML output is redirected to the null device.  TraCI values remain
    available, while sequential validation and parallel evaluation cannot
    collide on one detector output file.
    """

    if not presslight_add_xml:
        return sumo_cfg
    if not os.path.exists(presslight_add_xml):
        raise FileNotFoundError(
            f"PressLight detector additional file not found: {presslight_add_xml}"
        )
    os.makedirs(output_dir, exist_ok=True)
    null_path = "NUL" if os.name == "nt" else "/dev/null"

    add_tree = ET.parse(presslight_add_xml)
    for elem in add_tree.getroot().iter():
        if _local_name(elem.tag).lower() in {
            "laneareadetector", "e2detector", "inductionloop", "e1detector"
        } and "file" in elem.attrib:
            elem.set("file", null_path)
    runtime_add = os.path.abspath(os.path.join(output_dir, "presslight_e2_runtime.add.xml"))
    add_tree.write(runtime_add, encoding="utf-8", xml_declaration=True)

    cfg_tree = ET.parse(sumo_cfg)
    cfg_root = cfg_tree.getroot()
    input_node = next(
        (child for child in cfg_root if _local_name(child.tag).lower() == "input"),
        None,
    )
    if input_node is None:
        input_node = ET.SubElement(cfg_root, "input")
    additional_node = next(
        (
            child
            for child in input_node
            if _local_name(child.tag).lower() == "additional-files"
        ),
        None,
    )
    if additional_node is None:
        additional_node = ET.SubElement(input_node, "additional-files")

    cfg_dir = os.path.dirname(os.path.abspath(sumo_cfg))
    source_add_abs = os.path.normcase(os.path.abspath(presslight_add_xml))
    additional_files: list[str] = []
    for item in str(additional_node.get("value", "")).split(","):
        item = item.strip()
        if not item:
            continue
        absolute = os.path.abspath(item if os.path.isabs(item) else os.path.join(cfg_dir, item))
        if os.path.normcase(absolute) == source_add_abs:
            continue
        additional_files.append(absolute.replace("\\", "/"))
    if runtime_add.replace("\\", "/") not in additional_files:
        additional_files.append(runtime_add.replace("\\", "/"))
    additional_node.set("value", ",".join(additional_files))

    # Resolve all remaining SUMO file references before moving the wrapper.
    for elem in cfg_root.iter():
        tag = _local_name(elem.tag).lower()
        if tag == "additional-files" or "file" not in tag:
            continue
        values = [x.strip() for x in str(elem.get("value", "")).split(",") if x.strip()]
        if not values:
            continue
        absolute_values = [
            os.path.abspath(x if os.path.isabs(x) else os.path.join(cfg_dir, x)).replace("\\", "/")
            for x in values
        ]
        elem.set("value", ",".join(absolute_values))

    runtime_cfg = os.path.abspath(os.path.join(output_dir, "presslight_runtime.sumocfg"))
    cfg_tree.write(runtime_cfg, encoding="utf-8", xml_declaration=True)
    return runtime_cfg


if __name__ == "__main__":
    raise SystemExit(
        "Use train_presslight.py or evaluate_presslight.py; this module has no standalone defaults."
    )
