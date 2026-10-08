
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np

def build_yellow_state_from_green_state(green_state: str) -> str:
    return "".join("y" if ch == "G" else ch for ch in green_state)

@dataclass
class LaneInfo:
    lane_id:    str
    edge_id:    str
    index:      int    # 在所属 edge 中的 lane 编号（0-based）
    length:     float  # 车道长度（米）
    speed_free: float  # 自由流速度（m/s）
    width:      float  # 车道宽度（米）
    capacity:   int    # 车道容量（辆）= length / 7.5（标准车长含安全距离）

@dataclass
class EdgeInfo:
    edge_id:   str
    from_node: str
    to_node:   str
    length:    float
    num_lanes: int
    lane_ids:  List[str] = field(default_factory=list)

@dataclass
class PhaseInfo:
    name:              str        # 相位名称，如 "sn_straight"
    state:             str        # 完整 state 字符串，如 "gGrgrrgGrgrr"
    duration:          int        # 绿灯时长（秒）
    sumo_phase_index:  int = -1
    yellow_phase_index: Optional[int] = None
    yellow_state: Optional[str] = None
    green_link_indices: List[int] = field(default_factory=list)

@dataclass
class ConnectionInfo:
    tl_id: str
    from_edge: str
    from_lane: str    # lane_id，如 "nt21_nt11_0"
    to_edge:   str
    to_lane_idx: int  # toLane 编号
    to_lane: str
    link_index:  int  # tlLogic state 字符串中的位置
    direction:   str  # 'r'=右转, 's'=直行, 'l'=左转

@dataclass
class IntersectionInfo:
    tl_id:   str
    x:       float
    y:       float
    group:   str     # 参数共享组 ID

    inc_edges_nesw:  List[Optional[str]] = field(default_factory=list)
    inc_lanes_nesw:  List[List[str]]     = field(default_factory=list)

    neighbors_nesw:  List[Optional[str]] = field(default_factory=list)

    green_phases:    List[PhaseInfo]     = field(default_factory=list)
    connections:     List[ConnectionInfo] = field(default_factory=list)

    A_same: Optional[np.ndarray] = field(default=None, repr=False)
    A_diff: Optional[np.ndarray] = field(default=None, repr=False)
    structure_signature: str = ""
    structure_signature_dict: dict = field(default_factory=dict)

    @property
    def n_lanes(self) -> int:
        return sum(len(lanes) for lanes in self.inc_lanes_nesw)

    @property
    def n_green_phases(self) -> int:
        return len(self.green_phases)

    @property
    def state_len(self) -> int:
        return len(self.connections)

    def all_inc_lanes_flat(self) -> List[str]:
        result = []
        for lanes in self.inc_lanes_nesw:
            result.extend(lanes)
        return result

@dataclass
class NetworkInfo:

    intersection_ids:   List[str]                        = field(default_factory=list)
    intersections:      Dict[str, IntersectionInfo]      = field(default_factory=dict)

    lane_info:          Dict[str, LaneInfo]              = field(default_factory=dict)
    edge_info:          Dict[str, EdgeInfo]              = field(default_factory=dict)

    lane_to_e2:         Dict[str, str]                   = field(default_factory=dict)

    lane_to_e1_all:     Dict[str, str]                   = field(default_factory=dict)

    lane_to_e1_truck:   Dict[str, str]                   = field(default_factory=dict)

    net_xml_path:       str                              = ""
    add_xml_path:       str                              = ""
    groups_json_path:   str                              = ""

    def get_intersection(self, tl_id: str) -> IntersectionInfo:
        return self.intersections[tl_id]

    def get_neighbors(self, tl_id: str) -> List[Optional[str]]:
        return self.intersections[tl_id].neighbors_nesw

    def get_group(self, tl_id: str) -> str:
        return self.intersections[tl_id].group

    def unique_groups(self) -> List[str]:
        return sorted(set(info.group for info in self.intersections.values()))

class NetworkParser:

    VEHICLE_LENGTH_M: float = 7.5

    DIRECTIONS = ["N", "E", "S", "W"]

    def __init__(
        self,
        net_xml:     str,
        add_xml:     str,
        groups_json: Optional[str] = None,
        direction_overrides_json: Optional[str] = None,
    ) -> None:
        self.net_xml_path    = net_xml
        self.add_xml_path    = add_xml
        self.groups_json_path = groups_json or ""
        default_overrides = os.path.join(os.path.dirname(os.path.abspath(net_xml)), "edge_direction_overrides.json")
        self.direction_overrides_json_path = direction_overrides_json or default_overrides

        self._net_root: Optional[ET.Element] = None
        self._add_root: Optional[ET.Element] = None

        self._junctions:   Dict[str, Tuple[float, float]] = {}  # id -> (x, y)
        self._edge_info:   Dict[str, EdgeInfo]             = {}
        self._lane_info:   Dict[str, LaneInfo]             = {}
        self._tl_ids:      List[str]                       = []
        self._groups:      Dict[str, str]                  = {}  # tl_id -> group
        self._direction_overrides: Dict[str, Dict[str, str]] = {}
        self._tl_conns:    Dict[str, List[ConnectionInfo]] = defaultdict(list)
        self._tl_phases:   Dict[str, List[PhaseInfo]]      = {}

    def parse(self) -> NetworkInfo:
        self._load_xml()
        self._parse_junctions()
        self._parse_edges_and_lanes()
        self._parse_tl_phases()
        self._parse_connections()
        self._load_groups()
        self._load_direction_overrides()

        net_info = NetworkInfo(
            net_xml_path    = self.net_xml_path,
            add_xml_path    = self.add_xml_path,
            groups_json_path = self.groups_json_path,
            lane_info       = self._lane_info,
            edge_info       = self._edge_info,
        )

        for tl_id in self._tl_ids:
            iinfo = self._build_intersection_info(tl_id)
            net_info.intersections[tl_id] = iinfo
            net_info.intersection_ids.append(tl_id)

        self._parse_detectors(net_info)
        self._validate_group_structures(net_info)

        return net_info

    def _load_xml(self) -> None:
        if not os.path.exists(self.net_xml_path):
            raise FileNotFoundError(f"net.xml 不存在：{self.net_xml_path}")
        if not os.path.exists(self.add_xml_path):
            raise FileNotFoundError(f"add.xml 不存在：{self.add_xml_path}")

        self._net_root = ET.parse(self.net_xml_path).getroot()
        self._add_root = ET.parse(self.add_xml_path).getroot()

    def _parse_junctions(self) -> None:
        for junc in self._net_root.findall("junction"):
            jid  = junc.get("id")
            jtype = junc.get("type", "")
            x    = float(junc.get("x", 0))
            y    = float(junc.get("y", 0))

            self._junctions[jid] = (x, y)

            if jtype == "traffic_light" and jid.startswith("nt"):
                self._tl_ids.append(jid)

        self._tl_ids.sort()

    def _parse_edges_and_lanes(self) -> None:
        for edge in self._net_root.findall("edge"):
            eid = edge.get("id", "")
            if eid.startswith(":"):
                continue  # 跳过内部转向 edge

            from_node = edge.get("from", "")
            to_node   = edge.get("to", "")
            lanes_els = edge.findall("lane")
            if not lanes_els:
                continue

            lane_ids = []
            for lane_el in lanes_els:
                lid    = lane_el.get("id")
                index  = int(lane_el.get("index", 0))
                length = float(lane_el.get("length", 0))
                speed  = float(lane_el.get("speed", 13.9))
                width  = float(lane_el.get("width", 3.5))
                cap    = max(1, int(length / self.VEHICLE_LENGTH_M))

                self._lane_info[lid] = LaneInfo(
                    lane_id    = lid,
                    edge_id    = eid,
                    index      = index,
                    length     = length,
                    speed_free = speed,
                    width      = width,
                    capacity   = cap,
                )
                lane_ids.append(lid)

            edge_length = float(edge.get("length", lanes_els[0].get("length", 0)))

            self._edge_info[eid] = EdgeInfo(
                edge_id   = eid,
                from_node = from_node,
                to_node   = to_node,
                length    = edge_length,
                num_lanes = len(lanes_els),
                lane_ids  = lane_ids,
            )

    def _parse_tl_phases(self) -> None:
        for tl in self._net_root.findall("tlLogic"):
            tlid = tl.get("id")
            if tlid not in self._tl_ids:
                continue

            green_phases = []
            phases = tl.findall("phase")
            for idx, ph in enumerate(phases):
                state = ph.get("state", "")
                name  = ph.get("name", "")
                dur   = int(ph.get("duration", 0))

                if "y" in state:
                    continue
                if "g" not in state and "G" not in state:
                    continue

                green_link_idx = [
                    i for i, ch in enumerate(state)
                    if ch in ("g", "G")
                ]
                green_phases.append(PhaseInfo(
                    name               = name,
                    state              = state,
                    duration           = dur,
                    sumo_phase_index   = idx,
                    yellow_phase_index = idx + 1 if idx + 1 < len(phases) else None,
                    yellow_state       = build_yellow_state_from_green_state(state),
                    green_link_indices = green_link_idx,
                ))

            self._tl_phases[tlid] = green_phases

    def _parse_connections(self) -> None:
        raw: Dict[str, List[ConnectionInfo]] = defaultdict(list)

        for conn in self._net_root.findall("connection"):
            tl = conn.get("tl")
            if tl is None or tl not in self._tl_ids:
                continue

            from_edge  = conn.get("from", "")
            from_lane_idx = int(conn.get("fromLane", 0))
            to_edge    = conn.get("to", "")
            to_lane_idx = int(conn.get("toLane", 0))
            link_index = int(conn.get("linkIndex", 0))
            direction  = conn.get("dir", "s")

            from_lane_id = f"{from_edge}_{from_lane_idx}"

            raw[tl].append(ConnectionInfo(
                tl_id        = tl,
                from_edge    = from_edge,
                from_lane    = from_lane_id,
                to_edge      = to_edge,
                to_lane_idx  = to_lane_idx,
                to_lane      = f"{to_edge}_{to_lane_idx}",
                link_index   = link_index,
                direction    = direction,
            ))

        for tl_id, conns in raw.items():
            self._tl_conns[tl_id] = sorted(conns, key=lambda c: c.link_index)

    def _load_groups(self) -> None:
        if self.groups_json_path and os.path.exists(self.groups_json_path):
            with open(self.groups_json_path, "r", encoding="utf-8") as f:
                self._groups = json.load(f)
        else:
            self._groups = {tl_id: "standard" for tl_id in self._tl_ids}

    def _load_direction_overrides(self) -> None:
        path = self.direction_overrides_json_path
        if not path or not os.path.exists(path):
            self._direction_overrides = {}
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Direction overrides must be a JSON object: {path}")
        valid = set(self.DIRECTIONS)
        normalized: Dict[str, Dict[str, str]] = {}
        for tl_id, edge_map in data.items():
            if tl_id not in self._tl_ids or not isinstance(edge_map, dict):
                raise ValueError(f"Invalid direction override intersection: {tl_id}")
            normalized[tl_id] = {}
            for edge_id, direction in edge_map.items():
                edge = self._edge_info.get(edge_id)
                if edge is None or edge.to_node != tl_id or direction not in valid:
                    raise ValueError(
                        f"Invalid direction override: tl={tl_id}, edge={edge_id}, direction={direction}"
                    )
                normalized[tl_id][edge_id] = direction
        self._direction_overrides = normalized

    def _build_intersection_info(self, tl_id: str) -> IntersectionInfo:
        jx, jy = self._junctions[tl_id]
        group  = self._groups.get(tl_id, "standard")

        inc_edges = [
            eid for eid, einfo in self._edge_info.items()
            if einfo.to_node == tl_id and not eid.startswith(":")
        ]

        direction_map: Dict[str, Optional[str]] = {d: None for d in self.DIRECTIONS}
        neighbor_map:  Dict[str, Optional[str]] = {d: None for d in self.DIRECTIONS}

        for eid in inc_edges:
            from_node = self._edge_info[eid].from_node
            if from_node not in self._junctions:
                continue
            fx, fy = self._junctions[from_node]
            direction = self._direction_overrides.get(tl_id, {}).get(eid)
            if direction is None:
                direction = self._get_direction(jx, jy, fx, fy)
            if direction is not None:
                previous = direction_map[direction]
                if previous is not None and previous != eid:
                    raise ValueError(
                        f"Multiple incoming edges mapped to {tl_id} direction {direction}: "
                        f"{previous}, {eid}. Add an edge_direction_overrides.json entry."
                    )
                direction_map[direction] = eid
                if from_node in set(self._tl_ids):
                    neighbor_map[direction] = from_node

        inc_edges_nesw = [direction_map[d] for d in self.DIRECTIONS]
        neighbors_nesw = [neighbor_map[d]  for d in self.DIRECTIONS]

        inc_lanes_nesw: List[List[str]] = []
        for eid in inc_edges_nesw:
            if eid is None:
                inc_lanes_nesw.append([])
            else:
                lanes = sorted(
                    self._edge_info[eid].lane_ids,
                    key=lambda lid: self._lane_info[lid].index
                )
                inc_lanes_nesw.append(lanes)

        green_phases = list(self._tl_phases.get(tl_id, []))
        connections  = self._tl_conns.get(tl_id, [])

        all_lanes_flat = []
        for lanes in inc_lanes_nesw:
            all_lanes_flat.extend(lanes)

        # Canonicalize action indices by the N/E/S/W lane-release vector.
        # SUMO linkIndex order may be rotated between otherwise equivalent
        # intersections; sorting on the model-space mask makes shared Q-head
        # output k carry the same physical phase semantics throughout a group.
        lane_to_idx = {
            lane_id: index for index, lane_id in enumerate(all_lanes_flat)
        }
        link_to_lane = {
            connection.link_index: connection.from_lane
            for connection in connections
        }

        def _phase_lane_key(phase: PhaseInfo) -> tuple[int, ...]:
            mask = [0] * len(all_lanes_flat)
            for link_index in phase.green_link_indices:
                lane_id = link_to_lane.get(link_index)
                if lane_id in lane_to_idx:
                    mask[lane_to_idx[lane_id]] = 1
            return tuple(mask)

        green_phases.sort(key=_phase_lane_key, reverse=True)

        A_same, A_diff = self._build_conflict_matrices(
            all_lanes_flat, connections, green_phases
        )

        connection_counts_nesw = []
        for eid in inc_edges_nesw:
            if eid is None:
                connection_counts_nesw.append(0)
            else:
                connection_counts_nesw.append(sum(1 for c in connections if c.from_edge == eid))
        signature_dict = {
            "n_edges": sum(1 for eid in inc_edges_nesw if eid is not None),
            "n_lanes": sum(len(x) for x in inc_lanes_nesw),
            "n_green_phases": len(green_phases),
            "state_len": len(connections),
            "lane_counts_nesw": [len(x) for x in inc_lanes_nesw],
            "connection_counts_nesw": connection_counts_nesw,
            "green_phase_states": [p.state for p in green_phases],
            "phase_state_lengths": [len(p.state) for p in green_phases],
            "A_same_shape": list(A_same.shape),
            "A_diff_shape": list(A_diff.shape),
        }
        signature = json.dumps(signature_dict, sort_keys=True, ensure_ascii=False)

        return IntersectionInfo(
            tl_id           = tl_id,
            x               = jx,
            y               = jy,
            group           = group,
            inc_edges_nesw  = inc_edges_nesw,
            inc_lanes_nesw  = inc_lanes_nesw,
            neighbors_nesw  = neighbors_nesw,
            green_phases    = green_phases,
            connections     = connections,
            A_same          = A_same,
            A_diff          = A_diff,
            structure_signature = signature,
            structure_signature_dict = signature_dict,
        )

    def _parse_detectors(self, net_info: NetworkInfo) -> None:
        for det in self._add_root.findall("laneAreaDetector"):
            det_id  = det.get("id", "")
            lane_id = det.get("lane", "")
            if lane_id:
                if lane_id in net_info.lane_to_e2:
                    raise ValueError(
                        f"Duplicate E2 for lane {lane_id}: "
                        f"{net_info.lane_to_e2[lane_id]}, {det_id}; add.xml={self.add_xml_path}"
                    )
                lane_info = net_info.lane_info.get(lane_id)
                if lane_info is None:
                    raise ValueError(f"E2 {det_id} references unknown lane {lane_id}")
                pos = float(det.get("pos", "0"))
                length = float(det.get("length", "0"))
                lane_length = float(lane_info.length)
                if not (
                    0.0 <= pos < lane_length
                    and 0.0 < length <= lane_length
                    and pos + length <= lane_length + 1e-6
                ):
                    raise ValueError(
                        f"Invalid E2 geometry: id={det_id}, lane={lane_id}, pos={pos}, "
                        f"length={length}, lane_length={lane_length}"
                    )
                net_info.lane_to_e2[lane_id] = det_id

        for det in self._add_root.findall("inductionLoop"):
            det_id  = det.get("id", "")
            lane_id = det.get("lane", "")
            vtypes  = det.get("vTypes", "")

            if not lane_id:
                continue

            if vtypes == "truck":
                if lane_id in net_info.lane_to_e1_truck:
                    raise ValueError(f"Duplicate truck E1 for lane {lane_id}; add.xml={self.add_xml_path}")
                net_info.lane_to_e1_truck[lane_id] = det_id
            else:
                if lane_id in net_info.lane_to_e1_all:
                    raise ValueError(f"Duplicate all-vehicle E1 for lane {lane_id}; add.xml={self.add_xml_path}")
                net_info.lane_to_e1_all[lane_id] = det_id

    def _validate_group_structures(self, net_info: NetworkInfo) -> None:
        group_to_tls: Dict[str, List[str]] = defaultdict(list)
        for tl_id in net_info.intersection_ids:
            group_to_tls[net_info.get_group(tl_id)].append(tl_id)

        for group_id, tl_ids in group_to_tls.items():
            ref_id = tl_ids[0]
            ref = net_info.get_intersection(ref_id)
            ref_lane_counts = tuple(len(lanes) for lanes in ref.inc_lanes_nesw)
            ref_phase_lane_mask = build_phase_lane_mask(ref)
            ref_A_same = np.asarray(ref.A_same, dtype=np.float32)
            ref_A_diff = np.asarray(ref.A_diff, dtype=np.float32)
            for tl_id in tl_ids[1:]:
                current = net_info.get_intersection(tl_id)
                checks = {
                    "n_lanes": int(current.n_lanes) == int(ref.n_lanes),
                    "n_green_phases": (
                        int(current.n_green_phases)
                        == int(ref.n_green_phases)
                    ),
                    "lane_counts_nesw": (
                        tuple(len(lanes) for lanes in current.inc_lanes_nesw)
                        == ref_lane_counts
                    ),
                    "phase_lane_mask": np.array_equal(
                        build_phase_lane_mask(current),
                        ref_phase_lane_mask,
                    ),
                    "A_same": np.array_equal(
                        np.asarray(current.A_same, dtype=np.float32),
                        ref_A_same,
                    ),
                    "A_diff": np.array_equal(
                        np.asarray(current.A_diff, dtype=np.float32),
                        ref_A_diff,
                    ),
                }
                failed = [name for name, matched in checks.items() if not matched]
                if failed:
                    raise ValueError(
                        f"Group semantic structure mismatch: group_id={group_id}, "
                        f"reference={ref_id}, tl_id={tl_id}, failed={failed}"
                    )

    def _get_direction(
        self,
        jx: float, jy: float,
        fx: float, fy: float,
    ) -> Optional[str]:
        dx = fx - jx
        dy = fy - jy
        adx = abs(dx)
        ady = abs(dy)

        if adx < 1e-3 and ady < 1e-3:
            return None

        if adx >= ady:
            return "E" if dx > 0 else "W"
        else:
            return "N" if dy > 0 else "S"

    @staticmethod
    def _build_conflict_matrices(
        lanes_flat:   List[str],
        connections:  List[ConnectionInfo],
        green_phases: List[PhaseInfo],
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(lanes_flat)
        if n == 0:
            return np.eye(0, dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

        lane_to_idx: Dict[str, int] = {lid: i for i, lid in enumerate(lanes_flat)}

        link_to_lane: Dict[int, str] = {
            c.link_index: c.from_lane for c in connections
        }

        A_same = np.eye(n, dtype=np.float32)  # 对角线预设为 1

        for phase in green_phases:
            green_lanes = []
            for li in phase.green_link_indices:
                lane_id = link_to_lane.get(li)
                if lane_id is not None and lane_id in lane_to_idx:
                    green_lanes.append(lane_to_idx[lane_id])

            for i in green_lanes:
                for j in green_lanes:
                    A_same[i][j] = 1.0

        A_diff = 1.0 - np.clip(A_same, 0, 1)
        np.fill_diagonal(A_diff, 0.0)

        return A_same, A_diff

def parse_network(
    net_xml:     str,
    add_xml:     str,
    groups_json: Optional[str] = None,
    direction_overrides_json: Optional[str] = None,
) -> NetworkInfo:
    parser = NetworkParser(
        net_xml=net_xml,
        add_xml=add_xml,
        groups_json=groups_json,
        direction_overrides_json=direction_overrides_json,
    )
    return parser.parse()


@dataclass(frozen=True)
class Movement:
    """A signal-controlled movement, merged across parallel destination lanes."""
    movement_id: str
    tl_id: str
    from_lane: str
    direction: str
    to_edge: str
    to_lanes: Tuple[str, ...]
    link_indices: Tuple[int, ...]


def build_maxpressure_movements(net_info: NetworkInfo) -> Dict[str, Movement]:
    grouped = defaultdict(list)
    for tl_id in net_info.intersection_ids:
        for conn in net_info.get_intersection(tl_id).connections:
            grouped[(tl_id, conn.from_lane, conn.direction, conn.to_edge)].append(conn)
    movements: Dict[str, Movement] = {}
    for ordinal, (key, conns) in enumerate(sorted(grouped.items()), 1):
        tl_id, from_lane, direction, to_edge = key
        movement_id = f"mpm__{tl_id}__{ordinal:04d}"
        movements[movement_id] = Movement(
            movement_id, tl_id, from_lane, direction, to_edge,
            tuple(sorted({c.to_lane for c in conns})),
            tuple(sorted({c.link_index for c in conns})),
        )
    return movements


def build_phase_movement_map(net_info: NetworkInfo, movements=None):
    movements = movements or build_maxpressure_movements(net_info)
    by_tls = defaultdict(list)
    for movement in movements.values(): by_tls[movement.tl_id].append(movement)
    result = {}
    for tl_id in net_info.intersection_ids:
        phase_sets = []
        for phase in net_info.get_intersection(tl_id).green_phases:
            protected = {m.movement_id for m in by_tls[tl_id]
                         if any(i < len(phase.state) and phase.state[i] == "G" for i in m.link_indices)}
            phase_sets.append(tuple(sorted(protected)))
        result[tl_id] = tuple(phase_sets)
    return result


def build_turn_lookup(net_info: NetworkInfo, movements=None):
    movements = movements or build_maxpressure_movements(net_info)
    lookup = {}
    for movement in movements.values():
        key = (movement.from_lane, movement.to_edge)
        if key in lookup and lookup[key] != movement.movement_id:
            raise ValueError(f"Ambiguous Max-Pressure turn lookup: {key}")
        lookup[key] = movement.movement_id
    return lookup


def validate_maxpressure_network(net_info: NetworkInfo, movements=None, phase_map=None):
    movements = movements or build_maxpressure_movements(net_info)
    phase_map = phase_map or build_phase_movement_map(net_info, movements)
    connection_count = sum(len(net_info.get_intersection(t).connections) for t in net_info.intersection_ids)
    if sum(len(m.link_indices) for m in movements.values()) != connection_count:
        raise ValueError("Not every controlled connection belongs to exactly one movement")
    lane_movements = defaultdict(set)
    for m in movements.values(): lane_movements[m.from_lane].add(m.movement_id)
    return {"n_tls": len(net_info.intersection_ids), "n_connections": connection_count,
            "n_movements": len(movements),
            "n_shared_turn_lanes": sum(len(v) > 1 for v in lane_movements.values()),
            "n_multi_downstream_movements": sum(len(m.to_lanes) > 1 for m in movements.values()),
            "phase_counts": {t: len(phase_map[t]) for t in net_info.intersection_ids}}

@dataclass
class GroupMeta:
    group_id: str
    tl_ids: List[str]
    n_intersections: int
    n_lanes: int
    n_phases: int
    gk_dim: int
    q_output_dim: int
    A_same_shape: Tuple[int, int]
    A_diff_shape: Tuple[int, int]
    phase_lane_mask_shape: Tuple[int, int]

def build_phase_lane_mask(iinfo: IntersectionInfo) -> np.ndarray:
    lane_ids = iinfo.all_inc_lanes_flat()
    lane_to_idx = {lane_id: idx for idx, lane_id in enumerate(lane_ids)}
    link_to_lane = {c.link_index: c.from_lane for c in iinfo.connections}

    mask = np.zeros((iinfo.n_green_phases, len(lane_ids)), dtype=np.float32)
    for p_idx, phase in enumerate(iinfo.green_phases):
        for link_idx in phase.green_link_indices:
            lane_id = link_to_lane.get(link_idx)
            if lane_id is not None and lane_id in lane_to_idx:
                mask[p_idx, lane_to_idx[lane_id]] = 1.0
    return mask

def build_group_meta(net_info: NetworkInfo, node_update_dim: int = 64) -> Dict[str, GroupMeta]:
    groups: Dict[str, List[str]] = {}
    for tl_id in net_info.intersection_ids:
        groups.setdefault(net_info.get_group(tl_id), []).append(tl_id)

    meta: Dict[str, GroupMeta] = {}
    for group_id, tl_ids in sorted(groups.items()):
        ref = net_info.get_intersection(tl_ids[0])
        ref_n_lanes = ref.n_lanes
        ref_n_phases = ref.n_green_phases
        ref_A_same_shape = tuple(np.asarray(ref.A_same).shape)
        ref_A_diff_shape = tuple(np.asarray(ref.A_diff).shape)
        ref_phase_lane_shape = tuple(build_phase_lane_mask(ref).shape)

        for tl_id in tl_ids[1:]:
            iinfo = net_info.get_intersection(tl_id)
            if iinfo.n_lanes != ref_n_lanes:
                raise ValueError(f"group {group_id}: n_lanes 不一致: {tl_ids[0]}={ref_n_lanes}, {tl_id}={iinfo.n_lanes}")
            if iinfo.n_green_phases != ref_n_phases:
                raise ValueError(f"group {group_id}: n_phases 不一致: {tl_ids[0]}={ref_n_phases}, {tl_id}={iinfo.n_green_phases}")
            if tuple(np.asarray(iinfo.A_same).shape) != ref_A_same_shape:
                raise ValueError(f"group {group_id}: A_same shape 不一致: {tl_id}")
            if tuple(np.asarray(iinfo.A_diff).shape) != ref_A_diff_shape:
                raise ValueError(f"group {group_id}: A_diff shape 不一致: {tl_id}")
            if tuple(build_phase_lane_mask(iinfo).shape) != ref_phase_lane_shape:
                raise ValueError(f"group {group_id}: phase_lane_mask shape 不一致: {tl_id}")

        meta[group_id] = GroupMeta(
            group_id=group_id,
            tl_ids=list(tl_ids),
            n_intersections=len(tl_ids),
            n_lanes=int(ref_n_lanes),
            n_phases=int(ref_n_phases),
            gk_dim=int(ref_n_lanes * int(node_update_dim)),
            q_output_dim=int(ref_n_phases),
            A_same_shape=(int(ref_A_same_shape[0]), int(ref_A_same_shape[1])),
            A_diff_shape=(int(ref_A_diff_shape[0]), int(ref_A_diff_shape[1])),
            phase_lane_mask_shape=(int(ref_phase_lane_shape[0]), int(ref_phase_lane_shape[1])),
        )
    return meta
