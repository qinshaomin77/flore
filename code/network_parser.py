"""
network_parser.py
=================
路网静态信息解析模块。

从 SUMO 的 net.xml 和 add.xml 中提取所有训练所需的静态结构信息，
生成 NetworkInfo 对象供所有其他模块共享。

主要输出
--------
NetworkInfo 包含：
    - intersection_ids       : 所有信号控制路口 ID 列表
    - intersection_pos       : 路口 (x, y) 坐标字典
    - lane_info              : 每条 lane 的容量/自由流速度/所属 edge
    - edge_info              : 每条 edge 的长度/车道数/from/to 节点
    - phase_info             : 每个路口的绿灯相位定义
    - inc_lanes_ordered      : 按 N→E→S→W 排序的进口 lane 列表
    - inc_edges_ordered      : 按 N→E→S→W 排序的进口 edge 列表
    - conn_info              : 每个路口的 connection 列表（按 linkIndex 排序）
    - A_same / A_diff        : 每个路口的冲突矩阵
    - neighbors              : 每个路口的邻居 ID 列表（N/E/S/W 方向，缺失为 None）
    - lane_to_e2_detector    : lane_id -> E2 detector_id
    - lane_to_e1_all         : lane_id -> E1 all-vehicle detector_id
    - lane_to_e1_truck       : lane_id -> E1 truck detector_id
    - intersection_groups    : intersection_id -> group_id 字符串

坐标系说明（SUMO）
------------------
    x 增大 → 向东（列index增大）
    y 增大 → 向南（行index增大，nt00 在左上角）
    因此：北邻居 y 较大，南邻居 y 较小（与直觉相反，注意！）

依赖
----
    - config.py（仅用于路径配置，可选）
    - 标准库：xml.etree.ElementTree, json, os, dataclasses
    - numpy（构建矩阵）

验证方法
--------
    python network_parser.py
    会对 grid_6x6_net.xml 和 grid_6x6_add.xml 进行完整解析并打印摘要。
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np


def build_yellow_state_from_green_state(green_state: str) -> str:
    """Build yellow state from the old protected-green state."""
    return "".join("y" if ch == "G" else ch for ch in green_state)


# ═══════════════════════════════════════════════════════════════════════
# 数据结构定义
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LaneInfo:
    """单条 lane 的静态属性。"""
    lane_id:    str
    edge_id:    str
    index:      int    # 在所属 edge 中的 lane 编号（0-based）
    length:     float  # 车道长度（米）
    speed_free: float  # 自由流速度（m/s）
    width:      float  # 车道宽度（米）
    capacity:   int    # 车道容量（辆）= length / 7.5（标准车长含安全距离）


@dataclass
class EdgeInfo:
    """单条 edge 的静态属性。"""
    edge_id:   str
    from_node: str
    to_node:   str
    length:    float
    num_lanes: int
    lane_ids:  List[str] = field(default_factory=list)


@dataclass
class PhaseInfo:
    """单个绿灯相位的定义。"""
    name:              str        # 相位名称，如 "sn_straight"
    state:             str        # 完整 state 字符串，如 "gGrgrrgGrgrr"
    duration:          int        # 绿灯时长（秒）
    sumo_phase_index:  int = -1
    yellow_phase_index: Optional[int] = None
    yellow_state: Optional[str] = None
    green_link_indices: List[int] = field(default_factory=list)
    # state 中为 'g' 或 'G' 的 linkIndex 列表


@dataclass
class ConnectionInfo:
    tl_id: str
    """单条 connection 的属性。"""
    from_edge: str
    from_lane: str    # lane_id，如 "nt21_nt11_0"
    to_edge:   str
    to_lane_idx: int  # toLane 编号
    to_lane: str
    link_index:  int  # tlLogic state 字符串中的位置
    direction:   str  # 'r'=右转, 's'=直行, 'l'=左转


@dataclass
class IntersectionInfo:
    """单个路口的完整静态信息。"""
    tl_id:   str
    x:       float
    y:       float
    group:   str     # 参数共享组 ID

    # 按 N→E→S→W 排序的进口 edge（缺失方向为 None）
    inc_edges_nesw:  List[Optional[str]] = field(default_factory=list)
    # 按 N→E→S→W 排序的进口 lane（每方向是一个 list，缺失为 []）
    inc_lanes_nesw:  List[List[str]]     = field(default_factory=list)

    # 邻居路口 ID（N/E/S/W 顺序，无邻居为 None）
    neighbors_nesw:  List[Optional[str]] = field(default_factory=list)

    # 绿灯相位列表（过滤掉黄灯相位）
    green_phases:    List[PhaseInfo]     = field(default_factory=list)
    # 完整 connection 列表（按 linkIndex 升序）
    connections:     List[ConnectionInfo] = field(default_factory=list)

    # 冲突矩阵（shape: [n_lanes, n_lanes]，按 inc_lanes 展平顺序）
    A_same: Optional[np.ndarray] = field(default=None, repr=False)
    A_diff: Optional[np.ndarray] = field(default=None, repr=False)
    structure_signature: str = ""
    structure_signature_dict: dict = field(default_factory=dict)

    @property
    def n_lanes(self) -> int:
        """所有进口方向的 lane 总数。"""
        return sum(len(lanes) for lanes in self.inc_lanes_nesw)

    @property
    def n_green_phases(self) -> int:
        return len(self.green_phases)

    @property
    def state_len(self) -> int:
        """state 字符串长度 = connection 总数。"""
        return len(self.connections)

    def all_inc_lanes_flat(self) -> List[str]:
        """按 N→E→S→W 展平的进口 lane ID 列表。"""
        result = []
        for lanes in self.inc_lanes_nesw:
            result.extend(lanes)
        return result


@dataclass
class NetworkInfo:
    """路网全局静态信息容器。传递给所有其他模块。"""

    # ── 路口 ──────────────────────────────────────────────────────
    intersection_ids:   List[str]                        = field(default_factory=list)
    intersections:      Dict[str, IntersectionInfo]      = field(default_factory=dict)

    # ── lane / edge ───────────────────────────────────────────────
    lane_info:          Dict[str, LaneInfo]              = field(default_factory=dict)
    edge_info:          Dict[str, EdgeInfo]              = field(default_factory=dict)

    # ── 检测器映射 ────────────────────────────────────────────────
    lane_to_e2:         Dict[str, str]                   = field(default_factory=dict)
    # lane_id -> E2 laneAreaDetector id

    lane_to_e1_all:     Dict[str, str]                   = field(default_factory=dict)
    # lane_id -> E1 inductionLoop id（所有车辆）

    lane_to_e1_truck:   Dict[str, str]                   = field(default_factory=dict)
    # lane_id -> E1 inductionLoop id（仅货车）

    # ── 元信息 ────────────────────────────────────────────────────
    net_xml_path:       str                              = ""
    add_xml_path:       str                              = ""
    groups_json_path:   str                              = ""

    # ── 便捷访问 ──────────────────────────────────────────────────
    def get_intersection(self, tl_id: str) -> IntersectionInfo:
        return self.intersections[tl_id]

    def get_neighbors(self, tl_id: str) -> List[Optional[str]]:
        """返回 [N, E, S, W] 邻居 ID，无邻居位置为 None。"""
        return self.intersections[tl_id].neighbors_nesw

    def get_group(self, tl_id: str) -> str:
        return self.intersections[tl_id].group

    def unique_groups(self) -> List[str]:
        return sorted(set(info.group for info in self.intersections.values()))


# ═══════════════════════════════════════════════════════════════════════
# 主解析器
# ═══════════════════════════════════════════════════════════════════════

class NetworkParser:
    """
    从 SUMO net.xml 和 add.xml 解析路网静态信息。

    使用方式
    --------
        parser = NetworkParser(
            net_xml  = "network_36/network_36_net.xml",
            add_xml  = "network_36/network_36_add.xml",
            groups_json = "network_36/intersection_groups.json",  # 可选
        )
        net_info = parser.parse()
    """

    # 标准车长 + 安全距离（用于计算车道容量）
    VEHICLE_LENGTH_M: float = 7.5

    # SUMO 坐标系：y 增大 → 向南（与直觉相反）
    # 因此"北"邻居的 y 坐标 > 当前路口 y 坐标
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

        # 中间缓存
        self._junctions:   Dict[str, Tuple[float, float]] = {}  # id -> (x, y)
        self._edge_info:   Dict[str, EdgeInfo]             = {}
        self._lane_info:   Dict[str, LaneInfo]             = {}
        self._tl_ids:      List[str]                       = []
        self._groups:      Dict[str, str]                  = {}  # tl_id -> group
        self._direction_overrides: Dict[str, Dict[str, str]] = {}
        self._tl_conns:    Dict[str, List[ConnectionInfo]] = defaultdict(list)
        self._tl_phases:   Dict[str, List[PhaseInfo]]      = {}

    # ─────────────────────────────────────────────────────────────────
    # 公开接口
    # ─────────────────────────────────────────────────────────────────

    def parse(self) -> NetworkInfo:
        """
        执行完整解析，返回 NetworkInfo 对象。

        解析顺序（各步骤相互独立，顺序不可随意调换）：
        1. 加载 XML 根节点
        2. 解析路口坐标和 TL ID
        3. 解析 edge/lane 物理属性
        4. 解析 tlLogic 相位定义
        5. 解析 connection（linkIndex 排序）
        6. 加载参数共享分组
        7. 构建每个路口的 IntersectionInfo（含方向排序、矩阵）
        8. 解析检测器映射
        9. 组装 NetworkInfo 并返回
        """
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

        # 构建每个路口的完整信息
        for tl_id in self._tl_ids:
            iinfo = self._build_intersection_info(tl_id)
            net_info.intersections[tl_id] = iinfo
            net_info.intersection_ids.append(tl_id)

        # 解析检测器
        self._parse_detectors(net_info)
        self._validate_group_structures(net_info)

        return net_info

    # ─────────────────────────────────────────────────────────────────
    # 内部解析步骤
    # ─────────────────────────────────────────────────────────────────

    def _load_xml(self) -> None:
        """加载 XML 根节点。"""
        if not os.path.exists(self.net_xml_path):
            raise FileNotFoundError(f"net.xml 不存在：{self.net_xml_path}")
        if not os.path.exists(self.add_xml_path):
            raise FileNotFoundError(f"add.xml 不存在：{self.add_xml_path}")

        self._net_root = ET.parse(self.net_xml_path).getroot()
        self._add_root = ET.parse(self.add_xml_path).getroot()

    def _parse_junctions(self) -> None:
        """
        解析所有 junction，提取：
        - 信号控制路口列表（type="traffic_light"）
        - 所有路口坐标（含边界虚拟节点，用于方向判断）
        """
        for junc in self._net_root.findall("junction"):
            jid  = junc.get("id")
            jtype = junc.get("type", "")
            x    = float(junc.get("x", 0))
            y    = float(junc.get("y", 0))

            self._junctions[jid] = (x, y)

            if jtype == "traffic_light" and jid.startswith("nt"):
                self._tl_ids.append(jid)

        # 按路口 ID 排序保证确定性
        self._tl_ids.sort()

    def _parse_edges_and_lanes(self) -> None:
        """
        解析所有非内部 edge 和其 lane，提取物理属性。
        内部转向 edge 以 ':' 开头，跳过。
        """
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

            # edge 长度优先取 attribute，其次取第一条 lane 的长度
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
        """
        解析每个路口的 tlLogic，提取绿灯相位（过滤黄灯相位）。
        绿灯相位：state 中含有 'g' 或 'G'。
        """
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

                # 只保留纯绿灯相位：含 g/G 但不含 y（y 表示黄灯过渡）
                # 说明：右转 lane 在黄灯相位中也保持 g，因此不能只判断有无 g/G
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
        """
        解析 connection 元素，按 linkIndex 排序存入 _tl_conns。
        每条 connection 必须有 tl 属性（受信号控制）。
        """
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

            # 构造 from_lane_id
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

        # 按 linkIndex 升序排列
        for tl_id, conns in raw.items():
            self._tl_conns[tl_id] = sorted(conns, key=lambda c: c.link_index)

    def _load_groups(self) -> None:
        """
        加载参数共享分组文件（intersection_groups.json）。
        如果文件不存在，所有路口归为 "standard" 组。
        """
        if self.groups_json_path and os.path.exists(self.groups_json_path):
            with open(self.groups_json_path, "r", encoding="utf-8") as f:
                self._groups = json.load(f)
        else:
            # 无分组文件：全部归为 standard
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
        """
        构建单个路口的 IntersectionInfo，包含：
        - 按 N/E/S/W 排序的进口 edge/lane
        - 邻居路口 ID
        - 绿灯相位
        - A_same / A_diff 矩阵
        """
        jx, jy = self._junctions[tl_id]
        group  = self._groups.get(tl_id, "standard")

        # ── 按方向对进口 edge 分类 ─────────────────────────────────
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
                # 邻居路口：仅当 from_node 是信号控制路口时
                if from_node in set(self._tl_ids):
                    neighbor_map[direction] = from_node

        # ── 按 N→E→S→W 顺序构建列表 ──────────────────────────────
        inc_edges_nesw = [direction_map[d] for d in self.DIRECTIONS]
        neighbors_nesw = [neighbor_map[d]  for d in self.DIRECTIONS]

        # 每个方向的 lane 列表（按 lane index 升序）
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

        # ── 获取相位和连接 ────────────────────────────────────────
        green_phases = self._tl_phases.get(tl_id, [])
        connections  = self._tl_conns.get(tl_id, [])

        # ── 构建 A_same / A_diff 矩阵 ─────────────────────────────
        all_lanes_flat = []
        for lanes in inc_lanes_nesw:
            all_lanes_flat.extend(lanes)

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
        """
        解析 add.xml 中的检测器，建立 lane_id → detector_id 映射。

        检测器命名规则（来自 bulid_network_grid_e1.py）：
            E2 laneAreaDetector : det_{edge_id}_{lane_idx}
                                  例：det_nt21_nt11_0
            E1 all-vehicle      : e1_all_{edge_id}_{lane_idx}
                                  例：e1_all_nt11_nt21_0
            E1 truck-only       : e1_truck_{edge_id}_{lane_idx}
                                  例：e1_truck_nt11_nt21_0
        """
        for det in self._add_root.findall("laneAreaDetector"):
            det_id  = det.get("id", "")
            lane_id = det.get("lane", "")
            if lane_id:
                net_info.lane_to_e2[lane_id] = det_id

        for det in self._add_root.findall("inductionLoop"):
            det_id  = det.get("id", "")
            lane_id = det.get("lane", "")
            vtypes  = det.get("vTypes", "")

            if not lane_id:
                continue

            if vtypes == "truck":
                net_info.lane_to_e1_truck[lane_id] = det_id
            else:
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

    # ─────────────────────────────────────────────────────────────────
    # 辅助方法
    # ─────────────────────────────────────────────────────────────────

    def _get_direction(
        self,
        jx: float, jy: float,
        fx: float, fy: float,
    ) -> Optional[str]:
        """
        根据当前路口 (jx,jy) 和邻居/进口节点 (fx,fy) 的相对位置，
        判断进口方向。

        SUMO 坐标系：x 增大→东，y 增大→南（行index增大方向）。
        因此：
            N（北）= fy > jy（from 节点在更"南"，即 y 更大？不对！）
            重要：nt00 在左上角(0,0)，nt50 在左下角(0,1600)
            所以 y 增大 → 向下 → 在地图上是"南"方向
            但路网命名约定是"上北下南"，即：
                nt01(320,0) 是 nt11 的"南"方（y更小）
                nt21(320,640) 是 nt11 的"北"方（y更大）

            验证：nt11(320,320)，
                - nt21(320,640): dy=+320 → y更大 → "北"（row2比row1更北）
                - nt01(320,0):   dy=-320 → y更小 → "南"
                - nt12(640,320): dx=+320 → "东"
                - nt10(0,320):   dx=-320 → "西"

        结论：dx>0→东，dx<0→西，dy>0→北，dy<0→南（SUMO y轴与地图"北"同向）
        """
        dx = fx - jx
        dy = fy - jy
        adx = abs(dx)
        ady = abs(dy)

        # 需要明显的方向主导才确认方向（避免斜向连接误判）
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
        """
        构建 A_same 和 A_diff 矩阵。

        A_same[i][j] = 1：lane i 和 lane j 在至少一个绿灯相位中同时放行。
                           对角线 A_same[i][i] = 1（自身和自身同组）。
        A_diff[i][j] = 1：lane i 和 lane j 从未在同一绿灯相位中同时出现。
                           对角线 A_diff[i][i] = 0。

        实现逻辑
        --------
        对每个绿灯相位：
            1. 找出该相位放行的所有 linkIndex（green_link_indices）
            2. 找出这些 linkIndex 对应的 from_lane_id
            3. 这些 lane 两两标记为 A_same=1

        参数
        ----
        lanes_flat   : 按 N→E→S→W 展平的 lane_id 列表（定义矩阵行/列顺序）
        connections  : 按 linkIndex 排序的 ConnectionInfo 列表
        green_phases : 绿灯相位列表
        """
        n = len(lanes_flat)
        if n == 0:
            return np.eye(0, dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

        lane_to_idx: Dict[str, int] = {lid: i for i, lid in enumerate(lanes_flat)}

        # linkIndex -> from_lane_id 映射
        link_to_lane: Dict[int, str] = {
            c.link_index: c.from_lane for c in connections
        }

        A_same = np.eye(n, dtype=np.float32)  # 对角线预设为 1

        for phase in green_phases:
            # 该相位放行的 lane 集合（忽略不在 lanes_flat 中的 lane）
            green_lanes = []
            for li in phase.green_link_indices:
                lane_id = link_to_lane.get(li)
                if lane_id is not None and lane_id in lane_to_idx:
                    green_lanes.append(lane_to_idx[lane_id])

            # 两两标记为同组
            for i in green_lanes:
                for j in green_lanes:
                    A_same[i][j] = 1.0

        # A_diff：A_same 的补集（对角线保持 0）
        A_diff = 1.0 - np.clip(A_same, 0, 1)
        np.fill_diagonal(A_diff, 0.0)

        return A_same, A_diff


# ═══════════════════════════════════════════════════════════════════════
# 便捷函数：直接从配置获取 NetworkInfo
# ═══════════════════════════════════════════════════════════════════════

def parse_network(
    net_xml:     str,
    add_xml:     str,
    groups_json: Optional[str] = None,
    direction_overrides_json: Optional[str] = None,
) -> NetworkInfo:
    """
    快捷解析函数。

    示例
    ----
        from network_parser import parse_network
        net = parse_network(
            net_xml     = "data/grid36/truck_sensitive_grid36.net.xml",
            add_xml     = "data/grid36/truck_sensitive_grid36.add.xml",
            groups_json = "data/grid36/intersection_groups.json",
        )
        print(net.intersection_ids)
    """
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
    for movement in movements.values():
        by_tls[movement.tl_id].append(movement)
    result = {}
    for tl_id in net_info.intersection_ids:
        phase_sets = []
        for phase in net_info.get_intersection(tl_id).green_phases:
            protected = {
                movement.movement_id for movement in by_tls[tl_id]
                if any(idx < len(phase.state) and phase.state[idx] == "G"
                       for idx in movement.link_indices)
            }
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
    return {
        "n_tls": len(net_info.intersection_ids), "n_connections": connection_count,
        "n_movements": len(movements),
        "n_shared_turn_lanes": sum(len(v) > 1 for v in lane_movements.values()),
        "n_multi_downstream_movements": sum(len(m.to_lanes) > 1 for m in movements.values()),
        "phase_counts": {t: len(phase_map[t]) for t in net_info.intersection_ids},
    }


# ═══════════════════════════════════════════════════════════════════════
# 自检（python network_parser.py 直接运行）
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # 路径：优先接受命令行参数，否则使用默认路径
    NET_XML = sys.argv[1] if len(sys.argv) > 1 else \
        "/mnt/user-data/uploads/grid_6x6_net.xml"
    ADD_XML = sys.argv[2] if len(sys.argv) > 2 else \
        "/mnt/user-data/uploads/grid_6x6_add.xml"

    print("=" * 65)
    print("network_parser.py 自检")
    print("=" * 65)
    print(f"net.xml : {NET_XML}")
    print(f"add.xml : {ADD_XML}")
    print()

    net = parse_network(net_xml=NET_XML, add_xml=ADD_XML)

    # ── 基本统计 ────────────────────────────────────────────────
    print(f"[INFO] 路口总数          : {len(net.intersection_ids)}")
    print(f"[INFO] lane 总数         : {len(net.lane_info)}")
    print(f"[INFO] edge 总数         : {len(net.edge_info)}")
    print(f"[INFO] E2 检测器数       : {len(net.lane_to_e2)}")
    print(f"[INFO] E1-all 检测器数   : {len(net.lane_to_e1_all)}")
    print(f"[INFO] E1-truck 检测器数 : {len(net.lane_to_e1_truck)}")
    print()

    # ── 断言：共36个路口 ────────────────────────────────────────
    assert len(net.intersection_ids) == 36, \
        f"期望36个路口，实际 {len(net.intersection_ids)}"
    print("[OK] 路口数量 = 36")

    # ── 检查 nt11（标准路口）────────────────────────────────────
    nt11 = net.get_intersection("nt11")
    print(f"\n[INFO] nt11 详情：")
    print(f"  坐标        : ({nt11.x}, {nt11.y})")
    print(f"  分组        : {nt11.group}")
    print(f"  进口 lane 数: {nt11.n_lanes}")
    print(f"  绿灯相位数  : {nt11.n_green_phases}")
    print(f"  连接数      : {len(nt11.connections)}")
    print(f"  state 长度  : {nt11.state_len}")
    print(f"  A_same shape: {nt11.A_same.shape}")

    assert nt11.n_lanes == 12,         f"nt11 应有12条进口lane，实际 {nt11.n_lanes}"
    assert nt11.n_green_phases == 4,   f"nt11 应有4个绿灯相位，实际 {nt11.n_green_phases}"
    assert nt11.state_len == 12,       f"nt11 state 长度应为12，实际 {nt11.state_len}"
    assert nt11.A_same.shape == (12, 12), f"A_same shape 错误：{nt11.A_same.shape}"
    print("[OK] nt11 标准路口结构正确")

    # ── 检查 nt11 的 N/E/S/W 进口方向 ──────────────────────────
    print(f"\n[INFO] nt11 进口方向（N/E/S/W）：")
    for d, eid in zip(["N", "E", "S", "W"], nt11.inc_edges_nesw):
        lanes = nt11.inc_lanes_nesw[["N","E","S","W"].index(d)]
        print(f"  {d}: edge={eid}, lanes={lanes}")

    # 验证 N 方向是 nt21_nt11
    assert nt11.inc_edges_nesw[0] == "nt21_nt11", \
        f"nt11 北进口应为 nt21_nt11，实际 {nt11.inc_edges_nesw[0]}"
    assert nt11.inc_edges_nesw[1] == "nt12_nt11", \
        f"nt11 东进口应为 nt12_nt11，实际 {nt11.inc_edges_nesw[1]}"
    assert nt11.inc_edges_nesw[2] == "nt01_nt11", \
        f"nt11 南进口应为 nt01_nt11，实际 {nt11.inc_edges_nesw[2]}"
    assert nt11.inc_edges_nesw[3] == "nt10_nt11", \
        f"nt11 西进口应为 nt10_nt11，实际 {nt11.inc_edges_nesw[3]}"
    print("[OK] nt11 N/E/S/W 进口方向正确")

    # ── 检查 nt11 的邻居 ────────────────────────────────────────
    nbrs = net.get_neighbors("nt11")
    print(f"\n[INFO] nt11 邻居（N/E/S/W）：{nbrs}")
    assert nbrs[0] == "nt21", f"nt11 北邻居应为 nt21，实际 {nbrs[0]}"
    assert nbrs[1] == "nt12", f"nt11 东邻居应为 nt12，实际 {nbrs[1]}"
    assert nbrs[2] == "nt01", f"nt11 南邻居应为 nt01，实际 {nbrs[2]}"
    assert nbrs[3] == "nt10", f"nt11 西邻居应为 nt10，实际 {nbrs[3]}"
    print("[OK] nt11 邻居正确")

    # ── 检查 nt11 的 linkIndex 顺序 ────────────────────────────
    print(f"\n[INFO] nt11 connections（linkIndex 顺序）：")
    for c in nt11.connections:
        print(f"  [{c.link_index:2d}] {c.from_lane:20s} -> {c.to_edge:15s} toLane={c.to_lane_idx} dir={c.direction}")

    # linkIndex 0 应为 N 进口 lane0（右转）
    c0 = nt11.connections[0]
    assert c0.from_lane == "nt21_nt11_0", \
        f"linkIndex=0 应为 nt21_nt11_0（N进口右转），实际 {c0.from_lane}"
    assert c0.direction == "r", \
        f"linkIndex=0 方向应为 r，实际 {c0.direction}"
    print("[OK] linkIndex 排序正确，linkIndex=0 为北进口右转")

    # ── 检查 A_same 矩阵的语义 ──────────────────────────────────
    # nt11 的 sn_straight 相位：N-直行(idx=1) 和 S-直行(idx=7) 应同组
    # 找到对应 lane 在 all_lanes_flat 中的 index
    flat = nt11.all_inc_lanes_flat()
    # N进口 lane1 = "nt21_nt11_1"（直行），S进口 lane1 = "nt01_nt11_1"（直行）
    n_straight_idx = flat.index("nt21_nt11_1")
    s_straight_idx = flat.index("nt01_nt11_1")
    assert nt11.A_same[n_straight_idx][s_straight_idx] == 1.0, \
        "南北直行应在同一相位（A_same=1）"
    # 北直行和东直行不应同组
    e_straight_idx = flat.index("nt12_nt11_1")
    assert nt11.A_same[n_straight_idx][e_straight_idx] == 0.0, \
        "北直行和东直行不应同组（A_same=0）"
    # 对角线为 1
    assert nt11.A_same[0][0] == 1.0, "A_same 对角线应为1"
    # A_diff 对角线为 0
    assert nt11.A_diff[0][0] == 0.0, "A_diff 对角线应为0"
    print("[OK] A_same / A_diff 矩阵语义正确")

    # ── 检查 nt00（边角路口，有虚拟进口节点）──────────────────
    nt00 = net.get_intersection("nt00")
    print(f"\n[INFO] nt00 边角路口：")
    print(f"  进口方向 (N/E/S/W): {nt00.inc_edges_nesw}")
    print(f"  邻居     (N/E/S/W): {nt00.neighbors_nesw}")
    # nt00 有2个 nt 邻居（nt10 和 nt01）和2个虚拟边界节点
    real_nbrs = [n for n in nt00.neighbors_nesw if n is not None]
    assert len(real_nbrs) >= 2, f"nt00 至少应有2个真实邻居，实际 {real_nbrs}"
    print("[OK] nt00 边角路口邻居解析正确")

    # ── 检查 lane capacity ──────────────────────────────────────
    sample_lane = net.lane_info["nt21_nt11_0"]
    assert sample_lane.length == 320.0,    f"lane 长度应为320，实际 {sample_lane.length}"
    assert sample_lane.speed_free == 27.78, f"自由流速度应为27.78，实际 {sample_lane.speed_free}"
    assert sample_lane.capacity == int(320.0 / 7.5), \
        f"容量应为{int(320/7.5)}，实际 {sample_lane.capacity}"
    print(f"\n[OK] lane 容量：{sample_lane.capacity} 辆（320m / 7.5m）")

    # ── 检查参数共享分组（无 groups_json 时全为 standard）──────
    groups = net.unique_groups()
    print(f"\n[INFO] 参数共享分组：{groups}")
    assert "standard" in groups, "应有 standard 组"
    print("[OK] 分组信息加载正确")

    # ── E2 检测器检查 ───────────────────────────────────────────
    assert "nt21_nt11_0" in net.lane_to_e2, "nt21_nt11_0 应有 E2 检测器"
    print(f"\n[OK] E2 检测器示例：nt21_nt11_0 -> {net.lane_to_e2['nt21_nt11_0']}")

    # ── E1 检测器检查 ───────────────────────────────────────────
    # E1 装在出口 edge 上，取一个出口 lane 验证
    e1_lanes = list(net.lane_to_e1_all.keys())
    assert len(e1_lanes) > 0, "应有 E1 全车检测器"
    print(f"[OK] E1-all  示例：{e1_lanes[0]} -> {net.lane_to_e1_all[e1_lanes[0]]}")

    truck_lanes = list(net.lane_to_e1_truck.keys())
    assert len(truck_lanes) > 0, "应有 E1 货车检测器"
    print(f"[OK] E1-truck示例：{truck_lanes[0]} -> {net.lane_to_e1_truck[truck_lanes[0]]}")

    print()
    print("=" * 65)
    print("network_parser.py 自检全部通过。")
    print("=" * 65)

# ═══════════════════════════════════════════════════════════════════════
# MGMQ-DDQN 复现版增强工具
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GroupMeta:
    """参数共享组的结构摘要。"""
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
    """
    构造 [n_phases, n_lanes] 的 lane_phase 矩阵。

    phase_lane_mask[p, i] = 1 表示第 p 个 green_phase 下，第 i 条进口 lane 可放行；
    lane 顺序与 iinfo.all_inc_lanes_flat() 完全一致。
    """
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
    """
    按 group_id 汇总 MGMQ 模型需要的组结构信息。

    要求同一 group 内 n_lanes、n_green_phases、A_same/A_diff shape 一致；
    不同 group 之间可以不同，因此每个 group 会建立独立 Q-network。
    """
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
