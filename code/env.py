"""
env.py
======
SUMO 仿真环境封装 + Baseline 预跑。

职责
----
1. SumoEnv  ：封装所有 TraCI 调用，管理仿真生命周期，执行信号动作（含黄灯时序），
              返回每步的原始物理观测数据（raw_obs）。
2. BaselineRunner：运行 baseline 仿真，标定单一全网 NOx_max，
                   结果缓存为 JSON。

不负责
------
- 观测特征归一化（obs_reward.py 负责）
- Reward 计算（obs_reward.py 负责）
- 模型推理（model.py / agent.py 负责）

相位执行逻辑（重要）
--------------------
决策周期 = 10s，黄灯 = 3s，最短绿灯 = 5s（从黄灯结束起计时）

  保持相位：直接执行 5s 绿灯
  切换相位（A→B）：
    t=0: 决策切换 A→B
    t=0~3s: 黄灯（旧相位 A 的黄灯）
    t=3~5s: 新相位 B 绿灯（仅 2s，不足 min_green，下步强制保持 B）
    t=5: 下一决策时刻

TraCI 信号控制说明
------------------
SUMO 中通过 traci.trafficlight 控制信号：
  - setRedYellowGreenState(tl_id, state)：设置当前信号 state

为了精确执行黄灯时序，本文件在 _signal_state 中记录每个路口的
当前执行状态（绿灯中/黄灯中/黄灯剩余秒数），并在 simulationStep 级别
（每秒）精细控制。

实际仿真推进策略
----------------
每个决策步 = decision_interval 次 traci.simulationStep（每次推进1秒）。
信号切换在第 0 秒设置黄灯，第 3 秒切换为新相位绿灯。
观测数据在当前 decision_interval 结束时（即下一决策时刻前）采集。

依赖
----
    config.py         : EnvConfig
    network_parser.py : NetworkInfo, IntersectionInfo
    traci             : SUMO Python API（需安装 SUMO 并设置 SUMO_HOME）
    numpy, json, os
"""

from __future__ import annotations

import copy
import csv
import json
import os
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# ── 本地模块 ─────────────────────────────────────────────────────
from config import EnvConfig, ThresholdConfig, get_config
from network_parser import (
    NetworkInfo,
    IntersectionInfo,
    build_phase_lane_mask,
    parse_network,
)
from emission_lookup import (
    DecisionStepEmission,
    EmissionEpisodeRecorder,
    EmissionFactorLookup,
    build_emission_lookup,
)
from vehicle_state import VehicleStateParquetWriter, validate_hbefa4_vehicle_types


def _console(message: str) -> None:
    """Print one runtime status line with the current system time."""
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}",
        flush=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# TraCI 延迟导入（SUMO 未安装时不报错，仅在实际启动时检查）
# ═══════════════════════════════════════════════════════════════════════

def _import_traci(use_gui: bool = False) -> Tuple[object, bool]:
    """延迟导入 SUMO Python API；默认优先 libsumo，必要时回退 traci。"""
    sumo_home = os.environ.get("SUMO_HOME", "")
    if not sumo_home:
        # On Windows the SUMO installer commonly adds sumo.exe to PATH without
        # defining SUMO_HOME. Recover the installation root so SUMO can find
        # its local XML schemas and does not repeat a warning every episode.
        sumo_executable = shutil.which("sumo")
        if sumo_executable:
            candidate = os.path.dirname(os.path.dirname(os.path.abspath(sumo_executable)))
            if os.path.isdir(os.path.join(candidate, "tools")):
                sumo_home = candidate
                os.environ["SUMO_HOME"] = candidate
    if sumo_home:
        tools = os.path.join(sumo_home, "tools")
        if tools not in sys.path:
            sys.path.append(tools)

    force_traci = os.environ.get("SUMO_FORCE_TRACI", "") in {"1", "true", "True"}
    errors: List[str] = []

    if not use_gui and not force_traci:
        try:
            import libsumo
            import libsumo.constants  # noqa: F401
            return libsumo, True
        except ImportError as exc:
            errors.append(f"libsumo: {exc}")

    try:
        import traci
        import traci.constants  # noqa: F401
        return traci, False
    except ImportError as exc:
        errors.append(f"traci: {exc}")

    detail = "\n".join(f"  - {err}" for err in errors)
    raise ImportError(
        "无法导入 libsumo/traci。请确认：\n"
        "  1. 已安装 SUMO（https://sumo.dlr.de/docs/Installing/index.html）\n"
        "  2. 设置了环境变量 SUMO_HOME，例如：\n"
        "     export SUMO_HOME=/usr/share/sumo\n"
        "  3. 或执行 pip install eclipse-sumo\n"
        f"导入错误：\n{detail}"
    )


# ═══════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SignalState:
    """
    单个路口的信号执行状态，每步更新。

    用于精确控制黄灯时序和计算 action mask 所需的时间量。
    """
    tl_id:            str
    current_phase:    int    # 当前绿灯相位 index（0-based，在 green_phases 列表中）
    elapsed_green:    float  # 当前绿灯已持续秒数（从黄灯结束后开始计时）
    in_yellow:        bool   # 是否正在执行黄灯
    yellow_remaining: float  # 黄灯剩余秒数（in_yellow=True 时有效）
    pending_phase:    int    # 黄灯结束后即将切换到的相位（-1 表示无待切换）
    last_green_start: float  # 当前绿灯开始的仿真时刻（秒）

    # 每个相位上次结束红灯的时刻（用于 min_red 约束）
    # 索引 = 绿灯相位 index，值 = 该相位上次变为红灯时的仿真时刻
    phase_last_went_red: List[float] = field(default_factory=list)


@dataclass
class StepRawObs:
    """
    单个路口单步的原始物理观测，直接来自 TraCI，未归一化。
    传给 obs_reward.py 做归一化和特征构建。
    """
    tl_id: str

    # ── Lane 级原始值（按 N→E→S→W 展平，长度 = n_lanes）──────────
    queue_veh:     List[float]  # 配置观测范围内排队车辆数（辆）
    speed_mps:     List[float]  # 当前平均速度（m/s）
    truck_count:   List[float]  # 近decision_interval秒E1通过货车 + 观测范围内驻留货车
    total_count:   List[float]  # 配置观测范围内进口 lane 车辆数
    pass_veh:      List[float]  # 当前决策间隔内通过车辆数
    demand_veh:    List[float]  # E1近decision_interval秒通过数 + 观测范围当前车辆数
    NOx_mg:        List[float]  # 当前决策步观测范围内 lane 累计NOx（mg）
    wait_vwt:      List[float]  # 当前决策间隔内观测范围等待车辆逐秒累计值（veh·s）
    green_service_s: List[float]  # 当前决策间隔内实际有效绿灯累计秒数

    # ── Edge 级原始值（按 N→E→S→W，只含实际存在的进口 edge）─────
    edge_NOx_mg:   List[float]  # 每条进口 edge 的 NOx 总量（mg）
    # 计算方式：对该 edge 所有进口 lane 的 NOx 求和

    # ── Phase 级信号状态 ───────────────────────────────────────────
    current_phase:    int    # 当前绿灯相位 index
    elapsed_green:    float  # 已绿灯时长（秒）
    in_yellow:        bool   # 是否在黄灯期
    phase_last_went_red: List[float]  # 每个绿灯相位上次变红的时刻

    # ── 仿真元信息 ─────────────────────────────────────────────────
    sim_time:      float  # 当前仿真时刻（秒）


    sumo_green_phase: int = -1  # SUMO real green phase index mapped to current_phase coding.
    sumo_raw_phase:   int = -1  # SUMO raw phase index, including yellow/all-red phases.
    pending_phase:    int = -1  # Target phase while a yellow transition is pending.
    sumo_state:       str = ""   # SUMO real signal state from getRedYellowGreenState.
    env_state:        str = ""   # Expected env-issued green/yellow state for comparison.

@dataclass
class EpisodeTrafficStats:
    """episode 结束后从 tripinfo.xml 解析的交通统计。"""
    avg_delay_s:        float = 0.0   # 平均延误（秒）
    avg_travel_time_s:  float = 0.0   # 平均行程时间（秒）
    total_arrived:      int   = 0     # 完成行程车辆数
    completion_rate:    float = 0.0   # = arrived / departed


@dataclass
class BaselineThresholds:
    """Baseline 标定产物：单一全网 NOx 风险尺度。"""
    NOx_max: float = 1.0
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    calibration_meta: Dict[str, object] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
# SumoEnv
# ═══════════════════════════════════════════════════════════════════════

class SumoEnv:
    """
    SUMO 仿真环境封装。

    使用方式
    --------
        env = SumoEnv(cfg.env, net_info)
        env.start(episode_id=0)
        for step in range(720):
            actions = {tl_id: phase_idx for tl_id in net_info.intersection_ids}
            raw_obs_dict = env.step(actions)
            # raw_obs_dict: Dict[str, StepRawObs]
        traffic_stats = env.close()

    设计约定
    --------
    - 仅负责"执行动作、推进仿真、返回原始数据"，不做任何归一化。
    - TraCI 连接通过 port 区分，支持并行多实例（不同 port）。
    - 所有 traci 调用集中在本类，其他模块不直接调用 traci。
    """

    # fixed-time 控制时使用的固定相位循环（baseline 预跑用）
    # 每个绿灯相位 30s，按 green_phases 顺序循环
    FIXED_TIME_GREEN_DURATION: int = 30

    def __init__(
        self,
        env_cfg:  EnvConfig,
        net_info: NetworkInfo,
        port:     int = 8813,
        use_gui:  bool = False,
        emission_lookup: Optional[EmissionFactorLookup] = None,
        emission_recorder: Optional[EmissionEpisodeRecorder] = None,
        evaluation_logger: Optional[Any] = None,
    ) -> None:
        self.env_cfg  = env_cfg
        self.net_info = net_info
        self.port     = port
        self.use_gui  = use_gui
        self.emission_lookup = emission_lookup
        self.emission_recorder = emission_recorder
        # Optional read-only evaluator.  Keeping this as an explicit hook avoids
        # replacing methods at runtime and guarantees identical collection for
        # RL, Max-Pressure and Gap-actuated stepping paths.
        self.evaluation_logger = evaluation_logger
        if self.emission_recorder is None:
            if self.emission_lookup is None and env_cfg.emission_factor_csv:
                csv_path = env_cfg.emission_factor_csv
                if os.path.exists(csv_path):
                    self.emission_lookup = build_emission_lookup(
                        csv_path,
                        default_pollutant=env_cfg.emission_pollutants[0],
                    )
            if self.emission_lookup is not None:
                self.emission_recorder = EmissionEpisodeRecorder(
                    self.emission_lookup,
                    pollutants=env_cfg.emission_pollutants,
                    output_unit="mg",
                )
        self._last_step_emission: Optional[DecisionStepEmission] = None
        self._interval_wait_accum: Dict[str, float] = {}
        self._last_wait_vwt_by_lane: Dict[str, float] = {}
        self._interval_pass_accum: Dict[str, float] = {}
        self._last_pass_by_lane: Dict[str, float] = {}
        self._interval_green_service_accum: Dict[str, float] = {}
        self._last_green_service_by_lane: Dict[str, float] = {}
        self._phase_lane_masks = {
            tl_id: build_phase_lane_mask(net_info.get_intersection(tl_id))
            for tl_id in net_info.intersection_ids
        }
        self._e2_vehicle_ids_by_lane: Dict[str, frozenset[str]] = {}
        self._e2_halting_by_lane: Dict[str, float] = {}

        # 延迟导入 traci（避免无 SUMO 环境时 import 报错）
        self._traci = None
        self._is_libsumo: bool = False

        # 信号状态字典：tl_id -> SignalState
        self._signal_states: Dict[str, SignalState] = {}
        self._last_sumo_phase: Dict[str, Tuple[int, int]] = {}

        # 仿真时间（秒）
        self._sim_time: float = 0.0

        # E1 计数器缓存（上一步的通过数，用于计算增量 demand）
        self._e1_prev_count: Dict[str, int] = {}

        # 当前 episode 的 tripinfo 路径
        self._tripinfo_path: str = ""
        self._fcd_output_path: str = ""
        self._tls_switch_states_path: str = ""
        self._tls_switch_times_path: str = ""
        self._vehicle_second_path: str = ""
        self._vehicle_second_file = None
        self._vehicle_second_writer = None
        self._vehicle_second_buffer: List[List[Any]] = []
        self._vehicle_state_writer: Optional[VehicleStateParquetWriter] = None
        self._current_vehicle_context: Mapping[str, Mapping[int, Any]] = {}
        self._current_episode_id: int = -1
        self._current_sumo_seed: int = -1

        self._running: bool = False

        # ── 性能优化：TraCI subscription 与车辆类型缓存 ─────────────────
        self._tc = None
        self._subscribed: bool = False
        self._context_anchor: Optional[str] = None

        # 全网进口 lane 列表，用于 lane subscription
        self._all_inc_lanes: List[str] = []

        # 车辆类型缓存：veh_id -> vType，避免逐车反复 getTypeID()
        self._veh_type_cache: Dict[str, str] = {}

        # E1 (induction loop) 单步通过量订阅集合；订阅失败时回退到直接查询。
        self._subscribed_e1_step: set = set()

    # ─────────────────────────────────────────────────────────────────
    # 生命周期
    # ─────────────────────────────────────────────────────────────────

    @property
    def fcd_output_path(self) -> str:
        """Return the current episode's SUMO-managed FCD output path."""
        return str(self._fcd_output_path)

    @property
    def tripinfo_output_path(self) -> str:
        """Return the current episode's SUMO-managed tripinfo path."""
        return str(self._tripinfo_path)

    @property
    def tls_switch_states_output_path(self) -> str:
        """Return the current episode's SUMO TLS switch-state output path."""
        return str(self._tls_switch_states_path)

    @property
    def tls_switch_times_output_path(self) -> str:
        """Return the current episode's SUMO TLS switch-time output path."""
        return str(self._tls_switch_times_path)

    @staticmethod
    def _sumo_path(path: str) -> str:
        return os.path.abspath(path).replace("\\", "/")

    @staticmethod
    def _null_output_path() -> str:
        return "NUL" if os.name == "nt" else "/dev/null"

    @staticmethod
    def _xml_local_name(tag: str) -> str:
        return str(tag).split("}", 1)[-1].lower()

    def _is_detector_file_attr(self, elem: ET.Element, file_value: str) -> bool:
        value = str(file_value).strip().lower()
        tag = self._xml_local_name(elem.tag)
        detector_tags = {
            "detector",
            "inductionloop",
            "instantinductionloop",
            "laneareadetector",
            "entryexitdetector",
            "e1detector",
            "e2detector",
            "e3detector",
        }
        return "detector_output" in value or (tag in detector_tags and value.endswith(".xml"))

    def _prepare_episode_sumo_config(
        self,
        episode_id: int,
        detector_output_path: str,
        detector_dir: str,
        tls_switch_states_path: str = "",
        tls_switch_times_path: str = "",
    ) -> str:
        """Build episode-local add/sumocfg files so detector outputs do not collide."""
        sumocfg_path = str(self.env_cfg.sumo_cfg)
        add_xml_path = str(getattr(self.env_cfg, "add_xml", "") or "")
        tls_outputs_required = bool(
            tls_switch_states_path or tls_switch_times_path
        )
        if not sumocfg_path or not add_xml_path:
            if tls_outputs_required:
                raise RuntimeError(
                    "Cannot enable SUMO TLS outputs without both sumo_cfg "
                    "and add_xml"
                )
            return sumocfg_path

        try:
            add_tree = ET.parse(add_xml_path)
            add_root = add_tree.getroot()
            tls_event_types = {
                "SaveTLSSwitchStates",
                "SaveTLSSwitchTimes",
            }
            for child in list(add_root):
                if (
                    self._xml_local_name(child.tag) == "timedevent"
                    and child.attrib.get("type") in tls_event_types
                ):
                    add_root.remove(child)

            # Detector XML output is intentionally disabled for every run. Keep
            # the config field and method argument only for old YAML/callers.
            detector_file_value = self._null_output_path()
            for elem in add_root.iter():
                file_value = elem.attrib.get("file")
                if not file_value or not self._is_detector_file_attr(elem, file_value):
                    continue
                elem.set("file", detector_file_value)

            if tls_switch_states_path:
                ET.SubElement(
                    add_root,
                    "timedEvent",
                    {
                        "type": "SaveTLSSwitchStates",
                        "dest": self._sumo_path(tls_switch_states_path),
                    },
                )
            if tls_switch_times_path:
                ET.SubElement(
                    add_root,
                    "timedEvent",
                    {
                        "type": "SaveTLSSwitchTimes",
                        "dest": self._sumo_path(tls_switch_times_path),
                    },
                )

            runtime_add_xml = os.path.join(detector_dir, f"runtime_add_{int(episode_id):04d}.add.xml")
            add_tree.write(runtime_add_xml, encoding="utf-8", xml_declaration=True)

            sumocfg_tree = ET.parse(sumocfg_path)
            sumocfg_root = sumocfg_tree.getroot()
            input_node = None
            for child in sumocfg_root:
                if self._xml_local_name(child.tag) == "input":
                    input_node = child
                    break
            if input_node is None:
                input_node = ET.SubElement(sumocfg_root, "input")

            output_node = None
            for child in sumocfg_root:
                if self._xml_local_name(child.tag) == "output":
                    output_node = child
                    break
            save_sumo_aux_outputs = bool(getattr(self.env_cfg, "save_sumo_aux_outputs", True))
            save_fcd_output = bool(getattr(self.env_cfg, "save_fcd_output", False))
            if output_node is not None:
                if not save_sumo_aux_outputs:
                    drop_outputs = {
                        "summary-output",
                        "vehroute-output",
                        "fcd-output",
                        "emission-output",
                        "netstate-dump",
                    }
                    for child in list(output_node):
                        if self._xml_local_name(child.tag) in drop_outputs:
                            output_node.remove(child)
                # Never inherit an FCD destination from the source sumocfg.
                # When explicitly enabled, an episode-specific destination is
                # supplied on the SUMO command line below.
                for child in list(output_node):
                    local_name = self._xml_local_name(child.tag)
                    if local_name == "fcd-output" or local_name.startswith("fcd-output."):
                        output_node.remove(child)

            additional_node = None
            for child in input_node:
                if self._xml_local_name(child.tag) == "additional-files":
                    additional_node = child
                    break
            if additional_node is None:
                additional_node = ET.SubElement(input_node, "additional-files")

            sumocfg_dir = os.path.dirname(os.path.abspath(sumocfg_path))
            # The runtime sumocfg is written under the run directory. Preserve
            # source output files there and create their parent directories;
            # otherwise relative paths such as output/summary.xml are resolved
            # beside the runtime sumocfg and SUMO exits before TraCI connects.
            output_file_tags = {
                "summary-output",
                "tripinfo-output",
                "vehroute-output",
                "fcd-output",
                "emission-output",
                "netstate-dump",
                "queue-output",
                "statistic-output",
            }
            if output_node is not None:
                for child in output_node:
                    if self._xml_local_name(child.tag) not in output_file_tags:
                        continue
                    value = str(child.attrib.get("value", "") or "").strip()
                    if not value or value.upper() == "NUL":
                        continue
                    output_path = value if os.path.isabs(value) else os.path.join(detector_dir, value)
                    output_path = os.path.abspath(output_path)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    child.set("value", self._sumo_path(output_path))

            add_xml_abs = os.path.abspath(add_xml_path)
            existing_additional_files: list[str] = []
            raw_additional_value = str(additional_node.attrib.get("value", "") or "")
            for item in raw_additional_value.split(","):
                item = item.strip()
                if not item:
                    continue
                item_abs = item if os.path.isabs(item) else os.path.join(sumocfg_dir, item)
                item_abs = os.path.abspath(item_abs)
                if os.path.normcase(item_abs) == os.path.normcase(add_xml_abs):
                    continue
                existing_additional_files.append(self._sumo_path(item_abs))

            additional_files = [self._sumo_path(runtime_add_xml)]
            for item in existing_additional_files:
                if item not in additional_files:
                    additional_files.append(item)
            additional_node.set("value", ",".join(additional_files))

            for elem in sumocfg_root.iter():
                tag = self._xml_local_name(elem.tag)
                if tag == "additional-files" or "file" not in tag:
                    continue
                value = str(elem.attrib.get("value", "") or "")
                items = [item.strip() for item in value.split(",") if item.strip()]
                if not items:
                    continue
                abs_items = [
                    self._sumo_path(item if os.path.isabs(item) else os.path.join(sumocfg_dir, item))
                    for item in items
                ]
                elem.set("value", ",".join(abs_items))

            runtime_sumocfg = os.path.join(detector_dir, f"runtime_sumocfg_{int(episode_id):04d}.sumocfg")
            sumocfg_tree.write(runtime_sumocfg, encoding="utf-8", xml_declaration=True)
            return runtime_sumocfg
        except Exception as exc:
            if tls_outputs_required:
                raise RuntimeError(
                    "Failed to build episode-specific SUMO config with "
                    f"TLS outputs: {exc}"
                ) from exc
            _console(
                "[WARN] Failed to build episode-specific detector SUMO config; "
                f"falling back to original sumocfg. Original error: {exc}"
            )
            return sumocfg_path

    def start(
        self,
        episode_id: int = 0,
        fixed_time: bool = False,
        seed: Optional[int] = None,
        control_tls: bool = True,
    ) -> None:
        """
        启动 SUMO 仿真并建立 TraCI 连接。

        参数
        ----
        episode_id  : 用于生成唯一的 tripinfo 输出文件名
        fixed_time  : True 时使用 fixed-time 控制（baseline 预跑用）
        """
        if self._traci is None:
            self._traci, self._is_libsumo = _import_traci(self.use_gui)
            self._tc = self._traci.constants

        # 确保 tripinfo 目录存在
        tripinfo_dir = self.env_cfg.tripinfo_dir
        os.makedirs(tripinfo_dir, exist_ok=True)
        run_dir = os.path.dirname(tripinfo_dir)
        save_fcd_output = bool(getattr(self.env_cfg, "save_fcd_output", False))
        save_tls_outputs = bool(
            getattr(self.env_cfg, "save_tls_phase_outputs", False)
        )
        self._fcd_output_path = ""
        self._tls_switch_states_path = ""
        self._tls_switch_times_path = ""
        if save_fcd_output:
            fcd_dir = os.path.join(run_dir, "fcd")
            os.makedirs(fcd_dir, exist_ok=True)
            self._fcd_output_path = os.path.abspath(
                os.path.join(fcd_dir, f"fcd_ep{int(episode_id):04d}.xml")
            )
        if save_tls_outputs:
            phase_state_root = os.path.join(run_dir, "phase_state")
            switch_states_dir = os.path.join(
                phase_state_root, "tls_switch_states"
            )
            switch_times_dir = os.path.join(
                phase_state_root, "tls_switch_times"
            )
            os.makedirs(switch_states_dir, exist_ok=True)
            os.makedirs(switch_times_dir, exist_ok=True)
            self._tls_switch_states_path = os.path.abspath(
                os.path.join(
                    switch_states_dir,
                    f"tls_switch_states_ep{int(episode_id):04d}.xml",
                )
            )
            self._tls_switch_times_path = os.path.abspath(
                os.path.join(
                    switch_times_dir,
                    f"tls_switch_times_ep{int(episode_id):04d}.xml",
                )
            )
        detector_dir = os.path.join(run_dir, "sumo_runtime")
        os.makedirs(detector_dir, exist_ok=True)
        detector_output_path = os.path.join(
            detector_dir,
            f"detector_output_{int(episode_id):04d}.xml",
        )
        self._tripinfo_path = os.path.join(
            tripinfo_dir, f"tripinfo_ep{episode_id:04d}.xml"
        )
        episode_sumocfg = self._prepare_episode_sumo_config(
            episode_id=episode_id,
            detector_output_path=detector_output_path,
            detector_dir=detector_dir,
            tls_switch_states_path=self._tls_switch_states_path,
            tls_switch_times_path=self._tls_switch_times_path,
        )

        sumo_bin = "sumo-gui" if self.use_gui else "sumo"
        cmd = [
            sumo_bin,
            "-c", episode_sumocfg,
            "--step-length", "1",           # 1秒步长（精细控制黄灯）
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--duration-log.disable", "true",
            "--time-to-teleport", "-1",     # 禁止 teleport
            "--tripinfo-output", self._tripinfo_path,
            "--tripinfo-output.write-unfinished", "true",
            "--collision.action", "warn",
        ]
        if seed is not None:
            cmd += ["--seed", str(int(seed))]
        if save_fcd_output:
            cmd += ["--fcd-output", self._sumo_path(self._fcd_output_path)]
            if bool(getattr(self.env_cfg, "fcd_output_acceleration", True)):
                cmd += ["--fcd-output.acceleration", "true"]
            cmd += [
                "--device.fcd.period",
                str(float(getattr(self.env_cfg, "fcd_output_period", 1.0))),
            ]

        self._close_stale_traci_connection()
        try:
            if self._is_libsumo:
                self._traci.start(cmd)
            else:
                self._traci.start(cmd, port=self.port, numRetries=20)
        except Exception as exc:
            self._running = False
            self._close_stale_traci_connection()
            raise RuntimeError(
                f"Failed to start SUMO TraCI session on port {self.port}. "
                f"Original error: {exc}. "
                "This may be caused by occupied port, stale SUMO process, or output file conflicts."
            ) from exc
        self._sim_time = 0.0
        self._running  = True
        self._current_episode_id = int(episode_id)
        self._current_sumo_seed = int(seed) if seed is not None else -1
        if bool(getattr(self.env_cfg, "save_vehicle_second_output", False)):
            vehicle_second_dir = os.path.join(run_dir, "vehicle_second")
            os.makedirs(vehicle_second_dir, exist_ok=True)
            self._vehicle_second_path = os.path.abspath(os.path.join(
                vehicle_second_dir,
                f"vehicle_second_ep{int(episode_id):04d}.csv",
            ))
            self._vehicle_second_file = open(
                self._vehicle_second_path,
                "w",
                newline="",
                encoding="utf-8",
            )
            self._vehicle_second_writer = csv.writer(self._vehicle_second_file)
            self._vehicle_second_writer.writerow([
                "episode", "sumo_seed", "simulation_time_s", "vehicle_id",
                "vehicle_type", "lane_id", "speed_mps", "acceleration_mps2",
                "time_loss_accumulated_s", "is_waiting",
            ])
            self._vehicle_second_buffer = []
        verbose_runtime_output = bool(
            getattr(self.env_cfg, "verbose_runtime_output", True)
        )
        if save_fcd_output and verbose_runtime_output:
            _console(f"FCD output enabled: {self._fcd_output_path}")
        if save_tls_outputs and verbose_runtime_output:
            _console(
                "TLS switch states output enabled: "
                f"{self._tls_switch_states_path}"
            )
            _console(
                "TLS switch times output enabled: "
                f"{self._tls_switch_times_path}"
            )
        self._last_step_emission = None
        self._interval_wait_accum = {}
        self._last_wait_vwt_by_lane = {}
        self._interval_pass_accum = {}
        self._last_pass_by_lane = {}
        self._interval_green_service_accum = {}
        self._last_green_service_by_lane = {}
        self._e2_vehicle_ids_by_lane = {}
        self._e2_halting_by_lane = {}
        self._subscribed = False
        self._context_anchor = None
        self._veh_type_cache.clear()
        if self.emission_recorder is not None:
            self.emission_recorder.start_episode(episode_id)

        if (
            not control_tls
            and getattr(self.env_cfg, "baseline_control_mode", "fixed_time") == "sumo_actuated"
        ):
            self._activate_tls_program("actuated")

        # 初始化所有路口的信号状态
        if control_tls:
            self._init_signal_states(fixed_time)
        else:
            self._init_passive_signal_states()

        # 初始化 E1 计数器基准
        self._init_e1_counters()
        self._validate_observation_detectors()
        self._setup_subscriptions()
        if bool(getattr(self.env_cfg, "save_vehicle_state_output", False)):
            if self.emission_recorder is None:
                raise RuntimeError("vehicle_state output requires the MOVES emission recorder")
            emission_classes = validate_hbefa4_vehicle_types(self._traci)
            self._vehicle_state_writer = VehicleStateParquetWriter(
                output_root=run_dir,
                episode=int(episode_id),
                sumo_seed=int(seed) if seed is not None else -1,
                emission_class_by_type=emission_classes,
                row_group_rows=int(
                    getattr(self.env_cfg, "vehicle_state_row_group_rows", 100_000)
                ),
                compression=str(
                    getattr(self.env_cfg, "vehicle_state_compression", "zstd")
                ),
                compression_level=int(
                    getattr(self.env_cfg, "vehicle_state_compression_level", 3)
                ),
            )
        if self.evaluation_logger is not None:
            self.evaluation_logger.start_episode(
                traci=self._traci,
                tc=self._tc,
                episode=int(episode_id),
                sumo_seed=int(seed if seed is not None else episode_id),
            )

    def close(self, parse_tripinfo: bool = True) -> Optional[EpisodeTrafficStats]:
        """
        关闭 TraCI 连接，解析 tripinfo.xml，返回 episode 交通统计。
        """
        self._close_vehicle_second_output()
        if not self._running:
            return None

        writer_error: Optional[BaseException] = None
        try:
            self._traci.close()
        except Exception:
            pass
        finally:
            self._running = False
            self._subscribed = False
            self._context_anchor = None
            time.sleep(0.8)

        if self._vehicle_state_writer is not None:
            try:
                self._vehicle_state_writer.close()
            except BaseException as exc:
                writer_error = exc
                self._vehicle_state_writer.abort()
            finally:
                self._vehicle_state_writer = None
        if writer_error is not None:
            raise RuntimeError("Failed to finalize vehicle_state Parquet output") from writer_error

        if not parse_tripinfo:
            return None
        self._validate_tls_phase_outputs()
        # 解析 tripinfo
        stats = self._parse_tripinfo(self._tripinfo_path)
        return stats

    def _flush_vehicle_second_buffer(self) -> None:
        if self._vehicle_second_writer is None or not self._vehicle_second_buffer:
            return
        self._vehicle_second_writer.writerows(self._vehicle_second_buffer)
        self._vehicle_second_buffer.clear()

    def _maybe_flush_vehicle_second_buffer(self) -> None:
        interval = max(1, int(self.env_cfg.decision_interval))
        if int(self._sim_time) % interval == 0:
            self._flush_vehicle_second_buffer()

    def _close_vehicle_second_output(self) -> None:
        self._flush_vehicle_second_buffer()
        if self._vehicle_second_file is not None:
            self._vehicle_second_file.flush()
            self._vehicle_second_file.close()
        self._vehicle_second_file = None
        self._vehicle_second_writer = None

    def _validate_tls_phase_outputs(self) -> None:
        if not bool(getattr(self.env_cfg, "save_tls_phase_outputs", False)):
            return
        expected = (
            (self._tls_switch_states_path, "tlsstates"),
            (self._tls_switch_times_path, "tlsswitches"),
        )
        for path, expected_root in expected:
            if not path or not os.path.isfile(path):
                raise RuntimeError(f"Missing SUMO TLS output: {path}")
            if os.path.getsize(path) <= 0:
                raise RuntimeError(f"Empty SUMO TLS output: {path}")
            try:
                root = ET.parse(path).getroot()
            except (ET.ParseError, OSError) as exc:
                raise RuntimeError(
                    f"Invalid SUMO TLS output XML: {path}: {exc}"
                ) from exc
            actual_root = self._xml_local_name(root.tag)
            if actual_root != expected_root:
                raise RuntimeError(
                    "Unexpected SUMO TLS output root: "
                    f"{path}: expected={expected_root}, actual={actual_root}"
                )

    def _close_stale_traci_connection(self) -> None:
        if self._traci is None:
            return
        try:
            is_loaded = bool(self._traci.isLoaded()) if hasattr(self._traci, "isLoaded") else False
        except Exception:
            is_loaded = False
        if not is_loaded:
            return
        try:
            self._traci.close(wait=False)
        except TypeError:
            try:
                self._traci.close()
            except Exception:
                pass
        except Exception:
            pass
        finally:
            self._running = False
            time.sleep(0.8)

    # ─────────────────────────────────────────────────────────────────
    # 主步进接口
    # ─────────────────────────────────────────────────────────────────

    def step(
        self,
        actions: Dict[str, int],
        decision_step: Optional[int] = None,
    ) -> Dict[str, StepRawObs]:
        """
        执行一个决策步（decision_interval秒）。

        参数
        ----
        actions : Dict[tl_id -> green_phase_index]
                  agent 输出的相位选择（0-based，在 green_phases 中的 index）

        返回
        ----
        Dict[tl_id -> StepRawObs]：所有路口的原始物理观测

        执行流程
        --------
        每个决策步 = decision_interval 次 simulationStep（每次 1 秒）：
          秒 0: 处理动作（若切换则设黄灯，若保持则续绿）
          秒 1-2: 黄灯期（若切换）
          秒 3: 黄灯结束，设置新相位绿灯（若切换）
          秒 4: 绿灯中
          decision_interval 结束（即下一步秒 0）: 采集观测
        """
        assert self._running, "env 未启动，请先调用 start()"

        # 秒 0：处理所有路口的动作（设黄灯或续绿）
        if decision_step is None:
            decision_step = int(self._sim_time // self.env_cfg.decision_interval)
        self._apply_actions(actions)

        # 推进当前 decision_interval（每秒 1 步）
        self._reset_interval_wait()
        self._reset_interval_pass()
        self._reset_interval_green_service()
        for sub_step in range(self.env_cfg.decision_interval):
            self._traci.simulationStep()
            self._sim_time += 1.0
            self._current_vehicle_context = {}
            self._accumulate_interval_wait()
            self._record_emissions_for_second(int(decision_step))
            self._record_evaluation_for_second(int(decision_step))
            self._accumulate_interval_pass()
            self._accumulate_interval_green_service()

            # 在每秒检查是否需要从黄灯切换到新相位绿灯
            self._update_yellow_transitions(sub_step + 1)

        self._last_wait_vwt_by_lane = dict(self._interval_wait_accum)
        self._last_pass_by_lane = dict(self._interval_pass_accum)
        self._last_green_service_by_lane = dict(self._interval_green_service_accum)

        # 刷新车辆类型缓存，供 _collect_obs() 计算 truck_count
        self._refresh_vehicle_type_cache()

        # 采集所有路口观测
        if self.emission_recorder is not None:
            self._last_step_emission = self.emission_recorder.finalize_decision_step(int(decision_step))
            if self._vehicle_state_writer is not None:
                self._vehicle_state_writer.commit_decision_step(
                    self.emission_recorder.last_detail_batch
                )
        else:
            self._last_step_emission = None

        # 先更新信号状态计时，确保 _collect_obs 读到的 elapsed_green 是当前步的准确值
        self._update_signal_timing()

        # Read SUMO's executed phase for diagnostics only; do not write signal states.
        self._last_sumo_phase = {}
        for tl_id in self._signal_states.keys():
            self._last_sumo_phase[tl_id] = self._read_sumo_phase_readonly(tl_id)

        raw_obs = {}
        for tl_id in self.net_info.intersection_ids:
            raw_obs[tl_id] = self._collect_obs(tl_id, self._last_step_emission)

        return raw_obs

    # ─────────────────────────────────────────────────────────────────
    # 信号控制
    # ─────────────────────────────────────────────────────────────────

    def step_passive(
        self,
        decision_step: Optional[int] = None,
    ) -> Dict[str, StepRawObs]:
        """
        仅推进 SUMO，不手动控制信号灯。
        用于 SUMO 原生 actuated baseline。

        每步结束后从 TraCI 同步真实相位状态到 _signal_states，
        使 _collect_obs() 返回的 StepRawObs 包含真实的相位信息
        （current_phase / elapsed_green / in_yellow / phase_last_went_red）。
        """
        assert self._running, "env 未启动，请先调用 start()"

        if decision_step is None:
            decision_step = int(self._sim_time // self.env_cfg.decision_interval)

        self._reset_interval_wait()
        self._reset_interval_pass()
        self._reset_interval_green_service()
        for _ in range(self.env_cfg.decision_interval):
            self._traci.simulationStep()
            self._sim_time += 1.0
            self._current_vehicle_context = {}
            self._accumulate_interval_wait()
            self._record_emissions_for_second(int(decision_step))
            self._record_evaluation_for_second(int(decision_step))
            self._accumulate_interval_pass()
            self._accumulate_interval_green_service()

        self._last_wait_vwt_by_lane = dict(self._interval_wait_accum)
        self._last_pass_by_lane = dict(self._interval_pass_accum)
        self._last_green_service_by_lane = dict(self._interval_green_service_accum)

        # 刷新车辆类型缓存，供 _collect_obs() 计算 truck_count
        self._refresh_vehicle_type_cache()

        if self.emission_recorder is not None:
            self._last_step_emission = self.emission_recorder.finalize_decision_step(
                int(decision_step)
            )
            if self._vehicle_state_writer is not None:
                self._vehicle_state_writer.commit_decision_step(
                    self.emission_recorder.last_detail_batch
                )
        else:
            self._last_step_emission = None

        # ── 同步 SUMO 真实相位状态到 _signal_states ──────────────────
        self._sync_passive_signal_states()

        # ── 采集观测（此时 _signal_states 已是真实状态）─────────────
        raw_obs = {}
        for tl_id in self.net_info.intersection_ids:
            raw_obs[tl_id] = self._collect_obs(tl_id, self._last_step_emission)

        return raw_obs

    def _init_signal_states(self, fixed_time: bool) -> None:
        """
        初始化每个路口的信号状态。

        将所有路口从相位 0 开始，清零计时器。
        fixed_time=True 时设置固定时长绿灯（不由 agent 控制）。
        """
        self._last_sumo_phase = {}
        for tl_id in self.net_info.intersection_ids:
            iinfo    = self.net_info.get_intersection(tl_id)
            n_phases = iinfo.n_green_phases

            ss = SignalState(
                tl_id               = tl_id,
                current_phase       = 0,
                elapsed_green       = 0.0,
                in_yellow           = False,
                yellow_remaining    = 0.0,
                pending_phase       = -1,
                last_green_start    = 0.0,
                phase_last_went_red = [-self.env_cfg.min_red] * n_phases,
                # 初始化为 -min_red，确保所有相位在 t=0 时均可选择
            )
            self._signal_states[tl_id] = ss

            # 在 SUMO 中设置初始相位（phase 0）
            # SUMO tlLogic 中 phase index 对应 _tl_phase_idx() 的转换
            green_state = self._get_green_state(tl_id, 0)
            self._set_tls_state(tl_id, green_state)

            if fixed_time:
                pass

    def _apply_actions(self, actions: Dict[str, int]) -> None:
        """
        在决策步开始时执行所有路口的动作。

        每个路口：
          - 若动作 = 当前相位 → 保持，设置 decision_interval 持续时间
          - 若动作 ≠ 当前相位 → 切换，设置黄灯 3s，记录待切换相位
        """
        yellow_dur = self.env_cfg.yellow_duration  # 3

        for tl_id, action in actions.items():
            ss    = self._signal_states[tl_id]
            iinfo = self.net_info.get_intersection(tl_id)

            if action == ss.current_phase:
                # 保持当前相位：续绿 decision_interval 秒
                green_state = self._get_green_state(tl_id, ss.current_phase)
                self._set_tls_state(tl_id, green_state)
                ss.in_yellow      = False
                ss.yellow_remaining = 0.0
                ss.pending_phase  = -1

            else:
                # 切换相位：先执行黄灯
                old_green_state = self._get_green_state(tl_id, ss.current_phase)
                yellow_state = self._build_yellow_state_from_old_green(old_green_state)
                self._set_tls_state(tl_id, yellow_state)

                ss.in_yellow        = True
                ss.yellow_remaining = float(yellow_dur)
                ss.pending_phase    = action

                # 记录当前相位变为红灯的时刻（用于 min_red 约束）
                ss.phase_last_went_red[ss.current_phase] = self._sim_time

    def _update_yellow_transitions(self, elapsed_in_step: int) -> None:
        """
        每推进 1 秒后检查是否有路口需要从黄灯切换到新相位绿灯。

        elapsed_in_step：当前决策步内已过去的秒数（1~5）
        """
        yellow_dur = self.env_cfg.yellow_duration  # 3

        for tl_id, ss in self._signal_states.items():
            if not ss.in_yellow:
                continue

            ss.yellow_remaining -= 1.0

            if ss.yellow_remaining <= 0:
                # 黄灯结束，切换到新相位绿灯
                new_phase = ss.pending_phase
                new_green_state = self._get_green_state(tl_id, new_phase)
                self._set_tls_state(tl_id, new_green_state)

                ss.current_phase    = new_phase
                ss.in_yellow        = False
                ss.yellow_remaining = 0.0
                ss.pending_phase    = -1
                ss.elapsed_green    = 0.0  # 绿灯计时从黄灯结束起
                ss.last_green_start = self._sim_time

    def _update_signal_timing(self) -> None:
        """
        每决策步结束后更新 elapsed_green 计时。
        绿灯时长 = 上次黄灯结束 ~ 当前时刻（ss.last_green_start 之差）
        """
        for tl_id, ss in self._signal_states.items():
            if not ss.in_yellow:
                ss.elapsed_green = self._sim_time - ss.last_green_start

    def _green_to_sumo_phase(self, tl_id: str, green_phase_idx: int) -> int:
        """
        调试用：返回解析阶段保留的 SUMO 原始绿灯 phase index。
        """
        iinfo = self.net_info.get_intersection(tl_id)
        return int(iinfo.green_phases[green_phase_idx].sumo_phase_index)

    def _yellow_phase_idx(self, tl_id: str, green_phase_idx: int) -> int:
        """
        调试用：返回解析阶段保留的黄灯 phase index；主控制不依赖它。
        """
        iinfo = self.net_info.get_intersection(tl_id)
        yellow_idx = iinfo.green_phases[green_phase_idx].yellow_phase_index
        if yellow_idx is None:
            return -1
        return int(yellow_idx)

    def _get_green_state(self, tl_id: str, green_phase_idx: int) -> str:
        iinfo = self.net_info.get_intersection(tl_id)
        if green_phase_idx < 0 or green_phase_idx >= iinfo.n_green_phases:
            raise ValueError(f"Invalid green_phase_index={green_phase_idx} for tl_id={tl_id}")
        return iinfo.green_phases[green_phase_idx].state

    def _build_yellow_state_from_old_green(self, green_state: str) -> str:
        return "".join("y" if ch == "G" else ch for ch in green_state)

    def _set_tls_state(self, tl_id: str, state: str) -> None:
        iinfo = self.net_info.get_intersection(tl_id)
        expected_len = iinfo.state_len
        if len(state) != expected_len:
            raise ValueError(
                f"TLS state length mismatch: tl_id={tl_id}, "
                f"len(state)={len(state)}, expected={expected_len}, state={state!r}"
            )
        self._traci.trafficlight.setRedYellowGreenState(tl_id, state)

    def _green_index_from_sumo_state(
        self,
        tl_id: str,
        sumo_phase: int,
        phase_state: str,
    ) -> Optional[int]:
        """将 SUMO 原始 phase index/state 映射为 green_phases 的 0-based 索引。

        注意：SUMO tlLogic 的 phase index 通常是 0,1,2,3,...，其中绿灯和黄灯
        交替出现；而模型动作空间使用的是过滤后的 green_phases 索引。不能简单
        用 `sumo_phase < n_green_phases` 判断。
        """
        iinfo = self.net_info.get_intersection(tl_id)
        for idx, ph in enumerate(iinfo.green_phases):
            if int(getattr(ph, "sumo_phase_index", -1)) == int(sumo_phase):
                return idx
            if str(getattr(ph, "state", "")) == str(phase_state):
                return idx
        return None

    def _yellow_origin_green_index_from_sumo_state(
        self,
        tl_id: str,
        sumo_phase: int,
        phase_state: str,
    ) -> Optional[int]:
        """黄灯相位映射为其来源绿灯相位索引。"""
        iinfo = self.net_info.get_intersection(tl_id)
        for idx, ph in enumerate(iinfo.green_phases):
            yidx = getattr(ph, "yellow_phase_index", None)
            ystate = getattr(ph, "yellow_state", None)
            if yidx is not None and int(yidx) == int(sumo_phase):
                return idx
            if ystate and str(ystate) == str(phase_state):
                return idx
        return None

    def _read_sumo_phase_readonly(self, tl_id: str) -> tuple[int, int]:
        """Read SUMO's real phase for diagnostics without mutating _signal_states."""
        traci_tl = self._traci.trafficlight
        try:
            sumo_raw = int(traci_tl.getPhase(tl_id))
            phase_state = traci_tl.getRedYellowGreenState(tl_id)
            green_idx = self._green_index_from_sumo_state(tl_id, sumo_raw, phase_state)
            if green_idx is None:
                green_idx = self._yellow_origin_green_index_from_sumo_state(
                    tl_id, sumo_raw, phase_state
                )
            return (int(green_idx) if green_idx is not None else -1, sumo_raw)
        except Exception:
            return (-1, -1)

    def _sync_passive_signal_states(self) -> None:
        """从 TraCI 同步 SUMO 原生 actuated 的真实相位状态。

        该函数只在 control_tls=False 的 baseline / 预实验被动模式下使用。
        它解决两类问题：
        1. SUMO 原始 phase index 与模型 green_phase_index 不一致；
        2. 黄灯相位需要保留来源绿灯相位，同时更新 in_yellow 和红灯计时。
        """
        traci_tl = self._traci.trafficlight
        for tl_id, ss in self._signal_states.items():
            iinfo = self.net_info.get_intersection(tl_id)
            n_phases = iinfo.n_green_phases
            if n_phases <= 0:
                continue

            old_phase = int(np.clip(ss.current_phase, 0, n_phases - 1))
            sumo_phase = int(traci_tl.getPhase(tl_id))
            phase_state = traci_tl.getRedYellowGreenState(tl_id)
            elapsed = float(traci_tl.getSpentDuration(tl_id))
            in_yellow = "y" in phase_state or "Y" in phase_state

            green_idx = self._green_index_from_sumo_state(tl_id, sumo_phase, phase_state)
            yellow_origin_idx = self._yellow_origin_green_index_from_sumo_state(
                tl_id, sumo_phase, phase_state
            )

            if green_idx is not None:
                if green_idx != old_phase:
                    ss.phase_last_went_red[old_phase] = self._sim_time
                ss.current_phase = int(green_idx)
                ss.in_yellow = False
                ss.yellow_remaining = 0.0
                ss.pending_phase = -1
                ss.elapsed_green = max(0.0, elapsed)
                ss.last_green_start = self._sim_time - ss.elapsed_green
                continue

            if in_yellow:
                # 黄灯来源于某个绿灯相位；保持/更新 current_phase 为该来源相位。
                if yellow_origin_idx is not None:
                    origin = int(yellow_origin_idx)
                    if origin != old_phase:
                        ss.phase_last_went_red[old_phase] = self._sim_time
                    elif not ss.in_yellow:
                        # 刚从该绿灯进入黄灯，记录该相位变红时刻。
                        ss.phase_last_went_red[origin] = self._sim_time - max(0.0, elapsed)
                    ss.current_phase = origin

                ss.in_yellow = True
                ss.elapsed_green = 0.0
                ss.yellow_remaining = max(0.0, float(self.env_cfg.yellow_duration) - max(0.0, elapsed))
                ss.pending_phase = -1
                continue

            # 无法识别的状态：不改变 current_phase，仅同步黄灯标记和计时，保证不崩溃。
            ss.in_yellow = False
            ss.yellow_remaining = 0.0
            ss.elapsed_green = max(0.0, elapsed)

    def _init_passive_signal_states(self) -> None:
        """用于 SUMO 原生 actuated baseline 的真实相位状态初始化。

        不向 SUMO 写入任何灯态，只先构造 SignalState，再调用
        _sync_passive_signal_states() 按 TraCI 真实 phase/state 同步。
        """
        self._signal_states = {}
        self._last_sumo_phase = {}
        for tl_id in self.net_info.intersection_ids:
            iinfo = self.net_info.get_intersection(tl_id)
            n_phases = iinfo.n_green_phases
            self._signal_states[tl_id] = SignalState(
                tl_id=tl_id,
                current_phase=0,
                elapsed_green=0.0,
                in_yellow=False,
                yellow_remaining=0.0,
                pending_phase=-1,
                last_green_start=0.0,
                phase_last_went_red=[-self.env_cfg.min_red] * n_phases,
            )

        self._sync_passive_signal_states()

    def _activate_tls_program(self, program_id: str) -> None:
        for tl_id in self.net_info.intersection_ids:
            logics = self._traci.trafficlight.getAllProgramLogics(tl_id)
            available = [getattr(logic, "programID", "") for logic in logics]
            if program_id not in available:
                raise ValueError(
                    f"TLS {tl_id} does not provide programID={program_id!r}; "
                    f"available programs={available}. Check baseline_sumo_cfg and actuated tll.xml."
                )
            self._traci.trafficlight.setProgram(tl_id, program_id)

    def _setup_subscriptions(self) -> None:
        """建立 lane 订阅 + 全网车辆 context 订阅，减少 TraCI 往返。"""
        if self._traci is None or self._tc is None:
            return

        traci = self._traci
        tc = self._tc

        # 1) 收集所有进口 lane，建立 lane subscription
        self._all_inc_lanes = []
        seen = set()
        for tl_id in self.net_info.intersection_ids:
            iinfo = self.net_info.get_intersection(tl_id)
            for lane_id in iinfo.all_inc_lanes_flat():
                if lane_id and lane_id not in seen:
                    seen.add(lane_id)
                    self._all_inc_lanes.append(lane_id)

        for lane_id in self._all_inc_lanes:
            try:
                traci.lane.subscribe(lane_id, [
                    tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
                    tc.LAST_STEP_MEAN_SPEED,
                    tc.LAST_STEP_VEHICLE_ID_LIST,
                ])
            except Exception:
                # 某些 SUMO 版本常量名可能不同，后续 _collect_obs 会 fallback 到旧 TraCI 调用
                pass

        # 2) 建立全网车辆 context subscription，用一个有效 lane 作为 anchor
        self._context_anchor = None
        if (
            self.emission_recorder is not None
            or self.evaluation_logger is not None
            or bool(getattr(self.env_cfg, "save_vehicle_second_output", False))
            or bool(getattr(self.env_cfg, "save_vehicle_state_output", False))
        ) and self._all_inc_lanes:
            anchor = self._all_inc_lanes[0]
            try:
                var_ids = [
                    tc.VAR_TYPE,
                    tc.VAR_SPEED,
                    tc.VAR_ACCELERATION,
                    tc.VAR_LANE_ID,
                    tc.VAR_ROAD_ID,
                    tc.VAR_DISTANCE,
                ]
                if bool(getattr(self.env_cfg, "save_vehicle_state_output", False)):
                    var_ids.append(tc.VAR_NOXEMISSION)
                if (
                    self.evaluation_logger is not None
                    or bool(getattr(self.env_cfg, "save_vehicle_second_output", False))
                    or bool(getattr(self.env_cfg, "save_vehicle_state_output", False))
                ):
                    var_ids.extend([
                        tc.VAR_TIMELOSS,
                        tc.VAR_WAITING_TIME,
                    ])
                traci.lane.subscribeContext(
                    anchor,
                    tc.CMD_GET_VEHICLE_VARIABLE,
                    1000000.0,
                    var_ids,
                )
                self._context_anchor = anchor
            except Exception:
                self._context_anchor = None

        # 3) 建立 induction loop (E1) 单步通过量订阅，供逐秒 pass 累加使用。
        self._subscribed_e1_step = set()
        try:
            e1_step_var = tc.LAST_STEP_VEHICLE_NUMBER
        except Exception:
            e1_step_var = None
        if e1_step_var is not None:
            e1_ids = {
                det_id for det_id in self.net_info.lane_to_e1_all.values() if det_id
            }
            for det_id in e1_ids:
                try:
                    traci.inductionloop.subscribe(det_id, [e1_step_var])
                    self._subscribed_e1_step.add(det_id)
                except Exception:
                    # 该检测器订阅失败时，_accumulate_interval_pass 会回退到直接查询。
                    pass

        self._subscribed = True

    def _validate_observation_detectors(self) -> None:
        """Require every detector needed by the configured RL observation scope."""
        use_e2 = str(self.env_cfg.observation_scope).strip().lower() == "upstream_300m"
        for lane_id in self._all_incoming_lane_ids():
            if lane_id not in self.net_info.lane_to_e1_all:
                raise ValueError(
                    f"lane_id={lane_id}, missing detector type=E1, "
                    f"add.xml={self.net_info.add_xml_path}"
                )
            if use_e2 and lane_id not in self.net_info.lane_to_e2:
                raise ValueError(
                    f"lane_id={lane_id}, missing detector type=E2 for "
                    "state.observation_scope='upstream_300m', "
                    f"add.xml={self.net_info.add_xml_path}"
                )

    def _refresh_vehicle_type_cache(self) -> None:
        """刷新车辆类型缓存，供 _collect_obs() 计算 truck_count 使用。"""
        traci = self._traci
        if traci is None:
            return

        # 如果 emission context subscription 已经在 _record_emissions_for_second() 中刷新类型缓存，
        # 这里不重复全量查询。
        if self.emission_recorder is not None and getattr(self, "_context_anchor", None):
            return

        try:
            current_ids = set(traci.vehicle.getIDList())
        except Exception:
            return

        # 清理离网车辆，避免缓存无限增长
        for vid in list(self._veh_type_cache.keys()):
            if vid not in current_ids:
                del self._veh_type_cache[vid]

        # 只对新出现车辆查询一次 getTypeID
        for vid in current_ids:
            if vid not in self._veh_type_cache:
                try:
                    self._veh_type_cache[vid] = traci.vehicle.getTypeID(vid)
                except Exception:
                    self._veh_type_cache[vid] = ""

    def _record_emissions_for_second(self, decision_step: int) -> None:
        save_vehicle_second = bool(
            getattr(self.env_cfg, "save_vehicle_second_output", False)
        )
        if self.emission_recorder is None and not save_vehicle_second:
            return
        tc = self._tc
        anchor = getattr(self, "_context_anchor", None)

        # 优先使用 context subscription 批量结果
        if tc is not None and anchor is not None:
            try:
                results = self._traci.lane.getContextSubscriptionResults(anchor) or {}
                self._current_vehicle_context = results
            except Exception as exc:
                if self._vehicle_state_writer is not None:
                    raise RuntimeError(
                        "Failed to read the vehicle context subscription required "
                        "for vehicle_state output"
                    ) from exc
                results = {}

            if results:
                current_ids = set(results.keys())

                # 清理离网车辆缓存
                for vid in list(self._veh_type_cache.keys()):
                    if vid not in current_ids:
                        del self._veh_type_cache[vid]

                for veh_id, vals in results.items():
                    vtype = vals.get(tc.VAR_TYPE, "") or self._veh_type_cache.get(veh_id, "")
                    if vtype:
                        self._veh_type_cache[veh_id] = str(vtype)

                    lane_id = vals.get(tc.VAR_LANE_ID, "")
                    edge_id = vals.get(tc.VAR_ROAD_ID, "")

                    speed_mps = float(vals.get(tc.VAR_SPEED, 0.0))
                    acceleration_mps2 = float(vals.get(tc.VAR_ACCELERATION, 0.0))
                    time_loss_s = float(vals.get(tc.VAR_TIMELOSS, 0.0))
                    waiting = float(vals.get(tc.VAR_WAITING_TIME, 0.0)) > 0.0
                    normalized_type = EmissionFactorLookup.normalize_vehicle_type(vtype)
                    in_observation_zone = (
                        str(veh_id) in self._e2_vehicle_ids_by_lane.get(str(lane_id), ())
                    )
                    moves_scope = bool(lane_id and edge_id and not str(edge_id).startswith(":"))
                    row_id = -1
                    if self._vehicle_state_writer is not None:
                        row_id = self._vehicle_state_writer.append_vehicle(
                            simulation_time_s=self._sim_time,
                            decision_step=int(decision_step),
                            vehicle_id=str(veh_id),
                            vehicle_type_raw=str(vtype),
                            vehicle_type_used=normalized_type,
                            lane_id=str(lane_id),
                            edge_id=str(edge_id),
                            speed_mps=speed_mps,
                            acceleration_mps2=acceleration_mps2,
                            time_loss_accumulated_s=time_loss_s,
                            is_waiting=waiting,
                            in_observation_zone=in_observation_zone,
                            observation_lane_id=str(lane_id),
                            included_in_moves_scope=moves_scope,
                            nox_hbefa4_mg_s=float(vals.get(tc.VAR_NOXEMISSION, 0.0)),
                        )

                    if save_vehicle_second and lane_id:
                        self._vehicle_second_buffer.append([
                            self._current_episode_id,
                            self._current_sumo_seed,
                            float(self._sim_time),
                            str(veh_id),
                            normalized_type,
                            str(lane_id),
                            speed_mps,
                            acceleration_mps2,
                            time_loss_s,
                            int(waiting),
                        ])

                    # 过滤无效车辆状态
                    if not moves_scope:
                        continue

                    if self.emission_recorder is None:
                        continue
                    self.emission_recorder.record_vehicle_state(
                        sim_time=self._sim_time,
                        decision_step=int(decision_step),
                        veh_id=veh_id,
                        vehicle_type=vtype,
                        speed_ms=speed_mps,
                        accel_ms2=acceleration_mps2,
                        lane_id=lane_id,
                        edge_id=edge_id,
                        dt=1.0,
                        distance_m=vals.get(tc.VAR_DISTANCE, 0.0),
                        in_observation_zone=in_observation_zone,
                        observation_lane_id=str(lane_id),
                        row_id=row_id,
                    )
                self._maybe_flush_vehicle_second_buffer()
                return

            if self._vehicle_state_writer is not None:
                # An empty mapping is valid during seconds with no active vehicles.
                return

        # fallback：旧逻辑，保证 subscription 失败时仍可运行
        traci = self._traci
        for veh_id in traci.vehicle.getIDList():
            try:
                vtype = self._veh_type_cache.get(veh_id)
                if not vtype:
                    vtype = traci.vehicle.getTypeID(veh_id)
                    self._veh_type_cache[veh_id] = vtype

                edge_id = traci.vehicle.getRoadID(veh_id)
                lane_id = traci.vehicle.getLaneID(veh_id)
                speed_mps = traci.vehicle.getSpeed(veh_id)
                acceleration_mps2 = traci.vehicle.getAcceleration(veh_id)
                time_loss_s = float(traci.vehicle.getTimeLoss(veh_id))
                waiting = float(traci.vehicle.getWaitingTime(veh_id)) > 0.0
                normalized_type = EmissionFactorLookup.normalize_vehicle_type(vtype)
                in_observation_zone = (
                    str(veh_id) in self._e2_vehicle_ids_by_lane.get(str(lane_id), ())
                )
                moves_scope = bool(
                    lane_id and edge_id and not str(edge_id).startswith(":")
                )
                row_id = -1
                if self._vehicle_state_writer is not None:
                    row_id = self._vehicle_state_writer.append_vehicle(
                        simulation_time_s=self._sim_time,
                        decision_step=int(decision_step),
                        vehicle_id=str(veh_id),
                        vehicle_type_raw=str(vtype),
                        vehicle_type_used=normalized_type,
                        lane_id=str(lane_id),
                        edge_id=str(edge_id),
                        speed_mps=float(speed_mps),
                        acceleration_mps2=float(acceleration_mps2),
                        time_loss_accumulated_s=time_loss_s,
                        is_waiting=waiting,
                        in_observation_zone=in_observation_zone,
                        observation_lane_id=str(lane_id),
                        included_in_moves_scope=moves_scope,
                        nox_hbefa4_mg_s=float(
                            traci.vehicle.getNOxEmission(veh_id)
                        ),
                    )
                if save_vehicle_second and lane_id:
                    self._vehicle_second_buffer.append([
                        self._current_episode_id,
                        self._current_sumo_seed,
                        float(self._sim_time),
                        str(veh_id),
                        normalized_type,
                        str(lane_id),
                        float(speed_mps),
                        float(acceleration_mps2),
                        time_loss_s,
                        int(waiting),
                    ])
                if not moves_scope:
                    continue
                if self.emission_recorder is None:
                    continue

                self.emission_recorder.record_vehicle_state(
                    sim_time=self._sim_time,
                    decision_step=int(decision_step),
                    veh_id=veh_id,
                    vehicle_type=vtype,
                    speed_ms=speed_mps,
                    accel_ms2=acceleration_mps2,
                    lane_id=lane_id,
                    edge_id=edge_id,
                    dt=1.0,
                    distance_m=traci.vehicle.getDistance(veh_id),
                    in_observation_zone=in_observation_zone,
                    observation_lane_id=str(lane_id),
                    row_id=row_id,
                )
            except Exception:
                # vehicle_state is an evaluation audit artifact: silently
                # dropping a vehicle here would produce a plausible but
                # incomplete Parquet file. Fail the episode instead.
                if self._vehicle_state_writer is not None:
                    raise
                continue
        self._maybe_flush_vehicle_second_buffer()

    def _record_evaluation_for_second(self, decision_step: int) -> None:
        """Forward the post-step vehicle snapshot to the optional evaluator."""
        if self.evaluation_logger is None:
            return
        vehicle_context: Mapping[str, Mapping[int, Any]] = self._current_vehicle_context
        anchor = getattr(self, "_context_anchor", None)
        if not vehicle_context and anchor is not None:
            try:
                vehicle_context = (
                    self._traci.lane.getContextSubscriptionResults(anchor) or {}
                )
            except Exception:
                vehicle_context = {}
        self.evaluation_logger.collect_second(
            sim_time=float(self._sim_time),
            decision_step=int(decision_step),
            vehicle_context=vehicle_context,
        )

    # ─────────────────────────────────────────────────────────────────
    # 观测数据采集
    # ─────────────────────────────────────────────────────────────────

    def _all_incoming_lane_ids(self) -> List[str]:
        """返回全网所有受控路口进口 lane，去重且保持路口/lane 顺序稳定。"""
        lanes: List[str] = []
        seen = set()
        for tl_id in self.net_info.intersection_ids:
            for lid in self.net_info.get_intersection(tl_id).all_inc_lanes_flat():
                if lid in seen:
                    continue
                seen.add(lid)
                lanes.append(lid)
        return lanes

    def _reset_interval_wait(self) -> None:
        """决策步开始时清零 lane 区间等待增量累加器。"""
        if not self._interval_wait_accum:
            for lane_id in self._all_incoming_lane_ids():
                self._interval_wait_accum[lane_id] = 0.0
        else:
            for lane_id in self._interval_wait_accum:
                self._interval_wait_accum[lane_id] = 0.0

    def _accumulate_interval_wait(self) -> None:
        """每推进 1 秒后累计配置观测范围内的等待车辆数。"""
        if not self._interval_wait_accum:
            self._reset_interval_wait()
        use_e2 = str(self.env_cfg.observation_scope).strip().lower() == "upstream_300m"
        for lane_id in list(self._interval_wait_accum.keys()):
            detector_id = self.net_info.lane_to_e2.get(lane_id, "") if use_e2 else ""
            if detector_id:
                n_wait = float(self._traci.lanearea.getLastStepHaltingNumber(detector_id))
                self._e2_halting_by_lane[lane_id] = n_wait
                self._e2_vehicle_ids_by_lane[lane_id] = frozenset(
                    self._traci.lanearea.getLastStepVehicleIDs(detector_id)
                )
            else:
                n_wait = float(self._traci.lane.getLastStepHaltingNumber(lane_id))
            self._interval_wait_accum[lane_id] = (
                float(self._interval_wait_accum.get(lane_id, 0.0)) + n_wait
            )

    def _reset_interval_pass(self) -> None:
        """决策步开始时清零 E1 逐秒通过数累加器。"""
        if not self._interval_pass_accum:
            for lane_id in self._all_incoming_lane_ids():
                self._interval_pass_accum[lane_id] = 0.0
        else:
            for lane_id in self._interval_pass_accum:
                self._interval_pass_accum[lane_id] = 0.0

    def _accumulate_interval_pass(self) -> None:
        """每推进 1 秒累计一次各 lane 对应 E1 检测器的通过车辆数。"""
        traci = self._traci
        tc = self._tc
        if not self._interval_pass_accum:
            self._reset_interval_pass()
        for lane_id in list(self._interval_pass_accum.keys()):
            e1_id = self.net_info.lane_to_e1_all.get(lane_id, "")
            n_pass = 0.0
            if e1_id:
                read_from_subscription = False
                if tc is not None and e1_id in self._subscribed_e1_step:
                    try:
                        var_id = tc.LAST_STEP_VEHICLE_NUMBER
                        sub = traci.inductionloop.getSubscriptionResults(e1_id) or {}
                        if var_id in sub:
                            n_pass = float(sub[var_id])
                            read_from_subscription = True
                    except Exception:
                        read_from_subscription = False
                if not read_from_subscription:
                    try:
                        n_pass = float(traci.inductionloop.getLastStepVehicleNumber(e1_id))
                    except Exception:
                        n_pass = 0.0
            self._interval_pass_accum[lane_id] = (
                float(self._interval_pass_accum.get(lane_id, 0.0)) + n_pass
            )

    def _reset_interval_green_service(self) -> None:
        """决策周期开始前清零每条进口 lane 的实际有效绿灯秒数。"""
        self._interval_green_service_accum = {
            lane_id: 0.0 for lane_id in self._all_incoming_lane_ids()
        }

    def _accumulate_interval_green_service(self) -> None:
        """按 SUMO 当前实际 TLS 状态累计合法绿色相位的服务秒数。"""
        if not self._interval_green_service_accum:
            self._reset_interval_green_service()
        for tl_id in self.net_info.intersection_ids:
            iinfo = self.net_info.get_intersection(tl_id)
            try:
                sumo_phase = int(self._traci.trafficlight.getPhase(tl_id))
                phase_state = str(
                    self._traci.trafficlight.getRedYellowGreenState(tl_id)
                )
            except Exception as exc:
                raise RuntimeError(
                    f"无法读取 TLS 实际状态以累计有效绿灯: tl_id={tl_id}"
                ) from exc
            green_idx = self._green_index_from_sumo_state(
                tl_id, sumo_phase, phase_state
            )
            if green_idx is None:
                # yellow、all-red 或无法映射的状态均不计有效绿灯。
                continue
            phase_lane_mask = self._phase_lane_masks[tl_id]
            if green_idx >= phase_lane_mask.shape[0]:
                raise ValueError(
                    f"绿色相位索引越界: tl_id={tl_id}, green_idx={green_idx}, "
                    f"mask_shape={phase_lane_mask.shape}"
                )
            lanes = iinfo.all_inc_lanes_flat()
            lane_mask = phase_lane_mask[green_idx]
            if len(lanes) != len(lane_mask):
                raise ValueError(
                    f"lane mask 长度不一致: tl_id={tl_id}, "
                    f"lanes={len(lanes)}, mask={len(lane_mask)}"
                )
            for lane_id, served in zip(lanes, lane_mask):
                if bool(served):
                    self._interval_green_service_accum[lane_id] += 1.0

    def _collect_obs(
        self,
        tl_id: str,
        step_emission: Optional[DecisionStepEmission] = None,
    ) -> StepRawObs:
        """
        从 TraCI 采集单个路口的原始物理观测数据。
        """
        iinfo   = self.net_info.get_intersection(tl_id)
        ss      = self._signal_states[tl_id]
        traci   = self._traci

        all_lanes = iinfo.all_inc_lanes_flat()  # N→E→S→W 展平 lane 列表

        queue_veh   = []
        speed_mps   = []
        truck_count = []
        total_count = []
        demand_veh  = []
        pass_veh    = []
        NOx_mg      = []
        wait_vwt    = []
        green_service_s = []

        pollutant = self.env_cfg.emission_pollutants[0] if self.env_cfg.emission_pollutants else "NOx"
        lane_emission = (
            step_emission.by_lane
            if step_emission is not None
            else {}
        )
        observation_lane_emission = (
            step_emission.by_lane_observation_zone
            if step_emission is not None
            else {}
        )
        edge_emission = step_emission.by_edge if step_emission is not None else {}

        for lane_id in all_lanes:
            tc = self._tc
            sub = {}
            if tc is not None:
                try:
                    sub = traci.lane.getSubscriptionResults(lane_id) or {}
                except Exception:
                    sub = {}

            if sub and tc is not None:
                spd = sub.get(tc.LAST_STEP_MEAN_SPEED, 0.0)
            else:
                spd = traci.lane.getLastStepMeanSpeed(lane_id)

            if sub and tc is not None:
                queue = sub.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0.0)
                veh_ids = sub.get(tc.LAST_STEP_VEHICLE_ID_LIST, ())
            else:
                queue = traci.lane.getLastStepHaltingNumber(lane_id)
                veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)

            use_e2 = str(self.env_cfg.observation_scope).strip().lower() == "upstream_300m"
            if use_e2:
                # Initial observations are collected before the first
                # simulationStep, so the per-second E2 cache may be empty.
                # Populate it lazily from the current detector snapshot.
                if lane_id not in self._e2_vehicle_ids_by_lane:
                    detector_id = self.net_info.lane_to_e2.get(lane_id, "")
                    if not detector_id:
                        raise KeyError(f"Missing E2 detector mapping for lane {lane_id!r}")
                    self._e2_vehicle_ids_by_lane[lane_id] = frozenset(
                        traci.lanearea.getLastStepVehicleIDs(detector_id)
                    )
                    self._e2_halting_by_lane[lane_id] = float(
                        traci.lanearea.getLastStepHaltingNumber(detector_id)
                    )
                veh_ids = self._e2_vehicle_ids_by_lane[lane_id]
                queue = self._e2_halting_by_lane[lane_id]

            queue_veh.append(float(queue))
            speed_mps.append(float(spd))

            veh_ids = list(veh_ids or ())
            # 当前滞留货车数（窗口末端滞留部分）
            trucks_resident = sum(
                1 for vid in veh_ids
                if self._veh_type_cache.get(vid, "") == "truck"
            )
            total = len(veh_ids)

            # 近decision_interval秒 E1 货车检测器通过数（流量口径，与 demand 的 e1_cur 同语义；
            # 货车检测器 period 已由 ensure_e1_period 修复为 decision_interval）
            e1_truck_id = self.net_info.lane_to_e1_truck.get(lane_id, "")
            e1_truck_cur = 0
            if e1_truck_id:
                try:
                    e1_truck_cur = traci.inductionloop.getLastIntervalVehicleNumber(e1_truck_id)
                except Exception:
                    e1_truck_cur = 0

            # demand 口径货车数 = 近decision_interval秒 通过货车 + 当前滞留货车
            truck_count.append(float(e1_truck_cur + trucks_resident))
            total_count.append(float(total))

            # Demand = 近decision_interval秒逐秒累计通过数（getLastStepVehicleNumber 累加，不依赖 E1 period）
            #          + 当前 lane 车辆数
            e1_cur = float(self._last_pass_by_lane.get(lane_id, 0.0))
            pass_veh.append(float(e1_cur))
            demand = e1_cur + total
            demand_veh.append(float(demand))

            # RL lane NOx follows the configured observation scope.
            NOx_mg.append(
                float(
                    observation_lane_emission.get((lane_id, pollutant), 0.0)
                    if use_e2
                    else lane_emission.get((lane_id, pollutant), 0.0)
                )
            )

            wait_vwt.append(float(self._last_wait_vwt_by_lane.get(lane_id, 0.0)))
            green_service_s.append(
                float(self._last_green_service_by_lane.get(lane_id, 0.0))
            )

        # Edge 级 NOx（对每条进口 edge 的所有 lane 求和）
        edge_NOx = []
        for eid in iinfo.inc_edges_nesw:
            if eid is None:
                edge_NOx.append(0.0)
                continue
            edge_info = self.net_info.edge_info[eid]
            edge_NOx.append(float(edge_emission.get((eid, pollutant), 0.0)))

        _sg, _sr = self._last_sumo_phase.get(tl_id, (-1, -1))
        try:
            _sumo_state = self._traci.trafficlight.getRedYellowGreenState(tl_id)
        except Exception:
            _sumo_state = ""
        try:
            if ss.in_yellow:
                _env_state = self._build_yellow_state_from_old_green(
                    self._get_green_state(tl_id, ss.current_phase)
                )
            else:
                _env_state = self._get_green_state(tl_id, ss.current_phase)
        except Exception:
            _env_state = ""

        return StepRawObs(
            tl_id              = tl_id,
            queue_veh          = queue_veh,
            speed_mps          = speed_mps,
            truck_count        = truck_count,
            total_count        = total_count,
            demand_veh         = demand_veh,
            pass_veh           = pass_veh,
            NOx_mg             = NOx_mg,
            wait_vwt           = wait_vwt,
            green_service_s    = green_service_s,
            edge_NOx_mg        = edge_NOx,
            current_phase      = ss.current_phase,
            elapsed_green      = (self._sim_time - ss.last_green_start) if not ss.in_yellow else 0.0,
            in_yellow          = ss.in_yellow,
            phase_last_went_red = list(ss.phase_last_went_red),
            sim_time           = self._sim_time,
            sumo_green_phase   = _sg,
            sumo_raw_phase     = _sr,
            pending_phase      = int(ss.pending_phase),
            sumo_state         = _sumo_state,
            env_state          = _env_state,
        )

    # ─────────────────────────────────────────────────────────────────
    # Action Mask 计算（供 obs_reward.py 或 agent.py 调用）
    # ─────────────────────────────────────────────────────────────────

    def compute_action_mask(self, tl_id: str) -> np.ndarray:
        """
        基于当前信号状态计算 action mask。

        返回 bool 数组 [n_phases]，True = 可行动作。

        约束
        ----
        1. min_green：elapsed_green < min_green 时，强制保持当前相位
        2. max_green：elapsed_green >= max_green 时，禁止保持当前相位
        3. min_red：某相位上次变红时刻距今 < min_red，禁止选该相位

        fallback：若所有相位都被禁止，强制保持当前相位（安全兜底）
        """
        iinfo    = self.net_info.get_intersection(tl_id)
        ss       = self._signal_states[tl_id]
        n_phases = iinfo.n_green_phases
        cfg      = self.env_cfg

        mask = np.ones(n_phases, dtype=bool)

        # 1. min_green 约束：绿灯不足最短时长，只能保持
        if ss.elapsed_green < cfg.min_green:
            mask[:] = False
            mask[ss.current_phase] = True
            return mask

        # 2. max_green 约束：绿灯超过最长时长，不能继续保持
        if ss.elapsed_green >= cfg.max_green:
            mask[ss.current_phase] = False

        # 3. min_red 约束：各相位等红灯时间不足
        for k in range(n_phases):
            if k == ss.current_phase:
                continue  # 当前相位不受 min_red 约束
            elapsed_red = self._sim_time - ss.phase_last_went_red[k]
            if elapsed_red < cfg.min_red:
                mask[k] = False

        # fallback：若全部被禁，强制保持
        if not mask.any():
            mask[ss.current_phase] = True

        return mask

    def get_lanearea_halting(self, detector_id: str) -> int:
        """Read an E2 last-step halt count without changing simulation state."""
        return int(self._traci.lanearea.getLastStepHaltingNumber(detector_id))

    def get_lanearea_vehicle_ids(self, detector_id: str) -> Tuple[str, ...]:
        return tuple(self._traci.lanearea.getLastStepVehicleIDs(detector_id))

    def get_lanearea_snapshot(self, detector_ids) -> Dict[str, Dict]:
        area = self._traci.lanearea
        snapshots = {}
        for detector_id in detector_ids:
            jam_veh = getattr(area, "getJamLengthVehicle", None)
            jam_m = getattr(area, "getJamLengthMeters", None)
            snapshots[detector_id] = {
                "detector_id": detector_id,
                "halting_number": int(area.getLastStepHaltingNumber(detector_id)),
                "vehicle_number": int(area.getLastStepVehicleNumber(detector_id)),
                "vehicle_ids": tuple(area.getLastStepVehicleIDs(detector_id)),
                "mean_speed_mps": float(area.getLastStepMeanSpeed(detector_id)),
                "occupancy_pct": float(area.getLastStepOccupancy(detector_id)),
                "jam_length_veh": float(jam_veh(detector_id)) if jam_veh else float("nan"),
                "jam_length_m": float(jam_m(detector_id)) if jam_m else float("nan"),
            }
        return snapshots

    def get_vehicle_type(self, veh_id: str) -> str:
        """Return the project's normalized vehicle type for a live vehicle.

        The TraCI type ID is cached in its original form so existing emission
        and observation code keeps the same cache semantics.  Normalization is
        centralized in :class:`EmissionFactorLookup`; controllers therefore do
        not need their own aliases for sedan/truck SUMO types.
        """
        vehicle_type = self._veh_type_cache.get(veh_id)
        if not vehicle_type:
            vehicle_type = str(self._traci.vehicle.getTypeID(veh_id))
            self._veh_type_cache[veh_id] = vehicle_type
        return EmissionFactorLookup.normalize_vehicle_type(vehicle_type)

    def get_vehicle_route_context(self, veh_id: str) -> Dict:
        vehicle = self._traci.vehicle
        return {"vehicle_id": veh_id, "route": tuple(vehicle.getRoute(veh_id)),
                "route_index": int(vehicle.getRouteIndex(veh_id)),
                "road_id": str(vehicle.getRoadID(veh_id)),
                "lane_id": str(vehicle.getLaneID(veh_id)),
                "speed_mps": float(vehicle.getSpeed(veh_id))}

    def get_phase_timing_info(self, tl_id: str) -> Dict:
        """
        返回 Phase 层观测所需的时序信息，供 obs_reward.py 构建 phase_feats。

        返回字段
        --------
        current_phase       : int
        elapsed_green       : float
        phase_last_went_red : List[float]
        sim_time            : float
        min_green           : int
        max_green           : int
        min_red             : int
        """
        ss = self._signal_states[tl_id]
        return {
            "current_phase"       : ss.current_phase,
            "elapsed_green"       : ss.elapsed_green,
            "phase_last_went_red" : list(ss.phase_last_went_red),
            "sim_time"            : self._sim_time,
            "min_green"           : self.env_cfg.min_green,
            "max_green"           : self.env_cfg.max_green,
            "min_red"             : self.env_cfg.min_red,
        }

    # ─────────────────────────────────────────────────────────────────
    # 全局指标
    # ─────────────────────────────────────────────────────────────────

    def get_arrived_count(self) -> int:
        """返回本步内完成行程的车辆数。"""
        return self._traci.simulation.getArrivedNumber()

    # ─────────────────────────────────────────────────────────────────
    # 初始化辅助
    # ─────────────────────────────────────────────────────────────────

    def _init_e1_counters(self) -> None:
        """初始化 E1 检测器基准计数（episode 开始时清零）。"""
        self._e1_prev_count = {}
        for lane_id, det_id in self.net_info.lane_to_e1_all.items():
            self._e1_prev_count[det_id] = 0

    # ─────────────────────────────────────────────────────────────────
    # Tripinfo 解析
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_tripinfo(tripinfo_path: str) -> EpisodeTrafficStats:
        """
        解析 tripinfo.xml，计算 episode 级交通统计。

        只统计已完成行程的车辆（departed + arrived）。
        """
        stats = EpisodeTrafficStats()
        if not os.path.exists(tripinfo_path):
            return stats

        try:
            root = ET.parse(tripinfo_path).getroot()
        except ET.ParseError:
            return stats

        delays       = []
        travel_times = []
        departed     = 0
        arrived      = 0

        for trip in root.findall("tripinfo"):
            departed += 1
            arr = trip.get("arrival", "-1")
            if arr == "-1":
                continue  # 未完成行程（write-unfinished 会记录，跳过）

            arrived += 1
            delay = float(trip.get("timeLoss", 0))
            tt    = float(trip.get("duration", 0))
            delays.append(delay)
            travel_times.append(tt)

        if arrived > 0:
            stats.avg_delay_s       = float(np.mean(delays))
            stats.avg_travel_time_s = float(np.mean(travel_times))

        stats.total_arrived   = arrived
        stats.completion_rate = arrived / departed if departed > 0 else 0.0
        return stats


# ═══════════════════════════════════════════════════════════════════════
# BaselineRunner
# ═══════════════════════════════════════════════════════════════════════

class BaselineRunner:
    """
    运行 fixed-time control baseline 仿真，获取归一化基准。

    输出
    ----
    BaselineThresholds：
        - NOx_max            : 全网 E2 观测区正 NOx 样本分位数
        - baseline_metrics   : fixed-time 交通指标

    使用方式
    --------
        runner  = BaselineRunner(cfg.env, net_info, threshold_cfg=cfg.threshold)
        thresh  = runner.run_or_load()  # 有缓存则直接加载
        # 保存路径：cfg.env.baseline_thresholds_json
    """

    def __init__(
        self,
        env_cfg: EnvConfig,
        net_info: NetworkInfo,
        threshold_cfg: Optional[ThresholdConfig] = None,
    ) -> None:
        self.env_cfg = env_cfg
        self.net_info = net_info
        self.threshold_cfg = threshold_cfg or ThresholdConfig()

    def run_or_load(
        self,
        n_episodes: int = 3,
        seeds: Optional[List[int]] = None,
        force_load: bool = False,
        force_rerun: bool = False,
    ) -> BaselineThresholds:
        """
        若缓存文件存在则直接加载，否则运行预跑并保存。

        参数
        ----
        n_episodes : 预跑的 episode 数（结果取多轮平均，减少随机误差）
        """
        cache_path = self.env_cfg.baseline_thresholds_json
        cache_exists = os.path.exists(cache_path)

        # code_main 默认直接使用 collect.py + calibrate.py 生成的外部预实验 JSON。
        # force_load=True 表示“只读指定 JSON，不因当前 threshold_cfg 差异而重新旧式 baseline”。
        if force_load:
            if not cache_exists:
                raise FileNotFoundError(
                    f"baseline_thresholds_json not found: {cache_path}\n"
                    "Please generate it first with code_calibration/collect.py and "
                    "code_calibration/calibrate.py."
                )
            thresholds = self._load(cache_path)
            self._validate_loaded_thresholds(thresholds)
            _console("BaselineRunner force_load=True，只读取预实验阈值 JSON，不重新运行 baseline。")
            _console(f"BaselineRunner baseline_thresholds_json={cache_path}")
            self._print_threshold_summary(thresholds)
            return thresholds

        if cache_exists and not force_rerun:
            thresholds = self._load(cache_path)
            if self._is_cache_compatible(thresholds):
                _console(f"BaselineRunner 使用已有兼容阈值缓存：{cache_path}")
                self._print_threshold_summary(thresholds)
                return thresholds
            _console(f"BaselineRunner WARN 阈值缓存不兼容，将重新预跑并覆盖：{cache_path}")

        reason = "强制重新标定" if force_rerun else ("缓存不存在" if not cache_exists else "缓存不兼容")
        _console(f"BaselineRunner {reason}，开始预跑 {n_episodes} 个 episode...")
        thresh = self._run(n_episodes, seeds=seeds)
        self._save(thresh, cache_path)
        _console(f"BaselineRunner 预跑完成，阈值已保存：{cache_path}")
        self._print_threshold_summary(thresh)
        return thresh

    def _run(self, n_episodes: int, seeds: Optional[List[int]] = None) -> BaselineThresholds:
        baseline_mode = getattr(self.env_cfg, "baseline_control_mode", "fixed_time")
        if baseline_mode not in {"fixed_time", "sumo_actuated"}:
            raise ValueError(f"Unknown baseline_control_mode={baseline_mode}")

        """运行 n_episodes 轮 fixed-time 仿真，收集排放数据并标定全网阈值。"""
        lane_NOx_all: Dict[str, List[float]] = {
            lid: [] for lid in self.net_info.lane_info
        }
        traffic_stats_list: List[EpisodeTrafficStats] = []

        for ep in range(n_episodes):
            baseline_env_cfg = self.env_cfg
            if baseline_mode == "sumo_actuated":
                baseline_env_cfg = copy.deepcopy(self.env_cfg)
                baseline_env_cfg.sumo_cfg = self.env_cfg.baseline_sumo_cfg
            _console(f"Baseline episode={ep + 1}/{n_episodes}")
            env = SumoEnv(
                env_cfg=baseline_env_cfg,
                net_info=self.net_info,
                port=8814,
            )
            stats = None
            try:
                baseline_seed = None if seeds is None or ep >= len(seeds) else seeds[ep]
                if baseline_mode == "sumo_actuated":
                    env.start(
                        episode_id=9000 + ep,
                        fixed_time=False,
                        seed=baseline_seed,
                        control_tls=False,
                    )
                else:
                    env.start(
                        episode_id=9000 + ep,
                        fixed_time=True,
                        seed=baseline_seed,
                        control_tls=True,
                    )

                for step in range(self.env_cfg.steps_per_episode):
                    if baseline_mode == "sumo_actuated":
                        raw_obs_dict = env.step_passive(decision_step=step)
                    else:
                        actions = self._fixed_time_actions(env)
                        raw_obs_dict = env.step(actions, decision_step=step)

                    for tl_id, obs in raw_obs_dict.items():
                        if not self._is_after_warmup(obs.sim_time):
                            continue

                        iinfo = self.net_info.get_intersection(tl_id)
                        lanes = iinfo.all_inc_lanes_flat()
                        for i, lid in enumerate(lanes):
                            lane_NOx_all[lid].append(obs.NOx_mg[i])
            finally:
                stats = env.close()
            if stats:
                traffic_stats_list.append(stats)

        # ── 计算分位数：只过滤标定样本，不裁剪标定值 ────────────────
        thresh = BaselineThresholds()

        all_nox_vals = []
        for vals in lane_NOx_all.values():
            all_nox_vals.extend(vals)
        all_nox_pos = self._filter_calibration_values(all_nox_vals)
        q_nox = float(self.threshold_cfg.nox_max_quantile) * 100.0
        thresh.NOx_max = float(np.percentile(all_nox_pos, q_nox)) if all_nox_pos else 1.0
        thresh.NOx_max = max(thresh.NOx_max, 1e-6)  # 避免分母为零

        # baseline 交通指标（多轮平均）
        if traffic_stats_list:
            thresh.baseline_metrics = {
                "avg_delay_s"      : float(np.mean([s.avg_delay_s for s in traffic_stats_list])),
                "avg_travel_time_s": float(np.mean([s.avg_travel_time_s for s in traffic_stats_list])),
                "total_arrived"    : float(np.mean([s.total_arrived for s in traffic_stats_list])),
                "completion_rate"  : float(np.mean([s.completion_rate for s in traffic_stats_list])),
            }

        thresh.calibration_meta = {
            "observation_scope": str(self.env_cfg.observation_scope),
            "observation_length": float(self.env_cfg.observation_length),
            "baseline_control_mode": str(getattr(self.env_cfg, "baseline_control_mode", "fixed_time")),
            "baseline_sumo_cfg": str(getattr(self.env_cfg, "baseline_sumo_cfg", "")),
            "baseline_net_xml": str(getattr(self.env_cfg, "baseline_net_xml", "")),
            "baseline_note": (
                "SUMO native actuated control is used. SumoEnv runs in passive mode without "
                "setRedYellowGreenState during baseline calibration."
                if baseline_mode == "sumo_actuated"
                else "Script-controlled fixed-time baseline is used."
            ),
            "exclude_zero_for_calibration": bool(self.threshold_cfg.exclude_zero_for_calibration),
            "zero_eps": float(self.threshold_cfg.zero_eps),
            "warmup_seconds": int(self.threshold_cfg.warmup_seconds),
            "nox_max_quantile": float(self.threshold_cfg.nox_max_quantile),
            "NOx_max": float(thresh.NOx_max),
            "n_lane_nox_samples_raw": int(len(all_nox_vals)),
            "n_lane_nox_samples_positive": int(len(all_nox_pos)),
            "note": (
                "One global NOx_max is calibrated from network-wide positive "
                "E2 observation samples after warm-up."
            ),
        }

        return thresh

    def _all_incoming_edge_ids(self) -> List[str]:
        edge_ids: List[str] = []
        seen = set()
        for tl_id in self.net_info.intersection_ids:
            iinfo = self.net_info.get_intersection(tl_id)
            for eid in iinfo.inc_edges_nesw:
                if eid is None or eid in seen:
                    continue
                seen.add(eid)
                edge_ids.append(eid)
        if not edge_ids:
            edge_ids = list(self.net_info.edge_info.keys())
        return edge_ids

    def _is_after_warmup(self, sim_time: float) -> bool:
        return float(sim_time) >= float(self.threshold_cfg.warmup_seconds)

    def _filter_calibration_values(self, values: Sequence[float]) -> List[float]:
        out: List[float] = []
        eps = float(self.threshold_cfg.zero_eps)
        for v in values:
            try:
                x = float(v)
            except Exception:
                continue
            if not np.isfinite(x):
                continue
            if self.threshold_cfg.exclude_zero_for_calibration and x <= eps:
                continue
            out.append(x)
        return out

    def _validate_loaded_thresholds(self, thresholds: BaselineThresholds) -> None:
        """加载外部 JSON 后只检查单一全局 NOx_max。"""
        if float(thresholds.NOx_max) <= 0:
            raise ValueError("NOx threshold JSON 中 NOx_max 必须为正值。")

    def _cfg_meta_value(self, key: str):
        return getattr(self.threshold_cfg, key)

    @staticmethod
    def _same_meta_value(a, b) -> bool:
        if isinstance(a, bool) or isinstance(b, bool):
            return bool(a) == bool(b)
        if isinstance(a, int) and not isinstance(a, bool):
            return int(a) == int(b)
        try:
            return abs(float(a) - float(b)) < 1e-9
        except Exception:
            return a == b

    def _is_cache_compatible(self, thresholds: BaselineThresholds) -> bool:
        meta = thresholds.calibration_meta or {}
        current_mode = getattr(self.env_cfg, "baseline_control_mode", "fixed_time")
        meta_mode = meta.get("baseline_control_mode")
        if meta_mode != current_mode:
            return False
        if current_mode == "sumo_actuated":
            current_cfg = os.path.abspath(str(getattr(self.env_cfg, "baseline_sumo_cfg", "")))
            cached_cfg = os.path.abspath(str(meta.get("baseline_sumo_cfg", "")))
            if current_cfg != cached_cfg:
                return False
        if not meta:
            return False

        keys_to_check = [
            "exclude_zero_for_calibration",
            "zero_eps",
            "warmup_seconds",
            "nox_max_quantile",
        ]
        for key in keys_to_check:
            if key not in meta:
                return False
            if not self._same_meta_value(meta[key], self._cfg_meta_value(key)):
                return False

        return True

    def _print_threshold_summary(self, thresh: BaselineThresholds) -> None:
        _console("BaselineRunner 阈值标定摘要")
        _console(f"NOx_max={thresh.NOx_max:.6g}")
        _console(f"calibration_meta={json.dumps(thresh.calibration_meta, ensure_ascii=False)}")

    def _fixed_time_actions(self, env: SumoEnv) -> Dict[str, int]:
        """
        生成 fixed-time 动作：按绿灯相位顺序循环。
        每个相位持续 FIXED_TIME_GREEN_DURATION 秒后切换到下一个。
        """
        actions = {}
        for tl_id, ss in env._signal_states.items():
            iinfo    = env.net_info.get_intersection(tl_id)
            n_phases = iinfo.n_green_phases

            # 如果当前绿灯时长超过固定时长，切换到下一相位
            if ss.elapsed_green >= SumoEnv.FIXED_TIME_GREEN_DURATION:
                next_phase = (ss.current_phase + 1) % n_phases
            else:
                next_phase = ss.current_phase

            actions[tl_id] = next_phase
        return actions

    def _save(self, thresh: BaselineThresholds, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "NOx_max": thresh.NOx_max,
            "baseline_metrics": thresh.baseline_metrics,
            "calibration_meta": thresh.calibration_meta,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load(self, path: str) -> BaselineThresholds:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nox_max = float(data.get("NOx_max", 1.0))
        return BaselineThresholds(
            NOx_max=nox_max,
            baseline_metrics=data.get("baseline_metrics", {}),
            calibration_meta=data.get("calibration_meta", {}),
        )


# ═══════════════════════════════════════════════════════════════════════
# 便捷工厂函数
# ═══════════════════════════════════════════════════════════════════════

def make_env(
    cfg=None,
    net_info: Optional[NetworkInfo] = None,
    port: int = 8813,
    use_gui: bool = False,
) -> SumoEnv:
    """
    快捷构建 SumoEnv。

    示例
    ----
        env = make_env()          # 使用默认配置
        env.start(episode_id=0)
        ...
    """
    if cfg is None:
        cfg = get_config()
    if net_info is None:
        net_info = parse_network(
            net_xml     = cfg.env.sumo_cfg.replace(".sumocfg", "_net.xml"),
            add_xml     = cfg.env.sumo_cfg.replace(".sumocfg", "_add.xml"),
            groups_json = cfg.env.intersection_groups_json,
        )
    return SumoEnv(cfg.env, net_info, port=port, use_gui=use_gui)


# ═══════════════════════════════════════════════════════════════════════
# 自检（不需要 SUMO，仅验证逻辑正确性）
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    _console("=" * 65)
    _console("env.py 自检（无需 SUMO，验证逻辑层）")
    _console("=" * 65)

    cfg     = get_config()
    net     = parse_network(
        net_xml = "/mnt/user-data/uploads/grid_6x6_net.xml",
        add_xml = "/mnt/user-data/uploads/grid_6x6_add.xml",
    )

    # ── 测试 1：SignalState 初始化 ──────────────────────────────
    n_phases = net.get_intersection("nt11").n_green_phases
    ss = SignalState(
        tl_id               = "nt11",
        current_phase       = 0,
        elapsed_green       = 0.0,
        in_yellow           = False,
        yellow_remaining    = 0.0,
        pending_phase       = -1,
        last_green_start    = 0.0,
        phase_last_went_red = [-cfg.env.min_red] * n_phases,
    )
    assert len(ss.phase_last_went_red) == n_phases
    _console(f"OK SignalState 初始化，n_phases={n_phases}")

    # ── 测试 2：_green_to_sumo_phase 转换 ──────────────────────
    env = SumoEnv(cfg.env, net, port=8813)
    # 绑定假路口状态（不启动 SUMO）
    env._signal_states["nt11"] = ss
    env._sim_time = 0.0

    for i in range(n_phases):
        sumo_idx = env._green_to_sumo_phase("nt11", i)
        assert sumo_idx == i * 2, f"绿灯 {i} → SUMO phase {sumo_idx}，期望 {i*2}"
    _console("OK _green_to_sumo_phase: 绿灯 i → SUMO phase i*2")

    for i in range(n_phases):
        yellow_idx = env._yellow_phase_idx("nt11", i)
        assert yellow_idx == i * 2 + 1
    _console("OK _yellow_phase_idx: 黄灯 i → SUMO phase i*2+1")

    # ── 测试 3：compute_action_mask 逻辑 ───────────────────────
    # Case A：elapsed_green < min_green → 只能保持当前相位
    ss.elapsed_green = 2.0  # < min_green=5
    ss.current_phase = 1
    mask_a = env.compute_action_mask("nt11")
    assert mask_a[1] == True,  "当前相位（1）应可选"
    assert mask_a[0] == False, "其他相位在 min_green 内应被禁"
    assert mask_a.sum() == 1,  "只有1个可行动作"
    _console("OK Action mask Case A（min_green约束）：只能保持当前相位")

    # Case B：正常范围，无 min_red 冲突
    ss.elapsed_green         = 10.0  # > min_green
    ss.current_phase         = 0
    ss.phase_last_went_red   = [-cfg.env.min_red] * n_phases  # 全部满足 min_red
    env._sim_time            = 100.0
    mask_b = env.compute_action_mask("nt11")
    assert mask_b.all(), "正常范围内所有相位均可选"
    _console("OK Action mask Case B（正常范围）：所有相位可选")

    # Case C：max_green 超限 → 不能保持当前相位
    ss.elapsed_green = cfg.env.max_green + 1  # > max_green=90
    ss.current_phase = 2
    mask_c = env.compute_action_mask("nt11")
    assert mask_c[2] == False, "超过 max_green，当前相位不可保持"
    _console("OK Action mask Case C（max_green约束）：当前相位被禁")

    # Case D：某相位 min_red 未满足 → 该相位被禁
    ss.elapsed_green       = 10.0
    ss.current_phase       = 0
    env._sim_time          = 100.0
    # 相位1刚变红3秒（不足 min_red=15s）
    ss.phase_last_went_red = [-cfg.env.min_red] * n_phases
    ss.phase_last_went_red[1] = 97.0  # 100 - 97 = 3s < 15s
    mask_d = env.compute_action_mask("nt11")
    assert mask_d[0] == True,  "当前相位（0）可选"
    assert mask_d[1] == False, "相位1 min_red 未满足，应被禁"
    if n_phases >= 3:
        assert mask_d[2] == True,  "相位2 无约束，可选"
    _console("OK Action mask Case D（min_red约束）：特定相位被禁")

    # Case E：fallback 兜底
    ss.elapsed_green = cfg.env.max_green + 1
    ss.current_phase = 0
    # 除当前相位外所有相位都被 min_red 禁止
    ss.phase_last_went_red = [env._sim_time] * n_phases  # 刚变红
    ss.phase_last_went_red[0] = -cfg.env.min_red         # 当前相位排除 min_red
    mask_e = env.compute_action_mask("nt11")
    # max_green 禁止保持（phase 0），但其他全被 min_red 禁，fallback 强制 phase 0
    assert mask_e[0] == True,  "fallback：强制保持当前相位"
    assert mask_e.sum() == 1,  "fallback 只有1个可行动作"
    _console("OK Action mask Case E（fallback兜底）")

    # ── 测试 4：get_phase_timing_info 返回结构 ──────────────────
    ss.elapsed_green = 7.0
    ss.current_phase = 2
    timing = env.get_phase_timing_info("nt11")
    assert timing["current_phase"] == 2
    assert timing["elapsed_green"] == 7.0
    assert "phase_last_went_red" in timing
    assert timing["min_green"] == cfg.env.min_green
    _console("OK get_phase_timing_info 返回结构正确")

    # ── 测试 5：BaselineThresholds 序列化/反序列化 ──────────────
    import tempfile
    thresh = BaselineThresholds(
        NOx_max = 15.3,
        baseline_metrics = {"avg_delay_s": 85.0},
    )
    runner = BaselineRunner(cfg.env, net)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = f.name
    runner._save(thresh, tmp_path)
    thresh_loaded = runner._load(tmp_path)
    assert abs(thresh_loaded.NOx_max - 15.3) < 1e-6
    os.unlink(tmp_path)
    _console("OK BaselineThresholds 序列化/反序列化正确")

    # ── 测试 6：_parse_tripinfo（用伪造 XML）───────────────────
    import tempfile
    fake_xml = """<?xml version="1.0"?>
<tripinfos>
  <tripinfo id="v0" depart="10.00" arrival="200.00" duration="190.00" timeLoss="40.00"/>
  <tripinfo id="v1" depart="20.00" arrival="250.00" duration="230.00" timeLoss="60.00"/>
  <tripinfo id="v2" depart="30.00" arrival="-1"/>
</tripinfos>"""
    with tempfile.NamedTemporaryFile(
        suffix=".xml", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write(fake_xml)
        tmp_xml = f.name
    stats = SumoEnv._parse_tripinfo(tmp_xml)
    assert stats.total_arrived == 2,                   f"应有2辆完成，实际 {stats.total_arrived}"
    assert abs(stats.avg_delay_s - 50.0) < 1e-5,      f"平均延误应为50，实际 {stats.avg_delay_s}"
    assert abs(stats.completion_rate - 2/3) < 1e-5
    os.unlink(tmp_xml)
    _console("OK _parse_tripinfo 解析正确（2辆完成，平均延误=50s）")

    print()
    _console("=" * 65)
    _console("env.py 自检全部通过（无需 SUMO）。")
    _console("实际仿真测试需要安装 SUMO 并设置 SUMO_HOME。")
    _console("=" * 65)
