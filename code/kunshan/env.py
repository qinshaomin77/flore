
from __future__ import annotations

import copy
import json
import os
import os
import subprocess
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import EnvConfig
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

def _import_traci(use_gui: bool = False) -> Tuple[object, bool]:
    sumo_home = os.environ.get("SUMO_HOME", "")
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

@dataclass
class SignalState:
    tl_id:            str
    current_phase:    int    # 当前绿灯相位 index（0-based，在 green_phases 列表中）
    elapsed_green:    float  # 当前绿灯已持续秒数（从黄灯结束后开始计时）
    in_yellow:        bool   # 是否正在执行黄灯
    yellow_remaining: float  # 黄灯剩余秒数（in_yellow=True 时有效）
    pending_phase:    int    # 黄灯结束后即将切换到的相位（-1 表示无待切换）
    last_green_start: float  # 当前绿灯开始的仿真时刻（秒）

    phase_last_went_red: List[float] = field(default_factory=list)

@dataclass
class StepRawObs:
    tl_id: str

    queue_veh:     List[float]  # 最后300m E2观测区内排队车辆数（辆）
    queue_full_lane: List[float]  # 整条 incoming lane 内排队车辆数（辆）
    speed_mps:     List[float]  # 当前平均速度（m/s）
    truck_count:   List[float]  # 近decision_interval秒E1通过货车 + E2区内驻留货车
    total_count:   List[float]  # 最后300m E2观测区内总车辆数
    pass_veh:      List[float]  # 当前决策间隔内通过车辆数
    green_service_s: List[float]  # 当前决策间隔内实际有效绿灯累计秒数
    demand_veh:    List[float]  # E1近decision_interval秒通过数 + E2区内当前车辆数
    NOx_mg:        List[float]  # 当前决策步最后300m E2观测区累计NOx（mg）
    wait_vwt:      List[float]  # 当前决策间隔内E2区等待车辆逐秒累计值（veh·s）
    wait_vwt_full_lane: List[float]  # 当前决策间隔内整lane排队累计（veh·s）
    NOx_full_lane_mg: List[float]  # 当前决策步整lane累计NOx（mg）
    NOx_signal_sensitive_mg: List[float]  # E2区信号敏感工况NOx（mg）

    edge_NOx_mg:   List[float]  # 每条进口 edge 的 NOx 总量（mg）

    current_phase:    int    # 当前绿灯相位 index
    elapsed_green:    float  # 已绿灯时长（秒）
    in_yellow:        bool   # 是否在黄灯期
    phase_last_went_red: List[float]  # 每个绿灯相位上次变红的时刻

    sim_time:      float  # 当前仿真时刻（秒）

    sumo_green_phase: int = -1  # SUMO real green phase index mapped to current_phase coding.
    sumo_raw_phase:   int = -1  # SUMO raw phase index, including yellow/all-red phases.
    pending_phase:    int = -1  # Target phase while a yellow transition is pending.
    sumo_state:       str = ""   # SUMO real signal state from getRedYellowGreenState.
    env_state:        str = ""   # Expected env-issued green/yellow state for comparison.

@dataclass
class EpisodeTrafficStats:
    avg_delay_s:        float = 0.0   # 平均延误（秒）
    avg_travel_time_s:  float = 0.0   # 平均行程时间（秒）
    total_arrived:      int   = 0     # 完成行程车辆数
    completion_rate:    float = 0.0   # = arrived / departed

@dataclass

class SumoEnv:

    FIXED_TIME_GREEN_DURATION: int = 30

    def __init__(
        self,
        env_cfg:  EnvConfig,
        net_info: NetworkInfo,
        port:     int = 8813,
        use_gui:  bool = False,
        emission_lookup: Optional[EmissionFactorLookup] = None,
        emission_recorder: Optional[EmissionEpisodeRecorder] = None,
        controlled_tl_ids: Optional[List[str]] = None,
    ) -> None:
        self.env_cfg  = env_cfg
        self.net_info = net_info
        self.port     = port
        self.use_gui  = use_gui
        self.emission_lookup = emission_lookup
        self.emission_recorder = emission_recorder
        all_tl_ids = tuple(self.net_info.intersection_ids)
        requested_tl_ids = all_tl_ids if controlled_tl_ids is None else tuple(controlled_tl_ids)
        unknown_tl_ids = sorted(set(requested_tl_ids) - set(all_tl_ids))
        if unknown_tl_ids:
            raise ValueError(f"controlled_tl_ids contains unknown TLS IDs: {unknown_tl_ids}")
        if len(requested_tl_ids) != len(set(requested_tl_ids)):
            raise ValueError("controlled_tl_ids contains duplicate TLS IDs")
        self.controlled_tl_ids = requested_tl_ids
        self._controlled_tl_id_set = set(requested_tl_ids)
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
                    record_vehicle_step=bool(
                        getattr(env_cfg, "record_vehicle_emission_steps", True)
                    ),
                    enable_lane_mechanism_metrics=bool(
                        getattr(env_cfg, "enable_lane_mechanism_metrics", False)
                    ),
                    signal_sensitive_halting_speed_mps=float(
                        getattr(env_cfg, "signal_sensitive_halting_speed_mps", 0.1)
                    ),
                    signal_sensitive_low_speed_mps=float(
                        getattr(env_cfg, "signal_sensitive_low_speed_mps", 5.0)
                    ),
                    signal_sensitive_restart_accel_ms2=float(
                        getattr(env_cfg, "signal_sensitive_restart_accel_ms2", 0.5)
                    ),
                )
        self._last_step_emission: Optional[DecisionStepEmission] = None
        self._interval_wait_accum: Dict[str, float] = {}
        self._last_wait_vwt_by_lane: Dict[str, float] = {}
        self._interval_full_wait_accum: Dict[str, float] = {}
        self._last_full_wait_vwt_by_lane: Dict[str, float] = {}
        self._interval_pass_accum: Dict[str, float] = {}
        self._last_pass_by_lane: Dict[str, float] = {}
        self._interval_truck_pass_accum: Dict[str, float] = {}
        self._last_truck_pass_by_lane: Dict[str, float] = {}
        self._interval_green_service_accum: Dict[str, float] = {}
        self._last_green_service_by_lane: Dict[str, float] = {}

        self._traci = None
        self._is_libsumo: bool = False

        self._signal_states: Dict[str, SignalState] = {}
        self._last_sumo_phase: Dict[str, Tuple[int, int]] = {}

        self._sim_time: float = 0.0

        self._e1_prev_count: Dict[str, int] = {}

        self._tripinfo_path: str = ""
        self._fcd_output_path: str = ""
        self._runtime_dir: str = ""

        self._running: bool = False

        self._tc = None
        self._subscribed: bool = False
        self._context_anchor: Optional[str] = None

        self._all_inc_lanes: List[str] = []

        self._veh_type_cache: Dict[str, str] = {}
        self._current_vehicle_speed_by_id: Dict[str, float] = {}

        self._subscribed_e1_step: set = set()
        self._vehicle_state_writer: Optional[VehicleStateParquetWriter] = None

    @property
    def fcd_output_path(self) -> str:
        return str(self._fcd_output_path)

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
        # SUMO detector outputs are not necessarily XML filenames. In
        # particular, training assets intentionally use the platform null
        # device ("NUL" on Windows or "/dev/null" on POSIX). Once an element
        # is known to be a detector and has a file attribute, that attribute
        # is an output target regardless of its extension.
        return tag in detector_tags or "detector_output" in value

    def _prepare_episode_sumo_config(
        self,
        episode_id: int,
        detector_output_path: str,
        detector_dir: str,
    ) -> str:
        sumocfg_path = str(self.env_cfg.sumo_cfg)
        add_xml_path = str(getattr(self.env_cfg, "add_xml", "") or "")
        if not sumocfg_path or not add_xml_path:
            return sumocfg_path

        try:
            add_tree = ET.parse(add_xml_path)
            add_root = add_tree.getroot()
            detector_file_value = self._null_output_path()
            changed_detector_files = 0
            for elem in add_root.iter():
                file_value = elem.attrib.get("file")
                if not file_value or not self._is_detector_file_attr(elem, file_value):
                    continue
                elem.set("file", detector_file_value)
                changed_detector_files += 1

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
                if save_fcd_output:
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
            add_xml_abs = os.path.abspath(add_xml_path)
            existing_additional_files: list[str] = []
            if not bool(getattr(self.env_cfg, "replace_sumocfg_additional_files", False)):
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
            if changed_detector_files == 0:
                print(
                    "[WARN] No detector file attributes were rewritten in temporary additional XML; "
                    f"using temporary SUMO config anyway: {runtime_sumocfg}"
                )
            return runtime_sumocfg
        except Exception as exc:
            print(
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
        if self._traci is None:
            self._traci, self._is_libsumo = _import_traci(self.use_gui)
            self._tc = self._traci.constants

        tripinfo_dir = self.env_cfg.tripinfo_dir
        os.makedirs(tripinfo_dir, exist_ok=True)
        run_dir = os.path.dirname(tripinfo_dir)
        save_fcd_output = bool(getattr(self.env_cfg, "save_fcd_output", False))
        self._fcd_output_path = ""
        if save_fcd_output:
            fcd_dir = os.path.join(run_dir, "fcd")
            os.makedirs(fcd_dir, exist_ok=True)
            self._fcd_output_path = os.path.abspath(
                os.path.join(fcd_dir, f"fcd_ep{int(episode_id):04d}.xml")
            )
        if bool(getattr(self.env_cfg, "keep_sumo_runtime_files", True)):
            detector_dir = os.path.join(run_dir, "sumo_runtime")
            os.makedirs(detector_dir, exist_ok=True)
        else:
            detector_dir = tempfile.mkdtemp(prefix=f"mgmq_sumo_ep{int(episode_id):04d}_")
        self._runtime_dir = detector_dir
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
        if save_fcd_output:
            print(f"[INFO] FCD output enabled: {self._fcd_output_path}", flush=True)
        self._last_step_emission = None
        self._interval_wait_accum = {}
        self._last_wait_vwt_by_lane = {}
        self._interval_full_wait_accum = {}
        self._last_full_wait_vwt_by_lane = {}
        self._interval_pass_accum = {}
        self._last_pass_by_lane = {}
        self._interval_truck_pass_accum = {}
        self._last_truck_pass_by_lane = {}
        self._interval_green_service_accum = {}
        self._last_green_service_by_lane = {}
        self._subscribed = False
        self._context_anchor = None
        self._veh_type_cache.clear()
        self._current_vehicle_speed_by_id.clear()
        if self.emission_recorder is not None:
            self.emission_recorder.start_episode(episode_id)

        if control_tls:
            self._init_signal_states(fixed_time)
        else:
            self._init_passive_signal_states()

        self._validate_controlled_detectors()
        self._init_e1_counters()
        self._setup_subscriptions()
        if bool(getattr(self.env_cfg, "save_vehicle_state_output", False)):
            if self.emission_recorder is None:
                raise RuntimeError("vehicle_state output requires the MOVES emission recorder")
            self._vehicle_state_writer = VehicleStateParquetWriter(
                output_root=run_dir,
                episode=int(episode_id),
                sumo_seed=int(seed) if seed is not None else -1,
                emission_class_by_type=validate_hbefa4_vehicle_types(self._traci),
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

    def close(self, parse_tripinfo: bool = True) -> Optional[EpisodeTrafficStats]:
        if not self._running:
            return None

        try:
            self._traci.close()
        except Exception:
            pass
        finally:
            self._running = False
            self._subscribed = False
            self._context_anchor = None
            time.sleep(0.8)

        writer_error: Optional[BaseException] = None
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

        stats = self._parse_tripinfo(self._tripinfo_path) if parse_tripinfo else None
        if not bool(getattr(self.env_cfg, "keep_sumo_runtime_files", True)) and self._runtime_dir:
            shutil.rmtree(self._runtime_dir, ignore_errors=True)
            self._runtime_dir = ""
        return stats

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

    def step(
        self,
        actions: Dict[str, int],
        decision_step: Optional[int] = None,
    ) -> Dict[str, StepRawObs]:
        assert self._running, "env 未启动，请先调用 start()"

        if decision_step is None:
            decision_step = int(self._sim_time // self.env_cfg.decision_interval)
        action_ids = set(actions)
        if action_ids != self._controlled_tl_id_set:
            missing = sorted(self._controlled_tl_id_set - action_ids)
            extra = sorted(action_ids - self._controlled_tl_id_set)
            raise ValueError(f"RL action TLS set mismatch: missing={missing}, extra={extra}")
        self._apply_actions(actions)

        self._reset_interval_wait()
        self._reset_interval_pass()
        self._reset_interval_green_service()
        for sub_step in range(self.env_cfg.decision_interval):
            self._traci.simulationStep()
            self._sim_time += 1.0
            self._record_emissions_for_second(int(decision_step))
            self._accumulate_interval_wait()
            self._accumulate_interval_pass()
            self._accumulate_interval_green_service()

            self._update_yellow_transitions(sub_step + 1)

        self._last_wait_vwt_by_lane = dict(self._interval_wait_accum)
        self._last_full_wait_vwt_by_lane = dict(self._interval_full_wait_accum)
        self._last_pass_by_lane = dict(self._interval_pass_accum)
        self._last_truck_pass_by_lane = dict(self._interval_truck_pass_accum)
        self._last_green_service_by_lane = dict(self._interval_green_service_accum)

        self._refresh_vehicle_type_cache()

        if self.emission_recorder is not None:
            self._last_step_emission = self.emission_recorder.finalize_decision_step(int(decision_step))
            if self._vehicle_state_writer is not None:
                self._vehicle_state_writer.commit_decision_step(
                    self.emission_recorder.last_detail_batch
                )
        else:
            self._last_step_emission = None

        self._update_signal_timing()

        self._last_sumo_phase = {}
        for tl_id in self._signal_states.keys():
            self._last_sumo_phase[tl_id] = self._read_sumo_phase_readonly(tl_id)

        raw_obs = {}
        for tl_id in self.controlled_tl_ids:
            raw_obs[tl_id] = self._collect_obs(tl_id, self._last_step_emission)

        return raw_obs

    def step_passive(
        self,
        decision_step: Optional[int] = None,
    ) -> Dict[str, StepRawObs]:
        assert self._running, "env 未启动，请先调用 start()"

        if decision_step is None:
            decision_step = int(self._sim_time // self.env_cfg.decision_interval)

        self._reset_interval_wait()
        self._reset_interval_pass()
        self._reset_interval_green_service()
        for _ in range(self.env_cfg.decision_interval):
            self._traci.simulationStep()
            self._sim_time += 1.0
            self._record_emissions_for_second(int(decision_step))
            self._accumulate_interval_wait()
            self._accumulate_interval_pass()
            self._accumulate_interval_green_service()

        self._last_wait_vwt_by_lane = dict(self._interval_wait_accum)
        self._last_full_wait_vwt_by_lane = dict(self._interval_full_wait_accum)
        self._last_pass_by_lane = dict(self._interval_pass_accum)
        self._last_truck_pass_by_lane = dict(self._interval_truck_pass_accum)
        self._last_green_service_by_lane = dict(self._interval_green_service_accum)

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

        self._sync_passive_signal_states()

        raw_obs = {}
        for tl_id in self.net_info.intersection_ids:
            raw_obs[tl_id] = self._collect_obs(tl_id, self._last_step_emission)

        return raw_obs

    def _init_signal_states(self, fixed_time: bool) -> None:
        self._last_sumo_phase = {}
        self._signal_states = {}
        for tl_id in self.controlled_tl_ids:
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
            )
            self._signal_states[tl_id] = ss

            green_state = self._get_green_state(tl_id, 0)
            self._set_tls_state(tl_id, green_state)

            if fixed_time:
                pass

    def _apply_actions(self, actions: Dict[str, int]) -> None:
        yellow_dur = self.env_cfg.yellow_duration  # 3

        for tl_id, action in actions.items():
            ss    = self._signal_states[tl_id]
            iinfo = self.net_info.get_intersection(tl_id)

            if action == ss.current_phase:
                green_state = self._get_green_state(tl_id, ss.current_phase)
                self._set_tls_state(tl_id, green_state)
                ss.in_yellow      = False
                ss.yellow_remaining = 0.0
                ss.pending_phase  = -1

            else:
                old_green_state = self._get_green_state(tl_id, ss.current_phase)
                yellow_state = self._build_yellow_state_from_old_green(old_green_state)
                self._set_tls_state(tl_id, yellow_state)

                ss.in_yellow        = True
                ss.yellow_remaining = float(yellow_dur)
                ss.pending_phase    = action

                ss.phase_last_went_red[ss.current_phase] = self._sim_time

    def _update_yellow_transitions(self, elapsed_in_step: int) -> None:
        yellow_dur = self.env_cfg.yellow_duration  # 3

        for tl_id, ss in self._signal_states.items():
            if not ss.in_yellow:
                continue

            ss.yellow_remaining -= 1.0

            if ss.yellow_remaining <= 0:
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
        for tl_id, ss in self._signal_states.items():
            if not ss.in_yellow:
                ss.elapsed_green = self._sim_time - ss.last_green_start

    def _green_to_sumo_phase(self, tl_id: str, green_phase_idx: int) -> int:
        iinfo = self.net_info.get_intersection(tl_id)
        return int(iinfo.green_phases[green_phase_idx].sumo_phase_index)

    def _yellow_phase_idx(self, tl_id: str, green_phase_idx: int) -> int:
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
                if yellow_origin_idx is not None:
                    origin = int(yellow_origin_idx)
                    if origin != old_phase:
                        ss.phase_last_went_red[old_phase] = self._sim_time
                    elif not ss.in_yellow:
                        ss.phase_last_went_red[origin] = self._sim_time - max(0.0, elapsed)
                    ss.current_phase = origin

                ss.in_yellow = True
                ss.elapsed_green = 0.0
                ss.yellow_remaining = max(0.0, float(self.env_cfg.yellow_duration) - max(0.0, elapsed))
                ss.pending_phase = -1
                continue

            ss.in_yellow = False
            ss.yellow_remaining = 0.0
            ss.elapsed_green = max(0.0, elapsed)

    def _init_passive_signal_states(self) -> None:
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

    def _setup_subscriptions(self) -> None:
        traci, tc = self._traci, self._tc
        if traci is None or tc is None:
            raise RuntimeError("TraCI is not initialized")
        self._all_inc_lanes = self._all_incoming_lane_ids()
        for lane_id in self._all_inc_lanes:
            traci.lane.subscribe(lane_id, [
                tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
                tc.LAST_STEP_MEAN_SPEED,
                tc.LAST_STEP_VEHICLE_ID_LIST,
            ])
        self._context_anchor = None
        if self.emission_recorder is not None or bool(
            getattr(self.env_cfg, "save_vehicle_state_output", False)
        ):
            self._context_anchor = self._all_inc_lanes[0]
            traci.lane.subscribeContext(
                self._context_anchor,
                tc.CMD_GET_VEHICLE_VARIABLE,
                1000000.0,
                [tc.VAR_TYPE, tc.VAR_SPEED, tc.VAR_ACCELERATION,
                 tc.VAR_LANE_ID, tc.VAR_ROAD_ID, tc.VAR_DISTANCE,
                 tc.VAR_NOXEMISSION, tc.VAR_TIMELOSS, tc.VAR_WAITING_TIME],
            )
        self._subscribed_e1_step = {
            self.net_info.lane_to_e1_all[lane_id]
            for lane_id in self._controlled_incoming_lane_ids()
        }
        for detector_id in self._subscribed_e1_step:
            traci.inductionloop.subscribe(
                detector_id,
                [tc.LAST_STEP_VEHICLE_NUMBER, tc.LAST_STEP_VEHICLE_ID_LIST],
            )
        self._subscribed = True

    def _validate_controlled_detectors(self) -> None:
        controlled = {
            (tl_id, lane_id)
            for tl_id in self.controlled_tl_ids
            for lane_id in self.net_info.get_intersection(tl_id).all_inc_lanes_flat()
        }
        for tl_id, lane_id in sorted(controlled):
            missing = []
            if lane_id not in self.net_info.lane_to_e2:
                missing.append("E2")
            if lane_id not in self.net_info.lane_to_e1_all:
                missing.append("E1")
            if missing:
                raise ValueError(
                    f"tl_id={tl_id}, lane_id={lane_id}, missing detector type={','.join(missing)}, "
                    f"add.xml={self.net_info.add_xml_path}"
                )

    def _refresh_vehicle_type_cache(self) -> None:
        traci = self._traci
        if traci is None:
            return

        if self.emission_recorder is not None and getattr(self, "_context_anchor", None):
            return

        try:
            current_ids = set(traci.vehicle.getIDList())
        except Exception:
            return

        for vid in list(self._veh_type_cache.keys()):
            if vid not in current_ids:
                del self._veh_type_cache[vid]

        for vid in current_ids:
            if vid not in self._veh_type_cache:
                try:
                    self._veh_type_cache[vid] = traci.vehicle.getTypeID(vid)
                except Exception:
                    self._veh_type_cache[vid] = ""

    def _record_emissions_for_second(self, decision_step: int) -> None:
        if self.emission_recorder is None:
            return
        tc, anchor = self._tc, self._context_anchor
        if tc is None or anchor is None:
            raise RuntimeError("Vehicle context subscription is not initialized")
        results = self._traci.lane.getContextSubscriptionResults(anchor) or {}
        self._current_vehicle_speed_by_id = {
            str(veh_id): float(values.get(tc.VAR_SPEED, 0.0))
            for veh_id, values in results.items()
        }
        vehicle_to_observation_lane: Dict[str, str] = {}
        for lane_id in self._controlled_incoming_lane_ids():
            e2_id = self.net_info.lane_to_e2[lane_id]
            for veh_id in self._traci.lanearea.getLastStepVehicleIDs(e2_id) or ():
                vehicle_to_observation_lane[str(veh_id)] = lane_id
        current_ids = set(results)
        self._veh_type_cache = {
            key: value for key, value in self._veh_type_cache.items() if key in current_ids
        }
        for veh_id, values in results.items():
            vtype = values.get(tc.VAR_TYPE, "")
            if vtype:
                self._veh_type_cache[veh_id] = str(vtype)
            lane_id = values.get(tc.VAR_LANE_ID, "")
            edge_id = values.get(tc.VAR_ROAD_ID, "")
            observation_lane_id = vehicle_to_observation_lane.get(str(veh_id), "")
            speed_mps = float(values.get(tc.VAR_SPEED, 0.0))
            accel_mps2 = float(values.get(tc.VAR_ACCELERATION, 0.0))
            normalized_type = EmissionFactorLookup.normalize_vehicle_type(vtype)
            # Keep the MOVES scope consistent with the grid36 evaluator:
            # internal junction edges (":" prefix) remain in vehicle_state
            # for auditing, but do not receive MOVES lookup values.
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
                    speed_mps=speed_mps,
                    acceleration_mps2=accel_mps2,
                    time_loss_accumulated_s=float(values.get(tc.VAR_TIMELOSS, 0.0)),
                    is_waiting=float(values.get(tc.VAR_WAITING_TIME, 0.0)) > 0.0,
                    in_observation_zone=bool(observation_lane_id),
                    observation_lane_id=observation_lane_id,
                    included_in_moves_scope=moves_scope,
                    nox_hbefa4_mg_s=float(values.get(tc.VAR_NOXEMISSION, 0.0)),
                )
            if not moves_scope:
                continue
            self.emission_recorder.record_vehicle_state(
                sim_time=self._sim_time,
                decision_step=int(decision_step),
                veh_id=veh_id,
                vehicle_type=vtype,
                speed_ms=speed_mps,
                accel_ms2=accel_mps2,
                lane_id=lane_id,
                edge_id=edge_id,
                dt=1.0,
                distance_m=values.get(tc.VAR_DISTANCE, 0.0),
                in_observation_zone=bool(observation_lane_id),
                observation_lane_id=observation_lane_id,
                row_id=row_id,
            )

    def _all_incoming_lane_ids(self) -> List[str]:
        lanes: List[str] = []
        seen = set()
        for tl_id in self.net_info.intersection_ids:
            for lid in self.net_info.get_intersection(tl_id).all_inc_lanes_flat():
                if lid in seen:
                    continue
                seen.add(lid)
                lanes.append(lid)
        return lanes

    def _controlled_incoming_lane_ids(self) -> List[str]:
        lanes: List[str] = []
        seen = set()
        for tl_id in self.controlled_tl_ids:
            for lane_id in self.net_info.get_intersection(tl_id).all_inc_lanes_flat():
                if lane_id not in seen:
                    seen.add(lane_id)
                    lanes.append(lane_id)
        return lanes

    def _is_truck(self, vehicle_type: str) -> bool:
        if self.emission_lookup is not None:
            return self.emission_lookup.normalize_vehicle_type(vehicle_type) == "truck"
        return str(vehicle_type).strip().lower() == "truck"

    def _reset_interval_wait(self) -> None:
        if not self._interval_wait_accum:
            for lane_id in self._controlled_incoming_lane_ids():
                self._interval_wait_accum[lane_id] = 0.0
                self._interval_full_wait_accum[lane_id] = 0.0
        else:
            for lane_id in self._interval_wait_accum:
                self._interval_wait_accum[lane_id] = 0.0
                self._interval_full_wait_accum[lane_id] = 0.0

    def _full_lane_halting_count(self, lane_id: str) -> float:
        tc = self._tc
        if tc is None:
            raise RuntimeError("TraCI constants are not initialized")
        sub = self._traci.lane.getSubscriptionResults(lane_id) or {}
        if tc.LAST_STEP_VEHICLE_ID_LIST not in sub:
            raise RuntimeError(
                f"Missing full-lane vehicle subscription result for lane_id={lane_id}"
            )
        threshold = float(
            getattr(self.env_cfg, "queue_halting_speed_threshold_mps", 5.0 / 3.6)
        )
        count = 0
        for veh_id_raw in sub[tc.LAST_STEP_VEHICLE_ID_LIST] or ():
            veh_id = str(veh_id_raw)
            speed = self._current_vehicle_speed_by_id.get(veh_id)
            if speed is None:
                try:
                    speed = float(self._traci.vehicle.getSpeed(veh_id))
                except Exception as exc:
                    raise RuntimeError(
                        f"Missing speed for full-lane vehicle veh_id={veh_id} "
                        f"lane_id={lane_id}"
                    ) from exc
            count += int(float(speed) <= threshold)
        return float(count)

    def _accumulate_interval_wait(self) -> None:
        if not self._interval_wait_accum:
            self._reset_interval_wait()
        for lane_id in self._interval_wait_accum:
            e2_id = self.net_info.lane_to_e2[lane_id]
            n_wait = self._traci.lanearea.getLastStepHaltingNumber(e2_id)
            self._interval_wait_accum[lane_id] += float(n_wait)
            self._interval_full_wait_accum[lane_id] += max(
                float(n_wait),
                self._full_lane_halting_count(lane_id),
            )

    def _reset_interval_pass(self) -> None:
        if not self._interval_pass_accum:
            for lane_id in self._controlled_incoming_lane_ids():
                self._interval_pass_accum[lane_id] = 0.0
                self._interval_truck_pass_accum[lane_id] = 0.0
        else:
            for lane_id in self._interval_pass_accum:
                self._interval_pass_accum[lane_id] = 0.0
                self._interval_truck_pass_accum[lane_id] = 0.0

    def _accumulate_interval_pass(self) -> None:
        if not self._interval_pass_accum:
            self._reset_interval_pass()
        for lane_id in self._interval_pass_accum:
            detector_id = self.net_info.lane_to_e1_all[lane_id]
            values = self._traci.inductionloop.getSubscriptionResults(detector_id) or {}
            passed_ids = {
                str(veh_id)
                for veh_id in values.get(self._tc.LAST_STEP_VEHICLE_ID_LIST, ()) or ()
                if str(veh_id)
            }
            self._interval_pass_accum[lane_id] += float(len(passed_ids))
            self._interval_truck_pass_accum[lane_id] += float(sum(
                self._is_truck(self._veh_type_cache.get(veh_id, ""))
                for veh_id in passed_ids
            ))

    def _reset_interval_green_service(self) -> None:
        """Clear actual effective-green seconds for every incoming lane."""
        self._interval_green_service_accum = {
            lane_id: 0.0 for lane_id in self._all_incoming_lane_ids()
        }

    def _accumulate_interval_green_service(self) -> None:
        """Accumulate one second for lanes served by SUMO's actual green."""
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
                continue
            phase_lane_mask = build_phase_lane_mask(iinfo)
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
        iinfo   = self.net_info.get_intersection(tl_id)
        ss      = self._signal_states[tl_id]
        traci   = self._traci

        all_lanes = iinfo.all_inc_lanes_flat()  # N→E→S→W 展平 lane 列表

        queue_veh   = []
        queue_full_lane = []
        speed_mps   = []
        truck_count = []
        total_count = []
        demand_veh  = []
        pass_veh    = []
        green_service_s = []
        NOx_mg      = []
        wait_vwt    = []
        wait_vwt_full_lane = []
        NOx_full_lane_mg = []
        NOx_signal_sensitive_mg = []

        pollutant = self.env_cfg.emission_pollutants[0] if self.env_cfg.emission_pollutants else "NOx"
        lane_emission = (
            step_emission.by_lane_analysis_full if step_emission is not None else {}
        )
        e2_lane_emission = (
            step_emission.by_lane_observation_zone if step_emission is not None else {}
        )
        e2_signal_sensitive_emission = (
            step_emission.by_lane_observation_zone_signal_sensitive
            if step_emission is not None
            else {}
        )
        edge_emission = step_emission.by_edge if step_emission is not None else {}

        for lane_id in all_lanes:
            tc = self._tc
            if tc is None:
                raise RuntimeError("TraCI constants are not initialized")
            sub = traci.lane.getSubscriptionResults(lane_id) or {}
            spd = sub[tc.LAST_STEP_MEAN_SPEED]
            speed_mps.append(float(spd))

            e2_id = self.net_info.lane_to_e2[lane_id]
            veh_ids = list(self._traci.lanearea.getLastStepVehicleIDs(e2_id) or ())
            queue = self._traci.lanearea.getLastStepHaltingNumber(e2_id)
            queue_veh.append(float(queue))
            queue_full_lane.append(
                max(float(queue), self._full_lane_halting_count(lane_id))
            )
            trucks_resident = sum(
                1 for vid in veh_ids
                if self._is_truck(self._veh_type_cache.get(vid, ""))
            )
            total = len(veh_ids)

            truck_pass = float(self._last_truck_pass_by_lane.get(lane_id, 0.0))
            truck_count.append(float(truck_pass + trucks_resident))
            total_count.append(float(total))

            e1_cur = float(self._last_pass_by_lane.get(lane_id, 0.0))
            pass_veh.append(float(e1_cur))
            demand = e1_cur + total
            demand_veh.append(float(demand))

            NOx_mg.append(float(e2_lane_emission.get((lane_id, pollutant), 0.0)))
            NOx_full_lane_mg.append(
                float(lane_emission.get((lane_id, pollutant), 0.0))
            )
            NOx_signal_sensitive_mg.append(
                float(
                    e2_signal_sensitive_emission.get((lane_id, pollutant), 0.0)
                )
            )

            wait_vwt.append(float(self._last_wait_vwt_by_lane.get(lane_id, 0.0)))
            wait_vwt_full_lane.append(
                float(self._last_full_wait_vwt_by_lane.get(lane_id, 0.0))
            )
            green_service_s.append(
                float(self._last_green_service_by_lane.get(lane_id, 0.0))
            )

        edge_NOx = []
        for eid in iinfo.inc_edges_nesw:
            if eid is None:
                edge_NOx.append(0.0)
                continue
            edge_info = self.net_info.edge_info[eid]
            edge_NOx.append(float(edge_emission.get((eid, pollutant), 0.0)))

        _sg, _sr = self._last_sumo_phase.get(tl_id, (-1, -1))
        _sumo_state = self._traci.trafficlight.getRedYellowGreenState(tl_id)
        _env_state = (
            self._build_yellow_state_from_old_green(
                self._get_green_state(tl_id, ss.current_phase)
            )
            if ss.in_yellow
            else self._get_green_state(tl_id, ss.current_phase)
        )

        return StepRawObs(
            tl_id              = tl_id,
            queue_veh          = queue_veh,
            queue_full_lane    = queue_full_lane,
            speed_mps          = speed_mps,
            truck_count        = truck_count,
            total_count        = total_count,
            demand_veh         = demand_veh,
            pass_veh           = pass_veh,
            green_service_s    = green_service_s,
            NOx_mg             = NOx_mg,
            wait_vwt           = wait_vwt,
            wait_vwt_full_lane = wait_vwt_full_lane,
            NOx_full_lane_mg   = NOx_full_lane_mg,
            NOx_signal_sensitive_mg = NOx_signal_sensitive_mg,
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

    def compute_action_mask(self, tl_id: str) -> np.ndarray:
        iinfo    = self.net_info.get_intersection(tl_id)
        ss       = self._signal_states[tl_id]
        n_phases = iinfo.n_green_phases
        cfg      = self.env_cfg

        mask = np.ones(n_phases, dtype=bool)

        if ss.elapsed_green < cfg.min_green:
            mask[:] = False
            mask[ss.current_phase] = True
            return mask

        if ss.elapsed_green >= cfg.max_green:
            mask[ss.current_phase] = False

        for k in range(n_phases):
            if k == ss.current_phase:
                continue  # 当前相位不受 min_red 约束
            elapsed_red = self._sim_time - ss.phase_last_went_red[k]
            if elapsed_red < cfg.min_red:
                mask[k] = False

        if not mask.any():
            mask[ss.current_phase] = True

        return mask

    def get_lanearea_halting(self, detector_id: str) -> int:
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
        """Return the cached, normalized sedan/truck type for a live vehicle."""
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

    def get_arrived_count(self) -> int:
        return self._traci.simulation.getArrivedNumber()

    def _init_e1_counters(self) -> None:
        self._e1_prev_count = {}
        for lane_id, det_id in self.net_info.lane_to_e1_all.items():
            self._e1_prev_count[det_id] = 0

    @staticmethod
    def _parse_tripinfo(tripinfo_path: str) -> EpisodeTrafficStats:
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
