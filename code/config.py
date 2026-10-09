# -*- coding: utf-8 -*-
"""
config.py
=========
FLORE 复现版全局配置。

设计目标
--------
1. 严格复现 MGMQ 的 lane-GAT + network Bi-GRU + Q-network + DDQN 训练流程。
2. 默认保持 4 维 lane state：[demand, queue, wait_vwt, lane_phase]；
   use_truck_count_state / use_nox_risk_state 独立控制 truck_count 与 nox_risk 是否进入模型输入。
3. 两类交叉口按 group 共享参数，组间不共享参数；每组保留自己的相位空间。
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# ═══════════════════════════════════════════════════════════════════════
# 环境配置
# ═══════════════════════════════════════════════════════════════════════

SUMO_SEED_MAX = 2000


def build_sumo_seed_list(
    global_seed: int = 42,
    total_episodes: int = 1,
    mode: str = "per_episode",
) -> list[int]:
    """Build deterministic SUMO seeds from one global seed.

    ``per_episode`` samples without replacement so every episode receives a
    distinct SUMO seed. ``fixed`` intentionally reuses the global seed.
    """
    n = int(total_episodes)
    if n <= 0:
        return []

    seed = int(global_seed)
    seed_mode = str(mode).strip().lower()
    if seed_mode == "fixed":
        return [seed for _ in range(n)]
    if seed_mode != "per_episode":
        raise ValueError("sumo_seed_mode must be fixed or per_episode")
    if n > SUMO_SEED_MAX:
        raise ValueError(
            f"per_episode supports at most {SUMO_SEED_MAX} unique SUMO seeds"
        )

    rng = random.Random(seed)
    return [int(value) for value in rng.sample(range(1, SUMO_SEED_MAX + 1), n)]


@dataclass
class EnvConfig:
    """SUMO 仿真与信号控制配置。"""

    sumo_cfg: str = "data/grid36/truck_sensitive_grid36.sumocfg"
    net_xml: str = "data/grid36/truck_sensitive_grid36.net.xml"
    add_xml: str = "data/grid36/truck_sensitive_grid36.add.xml"
    intersection_groups_json: str = "data/grid36/intersection_groups.json"

    # SumoEnv only receives EnvConfig; MasterConfig.validate() synchronizes
    # these values from StateConfig.
    observation_scope: str = "upstream_300m"
    observation_length: float = 300.0

    episode_duration: int = 4200
    decision_interval: int = 10
    yellow_duration: int = 3
    min_green: int = 5
    max_green: int = 90
    min_red: int = 15

    tripinfo_dir: str = "results/training/tripinfo"
    use_state_based_tls_control: bool = True
    sumo_seed_mode: str = "per_episode"
    baseline_seed_offset: int = 100000
    baseline_control_mode: str = "sumo_actuated"
    baseline_sumo_cfg: str = "data/grid36/truck_sensitive_grid36_actuated.sumocfg"
    baseline_net_xml: str = "data/grid36/truck_sensitive_grid36.net.xml"
    # Compatibility field only. Detector XML output is always disabled by env.py.
    save_detector_outputs: bool = False
    save_sumo_aux_outputs: bool = True

    # Whether to save SUMO native floating-car-data output.
    save_fcd_output: bool = False
    # Whether to save compact per-vehicle, per-second evaluation data.
    save_vehicle_second_output: bool = False
    # Stream auditable per-vehicle state and HBEFA4/MOVES NOx to Parquet.
    save_vehicle_state_output: bool = False
    vehicle_state_row_group_rows: int = 100_000
    vehicle_state_compression: str = "zstd"
    vehicle_state_compression_level: int = 3
    # Save SUMO-native TLS switch-state and switch-time XML outputs.
    save_tls_phase_outputs: bool = False
    # Include longitudinal acceleration in FCD vehicle records.
    fcd_output_acceleration: bool = True
    # FCD recording interval in seconds.
    fcd_output_period: float = 1.0
    # Print per-episode output paths when starting SUMO. Evaluation keeps this
    # disabled by default so the console only shows concise progress lines.
    verbose_runtime_output: bool = True

    # emission_lookup.py 继续保留；可通过 emission_risk 开关启用 NOx 风险 state/reward。
    emission_factor_csv: str = "data/emission/emission_factor_table.csv"
    emission_pollutants: tuple[str, ...] = ("NOx",)
    vehicle_type_policy: str = "sedan_truck_only"

    @property
    def steps_per_episode(self) -> int:
        return int(self.episode_duration // self.decision_interval)

    def validate(self) -> None:
        if self.decision_interval <= 0:
            raise ValueError("decision_interval must be > 0")
        if self.episode_duration % self.decision_interval != 0:
            raise ValueError("episode_duration must be divisible by decision_interval")
        if self.yellow_duration < 0:
            raise ValueError("yellow_duration must be non-negative")
        if self.sumo_seed_mode not in {"fixed", "per_episode"}:
            raise ValueError("sumo_seed_mode must be fixed or per_episode")
        if float(self.fcd_output_period) <= 0:
            raise ValueError("fcd_output_period must be > 0")


# 保留兼容 env.py 中旧 BaselineRunner 的 import；MGMQ 训练不使用。
@dataclass
class ThresholdConfig:
    nox_max_quantile: float = 0.80
    exclude_zero_for_calibration: bool = True
    zero_eps: float = 1e-9
    warmup_seconds: int = 300


# ═══════════════════════════════════════════════════════════════════════
# 状态 / 奖励 / 网络配置
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StateConfig:
    """MGMQ lane state 配置。

    默认 4 维：[demand, queue, wait_vwt, lane_phase]。
    MasterConfig.validate() 根据 use_truck_count_state / use_nox_risk_state 扩展为 5 或 6 维。
    """

    lane_input_dim: int = 4
    observation_scope: str = "upstream_300m"
    observation_length: float = 300.0
    lane_feature_names: tuple[str, ...] = (
        "demand",
        "queue",
        "wait_vwt",
        "lane_phase",
    )


@dataclass
class RewardConfig:
    """Traffic / NOx objective configuration."""

    gamma: float = 0.99
    wait_time_scale_k: float = 10.0

    # auto / traffic_only / nox_only / multi_objective
    #
    # auto preserves compatibility with existing YAML files:
    # - emission_risk.enabled_reward=False -> traffic_only
    # - emission_risk.enabled_reward=True  -> multi_objective
    # Internal reward tokens are retained: traffic_only = FLORE-TS/FLORE-TO; multi_objective = FLORE.
    objective_mode: str = "auto"




@dataclass
class EmissionRiskConfig:
    """Lane-level NOx 风险感知配置。

    thresholds_json 默认读取 configs/thresholds/grid36/baseline.json。
    其中单一全局 NOx_max 来自 baseline 正排放样本 P80 标定，
    在训练中作为 lane-level NOx risk scale，而不是法规超标阈值。
    """

    # 是否把 truck_count 作为模型 lane state 输入。
    use_truck_count_state: bool = False
    # 是否把 lane-level nox_risk 作为模型 lane state 输入。
    use_nox_risk_state: bool = False
    # 是否在 reward 中加入 NOx risk penalty。
    enabled_reward: bool = False
    # 是否在 lane_step.csv 中记录 NOx 原值、阈值、pressure、risk、warning/exceed。
    # 注意：只控制日志记录，不代表 nox_risk 进入 state 或 reward。
    log_lane_risk: bool = True

    thresholds_json: str = "configs/thresholds/grid36/baseline.json"
    risk_mode: str = "piecewise"
    pressure_beta: float = 1.0
    base_weight: float = 0.3
    tail_weight: float = 0.7
    # Pure reward-component ablation. This never changes observation risk.
    # full / no_tail / sum_only
    reward_risk_mode: str = "full"
    penalty_aggregate: str = "sum_max"
    penalty_time_normalize: bool = False

    # 风险函数参数：rho 是 P95 风险尺度的预警比例，不是 P80 分位数。
    rho: float = 0.8
    kappa: float = 2.0
    # state 与 reward 统一使用同一个截断上限。
    risk_clip: float = 10.0

    # 交叉口风险聚合：alpha * sum(risk) + (1-alpha) * max(risk)。
    reward_alpha: float = 0.3
    lambda_e: float = 0.3


@dataclass
class NetworkConfig:
    """MGMQ 网络结构配置。

    lane_input_dim 默认 4；由 use_truck_count_state / use_nox_risk_state 自动同步。
    """

    lane_input_dim: int = 4
    node_hidden_dim: int = 64
    gat_heads: int = 4
    node_update_dim: int = 64
    net_node_dim: int = 64
    bigru_hidden_dim: int = 64
    q_hidden_dim: int = 128
    leaky_relu_slope: float = 0.05

    # 五个方向投影：N/E/S/W/Self，每个方向独立 Dense 参数。
    directions: tuple[str, ...] = ("N", "E", "S", "W", "Self")

    # 两组交叉口各自建 Q-network；action_dim 使用该组 n_phases。
    use_group_parameter_sharing: bool = True


@dataclass
class DDQNConfig:
    """DDQN / replay memory 训练配置。"""

    total_episodes: int = 160
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 32
    replay_memory_size: int = 20000
    min_replay_size: int = 1000
    replay_sampling_mode: str = "usage_aware"  # uniform / usage_aware
    usage_penalty_alpha: float = 0.5
    usage_weight_min: float = 0.05
    online_update_interval: int = 1
    target_update_interval: int = 200
    max_grad_norm: float = 5.0
    loss_type: str = "huber"  # huber / mse

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    # Reach epsilon_end after 120 episodes at 4200 s / 10 s per decision.
    epsilon_decay_steps: int = 50400

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    def epsilon_at(self, global_step: int) -> float:
        if self.epsilon_decay_steps <= 0:
            return float(self.epsilon_end)
        frac = min(1.0, max(0.0, float(global_step) / float(self.epsilon_decay_steps)))
        return float(self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start))


@dataclass
class DualHeadConfig:
    """Risk-decomposed dual-head DDQN config."""

    enabled: bool = False
    nox_loss_beta: float = 1.0
    nox_q_softplus: bool = False


@dataclass
class LogConfig:
    log_root: str = "results/training"
    run_id: str = "flore"
    console_event_log: bool = True
    console_mode: str = "compact"  # full / compact / silent
    console_event_whitelist: tuple[str, ...] = (
        "run_start",
        "episode_start",
        "episode_summary_saved",
        "training_time_progress",
        "run_end",
        "exception",
    )
    console_step_interval: int = 10
    parameter_log_interval: int = 10
    detail_log_until_replay_ready: bool = False
    phase_step_log_interval_after_replay: int = 0
    lane_step_log_interval_after_replay: int = 0
    q_step_log_interval_after_replay: int = 0
    tripinfo_keep_interval_after_replay: int = 10
    step_progress_event_interval: int = 180
    update_event_interval: int = 500
    save_step_physical: bool = False
    save_reward_step: bool = True
    save_emission_step: bool = False
    emission_step_log_interval_train: int = 0
    emission_link_scope: str = "controlled_internal"
    save_q_step: bool = False
    save_parameter_stats: bool = False
    save_best: bool = True
    best_metric: str = "reward_mean"
    best_mode: str = "max"  # max / min

    @property
    def run_dir(self) -> str:
        return os.path.join(self.log_root, self.run_id)


@dataclass
class MaxPressureConfig:
    detector_length_m: float = 300.0
    halting_speed_threshold_mps: float = 0.1
    use_protected_green_only: bool = True
    protected_green_weight: float = 1.0
    permissive_green_weight: float = 0.0
    downstream_lane_weight_mode: str = "equal"
    tie_break_mode: str = "keep_current"
    use_e2_subscription: bool = False
    log_pressure_step: bool = True
    log_movement_step: bool = True
    log_shared_lane_classification: bool = True
    enable_spillback_constraint: bool = False


@dataclass
class MasterConfig:
    seed: int = 42
    device: str = "auto"
    env: EnvConfig = field(default_factory=EnvConfig)
    state: StateConfig = field(default_factory=StateConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    emission_risk: EmissionRiskConfig = field(default_factory=EmissionRiskConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ddqn: DDQNConfig = field(default_factory=DDQNConfig)
    dual_head: DualHeadConfig = field(default_factory=DualHeadConfig)
    log: LogConfig = field(default_factory=LogConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)  # 兼容旧 env.py
    max_pressure: MaxPressureConfig = field(default_factory=MaxPressureConfig)

    def validate(self) -> None:
        self.env.observation_scope = str(self.state.observation_scope).strip().lower()
        self.env.observation_length = float(self.state.observation_length)
        self.env.validate()
        er = self.emission_risk
        er.risk_mode = str(er.risk_mode).strip().lower()
        if er.risk_mode not in {"piecewise", "hybrid_pressure_tail"}:
            raise ValueError("emission_risk.risk_mode must be piecewise or hybrid_pressure_tail")
        if (
            er.risk_mode == "piecewise"
            and (bool(er.use_nox_risk_state) or bool(er.enabled_reward))
        ):
            warnings.warn(
                "risk_mode=piecewise uses tail risk only; "
                "base_weight and tail_weight are ignored.",
                UserWarning,
                stacklevel=2,
            )
        if float(er.pressure_beta) <= 0.0:
            raise ValueError("emission_risk.pressure_beta must be > 0")
        if float(er.base_weight) < 0.0:
            raise ValueError("emission_risk.base_weight must be >= 0")
        if float(er.tail_weight) < 0.0:
            raise ValueError("emission_risk.tail_weight must be >= 0")
        if er.risk_mode == "hybrid_pressure_tail" and float(er.base_weight) + float(er.tail_weight) <= 0.0:
            raise ValueError("emission_risk.base_weight + tail_weight must be > 0 for hybrid_pressure_tail")
        er.reward_risk_mode = str(er.reward_risk_mode).strip().lower()
        if er.reward_risk_mode not in {"full", "no_tail", "sum_only"}:
            raise ValueError(
                "emission_risk.reward_risk_mode must be full, no_tail, or sum_only"
            )
        er.penalty_aggregate = str(er.penalty_aggregate).strip().lower()
        if er.penalty_aggregate not in {"sum_max"}:
            raise ValueError("emission_risk.penalty_aggregate must be sum_max")
        er.penalty_time_normalize = bool(er.penalty_time_normalize)
        if float(er.rho) < 0.0 or float(er.rho) >= 1.0:
            raise ValueError("emission_risk.rho must be in [0, 1)")
        if float(er.kappa) < 0.0:
            raise ValueError("emission_risk.kappa must be >= 0")
        if float(er.risk_clip) <= 0.0:
            raise ValueError("emission_risk.risk_clip must be > 0")
        if float(er.reward_alpha) < 0.0 or float(er.reward_alpha) > 1.0:
            raise ValueError("emission_risk.reward_alpha must be in [0, 1]")
        if float(er.lambda_e) < 0.0:
            raise ValueError("emission_risk.lambda_e must be >= 0")

        mode = str(getattr(self.reward, "objective_mode", "auto")).strip().lower()
        valid_modes = {
            "auto",
            "traffic_only",
            "nox_only",
            "multi_objective",
        }
        if mode not in valid_modes:
            raise ValueError(
                "reward.objective_mode must be one of: "
                "auto, traffic_only, nox_only, multi_objective"
            )
        if mode == "auto":
            mode = "multi_objective" if bool(er.enabled_reward) else "traffic_only"
        self.reward.objective_mode = mode
        if mode in {"nox_only", "multi_objective"} and not bool(er.enabled_reward):
            raise ValueError(
                f"reward.objective_mode={mode!r} requires "
                "emission_risk.enabled_reward=True"
            )

        dh = self.dual_head
        if float(dh.nox_loss_beta) < 0.0:
            raise ValueError("dual_head.nox_loss_beta must be >= 0")
        if bool(dh.enabled) and not bool(er.enabled_reward):
            raise ValueError(
                "dual_head.enabled=True requires emission_risk.enabled_reward=True."
            )
        if bool(dh.enabled) and self.reward.objective_mode != "multi_objective":
            raise ValueError(
                "dual_head.enabled=True is only supported with "
                "reward.objective_mode='multi_objective'."
            )

        lane_feature_names = [
            "demand",
            "queue",
            "wait_vwt",
            "lane_phase",
        ]
        if bool(er.use_truck_count_state):
            lane_feature_names.append("truck_count")
        if bool(er.use_nox_risk_state):
            lane_feature_names.append("nox_risk")
        lane_feature_names = tuple(lane_feature_names)
        if str(self.state.observation_scope).strip().lower() not in {"full_lane", "upstream_300m"}:
            raise ValueError("state.observation_scope must be full_lane or upstream_300m")
        if float(self.state.observation_length) <= 0:
            raise ValueError("state.observation_length must be > 0")

        expected_lane_dim = len(lane_feature_names)
        self.state.lane_input_dim = int(expected_lane_dim)
        self.network.lane_input_dim = int(expected_lane_dim)
        self.state.lane_feature_names = lane_feature_names

        if self.ddqn.loss_type not in {"huber", "mse"}:
            raise ValueError("ddqn.loss_type must be huber or mse")
        self.ddqn.replay_sampling_mode = str(self.ddqn.replay_sampling_mode).strip().lower()
        if self.ddqn.replay_sampling_mode not in {"uniform", "usage_aware"}:
            raise ValueError("ddqn.replay_sampling_mode must be uniform or usage_aware")
        if float(self.ddqn.usage_penalty_alpha) < 0:
            raise ValueError("ddqn.usage_penalty_alpha must be >= 0")
        if float(self.ddqn.usage_weight_min) <= 0 or float(self.ddqn.usage_weight_min) > 1:
            raise ValueError("ddqn.usage_weight_min must be in (0, 1]")
        if self.log.best_mode not in {"max", "min"}:
            raise ValueError("log.best_mode must be max or min")
        self.log.console_mode = str(self.log.console_mode).strip().lower()
        if self.log.console_mode not in {"full", "compact", "silent"}:
            raise ValueError("log.console_mode must be full, compact, or silent")
        self.log.console_event_whitelist = tuple(str(x) for x in self.log.console_event_whitelist)
        if self.log.console_step_interval <= 0:
            raise ValueError("log.console_step_interval must be > 0")
        if int(self.log.parameter_log_interval) <= 0:
            raise ValueError("log.parameter_log_interval must be > 0")
        if int(self.log.step_progress_event_interval) <= 0:
            raise ValueError("log.step_progress_event_interval must be > 0")
        for name in [
            "phase_step_log_interval_after_replay",
            "lane_step_log_interval_after_replay",
            "q_step_log_interval_after_replay",
            "tripinfo_keep_interval_after_replay",
            "update_event_interval",
            "emission_step_log_interval_train",
        ]:
            if int(getattr(self.log, name)) < 0:
                raise ValueError(f"log.{name} must be >= 0")

    def resolve_device(self) -> str:
        if self.device and self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _deep_update_dataclass(obj: Any, updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown config field: {obj.__class__.__name__}.{key}")
        cur = getattr(obj, key)
        if is_dataclass(cur) and isinstance(value, dict):
            _deep_update_dataclass(cur, value)
        else:
            setattr(obj, key, value)


def _resolve_config_path(config_path: str) -> str:
    raw_path = os.path.expanduser(str(config_path))
    if os.path.isabs(raw_path):
        return raw_path

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        raw_path,
        os.path.join(here, raw_path),
        os.path.join(os.path.dirname(here), raw_path),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return raw_path


def get_config(config_path: Optional[str] = None) -> MasterConfig:
    """读取默认配置，可选用 YAML 覆盖。"""
    cfg = MasterConfig()
    if config_path:
        if yaml is None:
            raise ImportError("pyyaml is required to load yaml config")
        resolved_config_path = _resolve_config_path(str(config_path))
        with open(resolved_config_path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        if not isinstance(payload, dict):
            raise ValueError("config yaml must be a mapping")
        _deep_update_dataclass(cfg, payload)
    cfg.validate()
    return cfg


def make_run_id(base: str) -> str:
    base = str(base or "flore").strip()
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}"


if __name__ == "__main__":
    cfg = get_config()
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
