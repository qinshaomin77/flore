
from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

SUMO_SEED_MAX = 2000

def build_sumo_seed_list(
    global_seed: int = 42,
    total_episodes: int = 1,
    mode: str = "per_episode",
) -> list[int]:
    n = int(total_episodes)
    if n <= 0:
        return []

    seed = int(global_seed)
    seed_mode = str(mode).strip().lower()
    if seed_mode == "fixed":
        return [seed for _ in range(n)]
    if seed_mode != "per_episode":
        raise ValueError("sumo_seed_mode must be fixed or per_episode")

    rng = random.Random(seed)
    return [int(rng.randint(1, SUMO_SEED_MAX)) for _ in range(n)]

@dataclass
class EnvConfig:

    sumo_cfg: str = "data/kunshan/kunshan_freight_enhanced_eval.sumocfg"
    net_xml: str = "data/kunshan/kunshan.net.xml"
    add_xml: str = "data/kunshan/kunshan_eval.add.xml"
    intersection_groups_json: str = "data/kunshan/intersection_groups.json"
    replace_sumocfg_additional_files: bool = False

    episode_duration: int = 3600
    decision_interval: int = 10
    yellow_duration: int = 3
    min_green: int = 5
    max_green: int = 90
    min_red: int = 15

    tripinfo_dir: str = "results/training/tripinfo"
    use_state_based_tls_control: bool = True
    sumo_seed_mode: str = "per_episode"
    save_sumo_aux_outputs: bool = True
    keep_sumo_runtime_files: bool = True

    save_fcd_output: bool = False
    fcd_output_acceleration: bool = True
    fcd_output_period: float = 1.0
    save_vehicle_state_output: bool = False
    vehicle_state_row_group_rows: int = 100_000
    vehicle_state_compression: str = "zstd"
    vehicle_state_compression_level: int = 3

    emission_factor_csv: str = "data/emission/emission_factor_table.csv"
    emission_pollutants: tuple[str, ...] = ("NOx",)
    vehicle_type_policy: str = "sedan_truck_only"
    record_vehicle_emission_steps: bool = True

    # Evaluation-only lane mechanism diagnostics. These fields do not enter
    # the agent observation, reward, replay buffer, or checkpoint metadata.
    enable_lane_mechanism_metrics: bool = False
    queue_halting_speed_threshold_mps: float = 5.0 / 3.6
    signal_sensitive_halting_speed_mps: float = 0.1
    signal_sensitive_low_speed_mps: float = 5.0
    signal_sensitive_restart_accel_ms2: float = 0.5

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
        halting_speed = float(self.signal_sensitive_halting_speed_mps)
        low_speed = float(self.signal_sensitive_low_speed_mps)
        restart_accel = float(self.signal_sensitive_restart_accel_ms2)
        queue_halting_speed = float(self.queue_halting_speed_threshold_mps)
        if queue_halting_speed <= 0.0:
            raise ValueError("queue_halting_speed_threshold_mps must be > 0")
        if halting_speed < 0.0:
            raise ValueError("signal_sensitive_halting_speed_mps must be >= 0")
        if low_speed <= halting_speed:
            raise ValueError(
                "signal_sensitive_low_speed_mps must be greater than "
                "signal_sensitive_halting_speed_mps"
            )
        if restart_accel < 0.0:
            raise ValueError("signal_sensitive_restart_accel_ms2 must be >= 0")

@dataclass

@dataclass
class StateConfig:

    lane_input_dim: int = 4
    observation_scope: str = "E2_last_300m"
    lane_feature_names: tuple[str, ...] = (
        "demand",
        "queue",
        "wait_vwt",
        "lane_phase",
    )

@dataclass
class RewardConfig:

    gamma: float = 0.99
    wait_time_scale_k: float = 10.0

    objective_mode: str = "traffic_only"

@dataclass
class EmissionRiskConfig:

    use_truck_count_state: bool = False
    use_nox_risk_state: bool = False
    enabled_reward: bool = False
    log_lane_risk: bool = True

    thresholds_json: str = "configs/thresholds/kunshan/kunshan_w300.json"
    threshold_policy: str = "lane_type"  # global / lane_type
    risk_mode: str = "hybrid_pressure_tail"  # piecewise / hybrid_pressure_tail
    pressure_beta: float = 1.0
    base_weight: float = 0.3
    tail_weight: float = 0.7
    # Pure reward-component ablation; observation risk always remains full.
    reward_risk_mode: str = "full"  # full / no_tail / sum_only
    penalty_aggregate: str = "sum_max"
    penalty_time_normalize: bool = True

    rho: float = 0.8
    kappa: float = 2.0
    risk_clip: float = 10.0

    reward_alpha: float = 0.3
    lambda_e: float = 0.3

@dataclass
class NetworkConfig:

    lane_input_dim: int = 4
    node_hidden_dim: int = 64
    gat_heads: int = 4
    node_update_dim: int = 64
    net_node_dim: int = 64
    bigru_hidden_dim: int = 64
    q_hidden_dim: int = 128
    leaky_relu_slope: float = 0.05

    directions: tuple[str, ...] = ("N", "E", "S", "W", "Self")

    use_group_parameter_sharing: bool = True

@dataclass
class DDQNConfig:

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
    epsilon_decay_steps: int = 50000

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

    enabled: bool = False
    nox_loss_beta: float = 1.0
    nox_q_softplus: bool = False

@dataclass
class LogConfig:
    log_root: str = "results/training"
    run_id: str = "mgmq_ddqn"
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
    max_pressure: MaxPressureConfig = field(default_factory=MaxPressureConfig)

    def validate(self) -> None:
        self.env.validate()
        er = self.emission_risk
        er.threshold_policy = str(er.threshold_policy).strip().lower()
        if er.threshold_policy not in {"global", "lane_type"}:
            raise ValueError("emission_risk.threshold_policy must be global or lane_type")
        er.risk_mode = str(er.risk_mode).strip().lower()
        if er.risk_mode not in {"piecewise", "hybrid_pressure_tail"}:
            raise ValueError(
                "emission_risk.risk_mode must be piecewise or hybrid_pressure_tail"
            )
        if float(er.pressure_beta) <= 0.0:
            raise ValueError("emission_risk.pressure_beta must be > 0")
        if float(er.base_weight) < 0.0:
            raise ValueError("emission_risk.base_weight must be >= 0")
        if float(er.tail_weight) < 0.0:
            raise ValueError("emission_risk.tail_weight must be >= 0")
        if (
            er.risk_mode == "hybrid_pressure_tail"
            and float(er.base_weight) + float(er.tail_weight) <= 0.0
        ):
            raise ValueError(
                "emission_risk.base_weight + tail_weight must be > 0 "
                "for hybrid_pressure_tail"
            )
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

        mode = str(self.reward.objective_mode).strip().lower()
        valid_modes = {"traffic_only", "nox_only", "multi_objective"}
        if mode not in valid_modes:
            raise ValueError(
                "reward.objective_mode must be one of: "
                "traffic_only, nox_only, multi_objective"
            )
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
        valid_observation_scopes = {
            "E2_last_100m",
            "E2_last_300m",
            "E2_last_500m",
            "E2_full_lane",
        }
        if str(self.state.observation_scope) not in valid_observation_scopes:
            raise ValueError(
                "state.observation_scope must be one of: "
                + ", ".join(sorted(valid_observation_scopes))
            )

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
        os.path.join(os.path.dirname(os.path.dirname(here)), raw_path),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return raw_path

def get_config(config_path: Optional[str] = None) -> MasterConfig:
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
    base = str(base or "mgmq_ddqn").strip()
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}"
