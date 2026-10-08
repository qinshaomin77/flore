# -*- coding: utf-8 -*-
"""Configuration for the independent PressLight baseline.

The module intentionally keeps SUMO timing and demand settings in the existing
``config.py``.  It only defines settings that are specific to PressLight.
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


SEGMENT_NAMES = ("near", "middle", "far")
PROJECT_ROOT_FILES = ("config.py", "env.py", "logger.py", "network_parser.py")


def discover_project_root(script_dir: Path) -> Path:
    return Path(script_dir).resolve().parents[1]


@dataclass
class PressLightPathConfig:
    """Input and output paths.

    ``base_add_xml`` is parsed by the existing ``NetworkParser``.  The three
    PressLight E2 detectors stay in a separate additional file so they never
    overwrite ``NetworkInfo.lane_to_e2``.
    """

    sumo_cfg: str = ""
    net_xml: str = ""
    base_add_xml: str = ""
    intersection_groups_json: str = ""
    presslight_add_xml: str = ""
    detector_map_json: str = ""
    output_root: str = "results_presslight"
    train_output_dir: str = ""
    emission_thresholds_json: str = ""


@dataclass
class PressLightStateConfig:
    max_allowed_incoming_lanes: Optional[int] = None
    max_allowed_outgoing_lanes: Optional[int] = None
    max_allowed_phases: Optional[int] = None
    segment_order: tuple[str, ...] = SEGMENT_NAMES
    count_mode: str = "raw"
    require_all_segment_detectors: bool = True


@dataclass
class PressLightGroupingConfig:
    mode: str = "structural"
    canonicalize_rotation: bool = True
    allow_reflection: bool = False
    manual_groups_json: str = ""
    require_full_manual_coverage: bool = True


@dataclass
class PressLightPressureConfig:
    # strict: original PressLight; turn_ratio: engineering adaptation.
    variant: str = "strict"
    aggregate: str = "abs_sum"
    effective_vehicle_length_m: float = 7.5
    incoming_observation_length_m: float = 300.0
    beta_pseudocount: float = 1.0
    beta_tolerance: float = 1e-6
    prior_update_rate: float = 0.05
    reward_scale: float = 1.0


@dataclass
class PressLightDQNConfig:
    algorithm: str = "dqn"  # dqn / ddqn
    hidden_dims: tuple[int, ...] = (20, 20)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 32
    replay_memory_size: int = 20000
    min_replay_size: int = 1000
    online_update_interval: int = 1
    target_update_interval: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50000
    max_grad_norm: float = 5.0
    loss_type: str = "huber"
    # Deprecated compatibility field. Grouping is controlled only by ``grouping``.
    group_parameter_sharing: bool = True
    checkpoint_include_replay: bool = False

    def epsilon_at(self, global_decision_step: int) -> float:
        if self.epsilon_decay_steps <= 0:
            return float(self.epsilon_end)
        fraction = min(
            1.0,
            max(0.0, float(global_decision_step) / float(self.epsilon_decay_steps)),
        )
        return float(
            self.epsilon_start
            + fraction * (self.epsilon_end - self.epsilon_start)
        )


@dataclass
class PressLightTrainingConfig:
    total_episodes: int = 160
    checkpoint_interval: int = 10
    save_latest_each_episode: bool = False
    resume_checkpoint: str = ""
    record_emissions: bool = False

    # Validation is independent of checkpoint saving.  Every periodic
    # checkpoint is saved even when validation is disabled or fails.
    fixed_validation_enabled: bool = True
    validation_seeds: tuple[int, ...] = (10301, 10527, 10846, 11129, 11754)
    min_completion_rate: float = 0.95
    best_metric: str = "avg_travel_time_s"
    best_mode: str = "min"


@dataclass
class PressLightEvaluationConfig:
    episodes: int = 20
    seed: int = 19
    save_detail: bool = True
    save_fcd: bool = True
    save_vehicle_emission_step: bool = False


@dataclass
class PressLightLogConfig:
    save_train_step_detail: bool = False
    train_step_detail_interval: int = 0
    console_step_interval: int = 30
    save_reward_step: bool = True


@dataclass
class PressLightConfig:
    seed: int = 42
    device: str = "auto"
    paths: PressLightPathConfig = field(default_factory=PressLightPathConfig)
    state: PressLightStateConfig = field(default_factory=PressLightStateConfig)
    grouping: PressLightGroupingConfig = field(default_factory=PressLightGroupingConfig)
    pressure: PressLightPressureConfig = field(default_factory=PressLightPressureConfig)
    dqn: PressLightDQNConfig = field(default_factory=PressLightDQNConfig)
    training: PressLightTrainingConfig = field(default_factory=PressLightTrainingConfig)
    evaluation: PressLightEvaluationConfig = field(default_factory=PressLightEvaluationConfig)
    log: PressLightLogConfig = field(default_factory=PressLightLogConfig)

    @property
    def experiment_name(self) -> str:
        base = "presslight" if self.pressure.variant == "strict" else "presslight_tr"
        if self.dqn.algorithm == "ddqn":
            base += "_ddqn"
        return base

    @property
    def train_run_dir(self) -> str:
        if self.paths.train_output_dir:
            return self.paths.train_output_dir
        return os.path.join(self.paths.output_root, "train", self.experiment_name)

    @property
    def eval_run_dir(self) -> str:
        return os.path.join(self.paths.output_root, "evaluate", self.experiment_name)

    def validate(self, base_cfg: Optional[Any] = None) -> None:
        def finite(name: str, value: Any, *, minimum: float | None = None,
                   maximum: float | None = None, strict_minimum: bool = False) -> None:
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite number; actual={value!r}") from exc
            valid = math.isfinite(number)
            if minimum is not None:
                valid = valid and (number > minimum if strict_minimum else number >= minimum)
            if maximum is not None:
                valid = valid and number <= maximum
            if not valid:
                bounds = (
                    f" > {minimum}" if strict_minimum else f" >= {minimum}"
                ) if minimum is not None else ""
                if maximum is not None:
                    bounds += f" and <= {maximum}"
                raise ValueError(f"{name} must be finite{bounds}; actual={value!r}")

        for name, value in (
            ("state.max_allowed_incoming_lanes", self.state.max_allowed_incoming_lanes),
            ("state.max_allowed_outgoing_lanes", self.state.max_allowed_outgoing_lanes),
            ("state.max_allowed_phases", self.state.max_allowed_phases),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{name} must be None or an integer > 0; actual={value!r}")
        if tuple(self.state.segment_order) != SEGMENT_NAMES:
            raise ValueError(f"state.segment_order must equal {SEGMENT_NAMES!r}; actual={self.state.segment_order!r}")
        if self.state.count_mode != "raw":
            raise ValueError(f"state.count_mode must equal 'raw'; actual={self.state.count_mode!r}")
        if self.grouping.mode not in {"structural", "manual", "independent"}:
            raise ValueError(f"grouping.mode must be structural, manual, or independent; actual={self.grouping.mode!r}")
        if self.grouping.mode == "manual" and not self.grouping.manual_groups_json:
            raise ValueError(f"grouping.manual_groups_json is required in manual mode; actual={self.grouping.manual_groups_json!r}")
        if self.pressure.variant not in {"strict", "turn_ratio"}:
            raise ValueError(f"pressure.variant must be 'strict' or 'turn_ratio'; actual={self.pressure.variant!r}")
        if self.pressure.aggregate != "abs_sum":
            raise ValueError(f"pressure.aggregate must equal 'abs_sum'; actual={self.pressure.aggregate!r}")
        finite("pressure.effective_vehicle_length_m", self.pressure.effective_vehicle_length_m, minimum=0, strict_minimum=True)
        finite("pressure.incoming_observation_length_m", self.pressure.incoming_observation_length_m, minimum=0, strict_minimum=True)
        finite("pressure.beta_pseudocount", self.pressure.beta_pseudocount, minimum=0)
        finite("pressure.beta_tolerance", self.pressure.beta_tolerance, minimum=0, strict_minimum=True)
        finite("pressure.prior_update_rate", self.pressure.prior_update_rate, minimum=0, maximum=1)
        finite("pressure.reward_scale", self.pressure.reward_scale, minimum=0, strict_minimum=True)
        if self.dqn.algorithm not in {"dqn", "ddqn"}:
            raise ValueError(f"dqn.algorithm must be 'dqn' or 'ddqn'; actual={self.dqn.algorithm!r}")
        if tuple(self.dqn.hidden_dims) != (20, 20):
            raise ValueError(f"dqn.hidden_dims must equal (20, 20); actual={self.dqn.hidden_dims!r}")
        finite("dqn.learning_rate", self.dqn.learning_rate, minimum=0, strict_minimum=True)
        finite("dqn.gamma", self.dqn.gamma, minimum=0, maximum=1)
        for name, value in (
            ("dqn.replay_memory_size", self.dqn.replay_memory_size),
            ("dqn.batch_size", self.dqn.batch_size),
            ("dqn.min_replay_size", self.dqn.min_replay_size),
            ("dqn.online_update_interval", self.dqn.online_update_interval),
            ("dqn.target_update_interval", self.dqn.target_update_interval),
            ("dqn.epsilon_decay_steps", self.dqn.epsilon_decay_steps),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be an integer > 0; actual={value!r}")
        if not (self.dqn.batch_size <= self.dqn.min_replay_size <= self.dqn.replay_memory_size):
            raise ValueError(
                "dqn.batch_size <= dqn.min_replay_size <= dqn.replay_memory_size is required; "
                f"actual=({self.dqn.batch_size!r}, {self.dqn.min_replay_size!r}, {self.dqn.replay_memory_size!r})"
            )
        finite("dqn.epsilon_end", self.dqn.epsilon_end, minimum=0, maximum=1)
        finite("dqn.epsilon_start", self.dqn.epsilon_start, minimum=0, maximum=1)
        if float(self.dqn.epsilon_end) > float(self.dqn.epsilon_start):
            raise ValueError(f"dqn.epsilon_end must be <= dqn.epsilon_start; actual=({self.dqn.epsilon_end!r}, {self.dqn.epsilon_start!r})")
        finite("dqn.max_grad_norm", self.dqn.max_grad_norm, minimum=0, strict_minimum=True)
        if self.dqn.loss_type not in {"huber", "mse"}:
            raise ValueError(f"dqn.loss_type must be 'huber' or 'mse'; actual={self.dqn.loss_type!r}")
        for name, value in (("training.total_episodes", self.training.total_episodes), ("training.checkpoint_interval", self.training.checkpoint_interval)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be an integer > 0; actual={value!r}")
        if self.training.best_metric not in {
            "avg_travel_time_s", "avg_delay_s", "pressure_mean", "reward_mean"
        }:
            raise ValueError(f"training.best_metric is unsupported; actual={self.training.best_metric!r}")
        expected_mode = "max" if self.training.best_metric == "reward_mean" else "min"
        if self.training.best_mode != expected_mode:
            raise ValueError(f"training.best_mode must be {expected_mode!r} for training.best_metric={self.training.best_metric!r}; actual={self.training.best_mode!r}")
        finite("training.min_completion_rate", self.training.min_completion_rate, minimum=0, maximum=1)
        if self.training.fixed_validation_enabled and not self.training.validation_seeds:
            raise ValueError(f"training.validation_seeds cannot be empty when training.fixed_validation_enabled=True; actual={self.training.validation_seeds!r}")
        if len(set(self.training.validation_seeds)) != len(self.training.validation_seeds):
            raise ValueError(f"training.validation_seeds must be unique; actual={self.training.validation_seeds!r}")
        if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in self.training.validation_seeds):
            raise ValueError(f"training.validation_seeds must contain non-negative integers; actual={self.training.validation_seeds!r}")
        if not isinstance(self.evaluation.episodes, int) or isinstance(self.evaluation.episodes, bool) or self.evaluation.episodes <= 0:
            raise ValueError(f"evaluation.episodes must be an integer > 0; actual={self.evaluation.episodes!r}")
        if not isinstance(self.evaluation.seed, int) or isinstance(self.evaluation.seed, bool) or self.evaluation.seed < 0:
            raise ValueError(f"evaluation.seed must be a non-negative integer; actual={self.evaluation.seed!r}")
        for name, value in (("log.console_step_interval", self.log.console_step_interval), ("log.train_step_detail_interval", self.log.train_step_detail_interval)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0; actual={value!r}")
        if base_cfg is not None:
            if int(base_cfg.env.decision_interval) <= 0:
                raise ValueError(f"base_cfg.env.decision_interval must be > 0; actual={base_cfg.env.decision_interval!r}")
            if int(base_cfg.env.episode_duration) % int(base_cfg.env.decision_interval) != 0:
                raise ValueError(
                    "base_cfg.env.episode_duration must be divisible by "
                    f"base_cfg.env.decision_interval; actual=({base_cfg.env.episode_duration!r}, "
                    f"{base_cfg.env.decision_interval!r})"
                )

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2, ensure_ascii=False)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(v) for v in value]
    return value


def _deep_update_dataclass(target: Any, updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if not hasattr(target, key):
            raise ValueError(f"Unknown PressLight config field: {target.__class__.__name__}.{key}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _deep_update_dataclass(current, value)
        else:
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(target, key, value)


def _default_paths_from_base(base_cfg: Any) -> PressLightPathConfig:
    base_add = str(base_cfg.env.add_xml)
    network_dir = os.path.dirname(base_add) or "."
    return PressLightPathConfig(
        sumo_cfg=str(base_cfg.env.sumo_cfg),
        net_xml=str(base_cfg.env.net_xml),
        base_add_xml=base_add,
        intersection_groups_json=str(base_cfg.env.intersection_groups_json),
        presslight_add_xml=os.path.join(network_dir, "presslight_e2.add.xml"),
        detector_map_json=os.path.join(network_dir, "presslight_detector_map.json"),
    )


def get_presslight_config(
    config_path: Optional[str] = None,
    base_cfg: Optional[Any] = None,
) -> PressLightConfig:
    """Build PressLight config and optionally apply a YAML/JSON override."""

    cfg = PressLightConfig()
    if base_cfg is not None:
        cfg.seed = int(getattr(base_cfg, "seed", cfg.seed))
        cfg.device = str(getattr(base_cfg, "device", cfg.device))
        cfg.paths = _default_paths_from_base(base_cfg)
        cfg.training.total_episodes = int(
            getattr(getattr(base_cfg, "ddqn", None), "total_episodes", cfg.training.total_episodes)
        )

    if config_path:
        path = Path(config_path).expanduser()
        if not path.is_absolute() and not path.is_file():
            path = discover_project_root(Path(__file__).resolve().parent) / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(str(path))
        with path.open("r", encoding="utf-8") as file:
            if path.suffix.lower() == ".json":
                payload = json.load(file)
            else:
                if yaml is None:
                    raise ImportError("pyyaml is required for a PressLight YAML config")
                payload = yaml.safe_load(file) or {}
        if not isinstance(payload, Mapping):
            raise ValueError("PressLight config must be a mapping")
        _deep_update_dataclass(cfg, payload)

    cfg = copy.deepcopy(cfg)
    cfg.validate(base_cfg)
    return cfg


if __name__ == "__main__":
    print(json.dumps(PressLightConfig().to_dict(), indent=2, ensure_ascii=False))
