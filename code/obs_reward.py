# -*- coding: utf-8 -*-
"""
obs_reward.py
=============
MGMQ-DDQN 状态构建 + action mask + reward。

默认保持 MGMQ 复现版 4 维 lane 输入：
    [demand, queue, wait_vwt, lane_phase]

use_truck_count_state / use_nox_risk_state 分别追加 truck_count 与 lane-level NOx 风险评估值。

truck_count 只从 raw_obs.truck_count 读取，不在本文件中调用 TraCI 或重复计算。

NOx 风险评估统一在本文件中计算，logger.py 只读取 obs.debug["lane"] 写 CSV，
避免 state、reward、日志三处重复计算导致不一致。
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from config import MasterConfig
from env import StepRawObs
from network_parser import IntersectionInfo, NetworkInfo, build_phase_lane_mask


@dataclass
class AgentObservation:
    """单个路口的 MGMQ 模型输入。"""

    tl_id: str
    group_id: str
    lane_features: np.ndarray       # [n_lanes, F], F=4/5/6
    A_same: np.ndarray              # [n_lanes, n_lanes]
    A_diff: np.ndarray              # [n_lanes, n_lanes]
    action_mask: np.ndarray         # [n_phases], bool
    current_phase: int
    lane_feature_names: Tuple[str, ...] = (
        "demand",
        "queue",
        "wait_vwt",
        "lane_phase",
    )
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardComponents:
    """MGMQ traffic reward 与 NOx risk reward 分量。"""

    tl_id: str
    rp: float
    rwn: float
    rpn: float
    rwt: float

    # 原始 MGMQ 交通效率 reward
    traffic_reward: float = 0.0

    # NOx diagnostics; lambda_e decides whether the penalty affects reward.
    emission_penalty: float = 0.0
    lambda_e: float = 0.0
    nox_risk_mean: float = 0.0
    nox_risk_sum: float = 0.0
    nox_risk_max: float = 0.0
    risk_reward_mode: str = "full"
    reward_risk_sum: float = 0.0
    reward_risk_max: float = 0.0
    mean_lane_risk: float = 0.0
    p95_lane_risk: float = 0.0
    nox_pressure_mean: float = 0.0
    nox_pressure_max: float = 0.0
    nox_warning_lane_count: int = 0
    nox_exceed_lane_count: int = 0

    # Final reward depends on cfg.reward.objective_mode:
    # traffic_only    : traffic_reward
    # nox_only        : -emission_penalty
    # multi_objective : traffic_reward - lambda_e * emission_penalty
    reward: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rp": float(self.rp),
            "rwn": float(self.rwn),
            "rpn": float(self.rpn),
            "rwt": float(self.rwt),
            "traffic_reward": float(self.traffic_reward),
            "emission_penalty": float(self.emission_penalty),
            "lambda_e": float(self.lambda_e),
            "nox_risk_mean": float(self.nox_risk_mean),
            "nox_risk_sum": float(self.nox_risk_sum),
            "nox_risk_max": float(self.nox_risk_max),
            "risk_reward_mode": str(self.risk_reward_mode),
            "reward_risk_sum": float(self.reward_risk_sum),
            "reward_risk_max": float(self.reward_risk_max),
            "mean_lane_risk": float(self.mean_lane_risk),
            "p95_lane_risk": float(self.p95_lane_risk),
            "nox_pressure_mean": float(self.nox_pressure_mean),
            "nox_pressure_max": float(self.nox_pressure_max),
            "nox_warning_lane_count": float(self.nox_warning_lane_count),
            "nox_exceed_lane_count": float(self.nox_exceed_lane_count),
            "reward": float(self.reward),
        }


@dataclass
class StepObsReward:
    observations: Dict[str, AgentObservation]
    rewards: Dict[str, float]
    reward_components: Dict[str, RewardComponents]


def _as_float_array(x: Sequence[float], n: int, name: str, tl_id: str) -> np.ndarray:
    arr = np.asarray(list(x), dtype=np.float32)
    if arr.shape[0] != n:
        raise ValueError(f"{tl_id}: {name} 长度 {arr.shape[0]} 与 n_lanes={n} 不一致")
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _safe_positive_float(value: Any, default: float = 1.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = float(default)
    if not np.isfinite(v) or v <= 0.0:
        v = float(default)
    return max(v, 1e-6)


def _resolve_effective_risk_weights(
    base_weight: float,
    tail_weight: float,
) -> tuple[float, float]:
    """Validate and normalize hybrid NOx-risk composition weights."""
    base = float(base_weight)
    tail = float(tail_weight)
    if not np.isfinite(base) or not np.isfinite(tail):
        raise ValueError("NOx risk weights must be finite")
    if base < 0.0 or tail < 0.0:
        raise ValueError("NOx risk weights must be non-negative")
    weight_sum = base + tail
    if weight_sum <= 0.0:
        raise ValueError("NOx risk base_weight + tail_weight must be > 0")
    return float(base / weight_sum), float(tail / weight_sum)


class ObsRewardBuilder:
    """构造 lane state、action mask 和 MGMQ / NOx-risk reward。"""

    def __init__(self, cfg: MasterConfig, net_info: NetworkInfo) -> None:
        self.cfg = cfg
        self.net_info = net_info
        self.k = float(cfg.reward.wait_time_scale_k)
        self._phase_lane_masks: Dict[str, np.ndarray] = {
            tl_id: build_phase_lane_mask(net_info.get_intersection(tl_id))
            for tl_id in net_info.intersection_ids
        }

        self._nox_tau_global: float = 1.0
        self._thresholds_path: str = ""
        self._risk_cache_sim_time: float | None = None
        self._risk_cache: Dict[Tuple[str, float], Dict[str, np.ndarray]] = {}
        if self._need_emission_risk():
            self._load_emission_thresholds()

    # ───────────────────────────────────────────────────────────────
    # NOx risk helpers
    # ───────────────────────────────────────────────────────────────

    def _need_emission_risk(self) -> bool:
        er = getattr(self.cfg, "emission_risk", None)
        if er is None:
            return False
        return bool(er.use_nox_risk_state or er.enabled_reward or er.log_lane_risk)

    def _resolve_thresholds_path(self) -> Path:
        er = self.cfg.emission_risk
        raw_path = Path(str(er.thresholds_json)).expanduser()
        if raw_path.is_absolute():
            return raw_path

        here = Path(__file__).resolve().parent
        candidates = [
            Path.cwd() / raw_path,
            here / raw_path,
            here.parent / raw_path,
        ]
        for p in candidates:
            if p.exists():
                return p.resolve()
        # 返回最符合用户项目结构的默认候选，便于报错定位。
        return (here.parent / raw_path).resolve()

    def _load_emission_thresholds(self) -> None:
        path = self._resolve_thresholds_path()
        if not path.exists():
            raise FileNotFoundError(
                "NOx risk threshold JSON not found. "
                f"Expected cfg.emission_risk.thresholds_json={self.cfg.emission_risk.thresholds_json!r}; "
                f"resolved path={path}"
            )
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        calibration_meta = payload.get("calibration_meta", {}) or {}
        observation_scope = payload.get(
            "observation_scope",
            calibration_meta.get("observation_scope"),
        )
        current_scope = str(self.cfg.state.observation_scope).strip().lower()
        if observation_scope not in {None, current_scope}:
            warnings.warn(
                "NOx threshold file observation scope does not match the current "
                f"{current_scope!r} observations. "
                "Recalibrate lane NOx thresholds before production retraining; "
                f"threshold_scope={observation_scope!r}, path={path}",
                RuntimeWarning,
                stacklevel=2,
            )
        self._thresholds_path = str(path)
        self._nox_tau_global = _safe_positive_float(payload.get("NOx_max", 1.0), 1.0)

    def _lane_nox_tau(self, lane_id: str) -> float:
        if not self._need_emission_risk():
            return 1.0
        return _safe_positive_float(self._nox_tau_global, 1.0)

    def compute_lane_nox_risk(
        self,
        lane_ids: Sequence[str],
        nox_mg: Sequence[float],
    ) -> Dict[str, np.ndarray]:
        """计算 lane-level NOx pressure / risk / warning / exceed。"""
        n = len(lane_ids)
        nox = _as_float_array(nox_mg, n, "NOx_mg", "NOxRisk")
        nox = np.maximum(nox, 0.0).astype(np.float32)

        tau_vals = []
        for lane_id in lane_ids:
            tau_vals.append(float(self._lane_nox_tau(str(lane_id))))
        tau = np.asarray(tau_vals, dtype=np.float32)
        tau = np.maximum(tau, 1e-6)
        pressure = nox / tau
        pressure[~np.isfinite(pressure)] = 0.0

        er = self.cfg.emission_risk
        rho = float(er.rho)
        kappa = float(er.kappa)
        risk_clip = float(er.risk_clip)

        tail_risk = np.zeros_like(pressure, dtype=np.float32)
        warn_mask = pressure >= rho
        mid_mask = (pressure >= rho) & (pressure < 1.0)
        high_mask = pressure >= 1.0
        denom = max(1.0 - rho, 1e-6)
        tail_risk[mid_mask] = ((pressure[mid_mask] - rho) / denom) ** 2
        tail_risk[high_mask] = 1.0 + kappa * ((pressure[high_mask] - 1.0) ** 2)
        tail_risk[~np.isfinite(tail_risk)] = 0.0

        pressure_beta = float(getattr(er, "pressure_beta", 1.0))
        base_risk = np.power(pressure, pressure_beta).astype(np.float32)
        base_risk[~np.isfinite(base_risk)] = 0.0

        risk_mode = str(getattr(er, "risk_mode", "piecewise")).strip().lower()
        if risk_mode == "hybrid_pressure_tail":
            effective_base_weight, effective_tail_weight = (
                _resolve_effective_risk_weights(
                    float(getattr(er, "base_weight", 0.3)),
                    float(getattr(er, "tail_weight", 0.7)),
                )
            )
            risk = (
                effective_base_weight * base_risk
                + effective_tail_weight * tail_risk
            )
        else:
            effective_base_weight = 0.0
            effective_tail_weight = 1.0
            risk = tail_risk

        risk[~np.isfinite(risk)] = 0.0
        risk = np.clip(risk, 0.0, risk_clip).astype(np.float32)

        return {
            "nox_mg": nox.astype(np.float32),
            "nox_tau_mg": tau.astype(np.float32),
            "nox_pressure": pressure.astype(np.float32),
            "nox_risk": risk.astype(np.float32),
            "nox_risk_mode": np.asarray([risk_mode] * n, dtype=object),
            "nox_base_risk": base_risk.astype(np.float32),
            "nox_tail_risk": tail_risk.astype(np.float32),
            "nox_base_weight_effective": np.full(
                n, effective_base_weight, dtype=np.float32
            ),
            "nox_tail_weight_effective": np.full(
                n, effective_tail_weight, dtype=np.float32
            ),
            "nox_warning": warn_mask.astype(np.int32),
            "nox_exceed": high_mask.astype(np.int32),
        }

    def _get_lane_nox_risk_cached(
        self,
        raw: StepRawObs,
        iinfo: IntersectionInfo,
    ) -> Dict[str, np.ndarray]:
        """Return lane-level NOx risk info with an intra-step cache.

        This cache is only an intra-step optimization. It does not change the
        state, reward, or risk formula.
        """
        sim_time = float(getattr(raw, "sim_time", -1.0))

        if self._risk_cache_sim_time is None or float(self._risk_cache_sim_time) != sim_time:
            self._risk_cache.clear()
            self._risk_cache_sim_time = sim_time

        key = (str(raw.tl_id), sim_time)
        cached = self._risk_cache.get(key)
        if cached is not None:
            return cached

        risk_info = self.compute_lane_nox_risk(
            lane_ids=iinfo.all_inc_lanes_flat(),
            nox_mg=getattr(raw, "NOx_mg", [0.0] * iinfo.n_lanes),
        )
        self._risk_cache[key] = risk_info
        return risk_info

    # ───────────────────────────────────────────────────────────────
    # Observation
    # ───────────────────────────────────────────────────────────────

    def build_observations(self, raw_obs_dict: Mapping[str, StepRawObs]) -> Dict[str, AgentObservation]:
        return {tl_id: self.build_observation(raw) for tl_id, raw in raw_obs_dict.items()}

    def build_observation(self, raw: StepRawObs) -> AgentObservation:
        tl_id = raw.tl_id
        iinfo = self.net_info.get_intersection(tl_id)
        lane_features, lane_debug, lane_feature_names = self.build_lane_features(raw, iinfo)
        action_mask = self.compute_action_mask(raw, iinfo)
        A_same = np.asarray(iinfo.A_same, dtype=np.float32)
        A_diff = np.asarray(iinfo.A_diff, dtype=np.float32)

        if A_same.shape != (iinfo.n_lanes, iinfo.n_lanes):
            raise ValueError(f"{tl_id}: A_same shape={A_same.shape} 与 n_lanes={iinfo.n_lanes} 不一致")
        if A_diff.shape != (iinfo.n_lanes, iinfo.n_lanes):
            raise ValueError(f"{tl_id}: A_diff shape={A_diff.shape} 与 n_lanes={iinfo.n_lanes} 不一致")
        if action_mask.shape[0] != iinfo.n_green_phases:
            raise ValueError(f"{tl_id}: action_mask length mismatch")

        return AgentObservation(
            tl_id=tl_id,
            group_id=self.net_info.get_group(tl_id),
            lane_features=lane_features,
            A_same=A_same,
            A_diff=A_diff,
            action_mask=action_mask,
            current_phase=int(raw.current_phase),
            lane_feature_names=lane_feature_names,
            debug={"lane": lane_debug},
        )

    def build_lane_features(self, raw: StepRawObs, iinfo: IntersectionInfo) -> Tuple[np.ndarray, Dict[str, Any], Tuple[str, ...]]:
        n = iinfo.n_lanes
        demand = _as_float_array(raw.demand_veh, n, "demand_veh", raw.tl_id)
        queue = _as_float_array(raw.queue_veh, n, "queue_veh", raw.tl_id)
        wait_vwt = _as_float_array(raw.wait_vwt, n, "wait_vwt", raw.tl_id)
        truck_count = _as_float_array(
            getattr(raw, "truck_count", [0.0] * n),
            n,
            "truck_count",
            raw.tl_id,
        )
        phase_mask = self._phase_lane_masks[raw.tl_id]
        if int(raw.current_phase) < 0 or int(raw.current_phase) >= phase_mask.shape[0]:
            lane_phase = np.zeros(n, dtype=np.float32)
        else:
            lane_phase = phase_mask[int(raw.current_phase)].astype(np.float32)

        features = [demand, queue, wait_vwt, lane_phase]
        feature_names = ["demand", "queue", "wait_vwt", "lane_phase"]
        lane_debug: Dict[str, Any] = {
            "demand": demand,
            "queue": queue,
            "wait_vwt": wait_vwt,
            "lane_phase": lane_phase,
            "truck_count": truck_count,
        }

        risk_info: Dict[str, np.ndarray] | None = None
        if self._need_emission_risk():
            risk_info = self._get_lane_nox_risk_cached(raw, iinfo)
            lane_debug.update(risk_info)

        if bool(self.cfg.emission_risk.use_truck_count_state):
            features.append(truck_count)
            feature_names.append("truck_count")

        if bool(self.cfg.emission_risk.use_nox_risk_state):
            if risk_info is None:
                risk_info = self._get_lane_nox_risk_cached(raw, iinfo)
                lane_debug.update(risk_info)
            features.append(risk_info["nox_risk"])
            feature_names.append("nox_risk")

        lane_features = np.stack(features, axis=-1).astype(np.float32)
        expected_names = tuple(getattr(self.cfg.state, "lane_feature_names", tuple(feature_names)))
        if tuple(feature_names) != expected_names:
            raise ValueError(
                f"{raw.tl_id}: lane_feature_names mismatch, obs={tuple(feature_names)} cfg={expected_names}"
            )
        expected_dim = int(getattr(self.cfg.network, "lane_input_dim", len(feature_names)))
        if int(lane_features.shape[-1]) != expected_dim:
            raise ValueError(
                f"{raw.tl_id}: lane_features dim={lane_features.shape[-1]} but cfg.network.lane_input_dim={expected_dim}"
            )
        return lane_features, lane_debug, tuple(feature_names)

    def compute_action_mask(self, raw: StepRawObs, iinfo: IntersectionInfo) -> np.ndarray:
        """按照 min_green / max_green / min_red 计算 action mask。True=合法动作。"""
        n = iinfo.n_green_phases
        mask = np.ones(n, dtype=bool)
        cur = int(np.clip(int(raw.current_phase), 0, max(0, n - 1)))
        cfg = self.cfg.env

        if bool(raw.in_yellow):
            mask[:] = False
            mask[cur] = True
            return mask

        if float(raw.elapsed_green) < float(cfg.min_green):
            mask[:] = False
            mask[cur] = True
            return mask

        if float(raw.elapsed_green) >= float(cfg.max_green):
            mask[cur] = False

        sim_time = float(raw.sim_time)
        last_red = list(raw.phase_last_went_red or [])
        for p in range(n):
            if p == cur:
                continue
            if p < len(last_red):
                if sim_time - float(last_red[p]) < float(cfg.min_red):
                    mask[p] = False

        if not np.any(mask):
            mask[cur] = True
        return mask

    # ───────────────────────────────────────────────────────────────
    # Reward
    # ───────────────────────────────────────────────────────────────

    def compute_rewards(
        self,
        next_raw_obs_dict: Mapping[str, StepRawObs],
        actions: Mapping[str, int],
        previous_phases: Mapping[str, int],
    ) -> Tuple[Dict[str, float], Dict[str, RewardComponents]]:
        comps: Dict[str, RewardComponents] = {}
        objective_mode = str(
            getattr(self.cfg.reward, "objective_mode", "multi_objective")
        ).strip().lower()
        enabled_reward = bool(getattr(self.cfg.emission_risk, "enabled_reward", False))
        log_lane_risk = bool(getattr(self.cfg.emission_risk, "log_lane_risk", False))
        compute_emission_metrics = enabled_reward or log_lane_risk
        for tl_id, raw in next_raw_obs_dict.items():
            action = int(actions[tl_id])
            prev_phase = int(previous_phases[tl_id])
            rp = 1.0 if action == prev_phase else -1.0
            rwn = float(np.sum(np.asarray(raw.queue_veh, dtype=np.float32)))
            pass_veh = getattr(raw, "pass_veh", None)
            if pass_veh is None:
                # 兜底：旧 StepRawObs 无 pass_veh 时，用 demand-total_count 近似。
                pass_arr = np.asarray(raw.demand_veh, dtype=np.float32) - np.asarray(raw.total_count, dtype=np.float32)
                pass_arr = np.maximum(pass_arr, 0.0)
            else:
                pass_arr = np.asarray(pass_veh, dtype=np.float32)
            rpn = float(np.sum(pass_arr))
            rwt = float(np.sum(np.asarray(raw.wait_vwt, dtype=np.float32)))
            traffic_reward = float(rp - rwn + rpn - rwt / max(self.k, 1e-6))

            emission_penalty = 0.0
            nox_risk_mean = 0.0
            nox_risk_sum = 0.0
            nox_risk_max = 0.0
            risk_reward_mode = str(
                getattr(self.cfg.emission_risk, "reward_risk_mode", "full")
            ).strip().lower()
            reward_risk_sum = 0.0
            reward_risk_max = 0.0
            mean_lane_risk = 0.0
            p95_lane_risk = 0.0
            nox_pressure_mean = 0.0
            nox_pressure_max = 0.0
            nox_warning_lane_count = 0
            nox_exceed_lane_count = 0
            lambda_e = 0.0

            if compute_emission_metrics:
                iinfo = self.net_info.get_intersection(tl_id)
                risk_info = self._get_lane_nox_risk_cached(raw, iinfo)
                risk = np.asarray(risk_info["nox_risk"], dtype=np.float32)
                if risk_reward_mode == "no_tail":
                    reward_risk = np.clip(
                        np.asarray(risk_info["nox_base_risk"], dtype=np.float32),
                        0.0,
                        float(self.cfg.emission_risk.risk_clip),
                    ).astype(np.float32)
                else:
                    # full and sum_only keep the original full lane risk.
                    reward_risk = risk
                pressure = np.asarray(risk_info["nox_pressure"], dtype=np.float32)
                if risk.size:
                    nox_risk_mean = float(np.mean(risk))
                    nox_risk_sum = float(np.sum(risk))
                    nox_risk_max = float(np.max(risk))
                if reward_risk.size:
                    reward_risk_sum = float(np.sum(reward_risk))
                    reward_risk_max = float(np.max(reward_risk))
                    mean_lane_risk = float(np.mean(reward_risk))
                    p95_lane_risk = float(np.percentile(reward_risk, 95))
                if pressure.size:
                    nox_pressure_mean = float(np.mean(pressure))
                    nox_pressure_max = float(np.max(pressure))
                nox_warning_lane_count = int(np.sum(np.asarray(risk_info["nox_warning"], dtype=np.int32)))
                nox_exceed_lane_count = int(np.sum(np.asarray(risk_info["nox_exceed"], dtype=np.int32)))
                alpha = float(self.cfg.emission_risk.reward_alpha)
                penalty_aggregate = str(getattr(self.cfg.emission_risk, "penalty_aggregate", "sum_max")).strip().lower()
                if penalty_aggregate != "sum_max":
                    raise ValueError("emission_risk.penalty_aggregate must be sum_max")
                dt = max(float(getattr(self.cfg.env, "decision_interval", 1.0)), 1e-6)
                if bool(getattr(self.cfg.emission_risk, "penalty_time_normalize", False)):
                    # Only the accumulated sum component is normalized by decision_interval.
                    # The max component represents the peak lane-level risk within the decision step
                    # and should not be divided by time; otherwise hotspot penalties would be
                    # artificially weakened.
                    reward_risk_sum_for_penalty = float(reward_risk_sum) / dt
                else:
                    reward_risk_sum_for_penalty = float(reward_risk_sum)
                if risk_reward_mode == "sum_only":
                    emission_penalty = float(reward_risk_sum_for_penalty)
                else:
                    emission_penalty = float(
                        alpha * reward_risk_sum_for_penalty
                        + (1.0 - alpha) * float(reward_risk_max)
                    )
            if objective_mode == "traffic_only":
                # NOx may still enter state and diagnostics, but not final reward.
                lambda_e = 0.0
                reward = float(traffic_reward)
            elif objective_mode == "nox_only":
                # Keep logged weighted penalty on the same scale as final reward.
                lambda_e = 1.0
                reward = float(-emission_penalty)
            elif objective_mode == "multi_objective":
                if not enabled_reward:
                    raise RuntimeError(
                        "multi_objective requires emission_risk.enabled_reward=True"
                    )
                lambda_e = float(self.cfg.emission_risk.lambda_e)
                reward = float(traffic_reward - lambda_e * emission_penalty)
            else:
                raise ValueError(
                    f"Unsupported reward.objective_mode={objective_mode!r}"
                )
            comps[tl_id] = RewardComponents(
                tl_id=tl_id,
                rp=rp,
                rwn=rwn,
                rpn=rpn,
                rwt=rwt,
                traffic_reward=traffic_reward,
                emission_penalty=emission_penalty,
                lambda_e=lambda_e,
                nox_risk_mean=nox_risk_mean,
                nox_risk_sum=nox_risk_sum,
                nox_risk_max=nox_risk_max,
                risk_reward_mode=risk_reward_mode,
                reward_risk_sum=reward_risk_sum,
                reward_risk_max=reward_risk_max,
                mean_lane_risk=mean_lane_risk,
                p95_lane_risk=p95_lane_risk,
                nox_pressure_mean=nox_pressure_mean,
                nox_pressure_max=nox_pressure_max,
                nox_warning_lane_count=nox_warning_lane_count,
                nox_exceed_lane_count=nox_exceed_lane_count,
                reward=reward,
            )
        rewards = {tl_id: comp.reward for tl_id, comp in comps.items()}
        return rewards, comps

    def build_step(
        self,
        next_raw_obs_dict: Mapping[str, StepRawObs],
        actions: Mapping[str, int],
        previous_phases: Mapping[str, int],
    ) -> StepObsReward:
        observations = self.build_observations(next_raw_obs_dict)
        rewards, comps = self.compute_rewards(next_raw_obs_dict, actions, previous_phases)
        return StepObsReward(observations=observations, rewards=rewards, reward_components=comps)
