
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from obs_reward import AgentObservation

_FEATURE_NAME_CACHE: dict[tuple, tuple] = {}

@dataclass
class ReplayObservation:
    tl_id: str
    group_id: str
    lane_features: np.ndarray
    action_mask: np.ndarray
    current_phase: int
    lane_feature_names: tuple[str, ...] = field(default_factory=tuple)

@dataclass
class GlobalTransition:
    obs: Dict[str, ReplayObservation]
    actions: Dict[str, int]
    rewards: Dict[str, float]
    next_obs: Dict[str, ReplayObservation]
    done: bool
    traffic_rewards: Dict[str, float] = field(default_factory=dict)
    nox_penalties: Dict[str, float] = field(default_factory=dict)

@dataclass
class ReplayItem:
    transition: GlobalTransition
    sample_times: int = 0
    insert_index: int = 0

# 此处刻意不做拷贝，依赖 ObsRewardBuilder 每步新建数组且构造后只读；若修改该前提，必须同步恢复 .copy()。
def make_replay_observation(obs: AgentObservation | ReplayObservation) -> ReplayObservation:
    lane_features = np.asarray(obs.lane_features, dtype=np.float32)
    action_mask = np.asarray(obs.action_mask, dtype=bool)
    if os.environ.get("MGMQ_BUFFER_STRICT") == "1":
        lane_features.flags.writeable = False
        action_mask.flags.writeable = False
    lane_feature_names = tuple(getattr(obs, "lane_feature_names", ()))
    lane_feature_names = _FEATURE_NAME_CACHE.setdefault(lane_feature_names, lane_feature_names)
    return ReplayObservation(
        tl_id=str(obs.tl_id),
        group_id=str(obs.group_id),
        lane_features=lane_features,
        action_mask=action_mask,
        current_phase=int(obs.current_phase),
        lane_feature_names=lane_feature_names,
    )

def make_replay_transition(transition: GlobalTransition) -> GlobalTransition:
    traffic_rewards = getattr(transition, "traffic_rewards", None) or transition.rewards
    nox_penalties = getattr(transition, "nox_penalties", None) or {
        str(tl_id): 0.0 for tl_id in transition.rewards.keys()
    }
    return GlobalTransition(
        obs={str(tl_id): make_replay_observation(obs) for tl_id, obs in transition.obs.items()},
        actions={str(tl_id): int(action) for tl_id, action in transition.actions.items()},
        rewards={str(tl_id): float(reward) for tl_id, reward in transition.rewards.items()},
        next_obs={str(tl_id): make_replay_observation(obs) for tl_id, obs in transition.next_obs.items()},
        done=bool(transition.done),
        traffic_rewards={
            str(tl_id): float(value)
            for tl_id, value in traffic_rewards.items()
        },
        nox_penalties={
            str(tl_id): float(value)
            for tl_id, value in nox_penalties.items()
        },
    )

class ReplayMemory:

    def __init__(
        self,
        capacity: int,
        seed: int = 42,
        sampling_mode: str = "uniform",
        usage_penalty_alpha: float = 0.5,
        usage_weight_min: float = 0.05,
    ) -> None:
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("ReplayMemory capacity must be > 0")

        self.sampling_mode = str(sampling_mode).strip().lower()
        if self.sampling_mode not in {"uniform", "usage_aware"}:
            raise ValueError("ReplayMemory sampling_mode must be uniform or usage_aware")

        self.usage_penalty_alpha = float(usage_penalty_alpha)
        if self.usage_penalty_alpha < 0:
            raise ValueError("ReplayMemory usage_penalty_alpha must be >= 0")

        self.usage_weight_min = float(usage_weight_min)
        if self.usage_weight_min <= 0 or self.usage_weight_min > 1:
            raise ValueError("ReplayMemory usage_weight_min must be in (0, 1]")

        self._data: list[Optional[ReplayItem]] = [None] * self.capacity
        self._sample_times: np.ndarray = np.zeros(self.capacity, dtype=np.int64)
        self._insert_indices: np.ndarray = np.zeros(self.capacity, dtype=np.int64)
        self._size: int = 0
        self._start: int = 0
        self._rng = random.Random(int(seed))
        self._np_rng = np.random.default_rng(int(seed))
        self.push_count = 0
        self.sample_count = 0

    def __len__(self) -> int:
        return int(self._size)

    def _logical_to_physical_indices(self, logical_indices: np.ndarray) -> np.ndarray:
        logical_indices = np.asarray(logical_indices, dtype=np.int64)
        return (int(self._start) + logical_indices) % int(self.capacity)

    def _active_physical_indices(self) -> np.ndarray:
        if self._size <= 0:
            return np.asarray([], dtype=np.int64)
        logical = np.arange(int(self._size), dtype=np.int64)
        return self._logical_to_physical_indices(logical)

    def _items_from_physical_indices(self, indices: np.ndarray) -> list[ReplayItem]:
        items: list[ReplayItem] = []
        for idx in np.asarray(indices, dtype=np.int64):
            item = self._data[int(idx)]
            if item is None:
                raise RuntimeError("ReplayMemory internal state corrupted: sampled empty slot.")
            items.append(item)
        return items

    def push(self, transition: GlobalTransition) -> None:
        transition_copy = make_replay_transition(transition)

        if self._size < self.capacity:
            slot = int(self._size)
            self._size += 1
        else:
            slot = int(self._start)
            self._start = (int(self._start) + 1) % int(self.capacity)

        item = ReplayItem(
            transition=transition_copy,
            sample_times=0,
            insert_index=int(self.push_count),
        )
        self._data[slot] = item
        self._sample_times[slot] = 0
        self._insert_indices[slot] = int(self.push_count)

        self.push_count += 1

    def sample(self, batch_size: int) -> List[GlobalTransition]:
        if self._size == 0:
            return []

        n = min(int(batch_size), int(self._size))
        if n <= 0:
            return []

        self.sample_count += 1
        if self.sampling_mode == "usage_aware":
            items = self._weighted_sample_without_replacement(n)
        else:
            logical = np.asarray(
                self._rng.sample(range(int(self._size)), n),
                dtype=np.int64,
            )
            physical = self._logical_to_physical_indices(logical)
            items = self._items_from_physical_indices(physical)

            for idx, item in zip(physical, items):
                self._sample_times[int(idx)] += 1
                item.sample_times = int(self._sample_times[int(idx)])

        return [item.transition for item in items]

    def _weights(self, items: Optional[List[ReplayItem]] = None) -> List[float]:
        alpha = float(self.usage_penalty_alpha)
        min_weight = float(self.usage_weight_min)

        if items is None:
            active = self._active_physical_indices()
            if active.size == 0:
                return []
            st = self._sample_times[active].astype(np.float64)
            weights = np.maximum(
                min_weight,
                1.0 / np.power(1.0 + st, alpha),
            )
            return [float(x) for x in weights.tolist()]

        return [
            max(min_weight, 1.0 / ((1.0 + float(item.sample_times)) ** alpha))
            for item in items
        ]

    def _weighted_sample_without_replacement(self, n: int) -> List[ReplayItem]:
        n = min(int(n), int(self._size))
        if n <= 0:
            return []

        active = self._active_physical_indices()
        if active.size == 0:
            return []

        if n >= active.size:
            chosen_physical = active
        else:
            st = self._sample_times[active].astype(np.float64)
            alpha = float(self.usage_penalty_alpha)
            min_weight = float(self.usage_weight_min)

            weights = np.maximum(
                min_weight,
                1.0 / np.power(1.0 + st, alpha),
            )

            total = float(np.sum(weights))
            if (not np.isfinite(total)) or total <= 0.0:
                logical = np.asarray(
                    self._rng.sample(range(int(self._size)), n),
                    dtype=np.int64,
                )
                chosen_physical = self._logical_to_physical_indices(logical)
            else:
                u = self._np_rng.random(active.size)
                u = np.maximum(u, np.finfo(np.float64).tiny)
                keys = -np.log(u) / weights
                chosen_pos = np.argpartition(keys, n - 1)[:n]
                chosen_physical = active[chosen_pos]

        items = self._items_from_physical_indices(chosen_physical)

        for idx, item in zip(chosen_physical, items):
            self._sample_times[int(idx)] += 1
            item.sample_times = int(self._sample_times[int(idx)])

        return items

    def clear(self) -> None:
        self._data = [None] * self.capacity
        self._sample_times.fill(0)
        self._insert_indices.fill(0)
        self._size = 0
        self._start = 0
        self.push_count = 0
        self.sample_count = 0

    @staticmethod
    def _percentile(values: List[int], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return float(ordered[0])
        pos = (len(ordered) - 1) * float(pct) / 100.0
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(ordered[lo])
        frac = pos - lo
        return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)

    def stats(self) -> Dict[str, Any]:
        active = self._active_physical_indices()
        sample_times_arr = self._sample_times[active] if active.size else np.asarray([], dtype=np.int64)

        n = int(sample_times_arr.size)
        mean = float(np.mean(sample_times_arr)) if n else 0.0
        std = float(np.std(sample_times_arr, ddof=0)) if n else 0.0
        zero_count = int(np.sum(sample_times_arr == 0)) if n else 0
        sample_times = sample_times_arr.astype(int).tolist()

        return {
            "capacity": int(self.capacity),
            "size": int(self._size),
            "push_count": int(self.push_count),
            "sample_count": int(self.sample_count),
            "sampling_mode": str(self.sampling_mode),
            "usage_penalty_alpha": float(self.usage_penalty_alpha),
            "usage_weight_min": float(self.usage_weight_min),
            "sample_times_mean": mean,
            "sample_times_std": std,
            "sample_times_min": int(np.min(sample_times_arr)) if n else 0,
            "sample_times_max": int(np.max(sample_times_arr)) if n else 0,
            "sample_times_p50": self._percentile(sample_times, 50),
            "sample_times_p90": self._percentile(sample_times, 90),
            "sample_times_p95": self._percentile(sample_times, 95),
            "sample_times_zero_count": zero_count,
            "sample_times_zero_ratio": float(zero_count / n) if n else 0.0,
        }
