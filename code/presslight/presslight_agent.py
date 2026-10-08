# -*- coding: utf-8 -*-
"""Grouped PressLight DQN agents, local replay buffers, and checkpoints."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from presslight_config import PressLightConfig
from presslight_network import PressLightNetworkSpec
from presslight_observation import PressLightObservation


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class PressLightMLP(nn.Module):
    """Compact PressLight MLP with group-specific input and action dimensions."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        action_dim: int,
    ) -> None:
        super().__init__()
        dims = [int(input_dim), *[int(x) for x in hidden_dims], int(action_dim)]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


@dataclass
class PressLightTransition:
    tl_id: str
    group_id: str
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_action_mask: np.ndarray
    done: bool


@dataclass
class PressLightActionInfo:
    tl_id: str
    group_id: str
    action: int
    epsilon: float
    is_random_action: bool
    is_greedy_action: bool
    q_values: list[float]
    q_masked_values: list[float]
    q_selected: float
    q_max: float
    q_margin: float
    valid_action_count: int
    action_mask: list[bool]
    local_action: int
    group_input_dim: int
    group_action_dim: int


@dataclass
class PressLightUpdateStats:
    group_id: str
    update_index: int
    replay_size: int
    batch_size: int
    q_loss: float
    td_error_mean: float
    td_error_abs_mean: float
    q_pred_mean: float
    q_target_mean: float
    reward_batch_mean: float
    grad_norm: float
    target_synced: bool = False


class LocalReplayBuffer:
    """Fixed-shape ring buffer for one structure-sharing group."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, seed: int,
                 group_id: str) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.group_id = str(group_id)
        if self.capacity <= 0:
            raise ValueError("Replay capacity must be > 0")
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.next_masks = np.zeros((self.capacity, self.action_dim), dtype=bool)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.size = 0
        self.position = 0
        self.push_count = 0
        self.sample_count = 0
        self.rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return int(self.size)

    def push(self, transition: PressLightTransition) -> None:
        if transition.group_id != self.group_id:
            raise ValueError(
                f"transition.group_id must equal replay group {self.group_id!r}; actual={transition.group_id!r}"
            )
        state = np.asarray(transition.state, dtype=np.float32)
        next_state = np.asarray(transition.next_state, dtype=np.float32)
        next_mask = np.asarray(transition.next_action_mask, dtype=bool)
        if state.shape != (self.state_dim,):
            raise ValueError(f"transition.state.shape must equal {(self.state_dim,)}; actual={state.shape}")
        if next_state.shape != (self.state_dim,):
            raise ValueError(f"transition.next_state.shape must equal {(self.state_dim,)}; actual={next_state.shape}")
        if next_mask.shape != (self.action_dim,):
            raise ValueError(f"transition.next_action_mask.shape must equal {(self.action_dim,)}; actual={next_mask.shape}")
        if not next_mask.any():
            raise ValueError("transition.next_action_mask must contain at least one valid action; actual=all False")
        if not 0 <= int(transition.action) < self.action_dim:
            raise ValueError(f"transition.action must be in [0, {self.action_dim}); actual={transition.action!r}")
        slot = int(self.position)
        self.states[slot] = state
        self.actions[slot] = int(transition.action)
        self.rewards[slot] = float(transition.reward)
        self.next_states[slot] = next_state
        self.next_masks[slot] = next_mask
        self.dones[slot] = 1.0 if transition.done else 0.0
        self.position = (slot + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.push_count += 1

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if self.size <= 0:
            raise RuntimeError("Cannot sample an empty replay buffer")
        n = min(int(batch_size), int(self.size))
        indices = self.rng.choice(self.size, size=n, replace=False)
        self.sample_count += 1
        return {
            "states": self.states[indices].copy(),
            "actions": self.actions[indices].copy(),
            "rewards": self.rewards[indices].copy(),
            "next_states": self.next_states[indices].copy(),
            "next_masks": self.next_masks[indices].copy(),
            "dones": self.dones[indices].copy(),
        }

    def state_dict(self) -> dict[str, Any]:
        n = int(self.size)
        return {
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "group_id": self.group_id,
            "size": self.size,
            "position": self.position,
            "push_count": self.push_count,
            "sample_count": self.sample_count,
            "states": self.states[:n].copy(),
            "actions": self.actions[:n].copy(),
            "rewards": self.rewards[:n].copy(),
            "next_states": self.next_states[:n].copy(),
            "next_masks": self.next_masks[:n].copy(),
            "dones": self.dones[:n].copy(),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if int(payload.get("state_dim", -1)) != self.state_dim:
            raise ValueError("Replay state_dim mismatch")
        if int(payload.get("action_dim", -1)) != self.action_dim:
            raise ValueError("Replay action_dim mismatch")
        if str(payload.get("group_id", self.group_id)) != self.group_id:
            raise ValueError("Replay group_id mismatch")
        n = min(int(payload.get("size", 0)), self.capacity)
        self.states[:n] = np.asarray(payload["states"], dtype=np.float32)[:n]
        self.actions[:n] = np.asarray(payload["actions"], dtype=np.int64)[:n]
        self.rewards[:n] = np.asarray(payload["rewards"], dtype=np.float32)[:n]
        self.next_states[:n] = np.asarray(payload["next_states"], dtype=np.float32)[:n]
        self.next_masks[:n] = np.asarray(payload["next_masks"], dtype=bool)[:n]
        self.dones[:n] = np.asarray(payload["dones"], dtype=np.float32)[:n]
        self.size = n
        self.position = int(payload.get("position", n % self.capacity)) % self.capacity
        self.push_count = int(payload.get("push_count", n))
        self.sample_count = int(payload.get("sample_count", 0))
        if payload.get("rng_state") is not None:
            self.rng.bit_generator.state = payload["rng_state"]


class PressLightAgentManager:
    def __init__(
        self,
        cfg: PressLightConfig,
        network_spec: PressLightNetworkSpec,
        device: Optional[str | torch.device] = None,
    ) -> None:
        self.cfg = cfg
        self.network_spec = network_spec
        set_global_seed(cfg.seed)
        self.device = self._resolve_device(device or cfg.device)
        self.group_input_dims = {
            group_id: int(spec.input_dim)
            for group_id, spec in network_spec.group_specs.items()
        }
        self.group_action_dims = {
            group_id: int(spec.action_dim)
            for group_id, spec in network_spec.group_specs.items()
        }

        self.online_networks: dict[str, PressLightMLP] = {}
        self.target_networks: dict[str, PressLightMLP] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.replay_buffers: dict[str, LocalReplayBuffer] = {}
        for index, group_id in enumerate(network_spec.group_ids):
            input_dim = self.group_input_dims[group_id]
            action_dim = self.group_action_dims[group_id]
            online = PressLightMLP(
                input_dim, tuple(cfg.dqn.hidden_dims), action_dim
            ).to(self.device)
            target = PressLightMLP(
                input_dim, tuple(cfg.dqn.hidden_dims), action_dim
            ).to(self.device)
            target.load_state_dict(online.state_dict())
            target.eval()
            self.online_networks[group_id] = online
            self.target_networks[group_id] = target
            self.optimizers[group_id] = torch.optim.Adam(
                online.parameters(), lr=float(cfg.dqn.learning_rate)
            )
            self.replay_buffers[group_id] = LocalReplayBuffer(
                capacity=cfg.dqn.replay_memory_size,
                state_dim=input_dim,
                action_dim=action_dim,
                seed=int(cfg.seed) + index * 1009,
                group_id=group_id,
            )

        self.global_decision_step = 0
        self.update_count = 0
        self.target_update_count = 0
        self.last_target_sync_step = 0
        self._rng = random.Random(int(cfg.seed) + 17)

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if isinstance(device, torch.device):
            return device
        value = str(device).strip().lower()
        if value in {"", "auto"}:
            value = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(value)

    @property
    def epsilon(self) -> float:
        return self.cfg.dqn.epsilon_at(self.global_decision_step)

    def model_meta(self) -> dict[str, Any]:
        return {
            "algorithm": self.cfg.dqn.algorithm,
            "variant": self.cfg.pressure.variant,
            "groups": self.network_spec.group_ids,
            "group_architectures": self.group_architectures(),
            "network_spec_digest": self.network_spec.digest,
        }

    def group_architectures(self) -> dict[str, dict[str, Any]]:
        return {
            group_id: {
                "input_dim": int(spec.input_dim),
                "hidden_dims": [int(x) for x in self.cfg.dqn.hidden_dims],
                "action_dim": int(spec.action_dim),
                "tl_ids": list(spec.tl_ids),
                "structure_signature": spec.structure_signature,
            }
            for group_id, spec in sorted(self.network_spec.group_specs.items())
        }

    def set_training(self, training: bool) -> None:
        for network in self.online_networks.values():
            network.train(bool(training))
        for network in self.target_networks.values():
            network.eval()

    def act(
        self,
        observations: dict[str, PressLightObservation],
        epsilon: Optional[float] = None,
        deterministic: bool = False,
    ) -> tuple[dict[str, int], dict[str, PressLightActionInfo]]:
        eps = 0.0 if deterministic else float(self.epsilon if epsilon is None else epsilon)
        actions: dict[str, int] = {}
        infos: dict[str, PressLightActionInfo] = {}
        by_group: dict[str, list[PressLightObservation]] = {}
        for obs in observations.values():
            by_group.setdefault(obs.group_id, []).append(obs)

        for group_id, group_observations in by_group.items():
            input_dim = self.group_input_dims[group_id]
            action_dim = self.group_action_dims[group_id]
            for obs in group_observations:
                if np.asarray(obs.state).shape != (input_dim,):
                    raise RuntimeError(f"{obs.tl_id}: state.shape must equal {(input_dim,)}; actual={np.asarray(obs.state).shape}")
            state_batch = torch.as_tensor(
                np.stack([obs.state for obs in group_observations]),
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                q_batch = self.online_networks[group_id](state_batch).cpu().numpy()
            for obs, q_values in zip(group_observations, q_batch):
                mask = np.asarray(obs.action_mask, dtype=bool)
                if mask.shape != (action_dim,):
                    raise RuntimeError(
                        f"{obs.tl_id}: action_mask.shape must equal {(action_dim,)}; actual={mask.shape}"
                    )
                if np.asarray(q_values).shape != (action_dim,):
                    raise RuntimeError(
                        f"{obs.tl_id}: q_values.shape must equal {(action_dim,)}; actual={np.asarray(q_values).shape}"
                    )
                valid_actions = np.flatnonzero(mask)
                if valid_actions.size == 0:
                    raise RuntimeError(f"{obs.tl_id}: empty action mask")
                masked = np.where(mask, q_values, -1.0e9)
                random_action = bool(not deterministic and self._rng.random() < eps)
                if random_action:
                    action = int(self._rng.choice(valid_actions.tolist()))
                else:
                    action = int(np.argmax(masked))
                valid_q = np.asarray(q_values[valid_actions], dtype=np.float64)
                if valid_q.size >= 2:
                    top_two = np.partition(valid_q, -2)[-2:]
                    margin = float(top_two.max() - top_two.min())
                else:
                    margin = 0.0
                actions[obs.tl_id] = action
                infos[obs.tl_id] = PressLightActionInfo(
                    tl_id=obs.tl_id,
                    group_id=group_id,
                    action=action,
                    epsilon=eps,
                    is_random_action=random_action,
                    is_greedy_action=not random_action,
                    q_values=[float(x) for x in q_values],
                    q_masked_values=[float(x) for x in masked],
                    q_selected=float(q_values[action]),
                    q_max=float(masked.max()),
                    q_margin=margin,
                    valid_action_count=int(valid_actions.size),
                    action_mask=[bool(x) for x in mask],
                    local_action=int(self.network_spec.intersections[obs.tl_id].canonical_to_local_phase[action]),
                    group_input_dim=input_dim,
                    group_action_dim=action_dim,
                )
        return actions, infos

    def store_transition(self, transition: PressLightTransition) -> None:
        if transition.group_id not in self.replay_buffers:
            raise KeyError(f"Unknown PressLight group: {transition.group_id}")
        self.replay_buffers[transition.group_id].push(transition)

    def replay_sizes(self) -> dict[str, int]:
        return {group_id: len(buffer) for group_id, buffer in self.replay_buffers.items()}

    def can_update(self, group_id: str) -> bool:
        return len(self.replay_buffers[group_id]) >= int(self.cfg.dqn.min_replay_size)

    def _update_group(self, group_id: str) -> PressLightUpdateStats:
        buffer = self.replay_buffers[group_id]
        batch = buffer.sample(self.cfg.dqn.batch_size)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch["next_masks"], dtype=torch.bool, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        online = self.online_networks[group_id]
        target = self.target_networks[group_id]
        optimizer = self.optimizers[group_id]
        q_pred = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            target_next_all = target(next_states).masked_fill(~next_masks, -1.0e9)
            if self.cfg.dqn.algorithm == "ddqn":
                online_next = online(next_states).masked_fill(~next_masks, -1.0e9)
                next_actions = online_next.argmax(dim=1, keepdim=True)
                q_next = target_next_all.gather(1, next_actions).squeeze(1)
            else:
                q_next = target_next_all.max(dim=1).values
            q_target = rewards + float(self.cfg.dqn.gamma) * (1.0 - dones) * q_next

        if self.cfg.dqn.loss_type == "mse":
            loss = F.mse_loss(q_pred, q_target)
        else:
            loss = F.smooth_l1_loss(q_pred, q_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            online.parameters(), float(self.cfg.dqn.max_grad_norm)
        )
        optimizer.step()
        self.update_count += 1
        td_error = q_target.detach() - q_pred.detach()
        return PressLightUpdateStats(
            group_id=group_id,
            update_index=self.update_count,
            replay_size=len(buffer),
            batch_size=int(states.shape[0]),
            q_loss=float(loss.detach().cpu()),
            td_error_mean=float(td_error.mean().cpu()),
            td_error_abs_mean=float(td_error.abs().mean().cpu()),
            q_pred_mean=float(q_pred.detach().mean().cpu()),
            q_target_mean=float(q_target.detach().mean().cpu()),
            reward_batch_mean=float(rewards.detach().mean().cpu()),
            grad_norm=float(grad_norm_tensor.detach().cpu()),
        )

    def update_from_replay(self) -> list[PressLightUpdateStats]:
        if self.global_decision_step % int(self.cfg.dqn.online_update_interval) != 0:
            return []
        stats = [
            self._update_group(group_id)
            for group_id in self.network_spec.group_ids
            if self.can_update(group_id)
        ]
        synced = self.maybe_sync_target()
        if synced:
            for item in stats:
                item.target_synced = True
        return stats

    def sync_target(self) -> None:
        for group_id in self.network_spec.group_ids:
            self.target_networks[group_id].load_state_dict(
                self.online_networks[group_id].state_dict()
            )
            self.target_networks[group_id].eval()
        self.target_update_count += 1
        self.last_target_sync_step = int(self.global_decision_step)

    def maybe_sync_target(self) -> bool:
        interval = int(self.cfg.dqn.target_update_interval)
        if interval <= 0:
            return False
        if self.global_decision_step - self.last_target_sync_step < interval:
            return False
        self.sync_target()
        return True

    def increment_step(self) -> None:
        self.global_decision_step += 1

    def checkpoint_payload(
        self,
        episode: int,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": 2,
            "checkpoint_kind": "training",
            "episode": int(episode),
            "global_decision_step": int(self.global_decision_step),
            "update_count": int(self.update_count),
            "target_update_count": int(self.target_update_count),
            "last_target_sync_step": int(self.last_target_sync_step),
            "online_networks": {
                group_id: network.state_dict()
                for group_id, network in self.online_networks.items()
            },
            "target_networks": {
                group_id: network.state_dict()
                for group_id, network in self.target_networks.items()
            },
            "optimizers": {
                group_id: optimizer.state_dict()
                for group_id, optimizer in self.optimizers.items()
            },
            "resolved_config": self.cfg.to_dict(),
            "network_spec_digest": self.network_spec.digest,
            "group_meta": self.model_meta(),
            "group_architectures": self.group_architectures(),
            "group_to_tls": self.network_spec.group_to_tls,
            "intersection_mappings": self._intersection_mappings(),
            "rng_state": {
                "python_global": random.getstate(),
                "manager": self._rng.getstate(),
                "numpy_global": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            "extra": extra or {},
        }
        if self.cfg.dqn.checkpoint_include_replay:
            payload["replay_buffers"] = {
                group_id: buffer.state_dict()
                for group_id, buffer in self.replay_buffers.items()
            }
        return payload

    def save_checkpoint(
        self,
        path: str,
        episode: int,
        extra: Optional[dict[str, Any]] = None,
    ) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.checkpoint_payload(episode, extra), path)
        return path

    def evaluation_checkpoint_payload(self, episode: int) -> dict[str, Any]:
        """Return a weights-only-safe checkpoint for inference/evaluation."""

        return {
            "format_version": 2,
            "checkpoint_kind": "evaluation",
            "episode": int(episode),
            "online_networks": {
                group_id: network.state_dict()
                for group_id, network in self.online_networks.items()
            },
            "group_meta": self.model_meta(),
            "group_architectures": self.group_architectures(),
            "group_to_tls": self.network_spec.group_to_tls,
            "intersection_mappings": self._intersection_mappings(),
            "network_spec_digest": self.network_spec.digest,
            "config_meta": {
                "seed": int(self.cfg.seed),
                "algorithm": str(self.cfg.dqn.algorithm),
                "variant": str(self.cfg.pressure.variant),
            },
        }

    def _intersection_mappings(self) -> dict[str, dict[str, Any]]:
        return {
            tl_id: {
                "group_id": spec.group_id,
                "canonical_incoming_lanes": list(spec.canonical_incoming_lanes),
                "canonical_outgoing_lanes": list(spec.canonical_outgoing_lanes),
                "canonical_to_local_phase": list(spec.canonical_to_local_phase),
                "local_to_canonical_phase": list(spec.local_to_canonical_phase),
            }
            for tl_id, spec in sorted(self.network_spec.intersections.items())
        }

    def save_evaluation_checkpoint(self, path: str, episode: int) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.evaluation_checkpoint_payload(episode), path)
        return path

    def _validate_checkpoint(self, payload: dict[str, Any], path: str) -> None:
        if int(payload.get("format_version", 0)) != 2:
            raise ValueError(
                "legacy fixed-dimension checkpoint is incompatible; retraining is required"
            )
        meta = payload.get("group_meta", {})
        expected = self.model_meta()
        for key in ("algorithm", "variant", "groups"):
            if meta.get(key) != expected.get(key):
                raise ValueError(
                    f"Checkpoint compatibility error for {key}: "
                    f"checkpoint={meta.get(key)!r}, current={expected.get(key)!r}; path={path}"
                )
        if payload.get("group_architectures", meta.get("group_architectures")) != self.group_architectures():
            raise ValueError(
                "Checkpoint group architectures/signatures differ from the current structural groups"
            )
        digest = str(payload.get("network_spec_digest", ""))
        if digest and digest != self.network_spec.digest:
            # Accept relocation only when the original paths reproduce the full digest.
            saved_paths = payload.get("resolved_config", {}).get("paths", {})
            source_net = saved_paths.get("net_xml")
            source_detectors = saved_paths.get("detector_map_json")
            if (
                isinstance(source_net, str) and source_net
                and isinstance(source_detectors, str) and source_detectors
                and payload.get("intersection_mappings") == self._intersection_mappings()
            ):
                summary = self.network_spec.summary()
                summary.pop("digest", None)
                summary["source_net_xml"] = source_net
                summary["source_detector_map"] = source_detectors
                relocated_digest = hashlib.sha256(
                    json.dumps(summary, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest()
                if relocated_digest == digest:
                    return
            raise ValueError(
                "Checkpoint network_spec_digest differs from the current lane/phase mapping"
            )

    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = False,
        load_replay: bool = False,
        restore_rng: bool = False,
    ) -> dict[str, Any]:
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:  # older PyTorch
            payload = torch.load(path, map_location=self.device)
        self._validate_checkpoint(payload, path)
        for group_id in self.network_spec.group_ids:
            self.online_networks[group_id].load_state_dict(payload["online_networks"][group_id])
            self.target_networks[group_id].load_state_dict(
                payload.get("target_networks", payload["online_networks"])[group_id]
            )
            if load_optimizer and group_id in payload.get("optimizers", {}):
                self.optimizers[group_id].load_state_dict(payload["optimizers"][group_id])
            if load_replay and group_id in payload.get("replay_buffers", {}):
                self.replay_buffers[group_id].load_state_dict(
                    payload["replay_buffers"][group_id]
                )
        self.global_decision_step = int(payload.get("global_decision_step", 0))
        self.update_count = int(payload.get("update_count", 0))
        self.target_update_count = int(payload.get("target_update_count", 0))
        self.last_target_sync_step = int(payload.get("last_target_sync_step", 0))

        if restore_rng:
            rng = payload.get("rng_state", {})
            if rng.get("python_global") is not None:
                random.setstate(rng["python_global"])
            if rng.get("manager") is not None:
                self._rng.setstate(rng["manager"])
            if rng.get("numpy_global") is not None:
                np.random.set_state(rng["numpy_global"])
            if rng.get("torch_cpu") is not None:
                torch.set_rng_state(rng["torch_cpu"])
            if torch.cuda.is_available() and rng.get("torch_cuda"):
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
        return payload

    def load_evaluation_checkpoint(self, path: str) -> dict[str, Any]:
        """Load an evaluation checkpoint without enabling pickle deserialization."""

        try:
            payload = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "This PyTorch version does not support safe weights_only loading. "
                "Upgrade PyTorch, or explicitly use trusted-checkpoint mode for a "
                "checkpoint generated by this project from a trusted source."
            ) from exc
        except (pickle.UnpicklingError, RuntimeError) as exc:
            raise ValueError(
                "Safe weights-only checkpoint loading failed. The file may be a full/legacy "
                "training checkpoint or may contain unsupported objects. No unsafe fallback "
                "was attempted; use --trusted-checkpoint only for a trusted project checkpoint."
            ) from exc
        if not isinstance(payload, dict) or payload.get("checkpoint_kind") != "evaluation":
            raise ValueError(
                "Expected a safe evaluation checkpoint with checkpoint_kind='evaluation'. "
                "Use --trusted-checkpoint only for a trusted legacy/full training checkpoint."
            )
        self._validate_checkpoint(payload, path)
        online_payload = payload.get("online_networks", {})
        for group_id in self.network_spec.group_ids:
            if group_id not in online_payload:
                raise ValueError(f"Evaluation checkpoint is missing online_networks[{group_id!r}]")
            self.online_networks[group_id].load_state_dict(online_payload[group_id])
            self.target_networks[group_id].load_state_dict(online_payload[group_id])
            self.target_networks[group_id].eval()
        return payload


def q_step_rows(
    episode: int,
    step: int,
    infos: dict[str, PressLightActionInfo],
) -> list[dict[str, Any]]:
    return [
        {
            "episode": int(episode),
            "step": int(step),
            "tl_id": info.tl_id,
            "group_id": info.group_id,
            "selected_action": int(info.action),
            "canonical_action": int(info.action),
            "local_action": int(info.local_action),
            "group_input_dim": int(info.group_input_dim),
            "group_action_dim": int(info.group_action_dim),
            "epsilon": float(info.epsilon),
            "is_random_action": bool(info.is_random_action),
            "is_greedy_action": bool(info.is_greedy_action),
            "q_values": "|".join(f"{x:.8g}" for x in info.q_values),
            "q_masked_values": "|".join(f"{x:.8g}" for x in info.q_masked_values),
            "q_selected": float(info.q_selected),
            "q_max": float(info.q_max),
            "q_margin": float(info.q_margin),
            "valid_action_count": int(info.valid_action_count),
            "action_mask": "|".join("1" if x else "0" for x in info.action_mask),
        }
        for info in infos.values()
    ]
