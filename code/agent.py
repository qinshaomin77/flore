# -*- coding: utf-8 -*-
"""
agent.py
========
MGMQ-DDQN 智能体管理器。

职责
----
1. 构建 online / target GroupedQNetworkBank。
2. epsilon-greedy + action mask 选择动作。
3. 写入 replay memory。
4. 从 buffer 采样，按 group 计算 DDQN loss 并更新 online network。
5. 定期同步 target network，保存/加载 checkpoint。
"""

from __future__ import annotations

import json
import math
import os
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from buffer import GlobalTransition, ReplayMemory
from config import MasterConfig
from model import GroupedQNetworkBank, masked_q_values
from network_parser import NetworkInfo
from obs_reward import AgentObservation


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


@dataclass
class ActionInfo:
    tl_id: str
    group_id: str
    action: int
    epsilon: float
    is_random_action: bool
    is_greedy_action: bool
    q_values: List[float]
    q_masked_values: List[float]
    q_max: float
    q_selected: float
    q_margin: float
    valid_action_count: int
    action_mask: List[bool]
    q_total_values: List[float] = field(default_factory=list)
    q_traffic_values: List[float] = field(default_factory=list)
    q_nox_values: List[float] = field(default_factory=list)
    q_total_selected: float = 0.0
    q_traffic_selected: float = 0.0
    q_nox_selected: float = 0.0


@dataclass
class UpdateStats:
    group_id: str
    update_index: int
    batch_size: int
    replay_size: int
    q_loss: float
    td_error_mean: float
    td_error_abs_mean: float
    q_pred_mean: float
    q_target_mean: float
    reward_batch_mean: float
    grad_norm: float
    learning_rate: float
    traffic_loss: float = 0.0
    nox_loss: float = 0.0
    td_traffic_mean: float = 0.0
    td_nox_mean: float = 0.0
    q_traffic_pred_mean: float = 0.0
    q_nox_pred_mean: float = 0.0
    target_traffic_mean: float = 0.0
    target_nox_mean: float = 0.0
    traffic_reward_batch_mean: float = 0.0
    nox_penalty_batch_mean: float = 0.0
    target_synced: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class MGMQAgentManager:
    """DDQN agent manager。"""

    def __init__(self, cfg: MasterConfig, net_info: NetworkInfo, device: Optional[str | torch.device] = None) -> None:
        self.cfg = cfg
        self.net_info = net_info
        set_global_seed(int(cfg.seed))
        self.device = torch.device(device or cfg.resolve_device())
        self.online_bank = GroupedQNetworkBank.from_network_info(net_info, cfg).to_device(self.device)
        self.target_bank = GroupedQNetworkBank.from_network_info(net_info, cfg).to_device(self.device)
        self.global_step = 0
        self.update_count = 0
        self.target_update_count = 0
        self.sync_target()
        self.target_bank.eval()

        self.memory = ReplayMemory(
            capacity=int(cfg.ddqn.replay_memory_size),
            seed=int(cfg.seed),
            sampling_mode=str(cfg.ddqn.replay_sampling_mode),
            usage_penalty_alpha=float(cfg.ddqn.usage_penalty_alpha),
            usage_weight_min=float(cfg.ddqn.usage_weight_min),
        )
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        for group_id, net in self.online_bank.networks.items():
            self.optimizers[group_id] = torch.optim.Adam(
                net.parameters(),
                lr=float(cfg.ddqn.learning_rate),
                betas=(float(cfg.ddqn.adam_beta1), float(cfg.ddqn.adam_beta2)),
                eps=float(cfg.ddqn.adam_eps),
            )

        self._rng = random.Random(int(cfg.seed))

    # ───────────────────────────────────────────────────────────────
    # 动作选择
    # ───────────────────────────────────────────────────────────────

    def epsilon(self) -> float:
        return float(self.cfg.ddqn.epsilon_at(self.global_step))

    @torch.no_grad()
    def act(
        self,
        observations: Mapping[str, AgentObservation],
        epsilon: Optional[float] = None,
        deterministic: bool = False,
        record_q_values: bool = True,
    ) -> Tuple[Dict[str, int], Dict[str, ActionInfo]]:
        self.online_bank.eval()
        eps = 0.0 if deterministic else float(self.epsilon() if epsilon is None else epsilon)
        use_dual = bool(getattr(self.cfg.dual_head, "enabled", False))
        q_by_tl, _ = self.online_bank.forward_global(
            observations,
            self.net_info,
            self.device,
            return_components=use_dual,
        )
        actions: Dict[str, int] = {}
        infos: Dict[str, ActionInfo] = {}

        for tl_id, obs in observations.items():
            if use_dual:
                q_bundle = q_by_tl[tl_id]
                q_total = q_bundle.q_total.detach()
                q_traffic = q_bundle.q_traffic.detach()
                q_nox = q_bundle.q_nox.detach()
            else:
                q_total = q_by_tl[tl_id].detach()
                q_traffic = q_total
                q_nox = torch.zeros_like(q_total)
            mask = torch.as_tensor(obs.action_mask, dtype=torch.bool, device=self.device)
            mq = masked_q_values(q_total, mask)
            valid_indices = torch.nonzero(mask, as_tuple=False).view(-1).detach().cpu().numpy().astype(int).tolist()
            if not valid_indices:
                valid_indices = [int(obs.current_phase)]
            use_random = (not deterministic) and (self._rng.random() < eps)
            if use_random:
                action = int(self._rng.choice(valid_indices))
            else:
                action = int(torch.argmax(mq).item())
            actions[tl_id] = action

            sorted_vals = torch.sort(mq[mask], descending=True).values if torch.any(mask) else torch.tensor([mq[action]], device=self.device)
            q_max = float(sorted_vals[0].item()) if sorted_vals.numel() else float(mq[action].item())
            q_second = float(sorted_vals[1].item()) if sorted_vals.numel() >= 2 else q_max
            q_values = [float(x) for x in q_total.detach().cpu().tolist()] if record_q_values else []
            q_masked_values = [float(x) for x in mq.detach().cpu().tolist()] if record_q_values else []
            q_traffic_values = [float(x) for x in q_traffic.detach().cpu().tolist()] if record_q_values else []
            q_nox_values = [float(x) for x in q_nox.detach().cpu().tolist()] if record_q_values else []
            q_total_selected = float(q_total[action].item())
            q_traffic_selected = float(q_traffic[action].item())
            q_nox_selected = float(q_nox[action].item())
            infos[tl_id] = ActionInfo(
                tl_id=tl_id,
                group_id=obs.group_id,
                action=action,
                epsilon=eps,
                is_random_action=bool(use_random),
                is_greedy_action=not bool(use_random),
                q_values=q_values,
                q_masked_values=q_masked_values,
                q_max=q_max,
                q_selected=q_total_selected,
                q_margin=float(q_max - q_second),
                valid_action_count=int(len(valid_indices)),
                action_mask=[bool(x) for x in obs.action_mask.tolist()],
                q_total_values=q_values,
                q_traffic_values=q_traffic_values,
                q_nox_values=q_nox_values,
                q_total_selected=q_total_selected,
                q_traffic_selected=q_traffic_selected,
                q_nox_selected=q_nox_selected,
            )
        self.online_bank.train()
        return actions, infos

    # ───────────────────────────────────────────────────────────────
    # Replay memory
    # ───────────────────────────────────────────────────────────────

    def store_transition(self, transition: GlobalTransition) -> None:
        self.memory.push(transition)

    def can_update(self) -> bool:
        return len(self.memory) >= int(self.cfg.ddqn.min_replay_size)

    # ───────────────────────────────────────────────────────────────
    # DDQN update
    # ───────────────────────────────────────────────────────────────

    def update_from_replay(self) -> List[UpdateStats]:
        if not self.can_update():
            return []
        batch = self.memory.sample(int(self.cfg.ddqn.batch_size))
        if not batch:
            return []

        if bool(getattr(self.cfg.dual_head, "enabled", False)):
            return self._update_from_replay_dual_head(batch)

        return self._update_from_replay_single_head(batch)

    def _update_from_replay_single_head(self, batch: List[GlobalTransition]) -> List[UpdateStats]:
        obs_batch = [tr.obs for tr in batch]
        next_obs_batch = [tr.next_obs for tr in batch]
        q_online_batch = self.online_bank.forward_global_batch(obs_batch, self.net_info, self.device)
        with torch.no_grad():
            q_next_online_batch = self.online_bank.forward_global_batch(next_obs_batch, self.net_info, self.device)
            q_next_target_batch = self.target_bank.forward_global_batch(next_obs_batch, self.net_info, self.device)

        group_loss_vecs: Dict[str, List[torch.Tensor]] = {gid: [] for gid in self.online_bank.networks.keys()}
        group_pred_vecs: Dict[str, List[torch.Tensor]] = {gid: [] for gid in self.online_bank.networks.keys()}
        group_target_vecs: Dict[str, List[torch.Tensor]] = {gid: [] for gid in self.online_bank.networks.keys()}
        group_reward_vecs: Dict[str, List[torch.Tensor]] = {gid: [] for gid in self.online_bank.networks.keys()}

        for tl_id in q_online_batch.keys():
            gid = batch[0].obs[tl_id].group_id
            actions_vec = torch.as_tensor(
                [int(tr.actions[tl_id]) for tr in batch],
                dtype=torch.long,
                device=self.device,
            )
            rewards_vec = torch.as_tensor(
                [float(tr.rewards[tl_id]) for tr in batch],
                dtype=torch.float32,
                device=self.device,
            )
            next_mask = torch.stack([
                torch.as_tensor(tr.next_obs[tl_id].action_mask, dtype=torch.bool, device=self.device)
                for tr in batch
            ], dim=0)
            done_vec = torch.as_tensor(
                [float(tr.done) for tr in batch],
                dtype=torch.float32,
                device=self.device,
            )

            pred_vec = q_online_batch[tl_id].gather(1, actions_vec[:, None]).squeeze(1)
            next_masked = masked_q_values(q_next_online_batch[tl_id], next_mask)
            next_action_vec = next_masked.argmax(dim=1)
            target_next_vec = q_next_target_batch[tl_id].gather(1, next_action_vec[:, None]).squeeze(1)
            target_vec = rewards_vec + (1.0 - done_vec) * float(self.cfg.ddqn.gamma) * target_next_vec

            if self.cfg.ddqn.loss_type == "mse":
                loss_vec = F.mse_loss(pred_vec, target_vec.detach(), reduction="none")
            else:
                loss_vec = F.smooth_l1_loss(pred_vec, target_vec.detach(), reduction="none")

            group_loss_vecs[gid].append(loss_vec)
            group_pred_vecs[gid].append(pred_vec.detach())
            group_target_vecs[gid].append(target_vec.detach())
            group_reward_vecs[gid].append(rewards_vec.detach())

        valid_loss_vecs = [vec for vecs in group_loss_vecs.values() for vec in vecs]
        if not valid_loss_vecs:
            return []
        all_loss_flat = torch.cat(valid_loss_vecs)

        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=True)
        total_loss = all_loss_flat.mean()
        total_loss.backward()

        grad_norm_by_group: Dict[str, float] = {}
        for gid, net in self.online_bank.networks.items():
            grad_norm = torch.nn.utils.clip_grad_norm_(
                net.parameters(),
                max_norm=float(self.cfg.ddqn.max_grad_norm),
            )
            grad_norm_by_group[gid] = float(
                grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
        for opt in self.optimizers.values():
            opt.step()

        stats: List[UpdateStats] = []
        for gid, loss_vecs in group_loss_vecs.items():
            if not loss_vecs:
                continue
            loss_flat = torch.cat(loss_vecs)
            pred_arr = torch.cat(group_pred_vecs[gid]).detach().cpu().numpy()
            target_arr = torch.cat(group_target_vecs[gid]).detach().cpu().numpy()
            reward_arr = torch.cat(group_reward_vecs[gid]).detach().cpu().numpy()
            td_arr = target_arr - pred_arr
            self.update_count += 1
            stats.append(UpdateStats(
                group_id=gid,
                update_index=int(self.update_count),
                batch_size=int(loss_flat.numel()),
                replay_size=int(len(self.memory)),
                q_loss=float(loss_flat.detach().mean().cpu().item()),
                td_error_mean=float(np.mean(td_arr)) if td_arr.size else 0.0,
                td_error_abs_mean=float(np.mean(np.abs(td_arr))) if td_arr.size else 0.0,
                q_pred_mean=float(np.mean(pred_arr)) if pred_arr.size else 0.0,
                q_target_mean=float(np.mean(target_arr)) if target_arr.size else 0.0,
                reward_batch_mean=float(np.mean(reward_arr)) if reward_arr.size else 0.0,
                grad_norm=float(grad_norm_by_group.get(gid, 0.0)),
                learning_rate=float(self.cfg.ddqn.learning_rate),
                target_synced=False,
            ))
        return stats

    def _update_from_replay_dual_head(self, batch: List[GlobalTransition]) -> List[UpdateStats]:
        obs_batch = [tr.obs for tr in batch]
        next_obs_batch = [tr.next_obs for tr in batch]
        q_online_batch = self.online_bank.forward_global_batch(
            obs_batch,
            self.net_info,
            self.device,
            return_components=True,
        )
        with torch.no_grad():
            q_next_online_batch = self.online_bank.forward_global_batch(
                next_obs_batch,
                self.net_info,
                self.device,
                return_components=True,
            )
            q_next_target_batch = self.target_bank.forward_global_batch(
                next_obs_batch,
                self.net_info,
                self.device,
                return_components=True,
            )

        group_loss_vecs: Dict[str, List[torch.Tensor]] = {gid: [] for gid in self.online_bank.networks.keys()}
        group_traffic_loss_vecs: Dict[str, List[torch.Tensor]] = {gid: [] for gid in self.online_bank.networks.keys()}
        group_nox_loss_vecs: Dict[str, List[torch.Tensor]] = {gid: [] for gid in self.online_bank.networks.keys()}
        group_pred_total_vecs: Dict[str, List[torch.Tensor]] = {gid: [] for gid in self.online_bank.networks.keys()}
        group_target_total_vecs: Dict[str, List[torch.Tensor]] = {gid: [] for gid in self.online_bank.networks.keys()}
        group_pred_traffic_vecs: Dict[str, List[torch.Tensor]] = {gid: [] for gid in self.online_bank.networks.keys()}
        group_pred_nox_vecs: Dict[str, List[torch.Tensor]] = {gid: [] for gid in self.online_bank.networks.keys()}
        group_target_traffic_vecs: Dict[str, List[torch.Tensor]] = {gid: [] for gid in self.online_bank.networks.keys()}
        group_target_nox_vecs: Dict[str, List[torch.Tensor]] = {gid: [] for gid in self.online_bank.networks.keys()}
        group_reward_vecs: Dict[str, List[torch.Tensor]] = {gid: [] for gid in self.online_bank.networks.keys()}
        group_traffic_reward_vecs: Dict[str, List[torch.Tensor]] = {gid: [] for gid in self.online_bank.networks.keys()}
        group_nox_penalty_vecs: Dict[str, List[torch.Tensor]] = {gid: [] for gid in self.online_bank.networks.keys()}
        group_td_traffic_vecs: Dict[str, List[torch.Tensor]] = {gid: [] for gid in self.online_bank.networks.keys()}
        group_td_nox_vecs: Dict[str, List[torch.Tensor]] = {gid: [] for gid in self.online_bank.networks.keys()}

        gamma = float(self.cfg.ddqn.gamma)
        lambda_e = float(self.cfg.emission_risk.lambda_e)
        beta = float(self.cfg.dual_head.nox_loss_beta)

        for tl_id in q_online_batch.keys():
            gid = batch[0].obs[tl_id].group_id
            actions_vec = torch.as_tensor(
                [int(tr.actions[tl_id]) for tr in batch],
                dtype=torch.long,
                device=self.device,
            )
            traffic_rewards_vec = torch.as_tensor(
                [float(tr.traffic_rewards.get(tl_id, tr.rewards[tl_id])) for tr in batch],
                dtype=torch.float32,
                device=self.device,
            )
            nox_penalties_vec = torch.as_tensor(
                [float(tr.nox_penalties.get(tl_id, 0.0)) for tr in batch],
                dtype=torch.float32,
                device=self.device,
            )
            final_rewards_vec = torch.as_tensor(
                [float(tr.rewards[tl_id]) for tr in batch],
                dtype=torch.float32,
                device=self.device,
            )
            done_vec = torch.as_tensor(
                [float(tr.done) for tr in batch],
                dtype=torch.float32,
                device=self.device,
            )
            next_mask = torch.stack([
                torch.as_tensor(tr.next_obs[tl_id].action_mask, dtype=torch.bool, device=self.device)
                for tr in batch
            ], dim=0)

            q_bundle = q_online_batch[tl_id]
            pred_traffic = q_bundle.q_traffic.gather(1, actions_vec[:, None]).squeeze(1)
            pred_nox = q_bundle.q_nox.gather(1, actions_vec[:, None]).squeeze(1)
            pred_total = q_bundle.q_total.gather(1, actions_vec[:, None]).squeeze(1)

            next_total_online = q_next_online_batch[tl_id].q_total
            next_masked = masked_q_values(next_total_online, next_mask)
            next_action_vec = next_masked.argmax(dim=1)

            target_next_traffic = q_next_target_batch[tl_id].q_traffic.gather(
                1,
                next_action_vec[:, None],
            ).squeeze(1)
            target_next_nox = q_next_target_batch[tl_id].q_nox.gather(
                1,
                next_action_vec[:, None],
            ).squeeze(1)

            target_traffic = traffic_rewards_vec + (1.0 - done_vec) * gamma * target_next_traffic
            target_nox = nox_penalties_vec + (1.0 - done_vec) * gamma * target_next_nox
            target_total = target_traffic - lambda_e * target_nox

            if self.cfg.ddqn.loss_type == "mse":
                loss_traffic = F.mse_loss(pred_traffic, target_traffic.detach(), reduction="none")
                loss_nox = F.mse_loss(pred_nox, target_nox.detach(), reduction="none")
            else:
                loss_traffic = F.smooth_l1_loss(pred_traffic, target_traffic.detach(), reduction="none")
                loss_nox = F.smooth_l1_loss(pred_nox, target_nox.detach(), reduction="none")
            loss_vec = loss_traffic + beta * loss_nox

            group_loss_vecs[gid].append(loss_vec)
            group_traffic_loss_vecs[gid].append(loss_traffic.detach())
            group_nox_loss_vecs[gid].append(loss_nox.detach())
            group_pred_total_vecs[gid].append(pred_total.detach())
            group_target_total_vecs[gid].append(target_total.detach())
            group_pred_traffic_vecs[gid].append(pred_traffic.detach())
            group_pred_nox_vecs[gid].append(pred_nox.detach())
            group_target_traffic_vecs[gid].append(target_traffic.detach())
            group_target_nox_vecs[gid].append(target_nox.detach())
            group_reward_vecs[gid].append(final_rewards_vec.detach())
            group_traffic_reward_vecs[gid].append(traffic_rewards_vec.detach())
            group_nox_penalty_vecs[gid].append(nox_penalties_vec.detach())
            group_td_traffic_vecs[gid].append((target_traffic - pred_traffic).detach())
            group_td_nox_vecs[gid].append((target_nox - pred_nox).detach())

        valid_loss_vecs = [vec for vecs in group_loss_vecs.values() for vec in vecs]
        if not valid_loss_vecs:
            return []
        all_loss_flat = torch.cat(valid_loss_vecs)

        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=True)
        total_loss = all_loss_flat.mean()
        total_loss.backward()

        grad_norm_by_group: Dict[str, float] = {}
        for gid, net in self.online_bank.networks.items():
            grad_norm = torch.nn.utils.clip_grad_norm_(
                net.parameters(),
                max_norm=float(self.cfg.ddqn.max_grad_norm),
            )
            grad_norm_by_group[gid] = float(
                grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
        for opt in self.optimizers.values():
            opt.step()

        stats: List[UpdateStats] = []
        for gid, loss_vecs in group_loss_vecs.items():
            if not loss_vecs:
                continue
            loss_flat = torch.cat(loss_vecs)
            traffic_loss_arr = torch.cat(group_traffic_loss_vecs[gid]).detach().cpu().numpy()
            nox_loss_arr = torch.cat(group_nox_loss_vecs[gid]).detach().cpu().numpy()
            pred_total_arr = torch.cat(group_pred_total_vecs[gid]).detach().cpu().numpy()
            target_total_arr = torch.cat(group_target_total_vecs[gid]).detach().cpu().numpy()
            pred_traffic_arr = torch.cat(group_pred_traffic_vecs[gid]).detach().cpu().numpy()
            pred_nox_arr = torch.cat(group_pred_nox_vecs[gid]).detach().cpu().numpy()
            target_traffic_arr = torch.cat(group_target_traffic_vecs[gid]).detach().cpu().numpy()
            target_nox_arr = torch.cat(group_target_nox_vecs[gid]).detach().cpu().numpy()
            reward_arr = torch.cat(group_reward_vecs[gid]).detach().cpu().numpy()
            traffic_reward_arr = torch.cat(group_traffic_reward_vecs[gid]).detach().cpu().numpy()
            nox_penalty_arr = torch.cat(group_nox_penalty_vecs[gid]).detach().cpu().numpy()
            td_arr = target_total_arr - pred_total_arr
            td_traffic_arr = torch.cat(group_td_traffic_vecs[gid]).detach().cpu().numpy()
            td_nox_arr = torch.cat(group_td_nox_vecs[gid]).detach().cpu().numpy()
            self.update_count += 1
            stats.append(UpdateStats(
                group_id=gid,
                update_index=int(self.update_count),
                batch_size=int(loss_flat.numel()),
                replay_size=int(len(self.memory)),
                q_loss=float(loss_flat.detach().mean().cpu().item()),
                td_error_mean=float(np.mean(td_arr)) if td_arr.size else 0.0,
                td_error_abs_mean=float(np.mean(np.abs(td_arr))) if td_arr.size else 0.0,
                q_pred_mean=float(np.mean(pred_total_arr)) if pred_total_arr.size else 0.0,
                q_target_mean=float(np.mean(target_total_arr)) if target_total_arr.size else 0.0,
                reward_batch_mean=float(np.mean(reward_arr)) if reward_arr.size else 0.0,
                grad_norm=float(grad_norm_by_group.get(gid, 0.0)),
                learning_rate=float(self.cfg.ddqn.learning_rate),
                traffic_loss=float(np.mean(traffic_loss_arr)) if traffic_loss_arr.size else 0.0,
                nox_loss=float(np.mean(nox_loss_arr)) if nox_loss_arr.size else 0.0,
                td_traffic_mean=float(np.mean(td_traffic_arr)) if td_traffic_arr.size else 0.0,
                td_nox_mean=float(np.mean(td_nox_arr)) if td_nox_arr.size else 0.0,
                q_traffic_pred_mean=float(np.mean(pred_traffic_arr)) if pred_traffic_arr.size else 0.0,
                q_nox_pred_mean=float(np.mean(pred_nox_arr)) if pred_nox_arr.size else 0.0,
                target_traffic_mean=float(np.mean(target_traffic_arr)) if target_traffic_arr.size else 0.0,
                target_nox_mean=float(np.mean(target_nox_arr)) if target_nox_arr.size else 0.0,
                traffic_reward_batch_mean=float(np.mean(traffic_reward_arr)) if traffic_reward_arr.size else 0.0,
                nox_penalty_batch_mean=float(np.mean(nox_penalty_arr)) if nox_penalty_arr.size else 0.0,
                target_synced=False,
            ))
        return stats

    def sync_target(self) -> None:
        self.target_bank.load_state_dict(self.online_bank.state_dict())
        self.target_update_count += 1

    def maybe_sync_target(self) -> bool:
        if int(self.cfg.ddqn.target_update_interval) <= 0:
            return False
        if self.global_step > 0 and self.global_step % int(self.cfg.ddqn.target_update_interval) == 0:
            self.sync_target()
            return True
        return False

    # ───────────────────────────────────────────────────────────────
    # Checkpoint
    # ───────────────────────────────────────────────────────────────

    def _ablation_meta(self) -> Dict[str, Any]:
        er = self.cfg.emission_risk
        return {
            "lane_input_dim": int(self.cfg.network.lane_input_dim),
            "lane_feature_names": list(self.cfg.state.lane_feature_names),
            "observation_scope": str(self.cfg.state.observation_scope),
            "use_truck_count_state": bool(er.use_truck_count_state),
            "use_nox_risk_state": bool(er.use_nox_risk_state),
            "enabled_reward": bool(er.enabled_reward),
            "lambda_e": float(er.lambda_e),
            "objective_mode": str(self.cfg.reward.objective_mode),
            "dual_head_enabled": bool(getattr(self.cfg.dual_head, "enabled", False)),
            "nox_q_softplus": bool(getattr(self.cfg.dual_head, "nox_q_softplus", False)),
            "risk_mode": str(er.risk_mode),
            "reward_risk_mode": str(getattr(er, "reward_risk_mode", "full")),
            "base_weight": float(er.base_weight),
            "tail_weight": float(er.tail_weight),
            "pressure_beta": float(er.pressure_beta),
            "rho": float(er.rho),
            "kappa": float(er.kappa),
            "risk_clip": float(er.risk_clip),
            "reward_alpha": float(er.reward_alpha),
            "penalty_aggregate": str(er.penalty_aggregate),
            "penalty_time_normalize": bool(er.penalty_time_normalize),
            "thresholds_json": str(er.thresholds_json),
        }

    @staticmethod
    def _cfg_payload_ablation_meta(cfg_payload: Any) -> Dict[str, Any]:
        if not isinstance(cfg_payload, Mapping):
            return {}
        state_cfg = cfg_payload.get("state", {})
        network_cfg = cfg_payload.get("network", {})
        er_cfg = cfg_payload.get("emission_risk", {})
        reward_cfg = cfg_payload.get("reward", {})
        dh_cfg = cfg_payload.get("dual_head", {})
        if not isinstance(state_cfg, Mapping) or not isinstance(network_cfg, Mapping):
            return {}
        return {
            "lane_input_dim": network_cfg.get("lane_input_dim"),
            "lane_feature_names": state_cfg.get("lane_feature_names"),
            "observation_scope": state_cfg.get("observation_scope"),
            "use_truck_count_state": er_cfg.get("use_truck_count_state") if isinstance(er_cfg, Mapping) else None,
            "use_nox_risk_state": er_cfg.get("use_nox_risk_state") if isinstance(er_cfg, Mapping) else None,
            "enabled_reward": er_cfg.get("enabled_reward") if isinstance(er_cfg, Mapping) else None,
            "lambda_e": er_cfg.get("lambda_e") if isinstance(er_cfg, Mapping) else None,
            "objective_mode": reward_cfg.get("objective_mode") if isinstance(reward_cfg, Mapping) else None,
            "dual_head_enabled": dh_cfg.get("enabled") if isinstance(dh_cfg, Mapping) else None,
            "nox_q_softplus": dh_cfg.get("nox_q_softplus") if isinstance(dh_cfg, Mapping) else None,
            "risk_mode": er_cfg.get("risk_mode") if isinstance(er_cfg, Mapping) else None,
            "reward_risk_mode": (
                er_cfg.get("reward_risk_mode", "full")
                if isinstance(er_cfg, Mapping)
                else None
            ),
            "base_weight": er_cfg.get("base_weight") if isinstance(er_cfg, Mapping) else None,
            "tail_weight": er_cfg.get("tail_weight") if isinstance(er_cfg, Mapping) else None,
            "pressure_beta": er_cfg.get("pressure_beta") if isinstance(er_cfg, Mapping) else None,
            "rho": er_cfg.get("rho") if isinstance(er_cfg, Mapping) else None,
            "kappa": er_cfg.get("kappa") if isinstance(er_cfg, Mapping) else None,
            "risk_clip": er_cfg.get("risk_clip") if isinstance(er_cfg, Mapping) else None,
            "reward_alpha": er_cfg.get("reward_alpha") if isinstance(er_cfg, Mapping) else None,
            "penalty_aggregate": er_cfg.get("penalty_aggregate") if isinstance(er_cfg, Mapping) else None,
            "penalty_time_normalize": er_cfg.get("penalty_time_normalize") if isinstance(er_cfg, Mapping) else None,
            "thresholds_json": er_cfg.get("thresholds_json") if isinstance(er_cfg, Mapping) else None,
        }

    def _checkpoint_ablation_meta(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        meta_raw = payload.get("ablation_meta", {})
        meta = dict(meta_raw) if isinstance(meta_raw, Mapping) else {}
        fallback = self._cfg_payload_ablation_meta(payload.get("cfg", {}))
        for field, value in fallback.items():
            if (field not in meta or meta[field] is None) and value is not None:
                meta[field] = value
        return meta

    @staticmethod
    def _thresholds_file_identity(value: Any) -> str:
        """Compare threshold metadata independently of machine-specific roots."""
        return Path(str(value)).name.casefold()

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off", ""}:
                return False
        return bool(value)

    def _validate_checkpoint_compatibility(
        self,
        payload: Mapping[str, Any],
        path: str,
        allow_threshold_mismatch: bool = False,
    ) -> None:
        ckpt_meta = self._checkpoint_ablation_meta(payload)
        current_meta = self._ablation_meta()
        checkpoint_scope = ckpt_meta.get("observation_scope")
        if (
            checkpoint_scope is not None
            and checkpoint_scope != current_meta["observation_scope"]
        ):
            raise ValueError(
                "Checkpoint observation scope is incompatible with the current "
                "state definition: "
                f"path={path!r}, checkpoint={checkpoint_scope!r}, "
                f"current={current_meta['observation_scope']!r}"
            )
        float_fields = {
            "lambda_e",
            "base_weight",
            "tail_weight",
            "pressure_beta",
            "rho",
            "kappa",
            "risk_clip",
            "reward_alpha",
        }
        bool_fields = {
            "use_truck_count_state",
            "use_nox_risk_state",
            "enabled_reward",
            "dual_head_enabled",
            "nox_q_softplus",
            "penalty_time_normalize",
        }
        string_fields = {
            "objective_mode",
            "risk_mode",
            "reward_risk_mode",
            "penalty_aggregate",
        }

        for field, current_value in current_meta.items():
            checkpoint_value = ckpt_meta.get(field)
            if checkpoint_value is None:
                if (
                    field == "observation_scope"
                    and str(current_value) == "full_lane"
                ):
                    # Checkpoints created before observation scopes were
                    # introduced used the legacy full-lane state definition.
                    continue
                warnings.warn(
                    f"Checkpoint {path!r} is missing compatibility metadata "
                    f"field {field!r}; loading for backward compatibility.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            if field in float_fields:
                matches = math.isclose(
                    float(checkpoint_value),
                    float(current_value),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            elif field in bool_fields:
                matches = self._as_bool(checkpoint_value) == self._as_bool(current_value)
            elif field in string_fields:
                checkpoint_normalized = str(checkpoint_value).strip().lower()
                current_normalized = str(current_value).strip().lower()
                matches = (
                    field == "objective_mode"
                    and checkpoint_normalized == "auto"
                ) or checkpoint_normalized == current_normalized
            elif field == "thresholds_json":
                matches = self._thresholds_file_identity(
                    checkpoint_value
                ) == self._thresholds_file_identity(current_value)
            elif field == "lane_feature_names":
                matches = tuple(str(x) for x in checkpoint_value) == tuple(
                    str(x) for x in current_value
                )
            elif field == "lane_input_dim":
                matches = int(checkpoint_value) == int(current_value)
            elif field == "observation_scope":
                matches = str(checkpoint_value) == str(current_value)
            else:
                matches = checkpoint_value == current_value

            if not matches:
                if field == "thresholds_json" and allow_threshold_mismatch:
                    warnings.warn(
                        "Checkpoint threshold metadata is being explicitly "
                        "overridden for evaluation: "
                        f"path={path!r}, checkpoint={checkpoint_value!r}, "
                        f"current={current_value!r}",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                raise ValueError(
                    "Checkpoint compatibility mismatch: "
                    f"path={path!r}, field={field!r}, "
                    f"checkpoint={checkpoint_value!r}, "
                    f"current={current_value!r}"
                )

    def checkpoint_payload(self, episode: int, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "episode": int(episode),
            "global_step": int(self.global_step),
            "update_count": int(self.update_count),
            "target_update_count": int(self.target_update_count),
            "ablation_meta": self._ablation_meta(),
            "cfg": self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else {},
            "online_state_dict": self.online_bank.state_dict(),
            "target_state_dict": self.target_bank.state_dict(),
            "optimizers": {gid: opt.state_dict() for gid, opt in self.optimizers.items()},
            "extra": extra or {},
        }

    def save_checkpoint(self, path: str, episode: int, extra: Optional[Dict[str, Any]] = None) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.checkpoint_payload(episode, extra), path)
        return path

    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = False,
        allow_threshold_mismatch: bool = False,
    ) -> Dict[str, Any]:
        payload = torch.load(path, map_location=self.device)
        self._validate_checkpoint_compatibility(
            payload,
            path,
            allow_threshold_mismatch=allow_threshold_mismatch,
        )
        self.online_bank.load_state_dict(payload["online_state_dict"])
        self.target_bank.load_state_dict(payload.get("target_state_dict", payload["online_state_dict"]))
        if load_optimizer and "optimizers" in payload:
            for gid, state in payload["optimizers"].items():
                if gid in self.optimizers:
                    self.optimizers[gid].load_state_dict(state)
        self.global_step = int(payload.get("global_step", 0))
        self.update_count = int(payload.get("update_count", 0))
        self.target_update_count = int(payload.get("target_update_count", 0))
        return payload

    def increment_step(self) -> None:
        self.global_step += 1
