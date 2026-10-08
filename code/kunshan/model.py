
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MasterConfig, NetworkConfig
from network_parser import NetworkInfo, build_group_meta
from obs_reward import AgentObservation

DIRECTIONS = ("N", "E", "S", "W", "Self")

def _activation(cfg: NetworkConfig) -> nn.Module:
    return nn.LeakyReLU(negative_slope=float(cfg.leaky_relu_slope), inplace=True)

class MaskedGATLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, negative_slope: float = 0.05) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_head = int(d_model // num_heads)
        self.scale = math.sqrt(self.d_head)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.leaky_relu = nn.LeakyReLU(float(negative_slope))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, Dh = self.num_heads, self.d_head
        q = self.q(x).view(B, N, H, Dh).transpose(1, 2)
        k = self.k(x).view(B, N, H, Dh).transpose(1, 2)
        v = self.v(x).view(B, N, H, Dh).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        scores = self.leaky_relu(scores)
        if adj.dim() == 2:
            mask = adj.to(device=x.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        else:
            mask = adj.to(device=x.device, dtype=torch.bool).unsqueeze(1)
        if mask.shape[-1] == mask.shape[-2]:
            eye = torch.eye(mask.shape[-1], device=x.device, dtype=torch.bool).view(1, 1, mask.shape[-1], mask.shape[-1])
            mask = mask | eye
        scores = scores.masked_fill(~mask, -1e9)
        alpha = torch.softmax(scores, dim=-1)
        out = torch.matmul(alpha, v).transpose(1, 2).contiguous().view(B, N, D)
        return self.o(out)

class IntersectionEncoder(nn.Module):

    def __init__(self, n_lanes: int, cfg: NetworkConfig) -> None:
        super().__init__()
        self.n_lanes = int(n_lanes)
        self.cfg = cfg
        D = int(cfg.node_hidden_dim)
        Dp = int(cfg.node_update_dim)
        self.input_fc = nn.Sequential(
            nn.Linear(int(cfg.lane_input_dim), D),
            _activation(cfg),
        )
        self.gat_same = MaskedGATLayer(D, int(cfg.gat_heads), float(cfg.leaky_relu_slope))
        self.gat_diff = MaskedGATLayer(D, int(cfg.gat_heads), float(cfg.leaky_relu_slope))
        self.node_update = nn.Sequential(
            nn.Linear(3 * D, Dp),
            _activation(cfg),
        )
        self.gk_dim = self.n_lanes * Dp

    def forward(self, lane_features: torch.Tensor, A_same: torch.Tensor, A_diff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if lane_features.dim() != 3:
            raise ValueError(f"lane_features must be [B,N,F], got {tuple(lane_features.shape)}")
        if lane_features.shape[1] != self.n_lanes:
            raise ValueError(f"n_lanes mismatch: expected {self.n_lanes}, got {lane_features.shape[1]}")
        h = self.input_fc(lane_features)       # [B,N,D]
        h_same = self.gat_same(h, A_same)      # [B,N,D]
        h_diff = self.gat_diff(h, A_diff)      # [B,N,D]
        h_prime = self.node_update(torch.cat([h, h_same, h_diff], dim=-1))  # [B,N,Dp]
        gk = h_prime.reshape(h_prime.shape[0], -1)  # [B,N*Dp]
        return gk, h_prime

class DirectionProjector(nn.Module):

    def __init__(self, gk_dim: int, cfg: NetworkConfig) -> None:
        super().__init__()
        self.directions = DIRECTIONS
        self.proj = nn.ModuleDict({
            d: nn.Sequential(
                nn.Linear(int(gk_dim), int(cfg.net_node_dim)),
                _activation(cfg),
            )
            for d in self.directions
        })

    def forward(self, gk: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {d: layer(gk) for d, layer in self.proj.items()}

class NetworkBiGRU(nn.Module):

    def __init__(self, cfg: NetworkConfig) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=int(cfg.net_node_dim),
            hidden_size=int(cfg.bigru_hidden_dim),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.out_dim = 2 * int(cfg.bigru_hidden_dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(seq)
        return torch.cat([h_n[0], h_n[1]], dim=-1)  # [B,2H]

class QNetworkHead(nn.Module):

    def __init__(self, z_dim: int, n_actions: int, cfg: NetworkConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(z_dim), int(cfg.q_hidden_dim)),
            _activation(cfg),
            nn.Linear(int(cfg.q_hidden_dim), int(n_actions)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

@dataclass
class QValueBundle:
    q_total: torch.Tensor
    q_traffic: torch.Tensor
    q_nox: torch.Tensor

class GroupQNetwork(nn.Module):

    def __init__(
        self,
        group_id: str,
        n_lanes: int,
        n_phases: int,
        cfg: NetworkConfig,
        dual_head_enabled: bool = False,
        nox_q_softplus: bool = False,
    ) -> None:
        super().__init__()
        self.group_id = str(group_id)
        self.n_lanes = int(n_lanes)
        self.n_phases = int(n_phases)
        self.cfg = cfg
        self.intersection = IntersectionEncoder(n_lanes=n_lanes, cfg=cfg)
        self.direction_projector = DirectionProjector(self.intersection.gk_dim, cfg=cfg)
        self.network_bigru = NetworkBiGRU(cfg=cfg)
        self.z_dim = self.intersection.gk_dim + self.network_bigru.out_dim
        self.dual_head_enabled = bool(dual_head_enabled)
        self.nox_q_softplus = bool(nox_q_softplus)
        if self.dual_head_enabled:
            self.q_traffic_head = QNetworkHead(z_dim=self.z_dim, n_actions=n_phases, cfg=cfg)
            self.q_nox_head = QNetworkHead(z_dim=self.z_dim, n_actions=n_phases, cfg=cfg)
        else:
            self.q_head = QNetworkHead(z_dim=self.z_dim, n_actions=n_phases, cfg=cfg)

    @property
    def gk_dim(self) -> int:
        return self.intersection.gk_dim

    def encode_intersection(
        self,
        obs: Any,
        device: torch.device,
        A_same: Optional[Any] = None,
        A_diff: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        lane = torch.as_tensor(obs.lane_features, dtype=torch.float32, device=device).unsqueeze(0)
        if A_same is None:
            if not hasattr(obs, "A_same"):
                raise ValueError(f"observation for {getattr(obs, 'tl_id', '<unknown>')} has no A_same")
            A_same = getattr(obs, "A_same")
        if A_diff is None:
            if not hasattr(obs, "A_diff"):
                raise ValueError(f"observation for {getattr(obs, 'tl_id', '<unknown>')} has no A_diff")
            A_diff = getattr(obs, "A_diff")
        A_same_t = torch.as_tensor(A_same, dtype=torch.float32, device=device)
        A_diff_t = torch.as_tensor(A_diff, dtype=torch.float32, device=device)
        gk, h_prime = self.intersection(lane, A_same_t, A_diff_t)
        proj = self.direction_projector(gk)
        return gk, h_prime, proj

    def q_from_embeddings(
        self,
        gk: torch.Tensor,
        seq: torch.Tensor,
        lambda_e: float = 0.0,
        return_components: bool = False,
    ):
        Gk = self.network_bigru(seq)
        zk = torch.cat([gk, Gk], dim=-1)
        if self.dual_head_enabled:
            q_traffic = self.q_traffic_head(zk)
            q_nox = self.q_nox_head(zk)
            if self.nox_q_softplus:
                q_nox = F.softplus(q_nox)
            q_total = q_traffic - float(lambda_e) * q_nox
        else:
            q_total = self.q_head(zk)
            q_traffic = q_total
            q_nox = torch.zeros_like(q_total)

        if return_components:
            return QValueBundle(
                q_total=q_total,
                q_traffic=q_traffic,
                q_nox=q_nox,
            ), Gk, zk

        return q_total, Gk, zk

class GroupedQNetworkBank(nn.Module):

    def __init__(
        self,
        dual_head_enabled: bool = False,
        lambda_e: float = 0.0,
        nox_q_softplus: bool = False,
    ) -> None:
        super().__init__()
        self.dual_head_enabled = bool(dual_head_enabled)
        self.lambda_e = float(lambda_e)
        self.nox_q_softplus = bool(nox_q_softplus)
        self.networks = nn.ModuleDict()
        self.tl_to_group: Dict[str, str] = {}
        self.group_meta: Dict[str, Any] = {}
        self._static_cache_device: Optional[torch.device] = None
        self._static_A_same: Dict[str, torch.Tensor] = {}
        self._static_A_diff: Dict[str, torch.Tensor] = {}
        self._zero_neighbor: Dict[str, torch.Tensor] = {}

    @classmethod
    def from_network_info(cls, net_info: NetworkInfo, cfg: MasterConfig) -> "GroupedQNetworkBank":
        bank = cls(
            dual_head_enabled=bool(getattr(cfg.dual_head, "enabled", False)),
            lambda_e=float(cfg.emission_risk.lambda_e),
            nox_q_softplus=bool(getattr(cfg.dual_head, "nox_q_softplus", False)),
        )
        meta = build_group_meta(net_info, node_update_dim=int(cfg.network.node_update_dim))
        for group_id, gm in meta.items():
            bank.networks[group_id] = GroupQNetwork(
                group_id=group_id,
                n_lanes=gm.n_lanes,
                n_phases=gm.n_phases,
                cfg=cfg.network,
                dual_head_enabled=bool(getattr(cfg.dual_head, "enabled", False)),
                nox_q_softplus=bool(getattr(cfg.dual_head, "nox_q_softplus", False)),
            )
            bank.group_meta[group_id] = gm
        for tl_id in net_info.intersection_ids:
            bank.tl_to_group[tl_id] = net_info.get_group(tl_id)
        return bank

    def get_group_id(self, tl_id: str) -> str:
        return self.tl_to_group[tl_id]

    def get_network(self, tl_id_or_group: str) -> GroupQNetwork:
        key = str(tl_id_or_group)
        if key in self.networks:
            return self.networks[key]
        return self.networks[self.get_group_id(key)]

    def _ensure_static_cache(self, net_info: NetworkInfo, device: torch.device) -> None:
        if self._static_cache_device == device and self._static_A_same and self._zero_neighbor:
            return
        self._static_cache_device = device
        self._static_A_same = {}
        self._static_A_diff = {}
        self._zero_neighbor = {}
        for tl_id in net_info.intersection_ids:
            iinfo = net_info.get_intersection(tl_id)
            self._static_A_same[tl_id] = torch.as_tensor(iinfo.A_same, dtype=torch.float32, device=device)
            self._static_A_diff[tl_id] = torch.as_tensor(iinfo.A_diff, dtype=torch.float32, device=device)
        for group_id, net in self.networks.items():
            self._zero_neighbor[group_id] = torch.zeros(1, int(net.cfg.net_node_dim), dtype=torch.float32, device=device)

    def forward_global_batch(
        self,
        batch_observations: Sequence[Mapping[str, Any]],
        net_info: NetworkInfo,
        device: Optional[torch.device] = None,
        return_debug: bool = False,
        return_components: bool = False,
    ):
        if not batch_observations:
            empty: Dict[str, torch.Tensor] = {}
            return (empty, {}) if return_debug else empty
        if device is None:
            device = next(self.parameters()).device
        self._ensure_static_cache(net_info, device)

        tl_ids = list(batch_observations[0].keys())
        batch_size = int(len(batch_observations))
        gk_by_tl: Dict[str, torch.Tensor] = {}
        proj_by_tl: Dict[str, Dict[str, torch.Tensor]] = {}
        debug: Dict[str, Dict[str, Any]] = {}

        for tl_id in tl_ids:
            net = self.get_network(tl_id)
            lane_batch = torch.stack([
                torch.as_tensor(obs_map[tl_id].lane_features, dtype=torch.float32, device=device)
                for obs_map in batch_observations
            ], dim=0)
            A_same = self._static_A_same[tl_id]
            A_diff = self._static_A_diff[tl_id]
            gk, h_prime = net.intersection(lane_batch, A_same, A_diff)
            proj = net.direction_projector(gk)
            gk_by_tl[tl_id] = gk
            proj_by_tl[tl_id] = proj
            if return_debug:
                debug[tl_id] = {
                    "gk": gk,
                    "h_prime": h_prime,
                    "projection": proj,
                }

        q_batch_by_tl: Dict[str, Any] = {}
        for tl_id in tl_ids:
            center_net = self.get_network(tl_id)
            neighbors = net_info.get_neighbors(tl_id)
            seq_parts = []
            for direction, nb in zip(("N", "E", "S", "W"), neighbors):
                if nb is None or nb not in proj_by_tl:
                    zero = self._zero_neighbor[center_net.group_id].expand(batch_size, -1)
                    seq_parts.append(zero)
                else:
                    seq_parts.append(proj_by_tl[nb][direction])
            seq_parts.append(proj_by_tl[tl_id]["Self"])
            seq = torch.stack(seq_parts, dim=1)
            q_out, Gk, zk = center_net.q_from_embeddings(
                gk_by_tl[tl_id],
                seq,
                lambda_e=float(self.lambda_e),
                return_components=bool(return_components),
            )
            q_batch_by_tl[tl_id] = q_out
            if return_debug:
                if isinstance(q_out, QValueBundle):
                    debug[tl_id].update({
                        "seq": seq,
                        "Gk": Gk,
                        "zk": zk,
                        "q_total": q_out.q_total,
                        "q_traffic": q_out.q_traffic,
                        "q_nox": q_out.q_nox,
                    })
                else:
                    debug[tl_id].update({"seq": seq, "Gk": Gk, "zk": zk, "q": q_out})

        if return_debug:
            return q_batch_by_tl, debug
        return q_batch_by_tl

    def forward_global(
        self,
        observations: Mapping[str, AgentObservation],
        net_info: NetworkInfo,
        device: Optional[torch.device] = None,
        return_components: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        q_batch, debug = self.forward_global_batch(
            [observations],
            net_info,
            device=device,
            return_debug=True,
            return_components=return_components,
        )
        if return_components:
            q_by_tl = {
                tl_id: QValueBundle(
                    q_total=q_values.q_total.squeeze(0),
                    q_traffic=q_values.q_traffic.squeeze(0),
                    q_nox=q_values.q_nox.squeeze(0),
                )
                if isinstance(q_values, QValueBundle)
                else q_values.squeeze(0)
                for tl_id, q_values in q_batch.items()
            }
        else:
            q_by_tl = {tl_id: q_values.squeeze(0) for tl_id, q_values in q_batch.items()}
        return q_by_tl, debug

    def model_meta(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "_bank": {
                "dual_head_enabled": bool(self.dual_head_enabled),
                "lambda_e": float(self.lambda_e),
                "nox_q_softplus": bool(self.nox_q_softplus),
            }
        }
        for group_id, net in self.networks.items():
            n_params = sum(p.numel() for p in net.parameters())
            n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
            out[group_id] = {
                "n_lanes": net.n_lanes,
                "n_phases": net.n_phases,
                "lane_input_dim": int(net.cfg.lane_input_dim),
                "gk_dim": net.gk_dim,
                "net_node_dim": net.cfg.net_node_dim,
                "bigru_hidden_dim": net.cfg.bigru_hidden_dim,
                "z_dim": net.z_dim,
                "q_output_dim": net.n_phases,
                "dual_head_enabled": bool(getattr(net, "dual_head_enabled", False)),
                "nox_q_softplus": bool(getattr(net, "nox_q_softplus", False)),
                "directions": list(DIRECTIONS),
                "param_count": int(n_params),
                "trainable_param_count": int(n_trainable),
            }
        return out

    def to_device(self, device: torch.device | str) -> "GroupedQNetworkBank":
        self.to(device)
        return self

def masked_q_values(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    mask = action_mask.to(device=q_values.device, dtype=torch.bool)
    return q_values.masked_fill(~mask, -1e9)
