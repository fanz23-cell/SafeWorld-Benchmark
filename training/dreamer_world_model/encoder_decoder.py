"""
MLP encoder: obs vector → flat token for RSSM.
MLP decoder: latent (h, z) → obs reconstruction + auxiliary signal heads.
"""

import torch
import torch.nn as nn
from typing import List


def mlp(in_dim: int, hidden: List[int], out_dim: int,
        act=nn.SiLU, norm=nn.LayerNorm) -> nn.Sequential:
    layers = []
    dims = [in_dim] + hidden
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(norm(dims[i + 1]))
        layers.append(act())
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """obs (B, T, obs_dim) → embed (B, T, embed_dim)"""

    def __init__(self, obs_dim: int, hidden: List[int], embed_dim: int):
        super().__init__()
        self.net = mlp(obs_dim, hidden, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ObsDecoder(nn.Module):
    """latent (B, T, lat_dim) → obs_pred (B, T, obs_dim)"""

    def __init__(self, lat_dim: int, hidden: List[int], obs_dim: int):
        super().__init__()
        self.net = mlp(lat_dim, hidden, obs_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class ScalarHead(nn.Module):
    """latent (B, T, lat_dim) → scalar (B, T)"""

    def __init__(self, lat_dim: int, hidden: List[int]):
        super().__init__()
        self.net = mlp(lat_dim, hidden, 1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)


class RewardHead(ScalarHead):
    pass


class AuxDecoder(nn.Module):
    """One ScalarHead per auxiliary safety signal."""

    def __init__(self, lat_dim: int, hidden: List[int], aux_keys: List[str]):
        super().__init__()
        self.keys = aux_keys
        self.heads = nn.ModuleDict({
            k: ScalarHead(lat_dim, hidden) for k in aux_keys
        })

    def forward(self, latent: torch.Tensor):
        return {k: self.heads[k](latent) for k in self.keys}
