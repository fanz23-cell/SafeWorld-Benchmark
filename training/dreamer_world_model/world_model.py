"""
World model: Encoder + RSSM + Decoder + Reward head + Aux heads.
Computes all losses in a single forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .config import WorldModelConfig
from .rssm import RSSM
from .encoder_decoder import Encoder, ObsDecoder, RewardHead, AuxDecoder


class WorldModel(nn.Module):

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg

        embed_dim = cfg.enc_hidden[-1]   # encoder output dim = last hidden layer

        self.encoder = Encoder(cfg.obs_dim, cfg.enc_hidden[:-1], embed_dim)
        self.rssm = RSSM(
            embed_dim=embed_dim,
            act_dim=cfg.act_dim,
            deter_dim=cfg.deter_dim,
            stoch_dim=cfg.stoch_dim,
            stoch_classes=cfg.stoch_classes,
        )
        lat_dim = self.rssm.lat_dim

        self.obs_decoder  = ObsDecoder(lat_dim, cfg.dec_hidden, cfg.obs_dim)
        self.reward_head  = RewardHead(lat_dim, [256])
        self.aux_decoder  = AuxDecoder(lat_dim, [256], cfg.aux_keys)

    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            batch: dict with keys obs, action, reward, terminal, is_first, + aux_keys
                   all tensors shape (B, T, ...) or (B, T)

        Returns:
            loss (scalar), metrics dict
        """
        obs      = batch["obs"]       # (B, T, obs_dim)
        action   = batch["action"]    # (B, T, act_dim)
        reward   = batch["reward"]    # (B, T)
        is_first = batch["is_first"]  # (B, T)

        # --- encode ---
        embeds = self.encoder(obs)    # (B, T, embed_dim)

        # --- RSSM rollout ---
        rssm_out = self.rssm.rollout_posterior(embeds, action, is_first)
        latent       = rssm_out["latent"]       # (B, T, lat_dim)
        post_logits  = rssm_out["post_logits"]  # (B, T, stoch_dim, stoch_classes)
        prior_logits = rssm_out["prior_logits"] # (B, T, stoch_dim, stoch_classes)

        # --- decode ---
        obs_pred    = self.obs_decoder(latent)  # (B, T, obs_dim)
        reward_pred = self.reward_head(latent)  # (B, T)
        aux_preds   = self.aux_decoder(latent)  # {key: (B, T)}

        # --- losses ---
        cfg = self.cfg

        # obs reconstruction (MSE, symlog-normalised)
        obs_loss = F.mse_loss(obs_pred, obs)

        # reward prediction
        rew_loss = F.mse_loss(reward_pred, reward)

        # KL(posterior || prior) with free bits
        kl = _categorical_kl(post_logits, prior_logits)   # (B, T)
        kl_loss = torch.clamp(kl, min=cfg.kl_free).mean()

        # auxiliary safety signals
        aux_loss = torch.stack([
            F.mse_loss(aux_preds[k], batch[k]) for k in cfg.aux_keys
        ]).mean()

        loss = (cfg.loss_obs    * obs_loss
              + cfg.loss_reward * rew_loss
              + cfg.loss_kl     * kl_loss
              + cfg.loss_aux    * aux_loss)

        metrics = {
            "loss/total":  loss.item(),
            "loss/obs":    obs_loss.item(),
            "loss/reward": rew_loss.item(),
            "loss/kl":     kl_loss.item(),
            "loss/aux":    aux_loss.item(),
            "kl/raw":      kl.mean().item(),
        }
        for k in cfg.aux_keys:
            metrics[f"aux/{k}"] = F.mse_loss(aux_preds[k], batch[k]).item()

        return loss, metrics

    @property
    def lat_dim(self) -> int:
        return self.rssm.lat_dim


# ------------------------------------------------------------------
# KL helpers
# ------------------------------------------------------------------

def _categorical_kl(post: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """
    KL(post || prior) for straight-through categorical.
    post, prior: (B, T, stoch_dim, stoch_classes) — raw logits
    Returns (B, T)
    """
    post_probs  = F.softmax(post,  dim=-1)   # (B, T, D, C)
    prior_probs = F.softmax(prior, dim=-1)
    # KL = sum_c p * (log p - log q)
    kl = (post_probs * (post_probs.clamp(min=1e-8).log()
                        - prior_probs.clamp(min=1e-8).log())).sum(-1)  # (B, T, D)
    return kl.sum(-1)  # (B, T)
