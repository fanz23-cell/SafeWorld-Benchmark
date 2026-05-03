"""
RSSM (Recurrent State Space Model) — PyTorch implementation.

State:
  h  (B, deter_dim)                  deterministic, from GRU
  z  (B, stoch_dim * stoch_classes)  stochastic, categorical (straight-through)

Prior:  p(z_t | h_t)          — from h only (no obs)
Posterior: q(z_t | h_t, e_t)  — from h + encoder embed

Latent feature fed to decoder: cat(h, z)  shape (B, deter_dim + stoch_dim*stoch_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class RSSM(nn.Module):

    def __init__(self, embed_dim: int, act_dim: int,
                 deter_dim: int, stoch_dim: int, stoch_classes: int,
                 hidden_dim: int = 512):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.lat_dim = deter_dim + stoch_dim * stoch_classes

        # GRU input: cat(z_prev, action, embed)  — embed only for posterior
        # We use a single GRU for both; prior uses zeros for embed slot
        gru_in = stoch_dim * stoch_classes + act_dim
        self.gru_cell = nn.GRUCell(gru_in, deter_dim)

        # Prior: h → logits(z)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * stoch_classes),
        )

        # Posterior: cat(h, embed) → logits(z)
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * stoch_classes),
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def initial_state(self, batch_size: int, device: torch.device):
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim * self.stoch_classes, device=device)
        return h, z

    def _sample_categorical(self, logits: torch.Tensor) -> torch.Tensor:
        """Straight-through categorical sample. logits: (B, stoch_dim*stoch_classes)"""
        B = logits.shape[0]
        logits_r = logits.view(B, self.stoch_dim, self.stoch_classes)
        # straight-through: sample in forward, use soft in backward
        probs = F.softmax(logits_r, dim=-1)
        indices = torch.argmax(probs, dim=-1)                          # (B, stoch_dim)
        one_hot = F.one_hot(indices, self.stoch_classes).float()       # (B, stoch_dim, stoch_classes)
        # straight-through estimator
        z = (one_hot - probs).detach() + probs
        return z.view(B, self.stoch_dim * self.stoch_classes)

    # ------------------------------------------------------------------
    # single-step
    # ------------------------------------------------------------------

    def step_prior(self, h: torch.Tensor, z_prev: torch.Tensor,
                   action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One prior step (no observation). Returns (h_new, z_new, prior_logits)."""
        gru_in = torch.cat([z_prev, action], dim=-1)
        h_new = self.gru_cell(gru_in, h)
        prior_logits = self.prior_net(h_new)
        z_new = self._sample_categorical(prior_logits)
        return h_new, z_new, prior_logits

    def step_posterior(self, h: torch.Tensor, z_prev: torch.Tensor,
                       action: torch.Tensor, embed: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One posterior step (with observation embed). Returns (h_new, z_new, post_logits, prior_logits)."""
        gru_in = torch.cat([z_prev, action], dim=-1)
        h_new = self.gru_cell(gru_in, h)
        prior_logits = self.prior_net(h_new)
        post_logits = self.post_net(torch.cat([h_new, embed], dim=-1))
        z_new = self._sample_categorical(post_logits)
        return h_new, z_new, post_logits, prior_logits

    # ------------------------------------------------------------------
    # sequence rollout (used during training)
    # ------------------------------------------------------------------

    def rollout_posterior(self, embeds: torch.Tensor, actions: torch.Tensor,
                          is_first: torch.Tensor
                          ) -> Dict[str, torch.Tensor]:
        """
        Roll out posterior over a sequence.

        Args:
            embeds:   (B, T, embed_dim)
            actions:  (B, T, act_dim)   — action taken at step t (a_{t-1} for step t)
            is_first: (B, T)            — 1 at episode boundaries (reset state)

        Returns dict with:
            latent:       (B, T, lat_dim)   cat(h, z) for decoder
            h:            (B, T, deter_dim)
            post_logits:  (B, T, stoch_dim, stoch_classes)
            prior_logits: (B, T, stoch_dim, stoch_classes)
        """
        B, T, _ = embeds.shape
        device = embeds.device

        h, z = self.initial_state(B, device)

        hs, zs, post_ls, prior_ls = [], [], [], []

        for t in range(T):
            # reset state at episode boundaries
            reset_mask = is_first[:, t].unsqueeze(-1)          # (B, 1)
            h = h * (1.0 - reset_mask)
            z = z * (1.0 - reset_mask)

            act_t = actions[:, t]                              # (B, act_dim)
            emb_t = embeds[:, t]                               # (B, embed_dim)

            h, z, post_l, prior_l = self.step_posterior(h, z, act_t, emb_t)

            hs.append(h)
            zs.append(z)
            post_ls.append(post_l.view(B, self.stoch_dim, self.stoch_classes))
            prior_ls.append(prior_l.view(B, self.stoch_dim, self.stoch_classes))

        h_seq = torch.stack(hs, dim=1)                        # (B, T, deter_dim)
        z_seq = torch.stack(zs, dim=1)                        # (B, T, stoch_dim*stoch_classes)
        latent = torch.cat([h_seq, z_seq], dim=-1)            # (B, T, lat_dim)

        return {
            "latent":       latent,
            "h":            h_seq,
            "post_logits":  torch.stack(post_ls, dim=1),      # (B, T, stoch_dim, stoch_classes)
            "prior_logits": torch.stack(prior_ls, dim=1),     # (B, T, stoch_dim, stoch_classes)
        }
