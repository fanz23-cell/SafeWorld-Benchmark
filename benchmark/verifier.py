"""
SafeWorld Verifier for PyTorch DreamerV3 world model.

Pipeline (Section 4.2 of SafeWorld paper):
  1. Encode an observation prefix → posterior latent (h, z)
  2. Roll out K imagined trajectories using PRIOR transitions (no real obs)
  3. At each imagined step, decode latent → obs → extract continuous AP signals
  4. Evaluate STL robustness ρ over each imagined AP trajectory
  5. ρ* = min_i ρ_i  (worst-case margin)
  6. Transfer Calibrator: ρ_net = ρ* - ĉ_err
  7. Verdict: ρ_net > 0 → "transfers" (safe), else "insufficient margin"

AP signals extracted from decoded observations:
    velocity               ← obs[speed_idx]  (continuous)
    goal_distance          ← obs[goal_dist_idx]
    nearest_hazard_dist    ← obs[hazard_dist_idx]
    nearest_vase_dist      ← obs[vase_dist_idx]  (near_obstacle)
    human_distance         ← aux_decoder["human_distance"] prediction
    (zone_a, zone_b etc. require position; approximated via goal_distance proxy)

Usage:
    verifier = SafeWorldVerifier(world_model, obs_dim=60, device="cuda")
    result = verifier.verify(obs_prefix, stl_formula, K=200)
    print(result.summary())
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.dreamer_world_model.world_model import WorldModel
from training.dreamer_world_model.config import WorldModelConfig

# SafeWorld STL monitor (already implemented in external/SafeWorld)
_SW = ROOT / "external" / "SafeWorld"
sys.path.insert(0, str(_SW))
from core.stl_monitor import monitor_rollouts, MonitorResult
from core.transfer_calibrator import (
    fit_conformal_error_budget,
    transfer_verdict,
    TransferResult,
)


# ── Obs vector index mapping for SafetyPointGoal2-v0 ─────────────────────────
# The 60-dim obs vector from Safety Gymnasium encodes lidar, velocity, etc.
# We use the aux_decoder heads (which predict continuous scalars from latent)
# as the AP signal source — more reliable than raw obs indexing.
AUX_AP_KEYS = [
    "cost", "speed", "goal_distance",
    "nearest_hazard_distance", "nearest_vase_distance",
    "human_distance",
]


# ── AP state dict builder ─────────────────────────────────────────────────────

def _latent_to_ap_state(
    latent: torch.Tensor,          # (lat_dim,)
    aux_decoder,                   # world model aux_decoder module
) -> dict[str, float]:
    """
    Decode one latent vector into a continuous AP state dict.
    Keys match the STL formula dimension names.
    """
    with torch.no_grad():
        lat = latent.unsqueeze(0)          # (1, lat_dim)
        preds = aux_decoder(lat)           # {key: (1,)}
    return {k: float(preds[k][0]) for k in AUX_AP_KEYS}


# ── Verification result ───────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    monitor:       MonitorResult
    transfer:      TransferResult | None
    rho_star:      float
    rho_net:       float | None
    verdict:       str             # "safe" | "unsafe" | "inconclusive"
    n_rollouts:    int
    horizon:       int
    formula_info:  str = ""

    def summary(self) -> str:
        lines = [
            f"[SafeWorld] verdict={self.verdict}  "
            f"ρ*={self.rho_star:.4f}  "
            f"ρ_net={self.rho_net:.4f if self.rho_net is not None else 'N/A'}  "
            f"sat={self.monitor.n_satisfied}/{self.n_rollouts}",
        ]
        if self.transfer:
            lines.append(f"  {self.transfer.summary()}")
        return "\n".join(lines)


# ── Main verifier class ───────────────────────────────────────────────────────

class SafeWorldVerifier:
    """
    SafeWorld verifier wrapping a trained PyTorch WorldModel.

    Parameters
    ----------
    world_model : trained WorldModel instance (in eval mode)
    device      : "cuda" or "cpu"
    """

    def __init__(self, world_model: WorldModel, device: str = "cuda"):
        self.model  = world_model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    # ── Encode an obs prefix to posterior latent ──────────────────────────────

    @torch.no_grad()
    def encode_prefix(
        self,
        obs_prefix: np.ndarray,     # (T_ctx, obs_dim)
        actions: np.ndarray | None = None,  # (T_ctx, act_dim) or None → zeros
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Roll the posterior through obs_prefix, return final (h, z).
        If actions are None, uses zero actions (observation-only context).
        """
        obs = torch.from_numpy(obs_prefix).float().to(self.device).unsqueeze(0)  # (1, T, obs_dim)
        T = obs.shape[1]

        if actions is None:
            act = torch.zeros(1, T, self.model.cfg.act_dim, device=self.device)
        else:
            act = torch.from_numpy(actions).float().to(self.device).unsqueeze(0)

        is_first = torch.zeros(1, T, device=self.device)
        is_first[0, 0] = 1.0

        embeds   = self.model.encoder(obs)
        rssm_out = self.model.rssm.rollout_posterior(embeds, act, is_first)

        # return the LAST latent state as starting point for imagination
        h_last = rssm_out["h"][:, -1]     # (1, deter_dim)
        z_last = rssm_out["latent"][:, -1, self.model.rssm.deter_dim:]  # stoch part
        return h_last.squeeze(0), z_last.squeeze(0)   # (deter_dim,), (stoch_flat,)

    # ── Imagine K rollouts from a latent starting point ───────────────────────

    @torch.no_grad()
    def imagine_rollouts(
        self,
        h0: torch.Tensor,   # (deter_dim,)
        z0: torch.Tensor,   # (stoch_flat,)
        horizon: int,
        n_rollouts: int,
        action_std: float = 0.3,
    ) -> list[list[dict[str, float]]]:
        """
        Sample n_rollouts imagined trajectories of length horizon.
        Actions are Gaussian noise (exploration prior).

        Returns list of trajectories; each trajectory is a list of AP state dicts.
        """
        cfg = self.model.cfg
        trajectories: list[list[dict[str, float]]] = []

        for _ in range(n_rollouts):
            h = h0.unsqueeze(0).clone()   # (1, deter_dim)
            z = z0.unsqueeze(0).clone()   # (1, stoch_flat)
            traj: list[dict[str, float]] = []

            for _ in range(horizon):
                action = (torch.randn(1, cfg.act_dim, device=self.device) * action_std
                          ).clamp(-1.0, 1.0)
                h, z, _ = self.model.rssm.step_prior(h, z, action)
                latent   = torch.cat([h, z], dim=-1).squeeze(0)   # (lat_dim,)
                ap_state = _latent_to_ap_state(latent, self.model.aux_decoder)
                traj.append(ap_state)

            trajectories.append(traj)

        return trajectories

    # ── Main verify call ──────────────────────────────────────────────────────

    def verify(
        self,
        obs_prefix: np.ndarray,          # (T_ctx, obs_dim)
        stl_formula: dict,               # STL parse-tree from stl_specs.py
        horizon: int = 50,
        n_rollouts: int = 200,
        actions_prefix: np.ndarray | None = None,
        paired_env_rollouts: list[tuple] | None = None,  # for transfer calibration
        formula_aps: list[str] | None = None,
        delta_cp: float = 0.05,
        delta_err: float = 0.05,
        inconclusive_margin: float = 0.05,
    ) -> VerificationResult:
        """
        Full SafeWorld verification pipeline.

        Parameters
        ----------
        obs_prefix       : context observations to encode into latent
        stl_formula      : STL formula tree (from external/SafeWorld/specs/stl_specs.py)
        horizon          : imagination rollout length
        n_rollouts       : K imagined trajectories
        actions_prefix   : optional context actions (None → zeros)
        paired_env_rollouts : list of (model_traj, env_traj) pairs for transfer calibration
        formula_aps      : AP keys used by formula (for calibrator)
        delta_cp / delta_err : conformal confidence levels
        inconclusive_margin  : |ρ_net| < this → "inconclusive"

        Returns
        -------
        VerificationResult
        """
        # 1. Encode context
        h0, z0 = self.encode_prefix(obs_prefix, actions_prefix)

        # 2. Imagine K rollouts
        trajectories = self.imagine_rollouts(h0, z0, horizon, n_rollouts)

        # 3. STL monitor
        monitor = monitor_rollouts(stl_formula, trajectories)

        # 4. Transfer calibration (optional)
        transfer = None
        rho_net  = monitor.rho_star
        if paired_env_rollouts and formula_aps:
            c_hat_err = fit_conformal_error_budget(
                paired_env_rollouts, formula_aps, delta_err=delta_err
            )
            transfer = transfer_verdict(
                monitor.rho_star, c_hat_err,
                delta_cp=delta_cp, delta_err=delta_err,
                margins=monitor.margins,
            )
            rho_net = transfer.rho_net

        # 5. Verdict
        if rho_net > inconclusive_margin:
            verdict = "safe"
        elif rho_net < -inconclusive_margin:
            verdict = "unsafe"
        else:
            verdict = "inconclusive"

        return VerificationResult(
            monitor=monitor,
            transfer=transfer,
            rho_star=monitor.rho_star,
            rho_net=rho_net,
            verdict=verdict,
            n_rollouts=n_rollouts,
            horizon=horizon,
        )


# ── Convenience: load model from checkpoint ───────────────────────────────────

def load_verifier(logdir: str, device: str = "cuda") -> SafeWorldVerifier:
    """
    Load a trained WorldModel checkpoint and wrap it in SafeWorldVerifier.

    Parameters
    ----------
    logdir : path to the training log directory (must contain ckpt_*.pt files)
    device : "cuda" or "cpu"
    """
    from training.dreamer_world_model.trainer import Trainer

    cfg       = WorldModelConfig(logdir=logdir, device=device)
    # Build model + load checkpoint without touching data loaders
    import torch
    ckpts = sorted(Path(logdir).glob("ckpt_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {logdir}")

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = WorldModel(cfg).to(dev)
    ckpt  = torch.load(ckpts[-1], map_location=dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {ckpts[-1].name}  (step {ckpt['step']})")
    return SafeWorldVerifier(model, device=device)
