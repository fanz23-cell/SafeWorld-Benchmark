"""
Training loop for the Goal2 world model.
"""

import os
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import WorldModelConfig
from .dataset import make_dataloader
from .world_model import WorldModel


class Trainer:

    def __init__(self, cfg: WorldModelConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.logdir = Path(cfg.logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.model = WorldModel(cfg).to(self.device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda")

        self.step = 0
        self._load_checkpoint()

        self.train_loader = make_dataloader(
            cfg.data_dir, cfg.seq_len, cfg.batch_size, split="train")
        self.val_loader = make_dataloader(
            cfg.data_dir, cfg.seq_len, cfg.batch_size, split="val")

        self._train_iter = iter(self._cycle(self.train_loader))

        # simple CSV log
        self._log_path = self.logdir / "metrics.csv"
        if not self._log_path.exists():
            self._log_path.write_text("step,loss_total,loss_obs,loss_reward,loss_kl,loss_aux\n")

    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        print(f"Training on {self.device}  |  logdir: {self.logdir}")
        print(f"  model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  train windows: {len(self.train_loader.dataset)}")
        print(f"  val   windows: {len(self.val_loader.dataset)}")

        t0 = time.time()
        while self.step < cfg.total_steps:
            batch = next(self._train_iter)
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            self.model.train()
            with torch.amp.autocast("cuda"):
                loss, metrics = self.model(batch)

            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()

            self.step += 1

            if self.step % cfg.log_every == 0:
                elapsed = time.time() - t0
                sps = cfg.log_every / elapsed
                t0 = time.time()
                print(f"[{self.step:>7d}] "
                      f"loss={metrics['loss/total']:.4f}  "
                      f"obs={metrics['loss/obs']:.4f}  "
                      f"rew={metrics['loss/reward']:.4f}  "
                      f"kl={metrics['loss/kl']:.4f}  "
                      f"aux={metrics['loss/aux']:.4f}  "
                      f"({sps:.0f} steps/s)")
                with open(self._log_path, "a") as f:
                    f.write(f"{self.step},"
                            f"{metrics['loss/total']:.6f},"
                            f"{metrics['loss/obs']:.6f},"
                            f"{metrics['loss/reward']:.6f},"
                            f"{metrics['loss/kl']:.6f},"
                            f"{metrics['loss/aux']:.6f}\n")

            if self.step % cfg.eval_every == 0:
                val_metrics = self._evaluate()
                print(f"  [eval] val_loss={val_metrics['loss/total']:.4f}  "
                      f"val_kl={val_metrics['loss/kl']:.4f}  "
                      f"val_aux={val_metrics['loss/aux']:.4f}")

            if self.step % cfg.save_every == 0:
                self._save_checkpoint()

        self._save_checkpoint()
        print("Training complete.")

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        self.model.eval()
        agg: Dict[str, float] = {}
        count = 0
        for batch in self.val_loader:
            if count >= self.cfg.eval_batches:
                break
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda"):
                _, metrics = self.model(batch)
            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + v
            count += 1
        return {k: v / count for k, v in agg.items()}

    # ------------------------------------------------------------------

    def _save_checkpoint(self):
        path = self.logdir / f"ckpt_{self.step:07d}.pt"
        torch.save({
            "step":        self.step,
            "model":       self.model.state_dict(),
            "optimizer":   self.opt.state_dict(),
            "scaler":      self.scaler.state_dict(),
        }, path)
        # keep only the 3 most recent checkpoints
        ckpts = sorted(self.logdir.glob("ckpt_*.pt"))
        for old in ckpts[:-3]:
            old.unlink()
        print(f"  saved checkpoint: {path.name}")

    def _load_checkpoint(self):
        ckpts = sorted(self.logdir.glob("ckpt_*.pt"))
        if not ckpts:
            return
        path = ckpts[-1]
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step = ckpt["step"]
        print(f"Resumed from checkpoint: {path.name}  (step {self.step})")

    @staticmethod
    def _cycle(loader: DataLoader):
        while True:
            yield from loader
