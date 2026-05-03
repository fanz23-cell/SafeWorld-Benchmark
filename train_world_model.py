#!/usr/bin/env python3
"""
Entry point: train the Goal2 DreamerV3 world model on offline data.

Usage:
  python train_world_model.py
  python train_world_model.py --total_steps 200000 --batch_size 16
  python train_world_model.py --logdir logs/run2 --lr 1e-4
"""

import argparse
import sys
from pathlib import Path

# allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dreamer_world_model.config import WorldModelConfig
from training.dreamer_world_model.trainer import Trainer


def parse_args() -> WorldModelConfig:
    cfg = WorldModelConfig()
    p = argparse.ArgumentParser()
    for field_name, default in cfg.__dict__.items():
        if isinstance(default, list):
            p.add_argument(f"--{field_name}", nargs="+", default=default)
        elif isinstance(default, bool):
            p.add_argument(f"--{field_name}", default=default,
                           type=lambda x: x.lower() != "false")
        else:
            p.add_argument(f"--{field_name}", default=default, type=type(default))
    args = p.parse_args()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    print("Config:")
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")
    print()
    trainer = Trainer(cfg)
    trainer.train()
