from dataclasses import dataclass, field
from typing import List


@dataclass
class WorldModelConfig:
    # --- data ---
    data_dir: str = "datasets/goal2_e2_human"
    obs_dim: int = 60
    act_dim: int = 2
    seq_len: int = 64          # sequence length per training batch
    batch_size: int = 32

    # --- RSSM ---
    deter_dim: int = 512       # GRU hidden size (deterministic state h)
    stoch_dim: int = 32        # stochastic state z: num_classes
    stoch_classes: int = 32    # categorical: 32x32 = 1024-dim z
    # total latent = deter_dim + stoch_dim * stoch_classes = 512 + 1024 = 1536

    # --- encoder / decoder MLP ---
    enc_hidden: List[int] = field(default_factory=lambda: [512, 512, 512])
    dec_hidden: List[int] = field(default_factory=lambda: [512, 512, 512])

    # --- auxiliary decoder heads (safety signals) ---
    # each head predicts a scalar from the latent state
    aux_keys: List[str] = field(default_factory=lambda: [
        "cost", "speed", "goal_distance",
        "nearest_hazard_distance", "nearest_vase_distance",
        "human_distance",
    ])

    # --- loss weights ---
    loss_obs: float = 1.0      # obs reconstruction
    loss_reward: float = 1.0   # reward prediction
    loss_kl: float = 1.0       # KL(posterior || prior)
    loss_aux: float = 0.5      # auxiliary safety signals
    kl_free: float = 1.0       # free bits (nats)

    # --- optimizer ---
    lr: float = 3e-4
    grad_clip: float = 100.0
    weight_decay: float = 1e-6

    # --- training ---
    total_steps: int = 500_000
    log_every: int = 500
    save_every: int = 10_000
    eval_every: int = 5_000
    eval_batches: int = 20

    # --- paths ---
    logdir: str = "logs/goal2_world_model_v2"
    device: str = "cuda"
