from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    root: Path
    image_size: int = 64
    limit: Optional[int] = None
    shuffle: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-4
    noise_std: float = 1.0
    device: str = "cuda"
    amp: bool = False
    log_interval: int = 50
    sample_interval: int = 0
    max_tokens_per_pack: int = 512


@dataclass
class ModelConfig:
    patch: int = 8
    d_model: int = 256
    depth: int = 8
    n_head: int = 8
    mlp_ratio: float = 4.0


@dataclass
class ExperimentConfig:
    data: DataConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

