import argparse
from pathlib import Path

import yaml

from navit_rf import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RectifiedFlowTrainer,
    TrainConfig,
)


def load_config(path: Path) -> ExperimentConfig:
    data = yaml.safe_load(path.read_text())
    return ExperimentConfig(
        data=DataConfig(**data["data"]),
        train=TrainConfig(**data.get("train", {})),
        model=ModelConfig(**data.get("model", {})),
    )


def main():
    parser = argparse.ArgumentParser(description="Train NaViT Rectified Flow")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional extra checkpoint to save after training")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory for latest/best checkpoints")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Warm-start model weights from this file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.checkpoint_dir:
        cfg.train.checkpoint_dir = args.checkpoint_dir
    if args.init_checkpoint:
        cfg.train.init_checkpoint = args.init_checkpoint
    trainer = RectifiedFlowTrainer(cfg)
    trainer.train()
    if args.checkpoint:
        trainer.save(args.checkpoint)


if __name__ == "__main__":
    main()
