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
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer = RectifiedFlowTrainer(cfg)
    trainer.train()
    if args.checkpoint:
        trainer.save(args.checkpoint)


if __name__ == "__main__":
    main()
