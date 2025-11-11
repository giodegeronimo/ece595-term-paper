import argparse
from pathlib import Path

import torch
import yaml

from navit_rf import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
    ViTVelocity,
)
from navit_rf.sampling import sample_rectified_flow


def load_config(path: Path) -> ExperimentConfig:
    data = yaml.safe_load(path.read_text())
    return ExperimentConfig(
        data=DataConfig(**data["data"]),
        train=TrainConfig(**data.get("train", {})),
        model=ModelConfig(**data.get("model", {})),
    )


def main():
    parser = argparse.ArgumentParser(description="Sample from a trained NaViT Rectified Flow model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("samples.pt"))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.train.device)
    model = ViTVelocity(
        patch=cfg.model.patch,
        in_ch=3,
        d_model=cfg.model.d_model,
        depth=cfg.model.depth,
        n_head=cfg.model.n_head,
        mlp_ratio=cfg.model.mlp_ratio,
    )
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])

    samples = sample_rectified_flow(
        model,
        n=args.n,
        device=device,
        img_size=cfg.data.image_size,
        steps=args.steps,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, args.out)
    print(f"Saved samples to {args.out}")


if __name__ == "__main__":
    main()
