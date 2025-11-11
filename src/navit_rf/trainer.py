from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import ImageDataset, gather_image_paths, default_transform
from .model import ViTVelocity
from .navit import make_packing_collate
from .utils import linear_probability_path, velocity_target


class RectifiedFlowTrainer:
    """
    High-level training harness that wires datasets, NaViT packing, and the
    rectified flow objective into a reusable API.
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.train.device)

        img_paths = gather_image_paths(cfg.data.root)
        if cfg.data.limit is not None:
            img_paths = img_paths[: cfg.data.limit]
        transform = default_transform(cfg.data.image_size, noise_std=cfg.train.noise_std)
        dataset = ImageDataset(img_paths, transform=transform)

        collate = make_packing_collate(
            patch_size=cfg.model.patch,
            max_tokens_per_pack=cfg.train.max_tokens_per_pack,
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=cfg.data.shuffle,
            num_workers=4,
            collate_fn=collate,
        )

        self.model = ViTVelocity(
            patch=cfg.model.patch,
            in_ch=3,
            d_model=cfg.model.d_model,
            depth=cfg.model.depth,
            n_head=cfg.model.n_head,
            mlp_ratio=cfg.model.mlp_ratio,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.train.lr)
        self.criterion = nn.MSELoss()

    def _forward_batch(self, batch: Dict[str, object]) -> torch.Tensor:
        images = batch["images"].to(self.device)
        patch_hw = batch["patch_hw"].to(self.device)
        packs = batch["packs"]

        x0 = torch.randn_like(images) * self.cfg.train.noise_std
        t = torch.rand(images.size(0), device=self.device)
        xt = linear_probability_path(x0, images, t)
        target = velocity_target(x0, images)

        pred = self.model(
            xt,
            t,
            patch_hw=patch_hw,
            packs=packs,
        )
        loss = self.criterion(pred, target)
        return loss

    def train(self):
        for epoch in range(self.cfg.train.epochs):
            running = 0.0
            steps = 0
            for step, batch in enumerate(self.dataloader, start=1):
                steps = step
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._forward_batch(batch)
                loss.backward()
                self.optimizer.step()

                running += loss.item()
                if step % self.cfg.train.log_interval == 0:
                    avg = running / self.cfg.train.log_interval
                    print(f"[epoch {epoch+1}] step {step}: loss={avg:.4f}")
                    running = 0.0
            if running and steps % self.cfg.train.log_interval != 0:
                avg = running / (steps % self.cfg.train.log_interval)
                print(f"[epoch {epoch+1}] loss={avg:.4f}")

    def save(self, path: str | Path):
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.cfg),
        }
        torch.save(ckpt, Path(path))
