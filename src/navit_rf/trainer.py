from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import json

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import ImageDataset, gather_image_paths, default_transform
from .model import ViTVelocity
from .navit import make_packing_collate, make_reflow_collate
from .utils import linear_probability_path
from .reflow import build_or_load_reflow_dataset


class RectifiedFlowTrainer:
    """
    High-level training harness that wires datasets, NaViT packing, and the
    rectified flow objective into a reusable API.
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.train.device)

        self.data_root = Path(cfg.data.root)
        img_paths = gather_image_paths(self.data_root)
        if cfg.data.limit is not None:
            img_paths = img_paths[: cfg.data.limit]
        transform = default_transform(cfg.data.image_size, noise_std=cfg.train.noise_std)
        dataset = ImageDataset(img_paths, transform=transform)
        self.dataset = dataset

        self.model = ViTVelocity(
            patch=cfg.model.patch,
            in_ch=3,
            d_model=cfg.model.d_model,
            depth=cfg.model.depth,
            n_head=cfg.model.n_head,
            mlp_ratio=cfg.model.mlp_ratio,
        ).to(self.device)

        init_ckpt = cfg.train.init_checkpoint
        if init_ckpt:
            state = torch.load(init_ckpt, map_location=self.device)
            model_state = state.get("model", state)
            self.model.load_state_dict(model_state)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log_root = self.data_root.parent / "logs"
        self.log_dir = Path(cfg.train.log_dir) if cfg.train.log_dir else default_log_root / timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)

        def _json_default(obj):
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"{obj!r} is not JSON serializable")

        (self.log_dir / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2, default=_json_default))

        self.reflow_mode = cfg.train.reflow_only
        checkpoint_dir = cfg.train.checkpoint_dir
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.latest_path = self.checkpoint_dir / "latest.pth"
            self.best_path = self.checkpoint_dir / "best.pth"
        else:
            self.checkpoint_dir = None
            self.latest_path = None
            self.best_path = None

        if not self.reflow_mode:
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
            self.reflow_path = None
            self.loss_log = self.log_dir / "training_loss.csv"
        else:
            default_reflow = self.data_root.parent / f"{self.data_root.name}_reflow"
            reflow_root = Path(cfg.train.reflow_dir) if cfg.train.reflow_dir else default_reflow
            reflow_dataset, data_file = build_or_load_reflow_dataset(
                model=self.model,
                dataset=dataset,
                device=self.device,
                noise_std=cfg.train.noise_std,
                reflow_pairs=cfg.train.reflow_pairs,
                reflow_steps=cfg.train.reflow_steps,
                reflow_dir=reflow_root,
                tag=cfg.train.reflow_tag,
                source_checkpoint=cfg.train.init_checkpoint,
            )
            collate = make_reflow_collate(cfg.model.patch)
            self.dataloader = DataLoader(
                reflow_dataset,
                batch_size=cfg.train.batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=collate,
            )
            self.reflow_path = data_file
            self.loss_log = self.log_dir / "reflow_loss.csv"

        if not self.loss_log.exists():
            self.loss_log.write_text("epoch,loss\n")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.train.lr)
        self.criterion = nn.MSELoss()
        self.best_loss = float("inf")
        self.last_epoch_loss: Optional[float] = None

    def _forward_batch(self, batch: Dict[str, object]) -> torch.Tensor:
        if self.reflow_mode:
            x0 = batch["x0"].to(self.device)
            target = batch["target"].to(self.device)
            images = x0 + target
            patch_hw = batch["patch_hw"].to(self.device)
            orig_hw = batch["orig_hw"].to(self.device)
            packs = batch["packs"]
        else:
            images = batch["images"].to(self.device)
            patch_hw = batch["patch_hw"].to(self.device)
            orig_hw = batch["orig_hw"].to(self.device)
            packs = batch["packs"]
            x0 = torch.randn_like(images) * self.cfg.train.noise_std
            target = images - x0

        t = torch.rand(images.size(0), device=self.device)
        xt = linear_probability_path(x0, images, t)

        pred = self.model(
            xt,
            t,
            patch_hw=patch_hw,
            packs=packs,
            orig_hw=orig_hw,
        )
        loss = self.criterion(pred, target)
        return loss

    def _current_state(self, epoch: int):
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.cfg),
        }

    def _save_checkpoints(self, epoch: int, epoch_loss: Optional[float], is_best: bool):
        if not self.checkpoint_dir or self.latest_path is None or self.best_path is None:
            return
        state = self._current_state(epoch)
        torch.save(state, self.latest_path)
        if epoch_loss is not None and is_best:
            torch.save(state, self.best_path)
            print(f"[epoch {epoch}] updated best checkpoint -> {self.best_path}")

    def _log_epoch_loss(self, epoch: int, epoch_loss: float):
        with open(self.loss_log, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{epoch_loss:.6f}\n")

    def train(self):
        for epoch in range(self.cfg.train.epochs):
            running = 0.0
            total = 0.0
            steps = 0
            for step, batch in enumerate(self.dataloader, start=1):
                steps = step
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._forward_batch(batch)
                loss.backward()
                self.optimizer.step()

                running += loss.item()
                total += loss.item()
                if step % self.cfg.train.log_interval == 0:
                    avg = running / self.cfg.train.log_interval
                    print(f"[epoch {epoch+1}] step {step}: loss={avg:.4f}")
                    running = 0.0
            epoch_loss = total / max(steps, 1)
            print(f"[epoch {epoch+1}] loss={epoch_loss:.4f}")
            self.last_epoch_loss = epoch_loss
            is_best = epoch_loss < self.best_loss
            if is_best:
                self.best_loss = epoch_loss
            self._log_epoch_loss(epoch + 1, epoch_loss)
            self._save_checkpoints(epoch + 1, epoch_loss, is_best)

    def save(self, path: str | Path):
        ckpt = self._current_state(epoch=self.cfg.train.epochs)
        torch.save(ckpt, Path(path))
