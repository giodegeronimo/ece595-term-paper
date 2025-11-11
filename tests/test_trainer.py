from pathlib import Path

import torch
from PIL import Image

from navit_rf.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from navit_rf.trainer import RectifiedFlowTrainer


def _write_images(tmpdir: Path, count: int = 4):
    tmpdir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img = Image.fromarray((torch.rand(64, 64, 3).numpy() * 255).astype("uint8"))
        img.save(tmpdir / f"{i:03d}.png")


def test_trainer_init(tmp_path):
    data_dir = tmp_path / "imgs"
    _write_images(data_dir)
    cfg = ExperimentConfig(
        data=DataConfig(root=data_dir, image_size=32, limit=4, shuffle=False),
        train=TrainConfig(batch_size=2, epochs=1, max_tokens_per_pack=128, log_interval=1, device="cpu"),
        model=ModelConfig(patch=4, d_model=32, depth=2, n_head=4, mlp_ratio=2.0),
    )
    trainer = RectifiedFlowTrainer(cfg)
    trainer.train()
