from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from .sampling import generate_reflow_pairs
from .data import ReflowPairDataset


def _sample_shapes(dataset, count: int) -> List[Tuple[int, int]]:
    total = len(dataset)
    if total == 0:
        raise ValueError("dataset is empty")
    indices = torch.randint(0, total, (count,))
    shapes: List[Tuple[int, int]] = []
    for idx in indices:
        sample = dataset[int(idx.item())]
        if isinstance(sample, torch.Tensor):
            _, h, w = sample.shape
        elif isinstance(sample, dict):
            tensor = sample.get("images") or sample.get("image") or sample.get("x1") or sample.get("x0")
            if tensor is None:
                raise ValueError("Unable to infer shape from dataset sample")
            _, h, w = tensor.shape
        else:
            tensor = sample[0] if isinstance(sample, (list, tuple)) else sample
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("Unsupported sample type for shape sampling")
            _, h, w = tensor.shape
        shapes.append((int(h), int(w)))
    return shapes


def _resolve_reflow_dir(base: Path, tag: str | None) -> Path:
    if tag:
        run_dir = base / tag
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base / f"reflow_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_or_load_reflow_dataset(
    model,
    dataset,
    *,
    device: torch.device,
    noise_std: float,
    reflow_pairs: int,
    reflow_steps: int,
    reflow_dir: Path,
    tag: str | None = None,
) -> Tuple[ReflowPairDataset, Path]:
    base = reflow_dir
    base.mkdir(parents=True, exist_ok=True)
    run_dir = _resolve_reflow_dir(base, tag)
    data_file = run_dir / "pairs.pt"
    if data_file.exists():
        data = torch.load(data_file)
        return ReflowPairDataset(data["anchors"], data["targets"], data["shapes"]), data_file

    count = reflow_pairs if reflow_pairs > 0 else len(dataset)
    shapes = _sample_shapes(dataset, count)
    anchors, targets, shapes = generate_reflow_pairs(
        model,
        device=device,
        shapes=shapes,
        steps=reflow_steps,
        noise_std=noise_std,
    )
    payload = {
        "anchors": anchors,
        "targets": targets,
        "shapes": shapes,
    }
    torch.save(payload, data_file)
    return ReflowPairDataset(anchors, targets, shapes), data_file
