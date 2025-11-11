"""
NaViT-style utilities for batching variable-resolution images.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import torch
import torch.nn.functional as F


def _first_fit_pack(token_counts: Sequence[int], max_tokens: int) -> List[List[int]]:
    """
    Greedy first-fit-decreasing bin packing used in NaViT to keep token sequences dense.
    """
    order = sorted(range(len(token_counts)), key=lambda idx: token_counts[idx], reverse=True)
    packs: List[List[int]] = []
    remaining: List[int] = []

    for idx in order:
        tokens = token_counts[idx]
        placed = False
        for pack_idx, rem in enumerate(remaining):
            if tokens <= rem:
                packs[pack_idx].append(idx)
                remaining[pack_idx] -= tokens
                placed = True
                break
        if not placed:
            capacity = max(max_tokens, tokens)
            packs.append([idx])
            remaining.append(capacity - tokens)
    return packs


@dataclass
class PackedBatch:
    """
    Data container describing the packed batch layout (currently unused but kept for reference).
    """

    images: torch.Tensor         # (M, C, H_pad, W_pad)
    patch_hw: torch.Tensor       # (M, 2)
    orig_hw: torch.Tensor        # (M, 2)
    packs: List[List[int]]       # list of image indices per packed sequence


class PackingCollate:
    """
    Pickle-friendly callable that pads and packs samples NaViT-style.
    """

    def __init__(self, patch_size: int, max_tokens_per_pack: int, pad_value: float = 0.0):
        self.patch_size = patch_size
        self.max_tokens_per_pack = max_tokens_per_pack
        self.pad_value = pad_value

    def __call__(self, batch: Sequence[torch.Tensor]) -> Dict[str, object]:
        if not batch:
            raise ValueError("Empty batch passed to collate_fn")

        # Peel off tensors if the dataset returns (image, label, ...)
        imgs = []
        for item in batch:
            if isinstance(item, torch.Tensor):
                imgs.append(item)
            elif isinstance(item, (tuple, list)):
                imgs.append(item[0])
            else:
                raise TypeError("Unsupported sample type; expected Tensor or tuple")

        channels = imgs[0].shape[0]
        patch_hw = []
        orig_hw = []
        tokens = []

        for img in imgs:
            if img.dim() != 3 or img.shape[0] != channels:
                raise ValueError("Each sample must have shape (C,H,W) with consistent C")
            _, h, w = img.shape
            orig_hw.append((h, w))
            ph = math.ceil(h / self.patch_size)
            pw = math.ceil(w / self.patch_size)
            patch_hw.append((ph, pw))
            tokens.append(ph * pw)

        packs = _first_fit_pack(tokens, self.max_tokens_per_pack)

        max_h = max(ph for ph, _ in patch_hw) * self.patch_size
        max_w = max(pw for _, pw in patch_hw) * self.patch_size
        padded = []
        for img in imgs:
            _, h, w = img.shape
            pad_h = max_h - h
            pad_w = max_w - w
            padded.append(F.pad(img, (0, pad_w, 0, pad_h), value=self.pad_value))

        images = torch.stack(padded, dim=0)

        return {
            "images": images,
            "patch_hw": torch.tensor(patch_hw, dtype=torch.long),
            "orig_hw": torch.tensor(orig_hw, dtype=torch.long),
            "packs": packs,
        }


class PaddingCollate:
    """
    Padding-only collate (no packing) kept for ablations; also pickle-friendly.
    """

    def __init__(self, patch_size: int, pad_value: float = 0.0):
        self.patch_size = patch_size
        self.pad_value = pad_value

    def __call__(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Empty batch passed to collate_fn")

        imgs = []
        for item in batch:
            if isinstance(item, torch.Tensor):
                imgs.append(item)
            elif isinstance(item, (tuple, list)):
                imgs.append(item[0])
            else:
                raise TypeError("Unsupported sample type; expected Tensor or tuple")

        channels = imgs[0].shape[0]
        patch_hw = []
        orig_hw = []
        for img in imgs:
            if img.dim() != 3 or img.shape[0] != channels:
                raise ValueError("Each sample must have shape (C,H,W) with consistent C")
            _, h, w = img.shape
            orig_hw.append((h, w))
            patch_hw.append((math.ceil(h / self.patch_size), math.ceil(w / self.patch_size)))

        max_h = max(h for h, _ in patch_hw) * self.patch_size
        max_w = max(w for _, w in patch_hw) * self.patch_size
        padded = []
        for img in imgs:
            _, h, w = img.shape
            padded.append(F.pad(img, (0, max_w - w, 0, max_h - h), value=self.pad_value))

        images = torch.stack(padded, dim=0)
        return {
            "images": images,
            "patch_hw": torch.tensor(patch_hw, dtype=torch.long),
            "orig_hw": torch.tensor(orig_hw, dtype=torch.long),
            "packs": [[idx] for idx in range(len(imgs))],
        }


def make_packing_collate(
    patch_size: int,
    max_tokens_per_pack: int,
    pad_value: float = 0.0,
) -> Callable[[Sequence[torch.Tensor]], Dict[str, object]]:
    """
    Returns a pickle-friendly collate callable for NaViT packing.
    """

    return PackingCollate(patch_size, max_tokens_per_pack, pad_value)


def make_padding_collate(patch_size: int, pad_value: float = 0.0) -> Callable:
    """
    Simpler collate that only pads to the largest sample in the batch (no packing).
    Useful for debugging or ablations vs the packed version.
    """

    return PaddingCollate(patch_size, pad_value)

