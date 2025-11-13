import math
from typing import List, Literal, Optional, Sequence, Tuple

import torch


@torch.no_grad()
def _prepare_noise(
    n: int,
    shapes: Sequence[Tuple[int, int]],
    patch_size: int,
    device: torch.device,
    noise_std: float,
):
    patch_hw = []
    orig_hw = []
    max_patch_h = 0
    max_patch_w = 0
    for h, w in shapes:
        ph = math.ceil(h / patch_size)
        pw = math.ceil(w / patch_size)
        patch_hw.append((ph, pw))
        orig_hw.append((h, w))
        max_patch_h = max(max_patch_h, ph)
        max_patch_w = max(max_patch_w, pw)

    padded_h = max_patch_h * patch_size
    padded_w = max_patch_w * patch_size
    x = torch.zeros(n, 3, padded_h, padded_w, device=device)
    for i, (h, w) in enumerate(shapes):
        noise = torch.randn(1, 3, h, w, device=device) * noise_std
        x[i, :, :h, :w] = noise

    patch_hw_tensor = torch.tensor(patch_hw, device=device)
    orig_hw_tensor = torch.tensor(orig_hw, device=device)
    return x, patch_hw_tensor, orig_hw_tensor


def _integrate_flow(
    model,
    x: torch.Tensor,
    patch_hw: torch.Tensor,
    orig_hw: torch.Tensor,
    *,
    device: torch.device,
    steps: int,
    solver: Literal["euler", "heun"],
):
    n = x.shape[0]
    packs = [[i] for i in range(n)]
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    for i in range(steps):
        t_lo, t_hi = ts[i], ts[i + 1]
        dt = t_hi - t_lo
        t_batch = torch.full((n,), t_lo, device=device)
        v = model(
            x,
            t_batch,
            patch_hw=patch_hw,
            packs=packs,
            orig_hw=orig_hw,
        )
        if solver == "euler":
            x = x + v * dt
        elif solver == "heun":
            x_mid = x + v * dt
            v_mid = model(
                x_mid,
                torch.full_like(t_batch, t_hi),
                patch_hw=patch_hw,
                packs=packs,
                orig_hw=orig_hw,
            )
            x = x + 0.5 * (v + v_mid) * dt
        else:
            raise ValueError(f"Unknown solver '{solver}'")
    return x


@torch.no_grad()
def sample_rectified_flow(
    model,
    n: int,
    *,
    device: torch.device,
    img_size: int,
    steps: int = 100,
    noise_std: float = 1.0,
    solver: Literal["euler", "heun"] = "heun",
    shapes: Optional[Sequence[Tuple[int, int]]] = None,
    return_shapes: bool = False,
):
    """
    Numerical integration of dx/dt = v(x,t) from t=0 -> 1 for rectified flows.

    shapes: optional iterable of (height,width) requests. When omitted all samples
            default to img_size x img_size.
    """
    model.eval().to(device)

    if shapes is None:
        shapes = [(img_size, img_size)] * n
    if len(shapes) != n:
        raise ValueError("len(shapes) must match n")

    patch = getattr(model, "patch", 1)
    x, patch_hw_tensor, orig_hw_tensor = _prepare_noise(n, shapes, patch, device, noise_std)
    x = _integrate_flow(
        model,
        x,
        patch_hw_tensor,
        orig_hw_tensor,
        device=device,
        steps=steps,
        solver=solver,
    )

    samples: List[torch.Tensor] = []
    x = x.clamp(-1.0, 1.0)
    max_h = 0
    max_w = 0
    for (h, w) in shapes:
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    batch = torch.zeros(n, 3, max_h, max_w, device=device)
    for i, (h, w) in enumerate(shapes):
        sample = x[i, :, :h, :w]
        batch[i, :, :h, :w] = sample
        samples.append(sample)

    # min-max normalize each image independently
    eps = 1e-8
    for i in range(n):
        sample = batch[i]
        valid_mask = torch.zeros_like(sample, dtype=torch.bool)
        h, w = shapes[i]
        valid_mask[:, :h, :w] = True
        values = sample[valid_mask]
        min_val = values.min()
        max_val = values.max()
        denom = (max_val - min_val).clamp_min(eps)
        batch[i] = (sample - min_val) / denom

    if return_shapes:
        return batch, list(shapes)
    return batch


@torch.no_grad()
def generate_reflow_pairs(
    model,
    *,
    device: torch.device,
    shapes: Sequence[Tuple[int, int]],
    steps: int = 100,
    noise_std: float = 1.0,
    solver: Literal["euler", "heun"] = "heun",
):
    """
    Produce (x0, target) pairs by running the current model on noise shaped like the data.
    """
    if len(shapes) == 0:
        raise ValueError("shapes must be non-empty")
    patch = getattr(model, "patch", 1)
    n = len(shapes)
    anchors, patch_hw_tensor, orig_hw_tensor = _prepare_noise(n, shapes, patch, device, noise_std)
    x = _integrate_flow(
        model,
        anchors.clone(),
        patch_hw_tensor,
        orig_hw_tensor,
        device=device,
        steps=steps,
        solver=solver,
    )
    targets = x - anchors
    return anchors.cpu(), targets.cpu(), list(shapes)
