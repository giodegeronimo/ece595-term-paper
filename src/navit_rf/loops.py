import torch

from .utils import linear_probability_path, velocity_target


def train_loop(
    model,
    dataloader,
    optimizer,
    loss_fn,
    *,
    device: str = "cuda",
    noise_std: float = 1.0,
    print_every: int = 50,
):
    """
    Legacy training loop kept for notebook parity. Prefer RectifiedFlowTrainer for
    configurable experiments.
    """
    model.train()
    total_loss = 0.0
    count = 0

    for step, batch in enumerate(dataloader, start=1):
        images = batch["images"].to(device)
        patch_hw = batch.get("patch_hw")
        packs = batch.get("packs")

        if patch_hw is None or packs is None:
            raise ValueError("Dataloader must supply 'patch_hw' and 'packs' for NaViT mode")
        patch_hw = patch_hw.to(device)

        x0 = torch.randn_like(images) * noise_std
        t = torch.rand(images.size(0), device=device)
        xt = linear_probability_path(x0, images, t)
        target = velocity_target(x0, images)

        optimizer.zero_grad(set_to_none=True)
        pred = model(xt, t, patch_hw=patch_hw, packs=packs)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1
        if step % print_every == 0:
            print(f"[step {step}] loss={loss.item():.6f}")

    return total_loss / max(count, 1)

