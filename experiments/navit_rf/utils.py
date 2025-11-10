import torch


def train_loop(
    model,
    dataloader,
    optimizer,
    loss_fn,
    device="cuda",
    img_size=256,
    noise_std=1.0,
    print_every=50,
):
    '''
    Simple rectified flow train loop
    '''
    model.train()

    total_loss = 0.0
    count = 0

    for step, x1 in enumerate(dataloader):
        if isinstance(x1, (list, tuple)):
            x1 = x1[0]
        x1 = x1.to(device)  # (B,C,H,W)

        B, C, H, W = x1.shape
        assert H == img_size and W == img_size, "Resize before feeding into model."

        x0 = torch.randn_like(x1) * noise_std
        t = torch.rand(B, device=device)
        xt = (1.0 - t.view(B, 1, 1, 1)) * x0 + t.view(B, 1, 1, 1) * x1
        target = x1 - x0

        optimizer.zero_grad()
        v = model(xt, t)
        loss = loss_fn(v, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

        if (step + 1) % print_every == 0:
            print(f"[step {step+1}] loss = {loss.item():.6f}")

    return total_loss / max(count, 1)

@torch.no_grad()
def sample_rectified_flow(model, n=8, device="cuda", img_size=256, steps=20, noise_std=1.0):
    """
    Integrate dx/dt = v(x,t) forward from t=0 -> 1 (Euler).
    Start at x(0) ~ N(0, I) and march toward the data manifold.
    """
    model.eval().to(device)
    x = torch.randn(n, 3, img_size, img_size, device=device) * noise_std
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    for i in range(steps):
        t_lo, t_hi = ts[i], ts[i + 1]
        dt = t_hi - t_lo
        t_batch = torch.full((n,), t_lo, device=device)
        v = model(x, t_batch)
        x = x + v * dt

    # clamp to [-1,1] (training range) then map back to [0,1] for visualization
    x = x.clamp(-1.0, 1.0)
    return (x + 1.0) * 0.5
    
