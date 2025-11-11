# Rectified Flow + NaViT

This note summarises the equations implemented in `navit_rf`.

## Training objective

We learn a velocity field `vθ(xt, t)` that transports noise samples `x0 ~ N(0, I)` to data `x1`.

* Probability path: `xt = (1 - t) x0 + t x1`
* Analytic velocity: `dxt/dt = x1 - x0`
* Loss: `L = E‖vθ(xt, t) - (x1 - x0)‖²`

Implementation detail: data are normalised to `[-1, 1]` so the Gaussian anchors and targets are zero-mean.

## Sampling

Starting from fresh noise:

```
x_0 ~ N(0, I)
for i in range(steps):
    x_{t+Δ} = x_t + vθ(x_t, t) Δ
```

The code defaults to Heun integration (two evaluations per step) to reduce drift without large step counts. After integration the tensor is clamped in `[-1, 1]` and mapped back to `[0, 1]` for display.

## NaViT batching

NaViT batches heterogeneous resolutions by packing patch tokens instead of padding entire images:

1. Pad each image to the nearest multiple of the patch size (purely for the convolutional patch embedding).
2. Patchify independently and retain only the valid patch grid per image.
3. Greedily pack the token chunks into fixed-budget sequences (First-Fit Decreasing).
4. Feed the packed sequence through the transformer with a key-padding mask.
5. Scatter the outputs back to the original `(patch_h, patch_w)` grids.

`src/navit_rf/navit.py` implements packing, while `ViTVelocity` accepts the resulting metadata (`packs`, `patch_hw`) to reconstruct each sample. Continuous Fourier positional embeddings remove the need for learned resolution-specific tables. Continuous Fourier time embeddings avoid the numerical pathologies of `2^k π` frequencies.
