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

## Reflow datasets

After an initial round of training on real data, you can fine-tune purely on synthetic “reflow” pairs:

1. Sample `(H, W)` shapes from the transformed training set so the synthetic data match the original resolution distribution.
2. For each shape, run the current model with fresh Gaussian anchors to obtain `(x0, target)` where `target = vθ` integrates the flow.
3. Store the padded anchors/targets and their shapes under the sibling directory `data_root.parent / f"{data_root.name}_reflow" / <tag> / pairs.pt`.
4. Switch `TrainConfig.reflow_only=True` (or use the notebook template) to train exclusively on those cached pairs; the trainer now expects batches containing `x0`, `target`, `patch_hw`, and `orig_hw`.

The helper `build_or_load_reflow_dataset` automates steps 1–3, while `make_reflow_collate` repads the pairs for the transformer. Checkpoints follow the “latest/best” convention so you can resume base or reflow training without juggling dozens of files.

Each run also drops a `config.json` plus `training_loss.csv` / `reflow_loss.csv` into `logs/<timestamp>/`, making it easy to plot convergence curves or cite the exact hyper-parameters later.
