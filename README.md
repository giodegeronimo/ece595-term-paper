# ECE 595 Term Paper – Flow-Based and Transformer Models for Generative and Temporal Understanding

**Author:** Giovanni De Geronimo  
**Institution:** Purdue University  
**Course:** ECE 595 – Topics in AI and Machine Learning  
**Advisor:** Prof. [Name]  
**Date:** Fall 2025  

## Structure

Top-level LaTeX artefacts remain unchanged. The NaViT + Rectified Flow implementation now follows a standard `src/` layout:

- `src/navit_rf/` – installable package (model, embeddings, trainers, NaViT packing utilities).
- `configs/` – YAML configs for reproducible runs (see `navit_64.yaml`).
- `scripts/` – CLI entry points (`train.py`, `sample.py`).
- `docs/rectified_flow.md` – derivation and NaViT batching notes.
- `tests/` – lightweight pytest smoke tests for embeddings, packer, and trainer wiring.
- `experiments/navit_rf/` – legacy notebooks; they now import from `navit_rf.*` so they stay in sync with the package code.

## Build
Compile using TeXShop (⌘ T) or run:
```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Development environment

Install dependencies with either Conda or Python’s editable installs:

```bash
conda env create -f environment.yml
conda activate navit-rf
pip install -e .
pre-commit install
```

Run formatting/linting:

```bash
pre-commit run --all-files
pytest
```

## Implementation Plan – NaViT + Rectified Flow Hybrid
- **Goal:** Extend Rectified Flow \cite{liu2023flow} with a NaViT-style packed encoder so a single velocity field can transport variable-resolution, variable-aspect-ratio images.
- **Dataset:** CIFAR-10 with aggressive random resizing/aspect-ratio jitter to simulate heterogeneous inputs; fall back to MNIST if compute gets tight.
- **Baselines:** (1) Standard Rectified Flow with a fixed-resolution U-Net encoder. (2) Lightweight DDPM with matching parameter count for speed comparisons.
- **Architecture sketch:** Replace the U-Net encoder in Rectified Flow with a packed ViT backbone that borrows NaViT components—sequence packing in the dataloader, per-image attention masks, factorized positional embeddings, and optional token dropping. The velocity head stays identical so transport paths remain comparable.
- **Training plan:** Distill from a pretrained diffusion teacher (or reuse publicly available weights) while packing mixed-resolution batches. Measure FID on 10k CIFAR-10 samples plus wall-clock sampling time for each model.
- **Repository layout:** Core library lives in `src/navit_rf`, train/sample CLIs sit in `scripts/`, configs enjoy `configs/`, and notebooks remain under `experiments/navit_rf/`. Figures produced from these runs will populate `figures/`.
