# ECE 595 Term Paper – Flow-Based and Transformer Models for Generative and Temporal Understanding

**Author:** Giovanni De Geronimo  
**Institution:** Purdue University  
**Course:** ECE 595 – Topics in AI and Machine Learning  
**Advisor:** Prof. [Name]  
**Date:** Fall 2025  

## Structure
- `main.tex` – root document
- `sec/` – individual section files
- `refs.bib` – bibliography
- `notes/` – reading logs and scratch outlines

## Build
Compile using TeXShop (⌘ T) or run:
```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Implementation Plan – NaViT + Rectified Flow Hybrid
- **Goal:** Extend Rectified Flow \cite{liu2023flow} with a NaViT-style packed encoder so a single velocity field can transport variable-resolution, variable-aspect-ratio images.
- **Dataset:** CIFAR-10 with aggressive random resizing/aspect-ratio jitter to simulate heterogeneous inputs; fall back to MNIST if compute gets tight.
- **Baselines:** (1) Standard Rectified Flow with a fixed-resolution U-Net encoder. (2) Lightweight DDPM with matching parameter count for speed comparisons.
- **Architecture sketch:** Replace the U-Net encoder in Rectified Flow with a packed ViT backbone that borrows NaViT components—sequence packing in the dataloader, per-image attention masks, factorized positional embeddings, and optional token dropping. The velocity head stays identical so transport paths remain comparable.
- **Training plan:** Distill from a pretrained diffusion teacher (or reuse publicly available weights) while packing mixed-resolution batches. Measure FID on 10k CIFAR-10 samples plus wall-clock sampling time for each model.
- **Repository layout:** Hybrid experiment lives under `experiments/navit_rf/` with scripts for packing-aware dataloaders, training, and sampling. Figures produced from these runs will populate `figures/`.
