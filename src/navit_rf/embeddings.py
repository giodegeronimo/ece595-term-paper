import math
from typing import Optional

import torch
import torch.nn as nn


class FourierTimeEmbedding(nn.Module):
    """
    Bounded Fourier features for scalar timesteps t in [0, 1].
    Keeping frequencies in [f_min, f_max] avoids float32 saturation.
    """

    def __init__(
        self,
        d_model: int,
        n_frequencies: int = 32,
        f_min: float = 1.0,
        f_max: float = 1000.0,
    ):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        freqs = torch.linspace(f_min, f_max, n_frequencies) * math.pi
        self.register_buffer("freqs", freqs)
        in_dim = 2 * n_frequencies + 1  # [t, sin, cos]
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        angles = t * self.freqs
        feats = torch.cat([t, torch.sin(angles), torch.cos(angles)], dim=1)
        return self.proj(feats)


class FractionalFourierPositionalEmbedding(nn.Module):
    """
    Continuous 2-D positional embeddings parameterised by fractional Fourier bases.
    Works for arbitrary patch grids, enabling variable-resolution NaViT batches.
    """

    def __init__(
        self,
        dim: int,
        n_frequencies: int = 32,
        base: float = 2.0,
        exp_min: float = 0.0,
        exp_max: float = 8.0,
    ):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        exponents = torch.linspace(exp_min, exp_max, n_frequencies)
        freqs = (base ** exponents) * math.pi
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(4 * n_frequencies, dim)  # sin/cos for x and y

    def forward(
        self,
        Ht: int,
        Wt: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        y = torch.linspace(0.0, 1.0, Ht, device=device, dtype=dtype or torch.float32)
        x = torch.linspace(0.0, 1.0, Wt, device=device, dtype=dtype or torch.float32)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([grid_y, grid_x], dim=-1).view(-1, 2)

        angles_y = coords[:, 0:1] * self.freqs
        angles_x = coords[:, 1:2] * self.freqs
        feats = torch.cat(
            [
                torch.sin(angles_y),
                torch.cos(angles_y),
                torch.sin(angles_x),
                torch.cos(angles_x),
            ],
            dim=1,
        )
        return self.proj(feats)

