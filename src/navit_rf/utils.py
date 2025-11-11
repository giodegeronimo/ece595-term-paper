import torch


def linear_probability_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Convenience helper for x_t = (1 - t) * x0 + t * x1 with broadcasting over spatial dims.
    """
    while t.dim() < x0.dim():
        t = t.unsqueeze(-1)
    return (1.0 - t) * x0 + t * x1


def velocity_target(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    Analytic velocity for the straight-line probability path.
    """
    return x1 - x0

