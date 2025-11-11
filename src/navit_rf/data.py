from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode


class ImageDataset(Dataset):
    """
    Thin dataset that loads all image paths under a directory. Designed to keep the
    experiments light-weight while staying compatible with torchvision-style transforms.
    """

    def __init__(self, img_paths: Sequence[Path], transform: Callable | None = None):
        self.img_paths = list(img_paths)
        self.transform = transform or default_transform()

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def gather_image_paths(root: str | Path, suffixes: Iterable[str] = (".jpg", ".jpeg", ".png")) -> List[Path]:
    root = Path(root)
    suffixes = tuple(s.lower() for s in suffixes)
    paths = [p for p in root.iterdir() if p.suffix.lower() in suffixes]
    paths.sort()
    return paths


def _normalize_to_noise(noise_std: float) -> v2.Normalize:
    if noise_std <= 0:
        raise ValueError("noise_std must be positive")
    return v2.Normalize(mean=[0.0, 0.0, 0.0], std=[noise_std, noise_std, noise_std])


class RandomAspectPreservingResize(torch.nn.Module):
    """
    Scales height and width by the same random factor while keeping aspect ratio intact.
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.2, 1.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()
        if not 0 < scale_range[0] <= scale_range[1]:
            raise ValueError("scale_range must satisfy 0 < min <= max")
        self.scale_min, self.scale_max = scale_range
        self.interpolation = interpolation

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            if img.dim() < 2:
                raise ValueError("Tensor image must have at least 2 dimensions")
            h, w = img.shape[-2:]
        else:
            w, h = img.size  # PIL Image
        scale = torch.empty(1).uniform_(self.scale_min, self.scale_max).item()
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resize = v2.Resize((new_h, new_w), interpolation=self.interpolation, antialias=True)
        return resize(img)


def default_transform(image_size: int = 64, noise_std: float = 1.0):
    """
    Default preprocessing: convert to float tensor and match the Gaussian anchor statistics.
    """
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            _normalize_to_noise(noise_std),
        ]
    )


def random_resized_transform(
    noise_std: float = 1.0,
    scale_range: Tuple[float, float] = (0.2, 1.0),
):
    """
    Randomly resize while preserving aspect ratio; no cropping or fixed target size.
    """
    return v2.Compose(
        [
            v2.ToImage(),
            RandomAspectPreservingResize(scale_range=scale_range),
            v2.ToDtype(torch.float32, scale=True),
            _normalize_to_noise(noise_std),
        ]
    )
