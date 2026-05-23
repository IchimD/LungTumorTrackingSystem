import random
import torch


def random_flip(image: torch.Tensor, mask: torch.Tensor, p: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    if random.random() < p:
        image = torch.flip(image, dims=[1])
        mask = torch.flip(mask, dims=[1])
    if random.random() < p:
        image = torch.flip(image, dims=[2])
        mask = torch.flip(mask, dims=[2])
    return image, mask


def random_intensity_jitter(
    image: torch.Tensor,
    gain_range: tuple[float, float] = (0.9, 1.1),
    shift_range: tuple[float, float] = (-0.05, 0.05),
    p: float = 0.2,
) -> torch.Tensor:
    if random.random() < p:
        gain = random.uniform(*gain_range)
        shift = random.uniform(*shift_range)
        image = image * gain + shift
        image = image.clamp(0.0, 1.0)
    return image


def default_training_augmentations(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    image, mask = random_flip(image, mask, p=0.5)
    image = random_intensity_jitter(image, p=0.25)
    return image, mask
