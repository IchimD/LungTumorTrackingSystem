import glob
import os
import random
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom

from src.data.io import (
    SUPPORTED_EXTENSIONS,
    find_matching_mask,
    load_numpy_or_mhd,
    normalize_mask,
    patient_id_from_filename,
)

TransformFn = Callable[[torch.Tensor], torch.Tensor]
AugmentFn = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


class VolumeSliceDataset(Dataset):
    """PyTorch dataset for 2D axial slices extracted from preprocessed volumes."""

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        patient_ids: Optional[Sequence[str]] = None,
        transform: Optional[TransformFn] = None,
        target_transform: Optional[TransformFn] = None,
        augment_fn: Optional[AugmentFn] = None,
        background_ratio: float = 0.3,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        self.augment_fn = augment_fn
        self.background_ratio = background_ratio

        self._items: List[Tuple[str, str, str]] = []  # image_path, mask_path, patient_id
        self._discover_patients(patient_ids)

        if not self._items:
            raise ValueError(
                f"No valid training patients were found in {image_dir!r} / {mask_dir!r}."
            )

    def _discover_patients(self, patient_ids: Optional[Sequence[str]]) -> None:
        allowed = set(patient_ids) if patient_ids is not None else None
        image_paths = sorted(
            p
            for p in glob.glob(os.path.join(self.image_dir, "*"))
            if os.path.splitext(p)[1].lower() in SUPPORTED_EXTENSIONS
        )

        for image_path in image_paths:
            patient_id = patient_id_from_filename(image_path)
            if allowed is not None and patient_id not in allowed:
                continue

            mask_path = find_matching_mask(image_path, self.mask_dir)
            if mask_path is None:
                continue

            self._items.append((image_path, mask_path, patient_id))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):
        image_path, mask_path, patient_id = self._items[index]
        
        # Load mask volume directly
        mask_volume = normalize_mask(np.load(mask_path, mmap_mode='r'))
        positive_slices = [z for z in range(mask_volume.shape[0]) if mask_volume[z].any()]
        negative_slices = [z for z in range(mask_volume.shape[0]) if not mask_volume[z].any()]

        if positive_slices and negative_slices:
            if random.random() < self.background_ratio:
                slice_index = random.choice(negative_slices)
            else:
                slice_index = random.choice(positive_slices)
        elif positive_slices:
            slice_index = random.choice(positive_slices)
        else:
            slice_index = random.randint(0, mask_volume.shape[0] - 1)

        # Load image volume directly and extract slice
        image_volume = np.load(image_path, mmap_mode='r')
        image = image_volume[slice_index]
        mask = mask_volume[slice_index]

        # Resize to fixed size (512x512) for U-Net input using scipy zoom
        target_size = (512, 512)
        scale_factors = (target_size[0] / image.shape[0], target_size[1] / image.shape[1])
        image_resized = zoom(image.astype(np.float32), scale_factors, order=1)
        image_resized = (image_resized - image_resized.min()) / (image_resized.max() - image_resized.min() + 1e-8)
        mask_resized = zoom(mask.astype(np.float32), scale_factors, order=0)

        image_tensor = torch.from_numpy(np.asarray(image_resized, dtype=np.float32)).unsqueeze(0)
        mask_resized_binary = (mask_resized > 0.5).astype(np.float32)  # Ensure binary 0 or 1
        mask_tensor = torch.from_numpy(mask_resized_binary).unsqueeze(0)

        if self.augment_fn is not None:
            image_tensor, mask_tensor = self.augment_fn(image_tensor, mask_tensor)

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        if self.target_transform is not None:
            mask_tensor = self.target_transform(mask_tensor)

        return image_tensor, mask_tensor, patient_id

    def get_patient_ids(self) -> List[str]:
        return [patient_id for _, _, patient_id in self._items]
