import glob
import os
import random
import warnings
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


def safe_np_load(path: str) -> np.ndarray:
    """Load NumPy volumes safely, with a fallback for pickled objects."""
    try:
        return np.load(path, allow_pickle=False)
    except Exception as base_exc:
        warnings.warn(
            f"np.load failed for {path} with allow_pickle=False: {base_exc}. Retrying with allow_pickle=True.",
            UserWarning,
        )
        try:
            arr = np.load(path, allow_pickle=True)
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"Loaded object is not an ndarray: {type(arr)}")
            return arr
        except Exception:
            raise base_exc


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
        candidates = [
            os.path.join(self.image_dir, "*"),
            os.path.join(self.image_dir, "images", "*"),
            os.path.join(self.image_dir, "**", "*"),
        ]

        image_paths = []
        for pattern in candidates:
            try:
                matches = sorted(
                    p
                    for p in glob.glob(pattern, recursive="**" in pattern)
                    if os.path.splitext(p)[1].lower() in SUPPORTED_EXTENSIONS
                )
                if matches:
                    image_paths = matches
                    break
            except Exception:
                continue

        for image_path in image_paths:
            if not os.path.isfile(image_path):
                continue

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

        try:
            mask_volume = normalize_mask(safe_np_load(mask_path))
        except Exception as exc:
            raise RuntimeError(f"Failed to load mask for patient {patient_id}: {mask_path}") from exc

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
            raise RuntimeError(
                f"No valid mask slices found for patient {patient_id} in file {mask_path}."
            )

        try:
            image_volume = safe_np_load(image_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load image for patient {patient_id}: {image_path}") from exc

        if slice_index >= image_volume.shape[0]:
            raise RuntimeError(
                f"Slice index {slice_index} out of bounds for image {image_path} with shape {image_volume.shape}"
            )

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
