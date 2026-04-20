import glob
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import (
    SUPPORTED_EXTENSIONS,
    find_matching_mask,
    load_numpy_or_mhd,
    normalize_mask,
    patient_id_from_filename,
)


class LUNA16Dataset(Dataset):
    """PyTorch dataset for preprocessed LUNA16 images and segmentation masks.

    Loads pairs of image and mask files from specified directories.
    Supports filtering by patient IDs and optional data transforms.
    Can preload all data into memory for faster access during training.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        patient_ids: Optional[Sequence[str]] = None,
        transform=None,
        target_transform=None,
        preload: bool = False,
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        image_dir : str
            Directory containing image files.
        mask_dir : str
            Directory containing mask files.
        patient_ids : Optional[Sequence[str]], optional
            List of patient IDs to include, by default None (all patients).
        transform : optional
            Transform to apply to images, by default None.
        target_transform : optional
            Transform to apply to masks, by default None.
        preload : bool, optional
            Whether to load all data into memory at initialization, by default False.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        self.preload = preload

        self._items: List[Tuple[str, str, str]] = []
        self._preloaded_data: List[Tuple[np.ndarray, np.ndarray, str]] = []

        self._discover_pairs(patient_ids)

        if not self._items:
            raise ValueError(
                f"No image/mask pairs were found in {image_dir!r} / {mask_dir!r}."
            )

        if self.preload:
            self._preload_all()

    def _discover_pairs(self, patient_ids: Optional[Sequence[str]]) -> None:
        """
        Discover and pair image and mask files.

        Scans image_dir for supported files, filters by patient_ids if provided,
        and finds matching mask files in mask_dir.

        Parameters
        ----------
        patient_ids : Optional[Sequence[str]]
            List of patient IDs to include, or None for all.
        """
        image_paths = sorted(
            p
            for p in glob.glob(os.path.join(self.image_dir, "*"))
            if os.path.splitext(p)[1].lower() in SUPPORTED_EXTENSIONS
        )

        allowed = set(patient_ids) if patient_ids is not None else None
        for image_path in image_paths:
            patient_id = patient_id_from_filename(image_path)
            if allowed is not None and patient_id not in allowed:
                continue

            mask_path = find_matching_mask(image_path, self.mask_dir)
            if mask_path is None:
                continue

            self._items.append((image_path, mask_path, patient_id))

    def _preload_all(self) -> None:
        """
        Preload all image and mask data into memory.

        Loads all pairs into self._preloaded_data for faster access.
        """
        self._preloaded_data = []
        for image_path, mask_path, patient_id in self._items:
            image = load_numpy_or_mhd(image_path)
            mask = normalize_mask(load_numpy_or_mhd(mask_path))
            self._preloaded_data.append((image, mask, patient_id))

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of image/mask pairs.
        """
        return len(self._items)

    def __getitem__(self, index: int):
        """
        Get a sample from the dataset.

        Loads the image and mask, converts to tensors, adds channel dimensions,
        and applies transforms.

        Parameters
        ----------
        index : int
            Index of the sample.

        Returns
        -------
        tuple
            (image_tensor, mask_tensor, patient_id)
        """
        if self.preload:
            image, mask, patient_id = self._preloaded_data[index]
        else:
            image_path, mask_path, patient_id = self._items[index]
            image = load_numpy_or_mhd(image_path)
            mask = normalize_mask(load_numpy_or_mhd(mask_path))

        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.uint8))

        # Add channel dimension if needed (e.g., for 2D slices or 3D volumes)
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.ndim == 3 and image_tensor.shape[0] != 1:
            image_tensor = image_tensor.unsqueeze(0)

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        elif mask_tensor.ndim == 3 and mask_tensor.shape[0] != 1:
            mask_tensor = mask_tensor.unsqueeze(0)

        # Apply transforms if provided
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        if self.target_transform is not None:
            mask_tensor = self.target_transform(mask_tensor)

        return image_tensor, mask_tensor, patient_id

    def get_patient_ids(self) -> List[str]:
        """
        Return the list of patient IDs in the dataset.

        Returns
        -------
        List[str]
            List of patient IDs.
        """
        return [patient_id for _, _, patient_id in self._items]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect a preprocessed LUNA16 image/mask dataset."
    )
    parser.add_argument("--image_dir", required=True, help="Image directory")
    parser.add_argument("--mask_dir", required=True, help="Mask directory")
    parser.add_argument(
        "--patient_id",
        default=None,
        help="Optional patient ID to filter the dataset",
    )
    args = parser.parse_args()

    patient_ids = [args.patient_id] if args.patient_id else None
    dataset = LUNA16Dataset(
        args.image_dir,
        args.mask_dir,
        patient_ids=patient_ids,
        preload=False,
    )

    print(f"Found {len(dataset)} image/mask pairs.")
    print("Patient IDs:")
    for pid in sorted(set(dataset.get_patient_ids())):
        print(f"  {pid}")

    sample_image, sample_mask, sample_patient = dataset[0]
    print(f"Sample patient: {sample_patient}")
    print(f"Image tensor shape: {tuple(sample_image.shape)}")
    print(f"Mask tensor shape: {tuple(sample_mask.shape)}")
    print(f"Image dtype: {sample_image.dtype}")
    print(f"Mask dtype: {sample_mask.dtype}")


if __name__ == "__main__":
    main()
