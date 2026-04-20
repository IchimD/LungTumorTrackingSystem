import math
import os
from typing import Sequence, Tuple

import numpy as np
import SimpleITK as sitk


def normalize_hu(scan: np.ndarray, hu_min: float = -1000.0, hu_max: float = 400.0) -> np.ndarray:
    """
    Window and normalize CT scan intensities to [0, 1].

    Clips Hounsfield Units to the specified window and scales to [0, 1].
    This is a common preprocessing step for lung CT scans.

    Parameters
    ----------
    scan : np.ndarray
        Input CT scan array.
    hu_min : float, optional
        Lower HU bound, by default -1000.0.
    hu_max : float, optional
        Upper HU bound, by default 400.0.

    Returns
    -------
    np.ndarray
        Normalized scan array.
    """
    scan = np.clip(scan, hu_min, hu_max)
    scan = (scan - hu_min) / (hu_max - hu_min)
    return scan.astype(np.float32)


def resample_volume(
    volume: np.ndarray,
    spacing: Sequence[float],
    output_spacing: Sequence[float],
    interpolator: int = sitk.sitkLinear,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a 3D volume to a new voxel spacing using SimpleITK.

    Parameters
    ----------
    volume : np.ndarray
        Input 3D volume.
    spacing : Sequence[float]
        Current voxel spacing (Z, Y, X).
    output_spacing : Sequence[float]
        Desired voxel spacing (Z, Y, X).
    interpolator : int, optional
        SimpleITK interpolator, by default sitk.sitkLinear.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Resampled volume and updated spacing.
    """
    spacing = np.asarray(spacing, dtype=np.float64)
    output_spacing = np.asarray(output_spacing, dtype=np.float64)

    # Create SimpleITK image with correct spacing
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing(tuple(reversed(spacing.tolist())))

    original_size = image.GetSize()  # (x, y, z)
    original_spacing = image.GetSpacing()  # (x, y, z)
    output_spacing_sitk = tuple(reversed(output_spacing.tolist()))

    # Calculate new size to maintain physical dimensions
    new_size = [
        int(math.ceil(osz * osp / nsp))
        for osz, osp, nsp in zip(original_size, original_spacing, output_spacing_sitk)
    ]

    # Perform resampling
    resampled = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),  # Identity transform
        interpolator,
        image.GetOrigin(),
        output_spacing_sitk,
        image.GetDirection(),
        0.0,  # Default pixel value
        image.GetPixelID(),
    )

    resampled_volume = sitk.GetArrayFromImage(resampled).astype(np.float32)
    return resampled_volume, output_spacing


def save_volume(path: str, volume: np.ndarray) -> None:
    """
    Save a NumPy array to a .npy file, creating directories if needed.

    Parameters
    ----------
    path : str
        Output file path.
    volume : np.ndarray
        Array to save.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, volume)
