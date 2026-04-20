import glob
import os
from typing import Optional, Sequence

import numpy as np

from scan_io import load_scan

SUPPORTED_EXTENSIONS = (".npy", ".npz", ".mhd")


def load_numpy_or_mhd(path: str) -> np.ndarray:
    """
    Load a volume from a NumPy or MHD file.

    Supports .npy (NumPy array), .npz (compressed NumPy archive), and .mhd (MetaImage format).
    For .mhd files, uses the scan_io module to load the volume.

    Parameters
    ----------
    path : str
        Path to the file to load.

    Returns
    -------
    np.ndarray
        The loaded volume array.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mhd":
        # Load MHD file using SimpleITK via scan_io
        volume, _, _ = load_scan(path)
        return volume

    if ext == ".npz":
        # Load compressed NumPy archive, use first array or 'arr_0'
        with np.load(path) as archive:
            if "arr_0" in archive:
                return archive["arr_0"]
            return next(iter(archive.values()))

    if ext == ".npy":
        # Load standard NumPy array
        return np.load(path)

    raise ValueError(
        f"Unsupported file extension for dataset loading: {path!r}. "
        f"Supported extensions are: {SUPPORTED_EXTENSIONS}"
    )


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Normalize a mask array to binary uint8 format.

    Ensures the mask is in a consistent binary format for segmentation tasks.
    Converts boolean arrays to uint8, and thresholds any multi-value masks to 0/1.

    Parameters
    ----------
    mask : np.ndarray
        Input mask array.

    Returns
    -------
    np.ndarray
        Normalized binary mask as uint8.
    """
    if mask.dtype == np.bool_:
        # Convert boolean to uint8
        return mask.astype(np.uint8)

    if mask.ndim == 0:
        # Handle scalar input
        return np.asarray(mask, dtype=np.uint8)

    if np.max(mask) > 1:
        # Threshold to binary if values exceed 1
        mask = mask > 0
    return mask.astype(np.uint8)


def patient_id_from_filename(path: str) -> str:
    """
    Extract patient ID from a filename.

    Assumes the patient ID is the part before the first underscore in the filename stem.
    For example, '1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260_000.npy'
    becomes '1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260'.

    Parameters
    ----------
    path : str
        Path to the file.

    Returns
    -------
    str
        Extracted patient ID.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_")[0]


def find_matching_mask(image_path: str, mask_dir: str, extensions: Sequence[str] = SUPPORTED_EXTENSIONS) -> Optional[str]:
    """
    Find a matching mask file for a given image file.

    Looks for a file in mask_dir with the same stem as the image file,
    trying each supported extension in order.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    mask_dir : str
        Directory containing mask files.
    extensions : Sequence[str], optional
        File extensions to try, by default SUPPORTED_EXTENSIONS.

    Returns
    -------
    Optional[str]
        Path to the matching mask file, or None if not found.
    """
    image_stem = os.path.splitext(os.path.basename(image_path))[0]
    for ext in extensions:
        candidate = os.path.join(mask_dir, image_stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None
