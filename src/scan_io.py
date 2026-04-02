import os
import glob
import numpy as np
import SimpleITK as sitk


def list_all_patients(subset_path: str) -> list[str]:
    """
    Scan the subset folder and return every patient's series UID.

    How it works:
    - Uses glob to find all files ending in .mhd inside subset_path.
    - Strips the directory and the .mhd extension from each filename,
      leaving only the seriesuid (the long numeric string that uniquely
      identifies a CT scan in the LUNA16 dataset).
    - Returns the list sorted so iteration order is deterministic.

    Change from v1:
    - Raises ValueError if subset_path does not exist, giving an
      actionable error message instead of silently returning an empty list.

    Parameters
    ----------
    subset_path : str
        Path to a subset folder, e.g. 'D:/LICENTA2/DATASET/subset0'.

    Returns
    -------
    list[str]
        Sorted list of seriesuid strings, one per patient scan.

    Example
    -------
    >>> patients = list_all_patients(r'D:/LICENTA2/DATASET/subset0')
    >>> print(len(patients))   # 89
    >>> print(patients[0])     # 1.3.6.1.4.1.14519...
    """
    if not os.path.isdir(subset_path):
        raise ValueError(f"Subset directory not found: {subset_path!r}")

    # Build the search pattern and get all matching file paths
    pattern   = os.path.join(subset_path, "*.mhd")
    mhd_files = sorted(glob.glob(pattern))

    # Extract just the stem (filename without extension) as the patient ID
    patient_ids = [os.path.splitext(os.path.basename(f))[0] for f in mhd_files]

    return patient_ids


def load_scan(mhd_filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single CT scan from a .mhd / .raw file pair.

    How it works:
    - SimpleITK reads the .mhd header, which points to the paired .raw
      binary file containing the actual voxel data.
    - The voxel array is extracted as a NumPy array. SimpleITK stores
      axes in (X, Y, Z) order, but GetArrayFromImage transposes to
      (Z, Y, X), which is standard for NumPy medical imaging work.
    - Spacing and origin are reversed from ITK's (X, Y, Z) convention
      to (Z, Y, X) to stay consistent with the array.

    Changes from v1:
    - Raises ValueError if mhd_filepath does not exist, replacing a
      cryptic C-level ITK error with an actionable message.
    - Volume is explicitly cast to np.float32 for a consistent downstream
      dtype. SimpleITK can return int16, int32, or float64 depending on
      file metadata; inconsistent dtypes cause hard-to-trace bugs in
      training pipelines.
    - spacing and origin are now returned as np.ndarray (shape (3,))
      instead of plain tuples, enabling vectorised arithmetic in
      preprocessing / augmentation code without manual axis unpacking.

    Parameters
    ----------
    mhd_filepath : str
        Absolute path to the .mhd file.

    Returns
    -------
    volume : np.ndarray, dtype float32
        3D array of HU (Hounsfield Unit) values, shape (Z, Y, X).
    spacing : np.ndarray, shape (3,), dtype float64
        Voxel size in mm, ordered (Z, Y, X).
    origin : np.ndarray, shape (3,), dtype float64
        World coordinates of voxel [0, 0, 0], ordered (Z, Y, X).

    Example
    -------
    >>> volume, spacing, origin = load_scan(r'D:/LICENTA2/DATASET/subset0/1.3...mhd')
    >>> print(volume.shape)    # e.g. (133, 512, 512)
    >>> print(volume.dtype)    # float32
    >>> print(spacing)         # e.g. [2.5   0.703 0.703]
    """
    if not os.path.isfile(mhd_filepath):
        raise ValueError(f"MHD file not found: {mhd_filepath!r}")

    # Read the .mhd file; SimpleITK automatically locates and loads the .raw companion
    image = sitk.ReadImage(mhd_filepath)

    # Convert to float32 NumPy array — axes become (Z, Y, X)
    volume = sitk.GetArrayFromImage(image).astype(np.float32)

    # ITK returns (X, Y, Z); list(reversed(...)) converts to (Z, Y, X).
    # np.ndarray enables downstream vectorised arithmetic without manual unpacking.
    spacing = np.array(list(reversed(image.GetSpacing())), dtype=np.float64)
    origin  = np.array(list(reversed(image.GetOrigin())),  dtype=np.float64)

    return volume, spacing, origin
