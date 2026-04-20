import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .mask import create_nodule_mask
from .utils import parse_spacing, progress_iter
from .volume import normalize_hu, resample_volume, save_volume
from annotation_io import build_patient_index, load_annotations
from scan_io import list_all_patients, load_scan


def preprocess_patient(
    mhd_path: str,
    nodules: Sequence[dict],
    output_image_path: str,
    output_mask_path: str,
    output_spacing: Optional[Sequence[float]] = None,
    hu_window: Tuple[float, float] = (-1000.0, 400.0),
    overwrite: bool = False,
) -> None:
    """
    Preprocess a single patient scan.

    Loads the MHD scan, optionally resamples, creates mask, normalizes HU,
    and saves image and mask.

    Parameters
    ----------
    mhd_path : str
        Path to the .mhd file.
    nodules : Sequence[dict]
        Nodule annotations for this patient.
    output_image_path : str
        Output path for normalized image.
    output_mask_path : str
        Output path for mask.
    output_spacing : Optional[Sequence[float]], optional
        Target spacing, by default None.
    hu_window : Tuple[float, float], optional
        HU window bounds, by default (-1000.0, 400.0).
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    """
    if os.path.exists(output_image_path) and os.path.exists(output_mask_path) and not overwrite:
        return

    volume, spacing, origin = load_scan(mhd_path)
    if output_spacing is not None and not np.allclose(spacing, output_spacing):
        volume, spacing = resample_volume(volume, spacing, output_spacing)

    mask = create_nodule_mask(volume.shape, nodules, spacing, origin)
    volume = normalize_hu(volume, *hu_window)

    save_volume(output_image_path, volume)
    save_volume(output_mask_path, mask)


def preprocess_subset(
    subset_path: str,
    annotations_csv: str,
    output_dir: str,
    output_spacing: Optional[Sequence[float]] = None,
    hu_window: Tuple[float, float] = (-1000.0, 400.0),
    overwrite: bool = False,
    patient_ids: Optional[Sequence[str]] = None,
) -> None:
    """
    Preprocess all patients in a LUNA16 subset.

    Parameters
    ----------
    subset_path : str
        Path to subset directory with .mhd files.
    annotations_csv : str
        Path to annotations CSV.
    output_dir : str
        Output directory for images/ and masks/.
    output_spacing : Optional[Sequence[float]], optional
        Target spacing, by default None.
    hu_window : Tuple[float, float], optional
        HU window, by default (-1000.0, 400.0).
    overwrite : bool, optional
        Overwrite existing, by default False.
    patient_ids : Optional[Sequence[str]], optional
        Specific patients to process, by default None (all).
    """
    annotations = load_annotations(annotations_csv)
    index = build_patient_index(annotations)

    patients = list_all_patients(subset_path)
    if patient_ids is not None:
        allowed = set(patient_ids)
        patients = [pid for pid in patients if pid in allowed]

    image_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")

    iterator: Iterable[str] = progress_iter(patients, desc="Preprocessing")
    for patient_id in iterator:
        mhd_path = os.path.join(subset_path, f"{patient_id}.mhd")
        image_path = os.path.join(image_dir, f"{patient_id}.npy")
        mask_path = os.path.join(mask_dir, f"{patient_id}.npy")

        nodules = index.get(patient_id, [])
        preprocess_patient(
            mhd_path,
            nodules,
            image_path,
            mask_path,
            output_spacing=output_spacing,
            hu_window=hu_window,
            overwrite=overwrite,
        )


def main() -> None:
    """
    Command-line interface for preprocessing.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess LUNA16 scans into intensity-normalised images and binary nodule masks."
    )
    parser.add_argument("--subset", required=True, help="Path to a LUNA16 subset folder containing .mhd files.")
    parser.add_argument("--csv", required=True, help="Path to annotations.csv.")
    parser.add_argument("--output_dir", required=True, help="Directory to save preprocessed images and masks.")
    parser.add_argument(
        "--output_spacing",
        default="1,1,1",
        type=parse_spacing,
        help="Desired output voxel spacing in mm (one value or three comma-separated values).",
    )
    parser.add_argument(
        "--hu_min",
        default=-1000.0,
        type=float,
        help="Lower bound of the HU window used for normalization.",
    )
    parser.add_argument(
        "--hu_max",
        default=400.0,
        type=float,
        help="Upper bound of the HU window used for normalization.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing preprocessed files if they already exist.",
    )
    parser.add_argument(
        "--patient_ids",
        nargs="*",
        default=None,
        help="Optional list of patient IDs to preprocess. If not provided, all patients in the subset are processed.",
    )
    args = parser.parse_args()

    preprocess_subset(
        args.subset,
        args.csv,
        args.output_dir,
        output_spacing=args.output_spacing,
        hu_window=(args.hu_min, args.hu_max),
        overwrite=args.overwrite,
        patient_ids=args.patient_ids,
    )


if __name__ == "__main__":
    main()
