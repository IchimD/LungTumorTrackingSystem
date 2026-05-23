import argparse
import json
import os
import random
from typing import List

from src.data.io import SUPPORTED_EXTENSIONS, find_matching_mask, patient_id_from_filename


def collect_patient_ids(image_dir: str, mask_dir: str) -> List[str]:
    """Collect all patient IDs that have matching image/mask pairs."""
    image_paths = sorted(
        p
        for p in os.listdir(image_dir)
        if os.path.splitext(p)[1].lower() in SUPPORTED_EXTENSIONS
    )
    patient_ids = []
    for filename in image_paths:
        image_path = os.path.join(image_dir, filename)
        if find_matching_mask(image_path, mask_dir) is not None:
            patient_ids.append(patient_id_from_filename(image_path))
    return sorted(set(patient_ids))


def create_splits(
    patient_ids: List[str],
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Create train/val/test splits by patient (stratified).

    Parameters
    ----------
    patient_ids : List[str]
        All patient IDs.
    train_fraction : float
        Fraction for training (default 0.7).
    val_fraction : float
        Fraction for validation (default 0.15).
    test_fraction : float
        Fraction for testing (default 0.15).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys: 'train', 'val', 'test', each containing patient ID lists.
    """
    assert abs((train_fraction + val_fraction + test_fraction) - 1.0) < 1e-6, \
        "train_fraction + val_fraction + test_fraction must sum to 1.0"

    random.Random(seed).shuffle(patient_ids)

    n = len(patient_ids)
    train_end = int(n * train_fraction)
    val_end = train_end + int(n * val_fraction)

    return {
        "train": patient_ids[:train_end],
        "val": patient_ids[train_end:val_end],
        "test": patient_ids[val_end:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and save train/val/test patient splits.")
    parser.add_argument("--image_dir", required=True, help="Directory with preprocessed image volumes.")
    parser.add_argument("--mask_dir", required=True, help="Directory with preprocessed mask volumes.")
    parser.add_argument("--output_dir", default="data/splits", help="Directory to save split JSON files.")
    parser.add_argument("--train_fraction", type=float, default=0.7, help="Fraction of patients for training.")
    parser.add_argument("--val_fraction", type=float, default=0.15, help="Fraction of patients for validation.")
    parser.add_argument("--test_fraction", type=float, default=0.15, help="Fraction of patients for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Collecting patient IDs from {args.image_dir} and {args.mask_dir}...")
    patient_ids = collect_patient_ids(args.image_dir, args.mask_dir)
    print(f"Found {len(patient_ids)} patients with image/mask pairs.")

    print(f"Creating splits: {args.train_fraction:.0%} train / {args.val_fraction:.0%} val / {args.test_fraction:.0%} test...")
    splits = create_splits(
        patient_ids,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    print(f"  Train: {len(splits['train'])} patients")
    print(f"  Val:   {len(splits['val'])} patients")
    print(f"  Test:  {len(splits['test'])} patients")

    output_file = os.path.join(args.output_dir, "patient_splits.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"\nSplits saved to {output_file}")


if __name__ == "__main__":
    main()
