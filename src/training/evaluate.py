import argparse
import json
import os
from typing import List

import torch
from torch.utils.data import DataLoader

from src.data.utils import progress_iter
from src.models.unet import UNet2D
from src.training.dataset import VolumeSliceDataset
from src.training.metrics import dice_score, iou_score, precision_score, sensitivity_score


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    """Load model checkpoint and return model + metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNet2D(in_channels=1, out_channels=1, base_channels=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return {"model": model, "checkpoint": checkpoint}


def evaluate_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Evaluate model on a dataset and return per-batch and aggregate metrics."""
    model.eval()
    results = {
        "dice_scores": [],
        "iou_scores": [],
        "sensitivity_scores": [],
        "precision_scores": [],
        "patient_ids": [],
    }

    with torch.no_grad():
        for images, masks, patient_ids in progress_iter(dataloader, desc="Evaluating"):  # type: ignore[arg-type]
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            outputs = model(images)

            for i in range(outputs.shape[0]):
                logit = outputs[i : i + 1]
                mask = masks[i : i + 1]
                results["dice_scores"].append(float(dice_score(logit, mask, threshold)))
                results["iou_scores"].append(float(iou_score(logit, mask, threshold)))
                results["sensitivity_scores"].append(float(sensitivity_score(logit, mask, threshold)))
                results["precision_scores"].append(float(precision_score(logit, mask, threshold)))
                results["patient_ids"].append(patient_ids[i])

    return results


def summarize_results(results: dict) -> dict:
    """Compute aggregate statistics from per-batch results."""
    import numpy as np

    if not results["dice_scores"]:
        return {}

    return {
        "num_slices": len(results["dice_scores"]),
        "dice_mean": float(np.mean(results["dice_scores"])),
        "dice_std": float(np.std(results["dice_scores"])),
        "iou_mean": float(np.mean(results["iou_scores"])),
        "iou_std": float(np.std(results["iou_scores"])),
        "sensitivity_mean": float(np.mean(results["sensitivity_scores"])),
        "sensitivity_std": float(np.std(results["sensitivity_scores"])),
        "precision_mean": float(np.mean(results["precision_scores"])),
        "precision_std": float(np.std(results["precision_scores"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net model on preprocessed volumes.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt file).")
    parser.add_argument("--image_dir", required=True, help="Directory with preprocessed image volumes.")
    parser.add_argument("--mask_dir", required=True, help="Directory with preprocessed mask volumes.")
    parser.add_argument("--output_dir", required=True, help="Directory to save evaluation results.")
    parser.add_argument("--patient_ids", nargs="*", default=None, help="Optional list of patient IDs to evaluate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary predictions.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint from {args.checkpoint}...")
    loaded = load_checkpoint(args.checkpoint, device)
    model = loaded["model"]

    print(f"Loading dataset from {args.image_dir} / {args.mask_dir}...")
    dataset = VolumeSliceDataset(
        args.image_dir,
        args.mask_dir,
        patient_ids=args.patient_ids,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Evaluating {len(dataset)} slices...")
    results = evaluate_dataset(model, dataloader, device, threshold=args.threshold)
    summary = summarize_results(results)

    print("\n=== Evaluation Results ===")
    print(f"Evaluated {summary['num_slices']} slices")
    print(f"Dice:        {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}")
    print(f"IoU:         {summary['iou_mean']:.4f} ± {summary['iou_std']:.4f}")
    print(f"Sensitivity: {summary['sensitivity_mean']:.4f} ± {summary['sensitivity_std']:.4f}")
    print(f"Precision:   {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")

    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_slice_results": results}, f, indent=2)
    print(f"\nDetailed results saved to {results_file}")


if __name__ == "__main__":
    main()
