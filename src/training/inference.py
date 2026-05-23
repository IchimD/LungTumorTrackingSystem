import argparse
import os
from typing import Optional

import numpy as np
import torch

from src.models.unet import UNet2D
from src.data.io import load_numpy_or_mhd, save_volume


def load_checkpoint(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNet2D(in_channels=1, out_channels=1, base_channels=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def predict_volume(
    model: torch.nn.Module,
    volume: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Apply model to all slices of a 3D volume and return binary mask volume.

    Parameters
    ----------
    model : torch.nn.Module
        Trained U-Net model.
    volume : np.ndarray
        Input volume of shape (Z, Y, X).
    device : torch.device
        Compute device.
    threshold : float
        Probability threshold for binary predictions.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    np.ndarray
        Binary mask volume of same shape as input.
    """
    model.eval()
    mask_volume = np.zeros_like(volume, dtype=np.uint8)

    with torch.no_grad():
        for z_start in range(0, volume.shape[0], batch_size):
            z_end = min(z_start + batch_size, volume.shape[0])
            slices = volume[z_start:z_end]

            slices_tensor = torch.from_numpy(slices.astype(np.float32)).unsqueeze(1).to(device)
            logits = model(slices_tensor)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).cpu().numpy().astype(np.uint8)

            mask_volume[z_start:z_end] = preds[:, 0, :, :]

    return mask_volume


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on preprocessed volumes using a trained U-Net.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt file).")
    parser.add_argument("--image_path", required=True, help="Path to preprocessed image volume (.npy).")
    parser.add_argument("--output_path", required=True, help="Path to save predicted mask volume (.npy).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary predictions.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint from {args.checkpoint}...")
    model = load_checkpoint(args.checkpoint, device)

    print(f"Loading volume from {args.image_path}...")
    volume = load_numpy_or_mhd(args.image_path)
    print(f"  Volume shape: {volume.shape}")
    print(f"  Volume dtype: {volume.dtype}")
    print(f"  Value range: [{volume.min():.4f}, {volume.max():.4f}]")

    print(f"Running inference with threshold={args.threshold}...")
    mask_volume = predict_volume(model, volume, device, threshold=args.threshold, batch_size=args.batch_size)

    print(f"Saving mask to {args.output_path}...")
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    save_volume(args.output_path, mask_volume)
    print(f"  Mask shape: {mask_volume.shape}")
    print(f"  Mask dtype: {mask_volume.dtype}")
    print(f"  Mask value range: [{mask_volume.min()}, {mask_volume.max()}]")
    print("Done!")


if __name__ == "__main__":
    main()
