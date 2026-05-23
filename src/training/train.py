import argparse
import json
import os
import random
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.augmentation import default_training_augmentations
from src.data.utils import progress_iter
from src.models.unet import UNet2D
from src.training.dataset import VolumeSliceDataset
from src.training.loss import BCEDiceLoss
from src.training.metrics import dice_score, iou_score, precision_score, sensitivity_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_patient_ids(patient_ids: List[str], val_fraction: float, seed: int) -> tuple[List[str], List[str]]:
    random.Random(seed).shuffle(patient_ids)
    val_count = max(1, int(len(patient_ids) * val_fraction))
    return patient_ids[val_count:], patient_ids[:val_count]


def collect_patient_ids(image_dir: str, mask_dir: str) -> List[str]:
    from src.data.io import SUPPORTED_EXTENSIONS, find_matching_mask, patient_id_from_filename

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


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, masks, _ in progress_iter(dataloader, desc="Training"):  # type: ignore[arg-type]
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())

    return total_loss / len(dataloader)


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_sensitivity = 0.0
    total_precision = 0.0
    count = 0

    with torch.no_grad():
        for images, masks, _ in progress_iter(dataloader, desc="Validation"):  # type: ignore[arg-type]
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += float(loss.detach().cpu())
            total_dice += dice_score(outputs, masks)
            total_iou += iou_score(outputs, masks)
            total_sensitivity += sensitivity_score(outputs, masks)
            total_precision += precision_score(outputs, masks)
            count += 1

    return {
        "loss": total_loss / max(count, 1),
        "dice": total_dice / max(count, 1),
        "iou": total_iou / max(count, 1),
        "sensitivity": total_sensitivity / max(count, 1),
        "precision": total_precision / max(count, 1),
    }


def build_augmentation_fn(enable: bool):
    if not enable:
        return None

    from src.data.augmentation import default_training_augmentations

    def augmentation(image: torch.Tensor, mask: torch.Tensor):
        return default_training_augmentations(image, mask)

    return augmentation


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 2D U-Net segmentation network on preprocessed volumes.")
    parser.add_argument("--image_dir", required=True, help="Directory with preprocessed image volumes.")
    parser.add_argument("--mask_dir", required=True, help="Directory with preprocessed mask volumes.")
    parser.add_argument("--output_dir", required=True, help="Directory where checkpoints and logs will be saved.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--val_fraction", type=float, default=0.15, help="Fraction of patients used for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers.")
    parser.add_argument("--augment", action="store_true", help="Enable light data augmentation during training.")
    parser.add_argument("--resume", action="store_true", help="Resume training from best checkpoint in output_dir.")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    patient_ids = collect_patient_ids(args.image_dir, args.mask_dir)
    if not patient_ids:
        raise ValueError("No patient volumes found in the provided image/mask directories.")

    train_ids, val_ids = split_patient_ids(patient_ids, args.val_fraction, args.seed)
    print(f"Using {len(train_ids)} patients for training and {len(val_ids)} patients for validation.")

    train_dataset = VolumeSliceDataset(
        args.image_dir,
        args.mask_dir,
        patient_ids=train_ids,
        augment_fn=build_augmentation_fn(args.augment),
    )
    val_dataset = VolumeSliceDataset(
        args.image_dir,
        args.mask_dir,
        patient_ids=val_ids,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(in_channels=1, out_channels=1, base_channels=32).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_dice = 0.0
    start_epoch = 1
    
    # Resume from checkpoint if requested
    checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_dice = checkpoint.get("val_dice", 0.0)
        start_epoch = checkpoint.get("epoch", 1) + 1
        print(f"Resumed from epoch {checkpoint.get('epoch', 0)}, best_val_dice: {best_val_dice:.4f}")
    
    split_path = os.path.join(args.output_dir, "patient_split.json")
    with open(split_path, "w", encoding="utf-8") as handle:
        json.dump({"train": train_ids, "val": val_ids}, handle, indent=2)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        stats = evaluate(model, val_loader, criterion, device)

        print(f"  train_loss: {train_loss:.4f}")
        print(f"  val_loss: {stats['loss']:.4f}  dice: {stats['dice']:.4f}  iou: {stats['iou']:.4f}")
        print(f"  sensitivity: {stats['sensitivity']:.4f}  precision: {stats['precision']:.4f}")

        if stats["dice"] > best_val_dice:
            best_val_dice = stats["dice"]
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": best_val_dice,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"  Saved best checkpoint to {checkpoint_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
