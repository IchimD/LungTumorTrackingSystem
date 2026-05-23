import argparse
import json
import os
import random
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.utils import make_grid
from tqdm import tqdm

from src.models.unet import UNet2D
from src.training.dataset import VolumeSliceDataset
from src.training.loss import BCEDiceLoss
from src.training.metrics import dice_score, iou_score, precision_score, sensitivity_score
from src.data.augmentation import default_training_augmentations


class SafeVolumeSliceDataset(VolumeSliceDataset):
    """Wrapper dataset that skips corrupted files encountered during loading.

    If a file load fails we remove it from the internal item list and retry.
    """

    def __getitem__(self, index: int):
        attempts = 0
        while attempts < 10:
            if not self._items:
                raise RuntimeError("No valid items left in dataset (all files corrupted).")
            index = index % len(self._items)
            try:
                return super().__getitem__(index)
            except Exception as exc:  # pragma: no cover - runtime guard
                # drop problematic item and try another
                try:
                    del self._items[index]
                except Exception:
                    pass
                attempts += 1
        raise RuntimeError("Too many corrupted files encountered when loading dataset.")


def build_augmentation_fn(enable: bool):
    if not enable:
        return None

    import torch

    def augmentation(image: torch.Tensor, mask: torch.Tensor):
        # reuse existing augmentations (flip + intensity jitter)
        image, mask = default_training_augmentations(image, mask)

        # random 90-degree rotations
        k = random.choice([0, 1, 2, 3])
        if k != 0:
            image = torch.rot90(image, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])

        return image, mask

    return augmentation


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for images, masks, _ in pbar:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        pbar.set_postfix(loss=total_loss / (pbar.n + 1e-8))

    return total_loss / max(len(dataloader), 1)


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_sensitivity = 0.0
    total_precision = 0.0
    count = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Val", leave=False)
        for images, masks, _ in pbar:
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


def log_sample_images(writer: SummaryWriter, model: torch.nn.Module, dataset: VolumeSliceDataset, device: torch.device, epoch: int, tag: str = "val_samples", max_images: int = 4):
    model.eval()
    imgs = []
    gts = []
    preds = []
    indices = list(range(len(dataset)))
    if not indices:
        return
    random.shuffle(indices)
    with torch.no_grad():
        for idx in indices[:max_images]:
            try:
                image, mask, _ = dataset[idx]
            except Exception:
                continue
            image = image.to(device, dtype=torch.float32).unsqueeze(0)
            mask = mask.to(device, dtype=torch.float32).unsqueeze(0)
            out = model(image)
            prob = torch.sigmoid(out)
            pred = (prob > 0.5).float()

            # normalize image to [0,1] for logging
            img = image[0]
            if img.max() > 0:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            imgs.append(img)
            gts.append(mask[0])
            preds.append(pred[0])

    if not imgs:
        return

    # create a grid with rows = samples, cols = [image, gt, pred]
    rows = []
    for i in range(len(imgs)):
        rows.append(torch.cat([imgs[i], gts[i], preds[i]], dim=2) if imgs[i].ndim == 3 else torch.cat([imgs[i], gts[i], preds[i]], dim=1))

    grid = make_grid(torch.stack(rows, dim=0), nrow=1, normalize=False, pad_value=1.0)
    writer.add_image(tag, grid, epoch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab GPU training for 2D U-Net segmentation.")
    parser.add_argument("--image_dir", default="/content/drive/My Drive/LICENTA_COLAB ", help="Directory with preprocessed image volumes.")
    parser.add_argument("--mask_dir", default="/content/drive/My Drive/LICENTA_COLAB/masks_rclone", help="Directory with preprocessed mask volumes.")
    parser.add_argument("--logs_dir", default="/content/drive/My Drive/LICENTA_COLAB/logs", help="TensorBoard logs directory.")
    parser.add_argument("--results_dir", default="/content/drive/My Drive/LICENTA_COLAB/results", help="Directory for saving results and images.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--val_fraction", type=float, default=0.15, help="Fraction of patients used for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers.")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation (flip, rotate, intensity jitter).")
    parser.add_argument("--resume", action="store_true", help="Resume training from best checkpoint in output_dir.")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # collect patients similar to original training script
    from src.training.train import collect_patient_ids, split_patient_ids

    patient_ids = collect_patient_ids(args.image_dir, args.mask_dir)
    if not patient_ids:
        raise ValueError("No patient volumes found in the provided image/mask directories.")

    train_ids, val_ids = split_patient_ids(patient_ids, args.val_fraction, args.seed)
    print(f"Using {len(train_ids)} patients for training and {len(val_ids)} patients for validation.")

    train_dataset = SafeVolumeSliceDataset(
        args.image_dir,
        args.mask_dir,
        patient_ids=train_ids,
        augment_fn=build_augmentation_fn(args.augment),
    )
    val_dataset = SafeVolumeSliceDataset(
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

    writer = SummaryWriter(log_dir=args.logs_dir)

    best_val_dice = 0.0
    start_epoch = 1

    checkpoint_path = os.path.join(args.results_dir, "best_model.pt")
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore[index]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # type: ignore[index]
        best_val_dice = checkpoint.get("val_dice", 0.0)
        start_epoch = checkpoint.get("epoch", 1) + 1
        print(f"Resumed from epoch {checkpoint.get('epoch', 0)}, best_val_dice: {best_val_dice:.4f}")

    split_path = os.path.join(args.results_dir, "patient_split.json")
    with open(split_path, "w", encoding="utf-8") as handle:
        json.dump({"train": train_ids, "val": val_ids}, handle, indent=2)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        stats = evaluate(model, val_loader, criterion, device)

        lr = optimizer.param_groups[0]["lr"]

        print(f"  train_loss: {train_loss:.4f}")
        print(f"  val_loss: {stats['loss']:.4f}  dice: {stats['dice']:.4f}  iou: {stats['iou']:.4f}")
        print(f"  sensitivity: {stats['sensitivity']:.4f}  precision: {stats['precision']:.4f}")

        # TensorBoard scalars
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", stats["loss"], epoch)
        writer.add_scalar("val/dice", stats["dice"], epoch)
        writer.add_scalar("val/iou", stats["iou"], epoch)
        writer.add_scalar("val/sensitivity", stats["sensitivity"], epoch)
        writer.add_scalar("val/precision", stats["precision"], epoch)
        writer.add_scalar("train/lr", lr, epoch)

        # Image logging every 5 epochs
        if epoch % 5 == 0:
            log_sample_images(writer, model, val_dataset, device, epoch)

        # Checkpoint best model
        if stats["dice"] > best_val_dice:
            best_val_dice = stats["dice"]
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

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
