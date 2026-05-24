"""
GOOGLE COLAB TRAINING CELL - U-Net Lung Nodule Segmentation
Copy this entire cell into a Colab notebook cell and run it.

Key Features:
- Clones from GitHub
- Mounts Google Drive
- Uses fixed dataset.py (removed mmap_mode='r')
- Trains with proper data validation
- Saves checkpoints to Google Drive
- Logs to TensorBoard
"""

# ============================================================================
# 1. SETUP: Install dependencies, clone repo, mount Google Drive
# ============================================================================
import os
import sys

print("="*80)
print("STEP 1: Installing dependencies...")
print("="*80)

os.system("pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
os.system("pip install -q SimpleITK scipy tqdm tensorboard")

print("\n" + "="*80)
print("STEP 2: Mounting Google Drive...")
print("="*80)

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

print("\n" + "="*80)
print("STEP 3: Cloning repository...")
print("="*80)

repo_url = "https://github.com/IchimD/LungTumorTrackingSystem"
repo_dir = "/content/LungTumorTrackingSystem"

if not os.path.exists(repo_dir):
    os.system(f"git clone {repo_url} {repo_dir}")
    print(f"✓ Cloned {repo_url}")
else:
    print(f"✓ Repository already exists at {repo_dir}")

sys.path.insert(0, repo_dir)

# ============================================================================
# 2. IMPORTS & CONFIGURATION
# ============================================================================
import random
import json
import warnings
import shutil
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
from scipy.ndimage import zoom

# Import from cloned repo
from src.models.unet import UNet2D
from src.training.loss import BCEDiceLoss
from src.training.metrics import dice_score, iou_score, precision_score, sensitivity_score
from src.data.augmentation import default_training_augmentations
from src.data.io import (
    SUPPORTED_EXTENSIONS,
    find_matching_mask,
    normalize_mask,
    patient_id_from_filename,
)
import glob

# ============================================================================
# 3. CONFIGURATION
# ============================================================================
CONFIG = {
    "image_dir": "/content/drive/My Drive/LICENTA_COLAB",
    "mask_dir": "/content/drive/My Drive/LICENTA_COLAB/masks_rclone",
    "logs_dir": "/content/drive/My Drive/LICENTA_COLAB/logs",
    "results_dir": "/content/drive/My Drive/LICENTA_COLAB/results",
    "batch_size": 16,
    "epochs": 60,
    "lr": 1e-3,
    "val_fraction": 0.15,
    "seed": 42,
    "num_workers": 0,  # CRITICAL for Colab shared memory
    "augment": True,
    "resume": False,
}

print("\n" + "="*80)
print("CONFIGURATION")
print("="*80)
for key, val in CONFIG.items():
    print(f"  {key}: {val}")

# ============================================================================
# 4. DATASET WITH FIXED MEMORY-MAPPED ISSUE
# ============================================================================

def safe_np_load(path: str) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False)
    except Exception as base_exc:
        warnings.warn(
            f"np.load failed for {path} with allow_pickle=False: {base_exc}. "
            "Retrying with allow_pickle=True.",
            UserWarning,
        )
        try:
            arr = np.load(path, allow_pickle=True)
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"Loaded object is not ndarray: {type(arr)}")
            return arr
        except Exception as e:
            raise base_exc from e

class VolumeSliceDataset(Dataset):
    """PyTorch dataset for 2D axial slices extracted from preprocessed volumes.
    
    FIXED: Removed mmap_mode='r' to avoid silent failures on networked filesystems.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        patient_ids: Optional[List[str]] = None,
        augment_fn=None,
        background_ratio: float = 0.3,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment_fn = augment_fn
        self.background_ratio = background_ratio

        self._items: List[Tuple[str, str, str]] = []
        self._discover_patients(patient_ids)

        if not self._items:
            raise ValueError(
                f"No valid training patients found in {image_dir!r} / {mask_dir!r}."
            )

    def _discover_patients(self, patient_ids: Optional[List[str]]) -> None:
        allowed = set(patient_ids) if patient_ids is not None else None

        # Search preferred locations: image_dir, image_dir/images, then recursive
        candidates = [self.image_dir, os.path.join(self.image_dir, "images")]
        found = []

        for cand in candidates:
            try:
                if os.path.isdir(cand):
                    entries = sorted(
                        os.path.join(cand, p)
                        for p in os.listdir(cand)
                        if os.path.splitext(p)[1].lower() in SUPPORTED_EXTENSIONS
                    )
                    if entries:
                        found = entries
                        break
            except Exception:
                continue

        # Fallback: recursive glob search
        if not found:
            pattern = os.path.join(self.image_dir, "**", "*")
            try:
                found = sorted(
                    p for p in glob.glob(pattern, recursive=True)
                    if os.path.splitext(p)[1].lower() in SUPPORTED_EXTENSIONS
                )
            except Exception:
                found = []

        for image_path in found:
            # image_path may be full path; ensure it's a file
            if not os.path.isfile(image_path):
                continue

            patient_id = patient_id_from_filename(image_path)
            if allowed is not None and patient_id not in allowed:
                continue

            mask_path = find_matching_mask(image_path, self.mask_dir)
            if mask_path is None:
                continue

            self._items.append((image_path, mask_path, patient_id))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):
        # Handle corrupted files by retrying with different indices
        attempts = 0
        max_attempts = 10
        
        while attempts < max_attempts:
            if not self._items:
                raise RuntimeError(
                    "No valid dataset items remain. Check that Drive is mounted and files are accessible."
                )

            index = index % len(self._items)
            image_path, mask_path, patient_id = self._items[index]

            try:
                if not os.path.exists(image_path) or not os.path.exists(mask_path):
                    raise OSError(
                        f"Missing image or mask file for patient {patient_id}. "
                        f"image={image_path}, mask={mask_path}"
                    )

                # Load mask volume - FIXED: removed mmap_mode='r'
                mask_volume = normalize_mask(safe_np_load(mask_path))
                positive_slices = [z for z in range(mask_volume.shape[0]) if mask_volume[z].any()]
                negative_slices = [z for z in range(mask_volume.shape[0]) if not mask_volume[z].any()]

                if positive_slices and negative_slices:
                    if random.random() < self.background_ratio:
                        slice_index = random.choice(negative_slices)
                    else:
                        slice_index = random.choice(positive_slices)
                elif positive_slices:
                    slice_index = random.choice(positive_slices)
                else:
                    # Skip this item if no valid slices (all zero masks)
                    raise ValueError(f"No valid slices in {patient_id}")

                # Load image volume - FIXED: removed mmap_mode='r'
                image_volume = safe_np_load(image_path)
                image = image_volume[slice_index]
                mask = mask_volume[slice_index]

                break  # Success, exit retry loop

            except Exception as e:
                # Attempt to quarantine problematic files (move to corrupted_masks) to allow training to continue.
                try:
                    corrupted_dir = os.path.join(os.path.dirname(self.mask_dir), "corrupted_masks")
                    os.makedirs(corrupted_dir, exist_ok=True)

                    for p in (mask_path, image_path):
                        try:
                            if os.path.exists(p):
                                dst = os.path.join(corrupted_dir, os.path.basename(p))
                                shutil.move(p, dst)
                                warnings.warn(f"Moved corrupted file to {dst}", UserWarning)
                        except Exception:
                            # ignore move errors, we'll still remove entry
                            pass
                except Exception:
                    pass

                warnings.warn(
                    f"Skipping corrupted/inaccessible sample {patient_id}: {e}",
                    UserWarning,
                )
                # Remove the item from future consideration and continue
                try:
                    self._items.pop(index)
                except Exception:
                    pass
                attempts += 1
                if not self._items:
                    raise RuntimeError(
                        "No valid dataset items remain after skipping corrupted or inaccessible files. "
                        "Ensure Google Drive is mounted correctly and your dataset files are available."
                    )
                if attempts >= max_attempts:
                    raise RuntimeError(
                        f"Could not load valid data after {max_attempts} attempts. "
                        f"Your dataset may have corrupted files or the Drive mount may be unstable. "
                        f"Last error: {str(e)}"
                    )

        # Resize to 512x512 using scipy.ndimage.zoom
        target_size = (512, 512)
        scale_factors = (target_size[0] / image.shape[0], target_size[1] / image.shape[1])
        
        # Image: order=1 (bilinear)
        image_resized = zoom(image.astype(np.float32), scale_factors, order=1)
        image_resized = (image_resized - image_resized.min()) / (image_resized.max() - image_resized.min() + 1e-8)
        
        # Mask: order=0 (nearest neighbor) - CRITICAL for binary masks
        mask_resized = zoom(mask.astype(np.float32), scale_factors, order=0)
        mask_resized_binary = (mask_resized > 0.5).astype(np.float32)

        # Convert to tensors (keep as float32, not uint8)
        image_tensor = torch.from_numpy(np.asarray(image_resized, dtype=np.float32)).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_resized_binary).unsqueeze(0)

        # Data augmentation
        if self.augment_fn is not None:
            image_tensor, mask_tensor = self.augment_fn(image_tensor, mask_tensor)

        return image_tensor, mask_tensor, patient_id

    def get_patient_ids(self) -> List[str]:
        return [patient_id for _, _, patient_id in self._items]


# ============================================================================
# 5. TRAINING FUNCTIONS
# ============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_patient_ids(image_dir: str, mask_dir: str) -> List[str]:
    """Collect patient ids from `image_dir`.

    This helper will look directly in `image_dir`, then in a common `images`
    subfolder, and finally perform a recursive search if needed. It returns a
    sorted list of unique patient ids for which a matching mask exists in
    `mask_dir`.
    """
    # Candidate folders to search (prefer explicit images/ subfolder)
    candidates = [image_dir, os.path.join(image_dir, "images")]
    image_files = []
    chosen_dir = None

    for cand in candidates:
        try:
            if os.path.isdir(cand):
                files = [
                    p for p in os.listdir(cand)
                    if os.path.splitext(p)[1].lower() in SUPPORTED_EXTENSIONS
                ]
                if files:
                    image_files = sorted(files)
                    chosen_dir = cand
                    break
        except Exception:
            continue

    # Fallback: recursive glob search under image_dir
    if not image_files:
        pattern = os.path.join(image_dir, "**", "*")
        try:
            files = [
                p for p in glob.glob(pattern, recursive=True)
                if os.path.splitext(p)[1].lower() in SUPPORTED_EXTENSIONS
            ]
            if files:
                # use basenames but remember the parent dir
                files = sorted(files)
                image_files = [os.path.basename(p) for p in files]
                chosen_dir = os.path.dirname(files[0])
        except Exception:
            pass

    patient_ids = []
    base_dir = chosen_dir if chosen_dir is not None else image_dir
    for filename in sorted(image_files):
        image_path = os.path.join(base_dir, filename)
        if find_matching_mask(image_path, mask_dir) is not None:
            patient_ids.append(patient_id_from_filename(image_path))

    return sorted(set(patient_ids))


def split_patient_ids(
    patient_ids: List[str], val_fraction: float, seed: int
) -> Tuple[List[str], List[str]]:
    random.Random(seed).shuffle(patient_ids)
    val_count = max(1, int(len(patient_ids) * val_fraction))
    return patient_ids[val_count:], patient_ids[:val_count]


def build_augmentation_fn(enable: bool):
    if not enable:
        return None

    def augmentation(image: torch.Tensor, mask: torch.Tensor):
        # Use existing augmentations: flip + intensity jitter
        image, mask = default_training_augmentations(image, mask)

        # Add random 90-degree rotations
        k = random.choice([0, 1, 2, 3])
        if k != 0:
            image = torch.rot90(image, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])

        return image, mask

    return augmentation


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0

    data_iter = iter(dataloader)
    pbar = tqdm(data_iter, desc="Train", leave=False)
    batch_idx = 0
    while True:
        try:
            batch = next(pbar)
        except StopIteration:
            break
        except Exception as e:
            warnings.warn(
                f"Skipping training batch {batch_idx} due to DataLoader error: {e}",
                UserWarning,
            )
            batch_idx += 1
            continue

        try:
            images, masks, _ = batch
        except Exception as e:
            warnings.warn(
                f"Skipping training batch {batch_idx} due to data unpacking error: {e}",
                UserWarning,
            )
            batch_idx += 1
            continue

        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        count += 1
        pbar.set_postfix(loss=total_loss / max(count, 1))
        batch_idx += 1

        total_loss += float(loss.detach().cpu())
        count += 1
        pbar.set_postfix(loss=total_loss / max(count, 1))

    if count == 0:
        raise RuntimeError("No valid training batches were processed. Check the dataset for corrupted files.")
    return total_loss / count


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
        data_iter = iter(dataloader)
        pbar = tqdm(data_iter, desc="Val", leave=False)
        batch_idx = 0
        while True:
            try:
                batch = next(pbar)
            except StopIteration:
                break
            except Exception as e:
                warnings.warn(
                    f"Skipping validation batch {batch_idx} due to DataLoader error: {e}",
                    UserWarning,
                )
                batch_idx += 1
                continue

            try:
                images, masks, _ = batch
            except Exception as e:
                warnings.warn(
                    f"Skipping validation batch {batch_idx} due to data unpacking error: {e}",
                    UserWarning,
                )
                batch_idx += 1
                continue

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
            batch_idx += 1

    return {
        "loss": total_loss / max(count, 1),
        "dice": total_dice / max(count, 1),
        "iou": total_iou / max(count, 1),
        "sensitivity": total_sensitivity / max(count, 1),
        "precision": total_precision / max(count, 1),
    }


def log_sample_images(
    writer: SummaryWriter,
    model: torch.nn.Module,
    dataset: VolumeSliceDataset,
    device: torch.device,
    epoch: int,
    tag: str = "val_samples",
    max_images: int = 4,
):
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
            except Exception as e:
                print(f"Warning: Could not load sample {idx}: {e}")
                continue
            
            image = image.to(device, dtype=torch.float32).unsqueeze(0)
            mask = mask.to(device, dtype=torch.float32).unsqueeze(0)
            out = model(image)
            prob = torch.sigmoid(out)
            pred = (prob > 0.5).float()

            # Normalize image to [0,1] for logging
            img = image[0]
            if img.max() > 0:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            imgs.append(img)
            gts.append(mask[0])
            preds.append(pred[0])

    if not imgs:
        print("Warning: No valid samples to log")
        return

    # Create a grid with rows = samples, cols = [image, gt, pred]
    rows = []
    for i in range(len(imgs)):
        row = torch.cat([imgs[i], gts[i], preds[i]], dim=2)
        rows.append(row)

    grid = make_grid(torch.stack(rows, dim=0), nrow=1, normalize=False, pad_value=1.0)
    writer.add_image(tag, grid, epoch)


def log_model_weights(writer: SummaryWriter, model: torch.nn.Module, epoch: int):
    """Log histograms of all model weights and biases to TensorBoard."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Log weight/bias histograms
            writer.add_histogram(f"weights/{name}", param.data, epoch)
            
            # Log gradients if available
            if param.grad is not None:
                writer.add_histogram(f"gradients/{name}", param.grad, epoch)


# ============================================================================
# 6. MAIN TRAINING LOOP
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Setting up datasets and dataloaders...")
print("="*80)

set_seed(CONFIG["seed"])

# Create output directories
os.makedirs(CONFIG["logs_dir"], exist_ok=True)
os.makedirs(CONFIG["results_dir"], exist_ok=True)

# Collect and split patients
print("\nCollecting patient IDs...")
patient_ids = collect_patient_ids(CONFIG["image_dir"], CONFIG["mask_dir"])
print(f"✓ Found {len(patient_ids)} patients")

if not patient_ids:
    print("\nNo patients found — gathering diagnostics to help debug mounting/path issues:\n")

    def _list_dir(path: str, preview: int = 20):
        try:
            entries = sorted(os.listdir(path))
            print(f"Contents of {path} ({len(entries)} entries):")
            for e in entries[:preview]:
                print(f"  {e}")
            if len(entries) > preview:
                print(f"  ... (+{len(entries)-preview} more)")
        except Exception as ex:
            print(f"  Could not list {path}: {ex}")

    # Show configured paths
    print(f"  Config image_dir: {CONFIG['image_dir']}")
    print(f"  Config mask_dir: {CONFIG['mask_dir']}")
    _list_dir(CONFIG['image_dir'])
    _list_dir(CONFIG['mask_dir'])

    # Try common alternative mount path used by Colab: '/content/drive/MyDrive'
    alt_image = CONFIG['image_dir'].replace('My Drive', 'MyDrive')
    alt_mask = CONFIG['mask_dir'].replace('My Drive', 'MyDrive')
    if alt_image != CONFIG['image_dir'] or alt_mask != CONFIG['mask_dir']:
        print('\nTrying alternative common mount paths:')
        print(f"  alt image_dir: {alt_image}")
        print(f"  alt mask_dir: {alt_mask}")
        _list_dir(alt_image)
        _list_dir(alt_mask)

    # Also list top-level drive mount to help diagnose
    print('\nTop-level /content/drive contents:')
    _list_dir('/content/drive')

    raise RuntimeError(
        f"No patients found! Check paths and Drive mount.\n"
        f"  Image dir: {CONFIG['image_dir']}\n"
        f"  Mask dir: {CONFIG['mask_dir']}\n"
        f"If files are under '/content/drive/MyDrive', adjust CONFIG paths accordingly."
    )

train_ids, val_ids = split_patient_ids(
    patient_ids, CONFIG["val_fraction"], CONFIG["seed"]
)
print(f"✓ Train: {len(train_ids)}, Val: {len(val_ids)}")

# Create datasets
print("\nCreating datasets...")
train_dataset = VolumeSliceDataset(
    CONFIG["image_dir"],
    CONFIG["mask_dir"],
    patient_ids=train_ids,
    augment_fn=build_augmentation_fn(CONFIG["augment"]),
)
val_dataset = VolumeSliceDataset(
    CONFIG["image_dir"],
    CONFIG["mask_dir"],
    patient_ids=val_ids,
)
print(f"✓ Train dataset: {len(train_dataset)} items")
print(f"✓ Val dataset: {len(val_dataset)} items")

# ============================================================================
# 7. DATA VALIDATION - Test that masks are NOT all zeros
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Validating data integrity...")
print("="*80)

print("\nSampling 3 random training slices...")
for i in range(3):
    idx = np.random.randint(0, len(train_dataset))
    img, mask, patient_id = train_dataset[idx]
    mask_sum = mask.sum().item()
    print(f"  Sample {i}: Shape={img.shape}, Mask sum={mask_sum:.1f}, Patient={patient_id}")
    
    if mask_sum == 0:
        print(f"    ⚠️  WARNING: All-zero mask! Possible data issue.")

print("\nTesting DataLoader with batch size 16...")
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=CONFIG["num_workers"],
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    pin_memory=torch.cuda.is_available(),
)

try:
    batch = next(iter(train_loader))
    images, masks, patient_ids = batch
except Exception as e:
    raise RuntimeError(
        "CRITICAL: DataLoader failed while fetching the first batch. "
        "This often means files are inaccessible, the Drive mount is unstable, "
        "or corrupted data remains in the dataset. "
        f"Underlying error: {e}"
    ) from e

print(f"✓ Batch shapes: images={images.shape}, masks={masks.shape}")
print(f"  Mask statistics: min={masks.min():.4f}, max={masks.max():.4f}, sum={masks.sum():.1f}")
print(f"  Patient IDs (first 3): {patient_ids[:3]}")

if masks.sum() == 0:
    raise RuntimeError("CRITICAL: First batch has all-zero masks! Data loading failed.")

print("✓ Data validation passed!")

# ============================================================================
# 8. INITIALIZE MODEL, LOSS, OPTIMIZER
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Initializing model...")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Using device: {device}")

model = UNet2D(in_channels=1, out_channels=1, base_channels=32).to(device)
criterion = BCEDiceLoss(bce_weight=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
writer = SummaryWriter(log_dir=CONFIG["logs_dir"])

print(f"✓ Model: UNet2D with base_channels=32")
print(f"✓ Loss: BCEDiceLoss (bce_weight=0.5)")
print(f"✓ Optimizer: Adam (lr={CONFIG['lr']})")

# ============================================================================
# Log hyperparameters to TensorBoard
# ============================================================================
print(f"\nLogging hyperparameters to TensorBoard...")
hparams = {
    "batch_size": CONFIG["batch_size"],
    "learning_rate": CONFIG["lr"],
    "epochs": CONFIG["epochs"],
    "optimizer": "Adam",
    "loss_fn": "BCEDiceLoss",
    "bce_weight": 0.5,
    "augmentation": CONFIG["augment"],
    "val_fraction": CONFIG["val_fraction"],
}

metrics = {
    "best_dice": 0.0,
    "final_loss": 0.0,
}

writer.add_hparams(hparams, metrics)

# ============================================================================
# 9. TRAINING LOOP
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Starting training...")
print("="*80)

best_val_dice = 0.0
start_epoch = 1

# Try to resume from checkpoint
checkpoint_path = os.path.join(CONFIG["results_dir"], "best_model.pt")
if CONFIG["resume"] and os.path.exists(checkpoint_path):
    print(f"\nResuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_val_dice = checkpoint.get("val_dice", 0.0)
    start_epoch = checkpoint.get("epoch", 1) + 1
    print(f"✓ Resumed from epoch {checkpoint.get('epoch', 0)}, best_val_dice={best_val_dice:.4f}")

# Save patient split
split_path = os.path.join(CONFIG["results_dir"], "patient_split.json")
with open(split_path, "w", encoding="utf-8") as f:
    json.dump({"train": train_ids, "val": val_ids}, f, indent=2)
print(f"\n✓ Saved patient split to {split_path}")

print("\n" + "="*80)
print("TRAINING START")
print("="*80 + "\n")

for epoch in range(start_epoch, CONFIG["epochs"] + 1):
    print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
    
    # Train
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    
    # Evaluate
    stats = evaluate(model, val_loader, criterion, device)

    print(f"  train_loss: {train_loss:.4f}")
    print(f"  val_loss: {stats['loss']:.4f}")
    print(f"  val_dice: {stats['dice']:.4f}  iou: {stats['iou']:.4f}")
    print(f"  sensitivity: {stats['sensitivity']:.4f}  precision: {stats['precision']:.4f}")

    # Log to TensorBoard
    writer.add_scalar("train/loss", train_loss, epoch)
    writer.add_scalar("val/loss", stats["loss"], epoch)
    writer.add_scalar("val/dice", stats["dice"], epoch)
    writer.add_scalar("val/iou", stats["iou"], epoch)
    writer.add_scalar("val/sensitivity", stats["sensitivity"], epoch)
    writer.add_scalar("val/precision", stats["precision"], epoch)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

    # Log model weight/bias histograms every 5 epochs
    if epoch % 5 == 0:
        log_model_weights(writer, model, epoch)
        print(f"  ✓ Logged weight histograms to TensorBoard")

    # Log sample images every 5 epochs
    if epoch % 5 == 0:
        try:
            log_sample_images(writer, model, val_dataset, device, epoch)
            print(f"  ✓ Logged sample predictions to TensorBoard")
        except Exception as e:
            print(f"  ⚠️  Could not log sample images: {e}")

    # Save best model
    if stats["dice"] > best_val_dice:
        best_val_dice = stats["dice"]
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": best_val_dice,
                "config": CONFIG,
            },
            checkpoint_path,
        )
        print(f"  ✓ Saved best checkpoint (dice={best_val_dice:.4f})")

writer.close()

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Best validation Dice: {best_val_dice:.4f}")
print(f"Best model saved to: {checkpoint_path}")
print(f"TensorBoard logs: {CONFIG['logs_dir']}")

print("\n" + "="*80)
print("HOW TO VIEW TENSORBOARD METRICS IN COLAB")
print("="*80)
print("\n1. In a NEW COLAB CELL, run this command:")
print(f"\n   %tensorboard --logdir '{CONFIG['logs_dir']}'")
print("\n2. This will show you:")
print("   • Training and validation loss/Dice/IOU curves")
print("   • Learning rate schedule")
print("   • Sample predictions (input, ground truth, model output)")
print("   • Weight and bias histograms")
print("   • Hyperparameter summary")
print("\n3. You can also monitor training in real-time by re-running this command.")
print("\n4. To download TensorBoard logs to your computer, run in a new cell:")
print(f"\n   from google.colab import files")
print(f"   files.download_from_directory('{CONFIG['logs_dir']}')")

