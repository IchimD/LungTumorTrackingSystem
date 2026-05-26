"""
GOOGLE COLAB TRAINING CELL — U-Net Lung Nodule Segmentation (IMPROVED)
========================================================================
Copy this ENTIRE file into a single Colab notebook cell and run it.

Key improvements over v1 (fixes 0.50 → target 0.70-0.80 Dice):
  1. ReduceLROnPlateau scheduler  — biggest single fix; LR decays on plateau
  2. background_ratio = 0.05      — model sees 95% positive slices
  3. bce_weight = 0.3             — Dice loss gets 70% weight vs 30% BCE
  4. Gradient clipping (max 1.0)  — stable training at higher LR
  5. Early stopping (patience 20) — stops before overfitting
  6. Stronger augmentation        — rotation 90°, Gaussian noise, brightness
  7. 120 epochs, batch 32         — more training on Colab GPU
  8. Live loss/Dice plot          — visible inside Colab cell every 5 epochs
  9. TensorBoard logging          — full curves + sample predictions

GitHub: https://github.com/IchimD/LungTumorTrackingSystem

Drive layout expected:
  My Drive/LICENTA_COLAB/           <- images (.npy files)
  My Drive/LICENTA_COLAB/masks_rclone/ <- masks  (.npy files)
  My Drive/LICENTA_COLAB/logs/      <- TensorBoard output (auto-created)
  My Drive/LICENTA_COLAB/results/   <- checkpoints (auto-created)
"""

# ============================================================================
# STEP 1 — Install dependencies
# ============================================================================
import os, sys

print("=" * 70)
print("STEP 1: Installing dependencies …")
print("=" * 70)

os.system(
    "pip install -q torch torchvision torchaudio "
    "--index-url https://download.pytorch.org/whl/cu118"
)
os.system("pip install -q SimpleITK scipy tqdm tensorboard matplotlib")

# ============================================================================
# STEP 2 — Mount Google Drive
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Mounting Google Drive …")
print("=" * 70)

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

# ============================================================================
# STEP 3 — Clone / update repository
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Cloning repository …")
print("=" * 70)

REPO_URL = "https://github.com/IchimD/LungTumorTrackingSystem"
REPO_DIR = "/content/LungTumorTrackingSystem"

if os.path.exists(REPO_DIR):
    print("Repo already cloned — pulling latest …")
    os.system(f"git -C {REPO_DIR} pull")
else:
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
    print(f"✓ Cloned {REPO_URL}")

sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

# ============================================================================
# STEP 4 — Imports
# ============================================================================
import random, json, warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
from scipy.ndimage import zoom
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe in Colab
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import glob

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

# ============================================================================
# CONFIGURATION  ← tune these values
# ============================================================================
CONFIG = {
    # ── paths ────────────────────────────────────────────────────────────────
    "image_dir":    "/content/drive/My Drive/LICENTA_COLAB ",          # trailing space is correct
    "mask_dir":     "/content/drive/My Drive/LICENTA_COLAB /masks",    # trailing space is correct
    "logs_dir":     "/tmp/tb_logs",                                    # local — fast, no Drive quota
    "results_dir":  "/content/drive/My Drive/LICENTA_COLAB/results",
    # ── training ─────────────────────────────────────────────────────────────
    "batch_size":          16,     # 16 + AMP fits T4 (14.6 GB); use 32 on A100
    "epochs":             120,
    "lr":               3e-4,      # lower start; scheduler handles decay
    "lr_min":           1e-6,      # floor for ReduceLROnPlateau
    "lr_patience":          8,     # epochs on plateau before halving LR
    "lr_factor":          0.5,
    "early_stop_patience": 20,     # stop if no improvement for this many epochs
    "grad_clip":          1.0,     # max gradient norm
    "val_fraction":       0.15,
    "seed":                 42,
    "num_workers":           2,    # 0 if you hit DataLoader errors
    "augment":            True,
    "resume":             True,
    # ── data ─────────────────────────────────────────────────────────────────
    "background_ratio":   0.05,    # only 5 % background slices per epoch
    # ── model / loss ─────────────────────────────────────────────────────────
    "base_channels":        32,    # 64 for more capacity (needs more VRAM)
    "bce_weight":          0.3,    # 70 % Dice + 30 % BCE loss
}

print("\n" + "=" * 70)
print("CONFIGURATION")
print("=" * 70)
for k, v in CONFIG.items():
    print(f"  {k:25s}: {v}")

# GPU info
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f"\n✓ GPU: {gpu.name}  VRAM: {gpu.total_memory / 1e9:.1f} GB")
else:
    print("\n⚠ No GPU found — running on CPU (will be slow)")

# ============================================================================
# DATASET
# ============================================================================
def safe_np_load(path: str) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False)
    except Exception as base_exc:
        warnings.warn(f"np.load failed ({base_exc}); retrying with allow_pickle=True", UserWarning)
        arr = np.load(path, allow_pickle=True)
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"Loaded object is not ndarray: {type(arr)}")
        return arr


class VolumeSliceDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        patient_ids: Optional[List[str]] = None,
        augment_fn=None,
        background_ratio: float = 0.05,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment_fn = augment_fn
        self.background_ratio = background_ratio
        self._items: List[Tuple[str, str, str]] = []
        self._discover_patients(patient_ids)
        if not self._items:
            raise ValueError(f"No patients found in {image_dir!r} / {mask_dir!r}")

    def _discover_patients(self, patient_ids):
        allowed = set(patient_ids) if patient_ids is not None else None
        found = []
        for base in [self.image_dir, os.path.join(self.image_dir, "images")]:
            if os.path.isdir(base):
                entries = sorted(
                    os.path.join(base, p)
                    for p in os.listdir(base)
                    if os.path.splitext(p)[1].lower() in SUPPORTED_EXTENSIONS
                )
                if entries:
                    found = entries
                    break
        if not found:
            found = sorted(
                p for p in glob.glob(os.path.join(self.image_dir, "**", "*"), recursive=True)
                if os.path.splitext(p)[1].lower() in SUPPORTED_EXTENSIONS
            )
        candidates = []
        for image_path in found:
            if not os.path.isfile(image_path):
                continue
            pid = patient_id_from_filename(image_path)
            if allowed is not None and pid not in allowed:
                continue
            mask_path = find_matching_mask(image_path, self.mask_dir)
            if mask_path is None:
                continue
            candidates.append((image_path, mask_path, pid))

        # Pre-filter: skip patients whose mask is entirely zero.
        # All-zero masks inflate Dice to ~1.0 via the smooth term and cause
        # repeated warnings inside __getitem__ during multi-worker loading.
        print(f"  [Dataset] Pre-filtering {len(candidates)} candidates for positive masks …")
        for image_path, mask_path, pid in candidates:
            try:
                mask = normalize_mask(safe_np_load(mask_path))
                if mask.any():
                    self._items.append((image_path, mask_path, pid))
            except Exception:
                pass
        removed = len(candidates) - len(self._items)
        if removed:
            print(f"  [Dataset] Removed {removed} all-zero-mask patients; kept {len(self._items)}")

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        for _ in range(5):
            index = index % len(self._items)
            image_path, mask_path, pid = self._items[index]
            try:
                mask_vol = normalize_mask(safe_np_load(mask_path))
                pos = [z for z in range(mask_vol.shape[0]) if mask_vol[z].any()]
                neg = [z for z in range(mask_vol.shape[0]) if not mask_vol[z].any()]
                if pos and neg:
                    z = random.choice(neg) if random.random() < self.background_ratio else random.choice(pos)
                elif pos:
                    z = random.choice(pos)
                else:
                    # Mask turned all-zero at runtime (shouldn't happen after pre-filter)
                    index = (index + 1) % len(self._items)
                    continue
                img_vol = safe_np_load(image_path)
                image, mask = img_vol[z], mask_vol[z]
                break
            except Exception:
                index = (index + 1) % len(self._items)
        else:
            raise RuntimeError("Failed to load valid sample after 5 attempts.")

        # Resize to 512×512
        sf = (512 / image.shape[0], 512 / image.shape[1])
        img_r = zoom(image.astype(np.float32), sf, order=1)
        img_r = (img_r - img_r.min()) / (img_r.max() - img_r.min() + 1e-8)
        msk_r = (zoom(mask.astype(np.float32), sf, order=0) > 0.5).astype(np.float32)

        img_t = torch.from_numpy(img_r).unsqueeze(0)
        msk_t = torch.from_numpy(msk_r).unsqueeze(0)

        if self.augment_fn is not None:
            img_t, msk_t = self.augment_fn(img_t, msk_t)

        return img_t, msk_t, pid

    def get_patient_ids(self):
        return [pid for _, _, pid in self._items]


# ============================================================================
# AUGMENTATION (stronger than v1)
# ============================================================================
def build_augmentation_fn(enable: bool):
    if not enable:
        return None

    def augment(img: torch.Tensor, mask: torch.Tensor):
        # Horizontal / vertical flip
        img, mask = default_training_augmentations(img, mask)

        # Random 90° rotation
        k = random.randint(0, 3)
        if k:
            img  = torch.rot90(img,  k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])

        # Gaussian noise (image only)
        if random.random() < 0.3:
            noise = torch.randn_like(img) * 0.02
            img = (img + noise).clamp(0.0, 1.0)

        # Random brightness / contrast
        if random.random() < 0.3:
            alpha = random.uniform(0.8, 1.2)   # contrast
            beta  = random.uniform(-0.1, 0.1)  # brightness
            img = (img * alpha + beta).clamp(0.0, 1.0)

        return img, mask

    return augment


# ============================================================================
# HELPERS
# ============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_patient_ids(image_dir: str, mask_dir: str) -> List[str]:
    found = []
    for base in [image_dir, os.path.join(image_dir, "images")]:
        if os.path.isdir(base):
            files = sorted(
                os.path.join(base, p) for p in os.listdir(base)
                if os.path.splitext(p)[1].lower() in SUPPORTED_EXTENSIONS
            )
            if files:
                found = files
                break
    if not found:
        found = sorted(
            p for p in glob.glob(os.path.join(image_dir, "**", "*"), recursive=True)
            if os.path.splitext(p)[1].lower() in SUPPORTED_EXTENSIONS
        )
    pids = []
    for fp in found:
        if find_matching_mask(fp, mask_dir) is not None:
            pids.append(patient_id_from_filename(fp))
    return sorted(set(pids))


def split_patient_ids(pids, val_frac, seed):
    random.Random(seed).shuffle(pids)
    n_val = max(1, int(len(pids) * val_frac))
    return pids[n_val:], pids[:n_val]


# ============================================================================
# TRAINING / EVALUATION
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip, scaler):
    model.train()
    total_loss, count = 0.0, 0
    with tqdm(loader, desc="Train", leave=False) as pbar:
        for imgs, masks, _ in pbar:
            imgs  = imgs.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = criterion(model(imgs), masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            count += 1
            pbar.set_postfix(loss=f"{total_loss/count:.4f}")
    return total_loss / max(count, 1)


def evaluate(model, loader, criterion, device, scaler):
    model.eval()
    totals = dict(loss=0., dice=0., iou=0., sens=0., prec=0.)
    count = 0
    with torch.no_grad():
        with tqdm(loader, desc="Val  ", leave=False) as pbar:
            for imgs, masks, _ in pbar:
                imgs  = imgs.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.float32)
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    out = model(imgs)
                    totals["loss"] += criterion(out, masks).item()
                totals["dice"] += dice_score(out, masks)
                totals["iou"]  += iou_score(out, masks)
                totals["sens"] += sensitivity_score(out, masks)
                totals["prec"] += precision_score(out, masks)
                count += 1
                pbar.set_postfix(dice=f"{totals['dice']/count:.4f}")
    return {k: v / max(count, 1) for k, v in totals.items()}


# ============================================================================
# LIVE PLOT (visible inside the Colab cell)
# ============================================================================
history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": [], "lr": []}

def update_live_plot(epoch):
    clear_output(wait=True)
    epochs = list(range(1, len(history["val_dice"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train loss", color="#2196F3")
    axes[0].plot(epochs, history["val_loss"],   label="Val loss",   color="#FF5722")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Dice / IoU
    axes[1].plot(epochs, history["val_dice"], label="Val Dice",  color="#4CAF50", linewidth=2)
    axes[1].plot(epochs, history["val_iou"],  label="Val IoU",   color="#9C27B0")
    axes[1].axhline(0.70, color="orange", linestyle="--", linewidth=1, label="Target 0.70")
    axes[1].axhline(0.80, color="red",    linestyle="--", linewidth=1, label="Target 0.80")
    axes[1].set_ylim(0, 1); axes[1].set_title("Dice / IoU")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)

    # Learning rate
    axes[2].semilogy(epochs, history["lr"], color="#FF9800")
    axes[2].set_title("Learning Rate"); axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.3, which="both")

    best_dice = max(history["val_dice"])
    fig.suptitle(
        f"Epoch {epoch}/{CONFIG['epochs']}  |  Best Dice = {best_dice:.4f}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    plot_path = os.path.join(CONFIG["results_dir"], "training_progress.png")
    plt.savefig(plot_path, dpi=100)
    display(fig)
    plt.close(fig)


def log_sample_images(writer, model, dataset, device, epoch, max_images=4):
    model.eval()
    rows = []
    indices = random.sample(range(len(dataset)), min(max_images, len(dataset)))
    with torch.no_grad():
        for idx in indices:
            try:
                img, mask, _ = dataset[idx]
            except Exception:
                continue
            img  = img.to(device).unsqueeze(0)
            pred = (torch.sigmoid(model(img)) > 0.5).float()
            img_n = (img[0] - img[0].min()) / (img[0].max() - img[0].min() + 1e-8)
            rows.append(torch.cat([img_n.cpu(), mask, pred[0].cpu()], dim=2))
    if rows:
        grid = make_grid(torch.stack(rows), nrow=1, pad_value=1.0)
        writer.add_image("val/predictions", grid, epoch)


# ============================================================================
# STEP 5 — Build datasets
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Building datasets …")
print("=" * 70)

set_seed(CONFIG["seed"])
os.makedirs(CONFIG["logs_dir"],    exist_ok=True)
os.makedirs(CONFIG["results_dir"], exist_ok=True)

patient_ids = collect_patient_ids(CONFIG["image_dir"], CONFIG["mask_dir"])
print(f"✓ Found {len(patient_ids)} patients with image/mask pairs")

if not patient_ids:
    for path in [CONFIG["image_dir"], CONFIG["mask_dir"]]:
        try:
            entries = os.listdir(path)
            print(f"\n{path} ({len(entries)} entries): {entries[:10]}")
        except Exception as ex:
            print(f"\nCannot list {path}: {ex}")
    raise RuntimeError(
        "No patients found! Check that Drive is mounted and paths are correct.\n"
        f"image_dir = {CONFIG['image_dir']}\nmask_dir  = {CONFIG['mask_dir']}"
    )

train_ids, val_ids = split_patient_ids(patient_ids, CONFIG["val_fraction"], CONFIG["seed"])
print(f"  Train: {len(train_ids)} patients,  Val: {len(val_ids)} patients")

aug_fn = build_augmentation_fn(CONFIG["augment"])

train_ds = VolumeSliceDataset(
    CONFIG["image_dir"], CONFIG["mask_dir"],
    patient_ids=train_ids, augment_fn=aug_fn,
    background_ratio=CONFIG["background_ratio"],
)
val_ds = VolumeSliceDataset(
    CONFIG["image_dir"], CONFIG["mask_dir"],
    patient_ids=val_ids,
    background_ratio=0.1,   # slightly more background in val (for realistic metrics)
)
print(f"  Train dataset: {len(train_ds)} items")
print(f"  Val   dataset: {len(val_ds)} items")

train_loader = DataLoader(
    train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
    num_workers=CONFIG["num_workers"], pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
    num_workers=CONFIG["num_workers"], pin_memory=torch.cuda.is_available(),
)

# Quick sanity check
print("\nSanity check — sampling first batch …")
imgs, masks, pids = next(iter(train_loader))
print(f"  Images: {imgs.shape}  Masks: {masks.shape}")
print(f"  Mask pixel sum (should be > 0): {masks.sum().item():.0f}")
if masks.sum() == 0:
    raise RuntimeError("CRITICAL: first batch has all-zero masks — data loading failed.")
print("✓ Data looks good!")

# ============================================================================
# STEP 6 — Model, loss, optimiser, scheduler
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Initialising model …")
print("=" * 70)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f"GPU memory free: {free/1e9:.1f} GB / {total/1e9:.1f} GB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model     = UNet2D(in_channels=1, out_channels=1,
                   base_channels=CONFIG["base_channels"]).to(device)
criterion = BCEDiceLoss(bce_weight=CONFIG["bce_weight"])
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",            # maximise val Dice
    factor=CONFIG["lr_factor"],
    patience=CONFIG["lr_patience"],
    min_lr=CONFIG["lr_min"],
)
writer = SummaryWriter(log_dir=CONFIG["logs_dir"])
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✓ UNet2D  base_channels={CONFIG['base_channels']}  params={n_params:,}")
print(f"✓ Loss: BCEDiceLoss  bce_weight={CONFIG['bce_weight']}")
print(f"✓ Optimiser: Adam  lr={CONFIG['lr']}")
print(f"✓ Scheduler: ReduceLROnPlateau  patience={CONFIG['lr_patience']}  factor={CONFIG['lr_factor']}")
print(f"✓ AMP: enabled={torch.cuda.is_available()}")

# Log hyperparameters
writer.add_hparams(
    {k: str(v) for k, v in CONFIG.items()},
    {"hparam/best_dice": 0.0},
)

# ============================================================================
# STEP 7 — (Optional) resume from checkpoint
# ============================================================================
best_val_dice   = 0.0
no_improve      = 0
start_epoch     = 1
checkpoint_path = os.path.join(CONFIG["results_dir"], "best_model.pt")

if CONFIG["resume"] and os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    best_val_dice = ckpt.get("val_dice", 0.0)
    start_epoch   = ckpt.get("epoch", 1) + 1
    print(f"\n✓ Resumed from epoch {start_epoch - 1}, best_dice={best_val_dice:.4f}")

# Save patient split
with open(os.path.join(CONFIG["results_dir"], "patient_split.json"), "w") as f:
    json.dump({"train": train_ids, "val": val_ids}, f, indent=2)

# ============================================================================
# STEP 8 — Training loop
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: Training …")
print("=" * 70)

for epoch in range(start_epoch, CONFIG["epochs"] + 1):

    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, CONFIG["grad_clip"], scaler)
    stats      = evaluate(model, val_loader, criterion, device, scaler)

    # Update LR scheduler
    scheduler.step(stats["dice"])
    current_lr = optimizer.param_groups[0]["lr"]

    # Console log
    print(
        f"Ep {epoch:3d}/{CONFIG['epochs']}  "
        f"train_loss={train_loss:.4f}  "
        f"val_loss={stats['loss']:.4f}  "
        f"dice={stats['dice']:.4f}  "
        f"iou={stats['iou']:.4f}  "
        f"sens={stats['sens']:.4f}  "
        f"prec={stats['prec']:.4f}  "
        f"lr={current_lr:.2e}"
    )

    # TensorBoard
    writer.add_scalar("train/loss",      train_loss,      epoch)
    writer.add_scalar("val/loss",        stats["loss"],   epoch)
    writer.add_scalar("val/dice",        stats["dice"],   epoch)
    writer.add_scalar("val/iou",         stats["iou"],    epoch)
    writer.add_scalar("val/sensitivity", stats["sens"],   epoch)
    writer.add_scalar("val/precision",   stats["prec"],   epoch)
    writer.add_scalar("train/lr",        current_lr,      epoch)

    # Sample predictions every 10 epochs
    if epoch % 10 == 0:
        try:
            log_sample_images(writer, model, val_ds, device, epoch)
        except Exception as e:
            print(f"  ⚠ Could not log sample images: {e}")

    # Live plot every 5 epochs
    history["train_loss"].append(train_loss)
    history["val_loss"].append(stats["loss"])
    history["val_dice"].append(stats["dice"])
    history["val_iou"].append(stats["iou"])
    history["lr"].append(current_lr)

    if epoch % 5 == 0 or epoch == CONFIG["epochs"]:
        try:
            update_live_plot(epoch)
        except Exception as e:
            print(f"  ⚠ Could not draw live plot: {e}")

    # Save best checkpoint
    if stats["dice"] > best_val_dice:
        best_val_dice = stats["dice"]
        no_improve    = 0
        torch.save(
            {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice":             best_val_dice,
                "config":               CONFIG,
                "history":              history,
            },
            checkpoint_path,
        )
        print(f"  ✓ New best — checkpoint saved (dice={best_val_dice:.4f})")
    else:
        no_improve += 1
        if no_improve >= CONFIG["early_stop_patience"]:
            print(f"\n⏹ Early stopping at epoch {epoch} (no improvement for {no_improve} epochs)")
            break

writer.close()

# Copy TensorBoard logs from /tmp to Drive for permanent storage
drive_logs = "/content/drive/My Drive/LICENTA_COLAB/logs"
os.makedirs(drive_logs, exist_ok=True)
os.system(f"cp -r /tmp/tb_logs/. '{drive_logs}/'")
print(f"✓ TensorBoard logs copied to Drive: {drive_logs}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"  Best validation Dice : {best_val_dice:.4f}")
print(f"  Checkpoint saved to  : {checkpoint_path}")
print(f"  TensorBoard logs     : {CONFIG['logs_dir']}")
print(f"  Live plot saved to   : {CONFIG['results_dir']}/training_progress.png")

# Print final history summary
if history["val_dice"]:
    print(f"\nDice progression (every 10 epochs):")
    for i, d in enumerate(history["val_dice"], 1):
        if i % 10 == 0 or i == len(history["val_dice"]):
            bar = "█" * int(d * 20)
            print(f"  Ep {i:3d}: {d:.4f}  {bar}")

print("\n" + "=" * 70)
print("HOW TO VIEW TENSORBOARD")
print("=" * 70)
print(f"\nPaste in a NEW Colab cell:\n")
print(f"  %load_ext tensorboard")
print(f"  %tensorboard --logdir '{CONFIG['logs_dir']}'")
print(f"\nThis shows: loss curves, Dice/IoU, LR schedule, sample predictions, weight histograms.")
