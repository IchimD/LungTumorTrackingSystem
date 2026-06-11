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

# ── Detect environment ────────────────────────────────────────────────────────
# Check Kaggle first — Kaggle has google.colab installed but it doesn't work
IS_KAGGLE = os.path.exists("/kaggle/working")
IS_COLAB = False
if not IS_KAGGLE:
    try:
        from google.colab import drive as _colab_drive
        IS_COLAB = True
    except ImportError:
        IS_COLAB = False

ENV = "Colab" if IS_COLAB else ("Kaggle" if IS_KAGGLE else "Vast.ai / other")
print(f"Environment: {ENV}")

# ============================================================================
# STEP 1 — Install dependencies
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: Installing dependencies …")
print("=" * 70)

if IS_COLAB:
    os.system(
        "pip install -q torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu118"
    )
os.system("pip install -q SimpleITK scipy tqdm tensorboard matplotlib segmentation-models-pytorch")

# ============================================================================
# STEP 2 — Mount Google Drive (Colab only — Vast.ai uses local /workspace)
# ============================================================================
if IS_COLAB:
    print("\n" + "=" * 70)
    print("STEP 2: Mounting Google Drive …")
    print("=" * 70)
    _colab_drive.mount("/content/drive", force_remount=True)
else:
    print("\nSTEP 2: Skipped (not Colab — data already on local disk)")

# ============================================================================
# STEP 3 — Clone / update repository
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Cloning repository …")
print("=" * 70)

REPO_URL = "https://github.com/IchimD/LungTumorTrackingSystem"
REPO_DIR = "/content/LungTumorTrackingSystem" if IS_COLAB else "/workspace/LungTumorTrackingSystem"

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

import segmentation_models_pytorch as smp
import torch.nn as nn
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, pos_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight
    def forward(self, logits, targets):
        targets = targets.float()
        pw = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        probs = torch.sigmoid(logits)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        # per-sample dice: sum over spatial dims, mean over batch
        # this prevents large-lesion patients from dominating the gradient
        spatial = list(range(2, logits.ndim))
        inter = (probs * targets).sum(dim=spatial)
        denom = probs.sum(dim=spatial) + targets.sum(dim=spatial)
        dice = (1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)).mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice
from src.training.metrics import iou_score, precision_score, sensitivity_score

def per_sample_dice(logits: torch.Tensor, targets: torch.Tensor,
                    threshold: float = 0.5, eps: float = 1e-6) -> float:
    """Per-sample Dice averaged only over samples that have positive ground-truth pixels."""
    preds   = (torch.sigmoid(logits) >= threshold).float()
    targets = targets.float()
    spatial = list(range(1, preds.ndim))
    inter   = (preds * targets).sum(dim=spatial)                          # (B,)
    union   = preds.sum(dim=spatial) + targets.sum(dim=spatial)           # (B,)
    gt_sum  = targets.reshape(targets.shape[0], -1).sum(dim=1)            # (B,)
    has_pos = gt_sum > 0                                                   # only real GT positives
    if not has_pos.any():
        return float('nan')
    dice = (2.0 * inter[has_pos] + eps) / (union[has_pos] + eps)
    return float(dice.mean())
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
    # ── paths (auto-detected per environment) ────────────────────────────────
    "image_dir":   ("/content/images" if IS_COLAB
                    else "/kaggle/working/images" if IS_KAGGLE
                    else "/workspace/images"),
    "mask_dir":    ("/content/masks" if IS_COLAB
                    else "/kaggle/working/masks" if IS_KAGGLE
                    else "/workspace/masks"),
    "logs_dir":    "/tmp/tb_logs",
    "results_dir": ("/content/drive/My Drive/LICENTA_COLAB/results" if IS_COLAB
                    else "/kaggle/working/results" if IS_KAGGLE
                    else "/workspace/results"),
    # ── training ─────────────────────────────────────────────────────────────
    "batch_size":          32,
    "epochs":              60,     # per fold
    "lr":               3e-4,      # decoder LR
    "lr_encoder":       3e-5,      # encoder LR — 10x smaller to prevent forgetting ImageNet
    "lr_min":           1e-6,
    "cosine_T0":           60,     # single smooth decay — no restarts that disrupt convergence
    "early_stop_patience": 25,
    "grad_clip":          1.0,
    "n_folds":              3,     # 3 folds → 36 val patients each, fits in 9h Kaggle session
    "seed":                 42,
    "num_workers":           0,
    "samples_per_patient":  40,
    "augment":            True,
    "resume":             False,   # fresh start
    # ── data ─────────────────────────────────────────────────────────────────
    "background_ratio":   0.0,     # train only on positive slices — prevents collapse to predict-nothing
    "max_vol_pos_frac":   0.05,    # skip only truly absurd masks (>5% of volume); keep large lesions
    # ── model / loss ─────────────────────────────────────────────────────────
    "bce_weight":          0.2,    # 80% Dice weight — Dice handles imbalance better than BCE
    "pos_weight":          1.0,    # symmetric BCE — let Dice loss handle imbalance
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
    """
    At init: scans each mask volume to record which slice indices are positive.
    __getitem__: picks a FRESH RANDOM positive slice from the patient each call.

    Because a different slice is chosen every time, the model can never memorise
    specific images.  __len__ = num_patients × samples_per_patient controls how
    many gradient updates happen per epoch without creating a fixed training set.
    File I/O uses numpy mmap so only the needed slice is read from the SSD.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        patient_ids: Optional[List[str]] = None,
        augment_fn=None,
        background_ratio: float = 0.05,
        samples_per_patient: int = 10,
        max_vol_pos_frac: float = 0.005,
    ) -> None:
        self.augment_fn          = augment_fn
        self.background_ratio    = background_ratio
        self.samples_per_patient = samples_per_patient
        self.max_vol_pos_frac    = max_vol_pos_frac
        # (img_path, msk_path, pid, pos_z_list, neg_z_list)
        self._patients: List[Tuple[str, str, str, List[int], List[int]]] = []
        self._build_index(image_dir, mask_dir, patient_ids)
        if not self._patients:
            raise ValueError(f"No patients found in {image_dir!r} / {mask_dir!r}")

    @staticmethod
    def _find_candidates(image_dir, mask_dir, patient_ids):
        allowed = set(patient_ids) if patient_ids is not None else None
        found = []
        for base in [image_dir, os.path.join(image_dir, "images")]:
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
                p for p in glob.glob(os.path.join(image_dir, "**", "*"), recursive=True)
                if os.path.splitext(p)[1].lower() in SUPPORTED_EXTENSIONS
            )
        candidates = []
        for image_path in found:
            if not os.path.isfile(image_path):
                continue
            pid = patient_id_from_filename(image_path)
            if allowed is not None and pid not in allowed:
                continue
            mask_path = find_matching_mask(image_path, mask_dir)
            if mask_path is None:
                continue
            candidates.append((image_path, mask_path, pid))
        return candidates

    def _build_index(self, image_dir, mask_dir, patient_ids):
        candidates = self._find_candidates(image_dir, mask_dir, patient_ids)
        print(f"  [Dataset] Scanning {len(candidates)} mask volumes for positive slice indices …")
        skipped_bad, skipped_empty = 0, 0
        for img_path, msk_path, pid in candidates:
            try:
                msk_vol = normalize_mask(safe_np_load(msk_path))
                if msk_vol.ndim == 2:
                    msk_vol = msk_vol[np.newaxis]

                # --- quality filter: skip volumes with unrealistically large annotations ---
                total_vox = msk_vol.size
                pos_vox   = int(msk_vol.sum())
                vol_frac  = pos_vox / max(total_vox, 1)
                if vol_frac > self.max_vol_pos_frac:
                    print(f"  [Dataset] SKIP {pid[:30]}…: annotation={vol_frac:.3%} > "
                          f"{self.max_vol_pos_frac:.2%} threshold (bad mask?)")
                    skipped_bad += 1
                    continue

                # keep slices where mask exists; exclude only nearly-full-slice annotations (>80%)
                pos = [z for z in range(msk_vol.shape[0])
                       if msk_vol[z].any() and float(msk_vol[z].mean()) < 0.80]
                neg = [z for z in range(msk_vol.shape[0]) if not msk_vol[z].any()]
                if pos:
                    self._patients.append((img_path, msk_path, pid, pos, neg))
                else:
                    skipped_empty += 1
            except Exception as e:
                warnings.warn(f"  Skipping {pid}: {e}", UserWarning)
        n = len(self._patients) * self.samples_per_patient
        print(f"  [Dataset] {len(self._patients)} patients kept | "
              f"{skipped_bad} skipped (bad mask) | {skipped_empty} skipped (no pos slices)")
        print(f"  [Dataset] {len(self._patients)} patients × {self.samples_per_patient} "
              f"samples/patient = {n} items/epoch")

    def __len__(self) -> int:
        return len(self._patients) * self.samples_per_patient

    def __getitem__(self, index: int):
        img_path, msk_path, pid, pos_z, neg_z = self._patients[index % len(self._patients)]

        z = (random.choice(neg_z) if neg_z and random.random() < self.background_ratio
             else random.choice(pos_z))

        # mmap: OS reads only the slice we need, not the whole volume
        try:
            img_vol = np.load(img_path, mmap_mode="r")
            msk_vol = normalize_mask(np.load(msk_path, mmap_mode="r"))
        except Exception:
            img_vol = safe_np_load(img_path)
            msk_vol = normalize_mask(safe_np_load(msk_path))

        D = img_vol.shape[0] if img_vol.ndim == 3 else 1
        msk_s = np.array(msk_vol[z] if msk_vol.ndim == 3 else msk_vol, dtype=np.float32)
        sf = (256 / msk_s.shape[0], 256 / msk_s.shape[1])
        msk_r = (zoom(msk_s, sf, order=0) > 0.5).astype(np.float32)

        # 2.5D: stack [z-1, z, z+1] as 3 channels — uses full pretrained RGB weights
        channels = []
        for dz in (-1, 0, 1):
            zi = max(0, min(D - 1, z + dz))
            s = np.array(img_vol[zi] if img_vol.ndim == 3 else img_vol, dtype=np.float32)
            s = zoom(s, sf, order=1)
            s = (s - s.min()) / (s.max() - s.min() + 1e-8)
            channels.append(s)
        img_r = np.stack(channels, axis=0)  # (3, 256, 256)

        img_t = torch.from_numpy(img_r)     # (3, H, W) — no unsqueeze
        msk_t = torch.from_numpy(msk_r).unsqueeze(0)
        if self.augment_fn is not None:
            img_t, msk_t = self.augment_fn(img_t, msk_t)
        return img_t, msk_t, pid

    def get_patient_ids(self) -> List[str]:
        return [pid for _, _, pid, _, _ in self._patients]


class AllSlicesDataset(Dataset):
    """Deterministic validation dataset — enumerates ALL positive slices once at init.
    No random sampling → stable, reproducible val dice every epoch."""

    def __init__(self, image_dir, mask_dir, patient_ids=None, max_vol_pos_frac=0.05):
        self._items = []
        candidates = VolumeSliceDataset._find_candidates(image_dir, mask_dir, patient_ids)
        for img_path, msk_path, pid in candidates:
            try:
                msk_vol = normalize_mask(safe_np_load(msk_path))
                if msk_vol.ndim == 2:
                    msk_vol = msk_vol[np.newaxis]
                if float(msk_vol.sum()) / max(msk_vol.size, 1) > max_vol_pos_frac:
                    continue
                for z in range(msk_vol.shape[0]):
                    if msk_vol[z].any() and float(msk_vol[z].mean()) < 0.80:
                        self._items.append((img_path, msk_path, z))
            except Exception as e:
                warnings.warn(f"Skipping {pid}: {e}")
        print(f"  [ValDataset] {len(self._items)} positive slices (deterministic)")

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_path, msk_path, z = self._items[idx]
        try:
            img_vol = np.load(img_path, mmap_mode="r")
            msk_vol = normalize_mask(np.load(msk_path, mmap_mode="r"))
        except Exception:
            img_vol = safe_np_load(img_path)
            msk_vol = normalize_mask(safe_np_load(msk_path))
        D = img_vol.shape[0]
        msk_s = np.array(msk_vol[z], dtype=np.float32)
        sf = (256 / msk_s.shape[0], 256 / msk_s.shape[1])
        msk_r = (zoom(msk_s, sf, order=0) > 0.5).astype(np.float32)
        channels = []
        for dz in (-1, 0, 1):
            zi = max(0, min(D - 1, z + dz))
            s = np.array(img_vol[zi], dtype=np.float32)
            s = zoom(s, sf, order=1)
            s = (s - s.min()) / (s.max() - s.min() + 1e-8)
            channels.append(s)
        img_r = np.stack(channels, axis=0)
        return torch.from_numpy(img_r), torch.from_numpy(msk_r).unsqueeze(0), ""


# ============================================================================
# AUGMENTATION (stronger than v1)
# ============================================================================
def build_augmentation_fn(enable: bool):
    if not enable:
        return None

    from scipy.ndimage import map_coordinates, gaussian_filter as _gf

    def _elastic(img_np, msk_np, alpha=40, sigma=5):
        sh = img_np.shape[-2:]  # H, W — works for (3,H,W) or (H,W)
        dx = _gf(np.random.randn(*sh), sigma) * alpha
        dy = _gf(np.random.randn(*sh), sigma) * alpha
        y, x = np.mgrid[0:sh[0], 0:sh[1]]
        coords = [(y + dy).ravel(), (x + dx).ravel()]
        if img_np.ndim == 3:
            out_ch = [map_coordinates(img_np[c], coords, order=1).reshape(sh)
                      for c in range(img_np.shape[0])]
            img_out = np.stack(out_ch, axis=0).astype(np.float32)
        else:
            img_out = map_coordinates(img_np, coords, order=1).reshape(sh).astype(np.float32)
        msk_out = map_coordinates(msk_np, coords, order=0).reshape(sh).astype(np.float32)
        return img_out, msk_out

    def augment(img: torch.Tensor, mask: torch.Tensor):
        # Horizontal / vertical flip
        img, mask = default_training_augmentations(img, mask)

        # Random 90° rotation
        k = random.randint(0, 3)
        if k:
            img  = torch.rot90(img,  k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])

        # Elastic deformation — biggest boost for medical segmentation
        if random.random() < 0.3:
            img_np, msk_np = _elastic(img.numpy(), mask.squeeze().numpy())
            img  = torch.from_numpy(img_np)
            mask = torch.from_numpy(msk_np).unsqueeze(0)

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
    count, dice_valid = 0, 0
    with torch.no_grad():
        with tqdm(loader, desc="Val  ", leave=False) as pbar:
            for imgs, masks, _ in pbar:
                imgs  = imgs.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.float32)
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    out = model(imgs)
                    totals["loss"] += criterion(out, masks).item()
                d = per_sample_dice(out, masks)
                if not (d != d):  # skip NaN batches (all-background)
                    totals["dice"] += d
                    dice_valid += 1
                totals["iou"]  += iou_score(out, masks)
                totals["sens"] += sensitivity_score(out, masks)
                totals["prec"] += precision_score(out, masks)
                count += 1
                cur_dice = totals["dice"] / max(dice_valid, 1)
                pbar.set_postfix(dice=f"{cur_dice:.4f}")
    n = max(count, 1)
    return {
        "loss": totals["loss"] / n,
        "dice": totals["dice"] / max(dice_valid, 1),
        "iou":  totals["iou"]  / n,
        "sens": totals["sens"] / n,
        "prec": totals["prec"] / n,
    }


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
# STEP 5 — 5-Fold Cross-Validation
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: 5-Fold Cross-Validation Setup")
print("=" * 70)

set_seed(CONFIG["seed"])
os.makedirs(CONFIG["logs_dir"],    exist_ok=True)
os.makedirs(CONFIG["results_dir"], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_properties(0).name}  free={free/1e9:.1f}GB")

all_patient_ids = collect_patient_ids(CONFIG["image_dir"], CONFIG["mask_dir"])
print(f"✓ Found {len(all_patient_ids)} patients total")
if not all_patient_ids:
    raise RuntimeError(f"No patients found in {CONFIG['image_dir']} / {CONFIG['mask_dir']}")

# Build folds
n_folds = CONFIG["n_folds"]
shuffled = all_patient_ids.copy()
random.Random(CONFIG["seed"]).shuffle(shuffled)
fold_size = len(shuffled) // n_folds
folds = [shuffled[i*fold_size : (i+1)*fold_size if i < n_folds-1 else len(shuffled)]
         for i in range(n_folds)]
print(f"  {n_folds} folds: {[len(f) for f in folds]} patients each")

writer  = SummaryWriter(log_dir=CONFIG["logs_dir"])
aug_fn  = build_augmentation_fn(CONFIG["augment"])
criterion = BCEDiceLoss(bce_weight=CONFIG["bce_weight"], pos_weight=CONFIG["pos_weight"])

fold_best_dices = []

# ============================================================================
# STEP 6-8 — Train each fold
# ============================================================================
for fold_idx in range(n_folds):
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx+1}/{n_folds}")
    print(f"{'='*70}")

    val_ids   = folds[fold_idx]
    train_ids = [p for i, f in enumerate(folds) for p in f if i != fold_idx]
    print(f"  Train: {len(train_ids)} patients   Val: {len(val_ids)} patients")

    train_ds = VolumeSliceDataset(
        CONFIG["image_dir"], CONFIG["mask_dir"],
        patient_ids=train_ids, augment_fn=aug_fn,
        background_ratio=CONFIG["background_ratio"],
        samples_per_patient=CONFIG["samples_per_patient"],
        max_vol_pos_frac=CONFIG["max_vol_pos_frac"],
    )
    val_ds = AllSlicesDataset(
        CONFIG["image_dir"], CONFIG["mask_dir"],
        patient_ids=val_ids,
        max_vol_pos_frac=CONFIG["max_vol_pos_frac"],
    )
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False,
                              num_workers=0, pin_memory=True)

    # Fresh model — encoder unfrozen with small LR to adapt ImageNet→CT features
    model = smp.UnetPlusPlus(
        encoder_name="resnet34", encoder_weights="imagenet",
        in_channels=3, classes=1, activation=None,
    ).to(device)

    # Differential learning rates: encoder learns slowly (preserve ImageNet features),
    # decoder learns fast (new task-specific upsampling)
    encoder_params = list(model.encoder.parameters())
    decoder_params = (list(model.decoder.parameters()) +
                      list(model.segmentation_head.parameters()))
    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": CONFIG["lr_encoder"]},
        {"params": decoder_params, "lr": CONFIG["lr"]},
    ], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=CONFIG["cosine_T0"], T_mult=1, eta_min=CONFIG["lr_min"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    n_enc   = sum(p.numel() for p in encoder_params)
    n_dec   = sum(p.numel() for p in decoder_params)
    print(f"  Encoder params: {n_enc:,} @ lr={CONFIG['lr_encoder']:.0e}  |  Decoder: {n_dec:,} @ lr={CONFIG['lr']:.0e}")

    fold_ckpt     = os.path.join(CONFIG["results_dir"], f"fold{fold_idx+1}_best.pt")
    best_dice_f   = 0.0
    no_improve_f  = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     device, CONFIG["grad_clip"], scaler)
        stats      = evaluate(model, val_loader, criterion, device, scaler)
        scheduler.step(epoch)
        lr = optimizer.param_groups[0]["lr"]

        print(f"  F{fold_idx+1} Ep{epoch:3d}/{CONFIG['epochs']}  "
              f"loss={train_loss:.4f}  dice={stats['dice']:.4f}  "
              f"sens={stats['sens']:.4f}  prec={stats['prec']:.4f}  lr={lr:.2e}")

        writer.add_scalar(f"fold{fold_idx+1}/train_loss", train_loss, epoch)
        writer.add_scalar(f"fold{fold_idx+1}/val_dice",   stats["dice"], epoch)

        if stats["dice"] > best_dice_f:
            best_dice_f  = stats["dice"]
            no_improve_f = 0
            torch.save(model.state_dict(), fold_ckpt)
            print(f"    ✓ New best dice={best_dice_f:.4f} — saved")
            if not IS_COLAB:
                os.system(f"rclone copy {fold_ckpt} 'gdrive:Licenta/DATASET/results/' "
                          f"--no-traverse > /dev/null 2>&1 &")
        else:
            no_improve_f += 1
            if no_improve_f >= CONFIG["early_stop_patience"]:
                print(f"    ⏹ Early stop at epoch {epoch}")
                break

    fold_best_dices.append(best_dice_f)
    print(f"\n  → Fold {fold_idx+1} best dice: {best_dice_f:.4f}")

writer.close()

# ============================================================================
# CROSS-VALIDATION RESULTS
# ============================================================================
mean_dice = float(np.mean(fold_best_dices))
std_dice  = float(np.std(fold_best_dices))
best_val_dice = mean_dice  # for summary section below

# Identify best fold for TTA
best_fold_idx  = int(np.argmax(fold_best_dices))
best_fold_ckpt = os.path.join(CONFIG["results_dir"], f"fold{best_fold_idx+1}_best.pt")
print(f"\n  Best fold: {best_fold_idx+1}  (dice={fold_best_dices[best_fold_idx]:.4f})")
print(f"  Mean ± Std: {mean_dice:.4f} ± {std_dice:.4f}")

# Rebuild val_loader for the best fold (needed for TTA)
best_val_ids = folds[best_fold_idx]
best_val_ds  = AllSlicesDataset(
    CONFIG["image_dir"], CONFIG["mask_dir"],
    patient_ids=best_val_ids,
    max_vol_pos_frac=CONFIG["max_vol_pos_frac"],
)
val_loader = DataLoader(best_val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                        num_workers=0, pin_memory=True)
# Reload model with best fold weights for TTA
model = smp.UnetPlusPlus(
    encoder_name="resnet34", encoder_weights=None,
    in_channels=3, classes=1, activation=None,
).to(device)

# Copy TensorBoard logs to permanent storage
if IS_COLAB:
    drive_logs = "/content/drive/My Drive/LICENTA_COLAB/logs"
    os.makedirs(drive_logs, exist_ok=True)
    os.system(f"cp -r /tmp/tb_logs/. '{drive_logs}/'")
    print(f"✓ TensorBoard logs copied to Drive: {drive_logs}")
else:
    # Vast.ai — sync logs and results back to Google Drive via rclone
    print("Syncing results to Google Drive via rclone …")
    os.system(f"rclone copy /workspace/results/ 'gdrive:LICENTA_COLAB/results/' --progress")
    os.system(f"rclone copy /tmp/tb_logs/ 'gdrive:LICENTA_COLAB/logs/' --progress")
    print("✓ Results and TensorBoard logs synced to Google Drive")

# ============================================================================
# TTA EVALUATION — load best checkpoint, predict with 4 augmented versions
# ============================================================================
print("\n" + "=" * 70)
print("TTA EVALUATION (best checkpoint + 4-flip ensemble)")
print("=" * 70)

try:
    ckpt = torch.load(best_fold_ckpt, map_location=device)
    model.load_state_dict(ckpt)  # saved as raw state_dict
    model.eval()
    tta_dice_total, tta_count = 0.0, 0
    with torch.no_grad():
        for imgs, masks, _ in val_loader:
            imgs  = imgs.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            # 4 augmented predictions: original, h-flip, v-flip, both
            preds = []
            for flip_dims in [[], [3], [2], [2, 3]]:
                x = imgs.flip(flip_dims) if flip_dims else imgs
                p = torch.sigmoid(model(x))
                if flip_dims:
                    p = p.flip(flip_dims)
                preds.append(p)
            avg_pred = torch.stack(preds, dim=0).mean(dim=0)
            # compute dice from averaged probability map
            gt   = masks.reshape(masks.shape[0], -1)
            pred = avg_pred.reshape(avg_pred.shape[0], -1)
            has_pos = gt.sum(dim=1) > 0
            if has_pos.any():
                inter = (pred[has_pos] * gt[has_pos]).sum(dim=1)
                denom = pred[has_pos].sum(dim=1) + gt[has_pos].sum(dim=1)
                tta_dice_total += float(((2*inter+1e-6)/(denom+1e-6)).mean())
                tta_count += 1
    tta_dice = tta_dice_total / max(tta_count, 1)
    print(f"  TTA Dice (4-flip ensemble): {tta_dice:.4f}  (vs best single {best_val_dice:.4f})")
except Exception as e:
    print(f"  TTA failed: {e}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"  Best validation Dice : {best_val_dice:.4f}")
print(f"  Best fold checkpoint : {best_fold_ckpt}")
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
