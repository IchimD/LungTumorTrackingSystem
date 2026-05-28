"""
PREPROCESS_KAGGLE.py
====================
Downloads and preprocesses TWO datasets into /kaggle/working/images/ and /kaggle/working/masks/

Dataset 1 — Medical Segmentation Decathlon Task06_Lung (63 patients)
  Source: Kaggle dataset  vivekprajapati2048/medical-segmentation-decathlon-lung
  Add it via Notebook Settings → Add Data before running.

Dataset 2 — NSCLC-Radiomics-NIFTI (422 patients, lung cancer GTV for radiotherapy)
  Source: HuggingFace  farrell236/NSCLC-Radiomics-NIFTI
  Downloaded automatically via huggingface_hub.

Combined: ~485 patients with lung tumor segmentation masks.

Run this once before COLAB_TRAINING_CELL.py.
"""

import os, sys, glob, warnings
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom as nd_zoom
from tqdm import tqdm

# Save to /kaggle/temp/ — up to ~100 GB temp space, sufficient for ~485 patients
OUT_IMG  = "/kaggle/working/images"
OUT_MASK = "/kaggle/working/masks"
os.makedirs(OUT_IMG,  exist_ok=True)
os.makedirs(OUT_MASK, exist_ok=True)

HU_MIN, HU_MAX = -200, 300   # same window as Decathlon preprocessing
TARGET_HW = 256              # resize each slice to 256×256 — saves ~8× disk vs 512×512

# ── helpers ──────────────────────────────────────────────────────────────────

def load_nifti_as_DHW(path: str) -> np.ndarray:
    """Load NIfTI, reorient to canonical RAS, return (D, H, W) float32 array."""
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)   # (X, Y, Z) in RAS
    data = np.moveaxis(data, -1, 0)          # → (Z, X, Y) = (D, H, W)
    return data

def resize_volume(vol: np.ndarray, target_hw: int, order: int) -> np.ndarray:
    """Resize H and W to target_hw×target_hw; keep D unchanged."""
    D, H, W = vol.shape
    if H == target_hw and W == target_hw:
        return vol
    sf_h = target_hw / H
    sf_w = target_hw / W
    return nd_zoom(vol, (1.0, sf_h, sf_w), order=order)

def save_pair(ct: np.ndarray, mask: np.ndarray, stem: str) -> bool:
    """HU-clip, resize to 256×256 slices, binarise mask, skip if no tumor, save .npy."""
    ct   = np.clip(ct, HU_MIN, HU_MAX).astype(np.float32)
    mask = (mask > 0).astype(np.float32)

    # Resize both volumes to TARGET_HW × TARGET_HW per slice
    ct   = resize_volume(ct,   TARGET_HW, order=1).astype(np.float32)
    mask = resize_volume(mask, TARGET_HW, order=0).astype(np.uint8)   # nearest for binary

    if mask.sum() == 0:
        return False
    np.save(os.path.join(OUT_IMG,  stem + ".npy"), ct)
    np.save(os.path.join(OUT_MASK, stem + ".npy"), mask)
    return True


# ============================================================================
# DATASET 1 — Decathlon Task06_Lung
# ============================================================================
print("=" * 70)
print("DATASET 1: Medical Segmentation Decathlon — Task06_Lung")
print("=" * 70)

# Kaggle mounts the added dataset here; try multiple possible sub-paths
DECATHLON_ROOTS = [
    "/kaggle/input/medical-segmentation-decathlon-lung",
    "/kaggle/input/medical-segmentation-decathlon-lung/Task06_Lung",
]
decathlon_img_dir  = None
decathlon_mask_dir = None
for root in DECATHLON_ROOTS:
    candidate_img  = os.path.join(root, "imagesTr")
    candidate_mask = os.path.join(root, "labelsTr")
    if os.path.isdir(candidate_img) and os.path.isdir(candidate_mask):
        decathlon_img_dir  = candidate_img
        decathlon_mask_dir = candidate_mask
        break

if decathlon_img_dir is None:
    print("⚠  Decathlon dataset not found — add 'vivekprajapati2048/medical-segmentation-decathlon-lung'")
    print("   via Notebook Settings → Add Data, then re-run.")
else:
    img_files = sorted(glob.glob(os.path.join(decathlon_img_dir, "*.nii.gz")))
    print(f"  Found {len(img_files)} Decathlon CT files in {decathlon_img_dir}")
    ok, skip = 0, 0
    for img_path in tqdm(img_files, desc="Decathlon"):
        stem = os.path.basename(img_path).replace(".nii.gz", "")
        mask_path = os.path.join(decathlon_mask_dir, stem + ".nii.gz")
        if not os.path.exists(mask_path):
            skip += 1
            continue
        out_stem = f"decathlon_{stem}"
        if (os.path.exists(os.path.join(OUT_IMG, out_stem + ".npy")) and
                os.path.exists(os.path.join(OUT_MASK, out_stem + ".npy"))):
            ok += 1
            continue
        try:
            ct   = load_nifti_as_DHW(img_path)
            mask = load_nifti_as_DHW(mask_path)
            if save_pair(ct, mask, out_stem):
                ok += 1
            else:
                skip += 1
        except Exception as e:
            warnings.warn(f"  {stem}: {e}")
            skip += 1
    print(f"  ✓ Decathlon: {ok} saved, {skip} skipped (no mask/no tumor)")


# ============================================================================
# DATASET 2 — NSCLC-Radiomics-NIFTI (HuggingFace)
# ============================================================================
print("\n" + "=" * 70)
print("DATASET 2: NSCLC-Radiomics-NIFTI (422 patients) — downloading from HuggingFace")
print("=" * 70)

os.system("pip install -q huggingface_hub nibabel")

from huggingface_hub import hf_hub_download, list_repo_tree

REPO_ID   = "farrell236/NSCLC-Radiomics-NIFTI"
REPO_TYPE = "dataset"
HF_CACHE  = "/kaggle/working/hf_cache"
os.makedirs(HF_CACHE, exist_ok=True)

# Patient IDs in this dataset: LUNG1-001 … LUNG1-422
# We generate all IDs and skip ones that fail (some may be missing).
patient_ids = [f"LUNG1-{i:03d}" for i in range(1, 423)]

ok, skip = 0, 0
for pid in tqdm(patient_ids, desc="NSCLC-HF"):
    out_stem = f"nsclc_{pid.lower().replace('-', '_')}"  # e.g. nsclc_lung1_006
    if (os.path.exists(os.path.join(OUT_IMG,  out_stem + ".npy")) and
            os.path.exists(os.path.join(OUT_MASK, out_stem + ".npy"))):
        ok += 1
        continue

    ct_remote   = f"NSCLC-Radiomics-NIFTI/{pid}/image.nii.gz"
    mask_remote = f"NSCLC-Radiomics-NIFTI/{pid}/seg-GTV-1.nii.gz"

    try:
        ct_local = hf_hub_download(
            repo_id=REPO_ID, repo_type=REPO_TYPE,
            filename=ct_remote, cache_dir=HF_CACHE,
        )
        mask_local = hf_hub_download(
            repo_id=REPO_ID, repo_type=REPO_TYPE,
            filename=mask_remote, cache_dir=HF_CACHE,
        )
        ct   = load_nifti_as_DHW(ct_local)
        mask = load_nifti_as_DHW(mask_local)
        if save_pair(ct, mask, out_stem):
            ok += 1
        else:
            skip += 1
    except Exception as e:
        skip += 1
        # Some patients legitimately missing GTV — not an error
    finally:
        # Delete cached NIfTI files immediately to conserve disk space
        for f in glob.glob(os.path.join(HF_CACHE, "**", "*.nii.gz"), recursive=True):
            try: os.remove(f)
            except: pass

print(f"\n  ✓ NSCLC: {ok} saved, {skip} skipped")

# ============================================================================
# SUMMARY
# ============================================================================
total_img  = len(glob.glob(os.path.join(OUT_IMG,  "*.npy")))
total_mask = len(glob.glob(os.path.join(OUT_MASK, "*.npy")))
print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)
print(f"  Images : {total_img}  →  {OUT_IMG}")
print(f"  Masks  : {total_mask}  →  {OUT_MASK}")
print(f"\nNext step: run COLAB_TRAINING_CELL.py")
