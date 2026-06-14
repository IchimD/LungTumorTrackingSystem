"""
Preprocess Medical Segmentation Decathlon Task06_Lung
=====================================================
Run this ONCE on Vast.ai after downloading the data.
Converts .nii.gz volumes to .npy files the training cell expects.

Usage:
  python PREPROCESS_DECATHLON.py
  (or paste into a Jupyter cell)

Input:  /workspace/Task06_Lung/imagesTr/  and  /workspace/Task06_Lung/labelsTr/
Output: /workspace/images/  and  /workspace/masks/
"""

import os, glob
import numpy as np

os.system("pip install -q nibabel")
import nibabel as nib

SRC   = "/workspace/Task06_Lung"
IMGS  = "/workspace/images"
MSKS  = "/workspace/masks"
os.makedirs(IMGS, exist_ok=True)
os.makedirs(MSKS, exist_ok=True)

# CT window for lung tumors: clip [-200, 300] HU then normalize to [0, 1]
HU_MIN, HU_MAX = -200, 300

files = sorted(glob.glob(os.path.join(SRC, "imagesTr", "*.nii.gz")))
print(f"Found {len(files)} cases")

for img_path in files:
    fname   = os.path.basename(img_path)                      # lung_001.nii.gz
    stem    = fname.replace(".nii.gz", "")                    # lung_001
    msk_path = os.path.join(SRC, "labelsTr", fname)

    if not os.path.exists(msk_path):
        print(f"  SKIP {stem} — no matching label")
        continue

    # Load image — NIfTI spatial order is (H, W, D), transpose to (D, H, W)
    img_nib = nib.load(img_path)
    img = img_nib.get_fdata(dtype=np.float32)
    if img.ndim == 4:
        img = img[..., 0]
    img = img.transpose(2, 0, 1).copy()                       # (D, H, W)

    # Clip to lung-tumor HU window and normalize to [0, 1]
    img = np.clip(img, HU_MIN, HU_MAX)
    img = (img - HU_MIN) / (HU_MAX - HU_MIN)
    img = img.astype(np.float32)

    # Load label
    msk = nib.load(msk_path).get_fdata(dtype=np.float32)
    msk = msk.transpose(2, 0, 1).copy()                       # (D, H, W)
    msk = (msk > 0).astype(np.uint8)

    pos_slices = int(msk.any(axis=(1, 2)).sum())
    total      = msk.shape[0]
    vol_frac   = float(msk.sum()) / max(msk.size, 1)

    np.save(os.path.join(IMGS, f"{stem}.npy"), img)
    np.save(os.path.join(MSKS, f"{stem}.npy"), msk)
    print(f"  {stem}  shape={img.shape}  pos_slices={pos_slices}/{total}  vol_frac={vol_frac:.3%}")

print(f"\nDone. Images -> {IMGS}   Masks -> {MSKS}")
print("Now run the training cell — image_dir and mask_dir are already set correctly.")
