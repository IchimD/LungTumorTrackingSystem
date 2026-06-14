"""
export_dicom_slice.py
=====================
Exports one representative slice from each of the 10 breathing phases
as .npy files ready to upload in the demo app.

Run:  python export_dicom_slice.py
"""

import glob, os
import numpy as np
import pydicom

PATIENT_DIR = "E:/LICENTA2/manifest-1781345433420/4D-Lung/100_HM10395/07-02-2003-NA-p4-14571"
OUT_DIR     = "D:/LICENTA2/demo_uploads"

os.makedirs(OUT_DIR, exist_ok=True)

def load_phase_volume(phase_pct):
    all_dirs = [d for d in glob.glob(os.path.join(PATIENT_DIR, "*")) if os.path.isdir(d)]
    best_folder, best_n = None, 0
    for d in all_dirs:
        if f"Gated {phase_pct}.0A" in os.path.basename(d):
            n = len(glob.glob(os.path.join(d, "*.dcm")))
            if n > best_n:
                best_n, best_folder = n, d
    if best_folder is None:
        return None
    files = sorted(glob.glob(os.path.join(best_folder, "*.dcm")))
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda s: float(getattr(s, "ImagePositionPatient", [0,0,0])[2]))
    return np.stack([
        ds.pixel_array.astype(np.float32)
        * float(getattr(ds, "RescaleSlope", 1))
        + float(getattr(ds, "RescaleIntercept", 0))
        for ds in slices
    ])

# Corrected tumor slice positions (from lung-restricted tracking, z_mm ~16-19mm)
TUMOR_SLICES = {
    0:  93,
    10: 94,
    20: 93,
    30: 93,
    40: 93,
    50: 93,
    60: 94,
    70: 93,
    80: 93,
    90: 93,
}

saved = []
for phase_pct in range(0, 100, 10):
    print(f"Loading phase {phase_pct}%...", end=" ")
    vol = load_phase_volume(phase_pct)
    if vol is None:
        print("NOT FOUND — skipping")
        continue

    D = vol.shape[0]
    z = TUMOR_SLICES[phase_pct]
    z = max(1, min(D-2, z))  # clamp so z-1 and z+1 exist

    three = vol[z-1:z+2].astype(np.float32)  # shape (3, H, W)
    out   = os.path.join(OUT_DIR, f"phase{phase_pct:02d}_z{z:03d}.npy")
    np.save(out, three)
    saved.append((phase_pct, z, out))
    print(f"saved  (z={z}, shape={three.shape})")

print(f"\nExported {len(saved)} files to: {OUT_DIR}\n")
print("Upload any of these in the demo app 'Upload Your CT' tab:")
for pct, z, path in saved:
    label = "(inhale)" if pct == 0 else "(exhale)" if pct == 50 else ""
    print(f"  {os.path.basename(path)}  — phase {pct}% {label}")
