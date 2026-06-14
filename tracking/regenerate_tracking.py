"""
regenerate_tracking.py
========================
Re-runs tumor tracking across all 10 phases using LUNG-RESTRICTED inference
(only considers CT slices in the thorax region, z_mm > -50mm).
Then generates comparison plots vs. radiologist ground truth.

Run:  python regenerate_tracking.py
"""

import os, glob, json, time
import numpy as np
import torch
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import zoom, label, center_of_mass
import segmentation_models_pytorch as smp

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "D:/LICENTA2/results/fold2_best.pt"
PATIENT_DIR  = "E:/LICENTA2/manifest-1781345433420/4D-Lung/100_HM10395/07-02-2003-NA-p4-14571"
CACHE_DIR    = "D:/LICENTA2/demo_cache"
TRACKING_DIR = "D:/LICENTA2/tracking_results"
IMG_SIZE     = 384
HU_MIN, HU_MAX = -200, 300
LUNG_Z_MIN_MM = -50.0   # only use slices above this z level (thorax)

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TRACKING_DIR, exist_ok=True)

BG     = "#070b14"
CARD   = "#0d1117"
ACC    = "#388bfd"
RED    = "#ff6b6b"
GRN    = "#56d364"
TXT    = "#e6edf3"
SUB    = "#8b949e"
BORDER = "#1e2535"
GOLD   = "#f9c513"

CT_FOLDERS = {
    0:  "1.000000-P4P100S300I00003 Gated 0.0A-29193",
    10: "1.000000-P4P100S300I00004 Gated 10.0A-82400",
    20: "1.000000-P4P100S300I00005 Gated 20.0A-81534",
    30: "1.000000-P4P100S300I00006 Gated 30.0A-66113",
    40: "1.000000-P4P100S300I00007 Gated 40.0A-45229",
    50: "1.000000-P4P100S300I00008 Gated 50.0A-57212",
    60: "1.000000-P4P100S300I00009 Gated 60.0A-99194",
    70: "1.000000-P4P100S300I00010 Gated 70.0A-85451",
    80: "1.000000-P4P100S300I00011 Gated 80.0A-15277",
    90: "1.000000-P4P100S300I00012 Gated 90.0A-70956",
}
RT_FOLDERS = {
    0:  "1.000000-P4P100S300I00003 Gated 0.0A-423.1",
    10: "1.000000-P4P100S300I00004 Gated 10.0A-423.2",
    20: "1.000000-P4P100S300I00005 Gated 20.0A-423.3",
    30: "1.000000-P4P100S300I00006 Gated 30.0A-423.4",
    40: "1.000000-P4P100S300I00007 Gated 40.0A-423.5",
    50: "1.000000-P4P100S300I00008 Gated 50.0A-423.6",
    60: "1.000000-P4P100S300I00009 Gated 60.0A-423.7",
    70: "1.000000-P4P100S300I00010 Gated 70.0A-423.8",
    80: "1.000000-P4P100S300I00011 Gated 80.0A-423.9",
    90: "1.000000-P4P100S300I00012 Gated 90.0A-23.10",
}

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading model...")
model = smp.UnetPlusPlus(encoder_name="efficientnet-b4", encoder_weights=None,
                          in_channels=3, classes=1, activation=None)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()
print("Model loaded OK")


# ── DICOM helpers ──────────────────────────────────────────────────────────────
def load_phase(phase_pct):
    folder = os.path.join(PATIENT_DIR, CT_FOLDERS[phase_pct])
    files  = sorted(glob.glob(os.path.join(folder, "*.dcm")))
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    vol = np.stack([
        ds.pixel_array.astype(np.float32)
        * float(getattr(ds, "RescaleSlope", 1))
        + float(getattr(ds, "RescaleIntercept", 0))
        for ds in slices
    ])
    zs  = np.array([float(s.ImagePositionPatient[2]) for s in slices])
    ipp = np.array([float(v) for v in slices[0].ImagePositionPatient])
    ps  = np.array([float(v) for v in slices[0].PixelSpacing])
    return vol, zs, ipp, ps


def infer_slice(vol, z):
    D = vol.shape[0]
    ch = []
    for dz in (-1, 0, 1):
        zi = max(0, min(D-1, z+dz))
        s  = np.clip(vol[zi], HU_MIN, HU_MAX).astype(np.float32)
        sf = (IMG_SIZE/s.shape[0], IMG_SIZE/s.shape[1])
        s  = zoom(s, sf, order=1)
        s  = (s - HU_MIN) / (HU_MAX - HU_MIN)
        ch.append(s)
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(np.stack(ch)[None]))).squeeze().numpy()


def get_gt_centroid(phase_pct):
    folder  = os.path.join(PATIENT_DIR, RT_FOLDERS[phase_pct])
    rt_file = glob.glob(os.path.join(folder, "*.dcm"))[0]
    ds      = pydicom.dcmread(rt_file)
    tumor_num = None
    for roi in ds.StructureSetROISequence:
        if roi.ROIName.lower().startswith("tumor"):
            tumor_num = roi.ROINumber; break
    if tumor_num is None: return None
    all_pts = []
    for cr in ds.ROIContourSequence:
        if cr.ReferencedROINumber != tumor_num: continue
        for c in getattr(cr, "ContourSequence", []):
            pts = np.array(c.ContourData, dtype=float).reshape(-1, 3)
            all_pts.append(pts)
        break
    if not all_pts: return None
    return np.vstack(all_pts).mean(axis=0)   # (x_mm, y_mm, z_mm)


# ── Track all phases with lung restriction ─────────────────────────────────────
print("\nTracking with lung-region restriction (z_mm > -50mm)...\n")
phases = list(range(0, 100, 10))
results = {}

for ph in phases:
    print(f"Phase {ph:2d}%  ", end="", flush=True)
    vol, zs, ipp, ps = load_phase(ph)
    D = vol.shape[0]

    # Find lung slice indices (z_mm > LUNG_Z_MIN_MM)
    lung_slices = np.where(zs > LUNG_Z_MIN_MM)[0]
    print(f"lung slices {lung_slices[0]}-{lung_slices[-1]} (z_mm {zs[lung_slices[0]]:.0f} to {zs[lung_slices[-1]]:.0f})  ", end="", flush=True)

    # Run inference on lung slices only
    probs_lung = np.zeros((len(lung_slices), IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i, z in enumerate(lung_slices):
        probs_lung[i] = infer_slice(vol, z)

    mask_lung = (probs_lung >= 0.5).astype(np.uint8)

    if mask_lung.sum() == 0:
        print("NO DETECTION")
        results[ph] = None
        continue

    # Keep largest component in lung region
    labeled, n = label(mask_lung)
    if n > 0:
        sizes = [(labeled == i).sum() for i in range(1, n+1)]
        mask_lung = (labeled == np.argmax(sizes)+1).astype(np.uint8)

    # Center of mass in lung-restricted space
    cz_r, cy_r, cx_r = center_of_mass(mask_lung)

    # Map back to original slice index and 512-space
    orig_z  = lung_slices[int(round(cz_r))]
    scale   = vol.shape[1] / IMG_SIZE   # 512/384
    cy_512  = cy_r * scale
    cx_512  = cx_r * scale

    # Convert to mm
    z_mm = zs[orig_z]
    y_mm = ipp[1] + cy_512 * ps[0]
    x_mm = ipp[0] + cx_512 * ps[1]

    # GT centroid
    gt_mm = get_gt_centroid(ph)
    err_mm = None
    if gt_mm is not None:
        err_mm = float(np.sqrt((gt_mm[0]-x_mm)**2 + (gt_mm[1]-y_mm)**2 + (gt_mm[2]-z_mm)**2))

    results[ph] = {
        "orig_slice": int(orig_z),
        "cy_384": float(cy_r), "cx_384": float(cx_r),
        "cy_512": float(cy_512), "cx_512": float(cx_512),
        "z_mm": float(z_mm), "y_mm": float(y_mm), "x_mm": float(x_mm),
        "gt_x_mm": float(gt_mm[0]) if gt_mm is not None else None,
        "gt_y_mm": float(gt_mm[1]) if gt_mm is not None else None,
        "gt_z_mm": float(gt_mm[2]) if gt_mm is not None else None,
        "error_3d_mm": err_mm,
    }

    print(f"z={orig_z} z_mm={z_mm:.1f}  GT_z_mm={gt_mm[2]:.1f}  err={err_mm:.1f}mm")

# Save
with open(os.path.join(CACHE_DIR, "tracking_corrected.json"), "w") as f:
    json.dump(results, f, indent=2)

valid  = {ph: r for ph, r in results.items() if r is not None}
errs   = [r["error_3d_mm"] for r in valid.values() if r["error_3d_mm"] is not None]
mean_e = np.mean(errs) if errs else 0
print(f"\nMean 3D error: {mean_e:.2f} mm")


# ── Generate comparison plot ───────────────────────────────────────────────────
print("\nGenerating comparison plots...")

ph_list = sorted(valid.keys())
mod_z   = [valid[p]["z_mm"]   for p in ph_list]
mod_y   = [valid[p]["y_mm"]   for p in ph_list]
mod_x   = [valid[p]["x_mm"]   for p in ph_list]
gt_z    = [valid[p]["gt_z_mm"] for p in ph_list]
gt_y    = [valid[p]["gt_y_mm"] for p in ph_list]
gt_x    = [valid[p]["gt_x_mm"] for p in ph_list]
errs_l  = [valid[p]["error_3d_mm"] for p in ph_list]

fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor=BG)
fig.subplots_adjust(wspace=0.32, left=0.06, right=0.97, top=0.80, bottom=0.18)

axis_data = [
    ("Superior-Inferior (z)",   gt_z,  mod_z),
    ("Anterior-Posterior (y)",  gt_y,  mod_y),
    ("Left-Right (x)",          gt_x,  mod_x),
]

for ax, (title, gt_arr, mod_arr) in zip(axes, axis_data):
    ax.set_facecolor(CARD)
    for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
    ax.tick_params(colors=SUB, labelsize=9)

    ax.plot(ph_list, gt_arr,  "o-",  color=GOLD, lw=2.5, markersize=9,
            markerfacecolor=BG, markeredgewidth=2.5, zorder=3,
            label="Radiation oncologist (RTSTRUCT)")
    ax.plot(ph_list, mod_arr, "s--", color=ACC,  lw=2.5, markersize=8,
            markerfacecolor=BG, markeredgewidth=2.5, zorder=3,
            label="AI model (U-Net++)")

    ax.fill_between(ph_list, gt_arr, mod_arr, alpha=0.10, color=RED)

    gt_range  = max(gt_arr)  - min(gt_arr)
    mod_range = max(mod_arr) - min(mod_arr)
    ax.text(0.03, 0.97,
            f"Doctor range: {gt_range:.1f} mm\nAI range:     {mod_range:.1f} mm",
            transform=ax.transAxes, va="top", color=TXT, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc=CARD, ec=BORDER, alpha=0.95))

    ax.set_xlabel("Breathing Phase (%)", color=SUB, fontsize=10)
    ax.set_ylabel("Position (mm)", color=SUB, fontsize=10)
    ax.set_title(title, color=TXT, fontsize=12, fontweight="600", pad=10)
    ax.set_xticks(ph_list)
    ax.grid(color=BORDER, linewidth=0.8, alpha=0.5)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=11,
           facecolor=CARD, edgecolor=BORDER, labelcolor=TXT, bbox_to_anchor=(0.5, 0.01))

fig.text(0.5, 0.93,
         "AI vs. Radiologist: Tumor Position across Breathing Cycle",
         ha="center", color=TXT, fontsize=15, fontweight="700")
fig.text(0.5, 0.88,
         f"Mean 3D error: {mean_e:.1f} mm  |  Patient 100_HM10395  |  10 phases",
         ha="center", color=SUB, fontsize=11)

plt.savefig(os.path.join(TRACKING_DIR, "comparison_tracking.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("  Saved comparison_tracking.png")

# ── Error bar chart ────────────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
ax.set_facecolor(CARD)
for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
ax.tick_params(colors=SUB, labelsize=10)

std_e = np.std(errs_l)
bar_colors = [GRN if e < 5 else ACC if e < 15 else RED for e in errs_l]
bars = ax.bar(ph_list, errs_l, color=bar_colors, width=7, edgecolor=BORDER, linewidth=0.8)
ax.fill_between(
    [ph_list[0] - 5, ph_list[-1] + 5],
    mean_e - std_e, mean_e + std_e,
    color=GOLD, alpha=0.10, zorder=0,
)
ax.axhline(mean_e, color=GOLD, lw=2, ls="--", label=f"Mean: {mean_e:.1f} mm  ±{std_e:.1f} mm std")
ax.axhline(5,  color=GRN, lw=1, ls=":", alpha=0.7, label="< 5 mm (excellent)")
ax.axhline(15, color=RED, lw=1, ls=":", alpha=0.7, label="< 15 mm (acceptable)")

for bar, err in zip(bars, errs_l):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{err:.1f}", ha="center", va="bottom", color=TXT, fontsize=10, fontweight="600")

ax.set_xlabel("Breathing Phase (%)", color=SUB, fontsize=11)
ax.set_ylabel("3D Localization Error (mm)", color=SUB, fontsize=11)
ax.set_xticks(ph_list)
ax.set_xticklabels([f"{p}%" for p in ph_list])
ax.grid(axis="y", color=BORDER, linewidth=0.8, alpha=0.5)
ax.legend(fontsize=10, facecolor=CARD, edgecolor=BORDER, labelcolor=TXT)

fig2.text(0.5, 0.97,
          "Localization Error per Phase (AI vs. Radiologist GTV)",
          ha="center", color=TXT, fontsize=13, fontweight="700", va="top")
fig2.subplots_adjust(top=0.88, bottom=0.14, left=0.09, right=0.97)

plt.savefig(os.path.join(TRACKING_DIR, "error_per_phase.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig2)
print("  Saved error_per_phase.png")

print(f"\nDone. Mean 3D error: {mean_e:.1f} mm")
