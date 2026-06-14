"""
precompute_phases.py  (v2 — with zoom inset)
=============================================
Generates per-breathing-phase comparison panels.
Each panel = 3 columns:
  LEFT   — CT + doctor contour (green)
  MIDDLE — zoomed crop 4x magnified around tumor, both contours
  RIGHT  — CT + AI detection (red)
"""

import os, glob, json
import numpy as np
import pydicom
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, center_of_mass
import segmentation_models_pytorch as smp

MODEL_PATH  = "D:/LICENTA2/results/fold2_best.pt"
PATIENT_DIR = "E:/LICENTA2/manifest-1781345433420/4D-Lung/100_HM10395/07-02-2003-NA-p4-14571"
CACHE_DIR   = "D:/LICENTA2/demo_cache"
IMG_SIZE    = 384
HU_MIN, HU_MAX = -200, 300
TUMOR_Z_IDX = {0:93, 10:94, 20:93, 30:93, 40:93, 50:93, 60:94, 70:93, 80:93, 90:93}
CROP_HALF   = 55   # half-size of zoom crop in 384-pixels

BG   = "#070b14";  CARD = "#0d1117";  TXT = "#e6edf3"
SUB  = "#8b949e";  BORDER = "#1e2535"
GREEN = "#22ff66"; RED = "#ff6b6b"

CT_FOLDERS = {
    0:"1.000000-P4P100S300I00003 Gated 0.0A-29193",
    10:"1.000000-P4P100S300I00004 Gated 10.0A-82400",
    20:"1.000000-P4P100S300I00005 Gated 20.0A-81534",
    30:"1.000000-P4P100S300I00006 Gated 30.0A-66113",
    40:"1.000000-P4P100S300I00007 Gated 40.0A-45229",
    50:"1.000000-P4P100S300I00008 Gated 50.0A-57212",
    60:"1.000000-P4P100S300I00009 Gated 60.0A-99194",
    70:"1.000000-P4P100S300I00010 Gated 70.0A-85451",
    80:"1.000000-P4P100S300I00011 Gated 80.0A-15277",
    90:"1.000000-P4P100S300I00012 Gated 90.0A-70956",
}
RT_FOLDERS = {
    0:"1.000000-P4P100S300I00003 Gated 0.0A-423.1",
    10:"1.000000-P4P100S300I00004 Gated 10.0A-423.2",
    20:"1.000000-P4P100S300I00005 Gated 20.0A-423.3",
    30:"1.000000-P4P100S300I00006 Gated 30.0A-423.4",
    40:"1.000000-P4P100S300I00007 Gated 40.0A-423.5",
    50:"1.000000-P4P100S300I00008 Gated 50.0A-423.6",
    60:"1.000000-P4P100S300I00009 Gated 60.0A-423.7",
    70:"1.000000-P4P100S300I00010 Gated 70.0A-423.8",
    80:"1.000000-P4P100S300I00011 Gated 80.0A-423.9",
    90:"1.000000-P4P100S300I00012 Gated 90.0A-23.10",
}

os.makedirs(CACHE_DIR, exist_ok=True)

print("Loading model...")
model = smp.UnetPlusPlus(encoder_name="efficientnet-b4", encoder_weights=None,
                          in_channels=3, classes=1, activation=None)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()
print("OK\n")

def to_hu(ds):
    return ds.pixel_array.astype(np.float32)*float(getattr(ds,"RescaleSlope",1))+float(getattr(ds,"RescaleIntercept",0))

def norm_384(hu):
    s = np.clip(hu, HU_MIN, HU_MAX).astype(np.float32)
    sf = (IMG_SIZE/s.shape[0], IMG_SIZE/s.shape[1])
    return zoom((s-HU_MIN)/(HU_MAX-HU_MIN), sf, order=1)

def infer(hu_p, hu_c, hu_n):
    ch = np.stack([norm_384(hu_p), norm_384(hu_c), norm_384(hu_n)])
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(ch[None]))).squeeze().numpy()

def get_contours(phase_pct, z_mm_target, ipp, ps):
    folder  = os.path.join(PATIENT_DIR, RT_FOLDERS[phase_pct])
    rt_file = glob.glob(os.path.join(folder, "*.dcm"))[0]
    ds = pydicom.dcmread(rt_file)
    tumor_num = None
    for roi in ds.StructureSetROISequence:
        if roi.ROIName.lower().startswith("tumor"):
            tumor_num = roi.ROINumber; break
    if tumor_num is None: return []
    scale = IMG_SIZE / 512.0
    out = []
    for cr in ds.ROIContourSequence:
        if cr.ReferencedROINumber != tumor_num: continue
        for c in getattr(cr,"ContourSequence",[]):
            pts = np.array(c.ContourData,dtype=float).reshape(-1,3)
            if abs(pts[0,2] - z_mm_target) > 6: continue
            cols = (pts[:,0] - ipp[0]) / ps[1] * scale
            rows = (pts[:,1] - ipp[1]) / ps[0] * scale
            out.append(np.stack([cols, rows], axis=1))
        break
    return out

print("Generating panels...")
phase_meta = {}

for ph in range(0, 100, 10):
    z_idx = TUMOR_Z_IDX[ph]
    folder = os.path.join(PATIENT_DIR, CT_FOLDERS[ph])
    files  = sorted(glob.glob(os.path.join(folder,"*.dcm")))
    sls    = [pydicom.dcmread(f) for f in files]
    sls.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    ds_c = sls[z_idx]
    ds_p = sls[max(0, z_idx-1)]
    ds_n = sls[min(len(sls)-1, z_idx+1)]
    ipp  = np.array([float(v) for v in ds_c.ImagePositionPatient])
    ps   = np.array([float(v) for v in ds_c.PixelSpacing])
    z_mm = float(ds_c.ImagePositionPatient[2])

    hu_c, hu_p, hu_n = to_hu(ds_c), to_hu(ds_p), to_hu(ds_n)
    ct_384 = norm_384(hu_c)
    prob   = infer(hu_p, hu_c, hu_n)
    mask   = (prob >= 0.5).astype(np.float32)
    n_px   = int(mask.sum())
    conf   = float(prob[mask>0].mean())*100 if n_px>0 else 0

    contours = get_contours(ph, z_mm, ipp, ps)

    # Centroid positions
    if contours:
        all_pts = np.vstack(contours)
        doc_cx, doc_cy = float(all_pts[:,0].mean()), float(all_pts[:,1].mean())
    else:
        doc_cx, doc_cy = IMG_SIZE//2, IMG_SIZE//2

    if n_px > 0:
        ai_cy_f, ai_cx_f = center_of_mass(mask)
        ai_cx, ai_cy = float(ai_cx_f), float(ai_cy_f)
    else:
        ai_cx, ai_cy = doc_cx, doc_cy

    # Zoom crop center = average of both centroids
    ctr_x = int((doc_cx + ai_cx) / 2)
    ctr_y = int((doc_cy + ai_cy) / 2)
    x1 = max(0, ctr_x - CROP_HALF); x2 = min(IMG_SIZE, ctr_x + CROP_HALF)
    y1 = max(0, ctr_y - CROP_HALF); y2 = min(IMG_SIZE, ctr_y + CROP_HALF)

    ct_crop   = ct_384[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 7), facecolor=BG)
    ax1 = fig.add_axes([0.01, 0.08, 0.30, 0.82])   # Doctor full view
    ax2 = fig.add_axes([0.34, 0.08, 0.30, 0.82])   # Zoomed crop (both)
    ax3 = fig.add_axes([0.67, 0.08, 0.30, 0.82])   # AI full view
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(BG); ax.axis("off")

    phase_lbl = "Full Inhale" if ph==0 else "Full Exhale" if ph==90 else f"Phase {ph}%"

    # ── LEFT: Doctor ───────────────────────────────────────────────────────────
    ax1.imshow(ct_384, cmap="gray", vmin=0, vmax=1)
    for pts in contours:
        poly = plt.Polygon(pts, fill=True, facecolor=(0.1,0.9,0.3,0.2),
                           edgecolor=GREEN, linewidth=2.5)
        ax1.add_patch(poly)
    ax1.plot(doc_cx, doc_cy, "+", color=GREEN, markersize=18, markeredgewidth=3)
    # Zoom box indicator
    rect = plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False,
                          edgecolor="white", linewidth=1.5, linestyle="--", alpha=0.6)
    ax1.add_patch(rect)
    ax1.set_title(f"DOCTOR  (radiation oncologist)", color=GREEN,
                  fontsize=13, fontweight="700", pad=8)
    ax1.text(0.5, -0.05,
             f"Manual contour drawn by a specialist\nCentroid: ({doc_cx:.0f}, {doc_cy:.0f}) px",
             transform=ax1.transAxes, ha="center", color=SUB, fontsize=9.5)

    # ── MIDDLE: Zoomed crop ────────────────────────────────────────────────────
    ax2.imshow(ct_crop, cmap="gray", vmin=0, vmax=1, extent=[x1,x2,y2,y1])
    # Doctor contour crop
    for pts in contours:
        in_box = pts[(pts[:,0]>=x1)&(pts[:,0]<=x2)&(pts[:,1]>=y1)&(pts[:,1]<=y2)]
        if len(in_box) > 2:
            poly2 = plt.Polygon(in_box, fill=True, facecolor=(0.1,0.9,0.3,0.25),
                                edgecolor=GREEN, linewidth=3)
            ax2.add_patch(poly2)
    # AI mask crop overlay
    if n_px > 0:
        rgba = np.zeros((*mask_crop.shape, 4), dtype=np.float32)
        rgba[:,:,0]=mask_crop; rgba[:,:,1]=mask_crop*0.1; rgba[:,:,3]=mask_crop*0.5
        ax2.imshow(rgba, extent=[x1,x2,y2,y1])
    # Both centroids
    ax2.plot(doc_cx, doc_cy, "o", color=GREEN, markersize=14,
             markerfacecolor="none", markeredgewidth=3, label="Doctor")
    ax2.plot(ai_cx,  ai_cy,  "x", color=RED,   markersize=14,
             markeredgewidth=3, label="AI")
    ax2.set_xlim(x1, x2); ax2.set_ylim(y2, y1)
    ax2.legend(fontsize=10, facecolor=CARD, edgecolor=BORDER,
               labelcolor=TXT, loc="lower right")
    dist_px = np.sqrt((doc_cx-ai_cx)**2+(doc_cy-ai_cy)**2)
    ax2.set_title(f"ZOOM  (4× magnified tumor area)", color=TXT,
                  fontsize=13, fontweight="700", pad=8)
    ax2.text(0.5, -0.05,
             f"Green circle = doctor  |  Red X = AI  |  Distance: {dist_px:.1f} px",
             transform=ax2.transAxes, ha="center", color=SUB, fontsize=9.5)
    # Axes ticks in zoom to show position
    ax2.axis("on")
    ax2.tick_params(colors=SUB, labelsize=8)
    for spine in ax2.spines.values(): spine.set_edgecolor(BORDER)

    # ── RIGHT: AI ──────────────────────────────────────────────────────────────
    ax3.imshow(ct_384, cmap="gray", vmin=0, vmax=1)
    if n_px > 0:
        rgba_full = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba_full[:,:,0]=mask; rgba_full[:,:,1]=mask*0.1; rgba_full[:,:,3]=mask*0.5
        ax3.imshow(rgba_full)
        ax3.contour(mask, levels=[0.5], colors=[RED], linewidths=2.5)
    ax3.plot(ai_cx, ai_cy, "+", color=RED, markersize=18, markeredgewidth=3)
    # Same zoom box
    rect2 = plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False,
                           edgecolor="white", linewidth=1.5, linestyle="--", alpha=0.6)
    ax3.add_patch(rect2)
    ax3.set_title(f"AI MODEL  (U-Net++ automatic)", color=RED,
                  fontsize=13, fontweight="700", pad=8)
    ax3.text(0.5, -0.05,
             f"Detected automatically — no human input\n{n_px} pixels at {conf:.0f}% confidence",
             transform=ax3.transAxes, ha="center", color=SUB, fontsize=9.5)

    # Top title
    fig.text(0.5, 0.97,
             f"Breathing Phase {ph}%  —  {phase_lbl}  —  CT slice at z = {z_mm:.0f} mm",
             ha="center", color=TXT, fontsize=14, fontweight="700")
    fig.text(0.5, 0.93,
             "Dashed box = zoomed region shown in center panel",
             ha="center", color=SUB, fontsize=10)

    out = os.path.join(CACHE_DIR, f"phase_{ph:02d}_panel.png")
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    phase_meta[str(ph)] = {
        "z_idx": z_idx, "z_mm": z_mm,
        "ai_px": n_px, "ai_conf": round(conf,1),
        "ai_cx": round(ai_cx,1), "ai_cy": round(ai_cy,1),
        "doc_cx": round(doc_cx,1), "doc_cy": round(doc_cy,1),
        "dist_px": round(dist_px,1),
        "doc_contours": len(contours),
    }
    print(f"  Phase {ph:2d}%  AI=({ai_cx:.0f},{ai_cy:.0f})  DOC=({doc_cx:.0f},{doc_cy:.0f})  dist={dist_px:.1f}px  saved")

with open(os.path.join(CACHE_DIR, "phase_meta.json"), "w") as f:
    json.dump(phase_meta, f, indent=2)
print("\nDone.")
