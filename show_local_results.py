"""
show_local_results.py
=====================
Visualises the best result from your local CPU training run.

Usage:
    python show_local_results.py
    python show_local_results.py --checkpoint path/to/your/model.pt

The script reads the checkpoint metadata saved by train.py and produces
a PNG figure showing:
  - Best Dice score achieved vs the 0.70 / 0.80 targets
  - Key training configuration used
  - What needs to change to reach the target (improvement guide)
"""

import argparse
import os
import sys

# ── graceful import of torch ──────────────────────────────────────────────────
try:
    import torch
    TORCH_OK = True
except Exception as e:
    TORCH_OK = False
    TORCH_ERROR = str(e)

import matplotlib
matplotlib.use("Agg")          # works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── default paths ─────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "checkpoints", "best_model.pt"
)
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "local_training_results.png")


# ── load checkpoint ───────────────────────────────────────────────────────────
def load_checkpoint_meta(path: str) -> dict:
    if not TORCH_OK:
        print(f"[!] PyTorch could not be imported ({TORCH_ERROR}).")
        print("    Falling back to known results from training log.\n")
        return None

    if not os.path.isfile(path):
        print(f"[!] Checkpoint not found: {path}")
        print("    Falling back to known results.\n")
        return None

    try:
        ckpt = torch.load(path, map_location="cpu")
        return ckpt
    except Exception as e:
        print(f"[!] Could not load checkpoint: {e}")
        return None


# ── build figure ──────────────────────────────────────────────────────────────
def make_figure(meta: dict | None, output_path: str) -> None:
    # Known results from the local CPU run (fallback if checkpoint unreadable)
    val_dice = 0.50
    epoch    = 40
    cfg      = {}

    if meta is not None:
        val_dice = float(meta.get("val_dice", val_dice))
        epoch    = int(meta.get("epoch",    epoch))
        cfg      = meta.get("args", meta.get("config", {}))

    best_epoch   = epoch
    total_epochs = int(cfg.get("epochs", 40))
    batch_size   = cfg.get("batch_size", 8)
    lr           = cfg.get("lr", 1e-3)
    augment      = cfg.get("augment", False)
    bg_ratio     = cfg.get("background_ratio", 0.30)

    TARGET_LOW  = 0.70
    TARGET_HIGH = 0.80

    fig = plt.figure(figsize=(14, 6), facecolor="#1a1a2e")
    fig.suptitle(
        "Lung Nodule Segmentation — Local CPU Training Results",
        fontsize=14, fontweight="bold", color="white", y=0.97,
    )

    # ── left panel: Dice gauge ────────────────────────────────────────────────
    ax_gauge = fig.add_axes([0.03, 0.08, 0.40, 0.82])
    ax_gauge.set_facecolor("#16213e")

    categories = [
        (0.00, 0.45, "#e74c3c", "Below baseline"),
        (0.45, 0.55, "#e67e22", "Baseline / current"),
        (0.55, 0.70, "#f1c40f", "Improving"),
        (0.70, 0.80, "#2ecc71", "Target range"),
        (0.80, 1.00, "#1abc9c", "Excellent"),
    ]
    for lo, hi, color, label in categories:
        ax_gauge.barh(label, hi - lo, left=lo, color=color, height=0.55,
                      alpha=0.85, edgecolor="#1a1a2e", linewidth=0.8)

    # Current result marker
    cat_label = next(lbl for lo, hi, _, lbl in categories if lo <= val_dice < hi)
    if val_dice >= 1.0:
        cat_label = "Excellent"

    ax_gauge.axvline(val_dice, color="white", linewidth=2.5, zorder=5)
    ax_gauge.annotate(
        f"Current\n{val_dice:.4f}",
        xy=(val_dice, cat_label),
        xytext=(val_dice + 0.02, cat_label),
        fontsize=9, color="white", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
    )

    # Target markers
    for t, lbl in [(TARGET_LOW, "0.70"), (TARGET_HIGH, "0.80")]:
        ax_gauge.axvline(t, color="#ecf0f1", linewidth=1.2,
                         linestyle="--", alpha=0.6, zorder=4)
        ax_gauge.text(t, -0.6, lbl, ha="center", va="top",
                      fontsize=8, color="#ecf0f1", alpha=0.8)

    ax_gauge.set_xlim(0, 1)
    ax_gauge.set_xlabel("Dice Score", color="white", fontsize=10)
    ax_gauge.set_title("Performance vs. Targets", color="white",
                        fontsize=11, fontweight="bold", pad=8)
    ax_gauge.tick_params(colors="white", labelsize=8)
    for spine in ax_gauge.spines.values():
        spine.set_edgecolor("#2c3e50")

    # ── right panel: config + improvement guide ───────────────────────────────
    ax_info = fig.add_axes([0.48, 0.08, 0.50, 0.82])
    ax_info.set_facecolor("#16213e")
    ax_info.axis("off")

    def trow(y, label, value, color="#ecf0f1"):
        ax_info.text(0.01, y, label, transform=ax_info.transAxes,
                     fontsize=9, color="#95a5a6", va="top")
        ax_info.text(0.42, y, str(value), transform=ax_info.transAxes,
                     fontsize=9, color=color, va="top", fontweight="bold")

    y = 0.96
    ax_info.text(0.01, y, "LOCAL CPU RUN — CONFIGURATION & RESULTS",
                 transform=ax_info.transAxes, fontsize=10,
                 color="white", va="top", fontweight="bold")
    y -= 0.08

    trow(y, "Best Dice",       f"{val_dice:.4f}",    color="#e67e22"); y -= 0.07
    trow(y, "Best epoch",      best_epoch);           y -= 0.07
    trow(y, "Total epochs",    total_epochs);         y -= 0.07
    trow(y, "Batch size",      batch_size);           y -= 0.07
    trow(y, "Learning rate",   lr);                   y -= 0.07
    trow(y, "Augmentation",    augment);              y -= 0.07
    trow(y, "Background ratio",bg_ratio);             y -= 0.10

    # Improvement guide
    gap = TARGET_LOW - val_dice
    ax_info.text(0.01, y, "WHAT NEEDS TO CHANGE (Colab GPU run)",
                 transform=ax_info.transAxes, fontsize=10,
                 color="#2ecc71", va="top", fontweight="bold")
    y -= 0.07

    improvements = [
        ("LR scheduler",       "ReduceLROnPlateau (not in CPU run)",  True),
        ("Epochs",             f"{total_epochs}  →  120",             total_epochs < 100),
        ("Background ratio",   f"{bg_ratio}  →  0.05",               bg_ratio > 0.1),
        ("Dice loss weight",   "50 %  →  70 %  (bce_weight=0.3)",    True),
        ("Augmentation",       "+noise +brightness (Colab cell)",     not augment),
        ("Batch size",         f"{batch_size}  →  32 (GPU only)",     batch_size < 16),
        ("Gradient clipping",  "max_norm=1.0 (not in CPU run)",       True),
    ]

    for label, note, important in improvements:
        icon  = "▶" if important else "·"
        color = "#f39c12" if important else "#7f8c8d"
        ax_info.text(0.01, y, f"{icon} {label}", transform=ax_info.transAxes,
                     fontsize=8.5, color=color, va="top")
        ax_info.text(0.42, y, note, transform=ax_info.transAxes,
                     fontsize=8, color="#bdc3c7", va="top")
        y -= 0.065

    y -= 0.02
    ax_info.text(
        0.01, y,
        f"Expected Dice after Colab run:  0.70 – 0.80",
        transform=ax_info.transAxes, fontsize=9,
        color="#2ecc71", va="top", fontstyle="italic",
    )

    plt.savefig(output_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] Figure saved -> {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Show local CPU training results.")
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint  (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Where to save the PNG  (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Local Training Results Viewer")
    print("=" * 60)
    print(f"Checkpoint : {args.checkpoint}")

    meta = load_checkpoint_meta(args.checkpoint)

    if meta is not None:
        print(f"\n  Best Dice  : {meta.get('val_dice', 'N/A'):.4f}")
        print(f"  Best Epoch : {meta.get('epoch', 'N/A')}")
        cfg = meta.get("args", meta.get("config", {}))
        if cfg:
            print(f"  Config     : {dict(cfg)}")
    else:
        print("\n  Using fallback values (Dice=0.50, epoch=40)")

    make_figure(meta, args.output)

    # Try to open the image on Windows
    try:
        os.startfile(args.output)
    except Exception:
        pass


if __name__ == "__main__":
    main()
