import os
import argparse

# ---------------------------------------------------------------------------
# Re-export facade — keeps all existing imports working unchanged while the
# actual implementations live in the focused sub-modules:
#
#   scan_io.py       — CT volume filesystem I/O
#   annotation_io.py — CSV metadata loading and patient indexing
#   coords.py        — world ↔ voxel coordinate conversion
# ---------------------------------------------------------------------------
from scan_io import list_all_patients, load_scan
from annotation_io import (
    load_annotations,
    load_candidates,
    build_patient_index,
    get_patient_nodules,
)
from coords import world_to_voxel, voxel_to_world

__all__ = [
    "list_all_patients",
    "load_scan",
    "load_annotations",
    "load_candidates",
    "build_patient_index",
    "get_patient_nodules",
    "world_to_voxel",
    "voxel_to_world",
]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Change from v1: paths are now CLI arguments instead of hardcoded strings,
    # making the script portable across machines and dataset layouts.
    parser = argparse.ArgumentParser(description="Visualise LUNA16 nodule annotations.")
    parser.add_argument(
        "--subset",
        default=r"D:\LICENTA2\DATASET\subset0",
        help="Path to a LUNA16 subset folder containing .mhd files.",
    )
    parser.add_argument(
        "--csv",
        default=r"D:\LICENTA2\DATASET\annotations.csv",
        help="Path to annotations.csv.",
    )
    args = parser.parse_args()

    SUBSET = args.subset
    CSV    = args.csv

    # --- 1. List all patients ---
    patients = list_all_patients(SUBSET)
    print(f"Total patients in subset: {len(patients)}")
    print("Patient IDs:")
    for i, pid in enumerate(patients):
        print(f"  [{i:02d}] {pid}")

    # --- 2. Load annotations and build O(1) lookup index ---
    # Change from v1: build_patient_index replaces per-iteration O(n) DataFrame scans.
    annotations   = load_annotations(CSV)
    patient_index = build_patient_index(annotations)
    print(f"\nAnnotations: {len(annotations)} nodules across all subsets")

    # --- 3. Find first patient in the subset that has nodules ---
    target_pid     = None
    target_nodules = []
    for pid in patients:
        nodules = get_patient_nodules(pid, patient_index)  # O(1) via index
        if nodules:
            target_pid     = pid
            target_nodules = nodules
            break

    if target_pid is None:
        print("No patients with nodules found in this subset.")
    else:
        print(f"\nShowing scan for patient: {target_pid}")
        print(f"  Nodules found: {len(target_nodules)}")

        # --- 4. Load the scan ---
        mhd_path = os.path.join(SUBSET, target_pid + ".mhd")
        volume, spacing, origin = load_scan(mhd_path)
        print(f"  Volume shape (Z, Y, X): {volume.shape}")
        print(f"  Volume dtype:           {volume.dtype}")   # now always float32
        print(f"  Spacing (Z, Y, X) mm:   {spacing}")

        # --- 5. Convert first nodule to voxel coords ---
        n = target_nodules[0]
        vz, vy, vx = world_to_voxel(
            (n["coordX"], n["coordY"], n["coordZ"]), origin, spacing
        )

        # Round-trip check: voxel → world should recover the original CSV coords
        # within rounding error (≤ half a voxel). Uses the new voxel_to_world function.
        wx_rt, wy_rt, wz_rt = voxel_to_world((vz, vy, vx), origin, spacing)
        print(f"  Original world coords : ({n['coordX']:.2f}, {n['coordY']:.2f}, {n['coordZ']:.2f}) mm")
        print(f"  Round-trip world coords: ({wx_rt:.2f}, {wy_rt:.2f}, {wz_rt:.2f}) mm")

        diameter_voxels = n["diameter_mm"] / spacing[1]
        radius = int(diameter_voxels / 2)
        print(f"  Nodule centre voxel: z={vz}, y={vy}, x={vx}")
        print(f"  Diameter: {n['diameter_mm']:.1f} mm (~{int(diameter_voxels)} voxels)")

        # --- 6. Show 3 orthogonal slices with nodule marked ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"Patient: {target_pid[:40]}...\n"
            f"Nodule at voxel (z={vz}, y={vy}, x={vx}), "
            f"diameter={n['diameter_mm']:.1f} mm",
            fontsize=9,
        )

        # Axial slice (top-down view) at nodule z
        axes[0].imshow(volume[vz], cmap="gray", vmin=-1000, vmax=400)
        axes[0].plot(vx, vy, "r+", markersize=14, markeredgewidth=2)
        circle0 = plt.Circle((vx, vy), radius, color="red", fill=False, linewidth=1.5)
        axes[0].add_patch(circle0)
        axes[0].set_title(f"Axial  (z={vz})")
        axes[0].axis("off")

        # Coronal slice (front view) at nodule y
        axes[1].imshow(volume[:, vy, :], cmap="gray", vmin=-1000, vmax=400)
        axes[1].plot(vx, vz, "r+", markersize=14, markeredgewidth=2)
        circle1 = plt.Circle((vx, vz), radius, color="red", fill=False, linewidth=1.5)
        axes[1].add_patch(circle1)
        axes[1].set_title(f"Coronal  (y={vy})")
        axes[1].axis("off")

        # Sagittal slice (side view) at nodule x
        axes[2].imshow(volume[:, :, vx], cmap="gray", vmin=-1000, vmax=400)
        axes[2].plot(vy, vz, "r+", markersize=14, markeredgewidth=2)
        circle2 = plt.Circle((vy, vz), radius, color="red", fill=False, linewidth=1.5)
        axes[2].add_patch(circle2)
        axes[2].set_title(f"Sagittal  (x={vx})")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

