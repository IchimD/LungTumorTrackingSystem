import math
from typing import Sequence

import numpy as np

from coords import world_to_voxel


def create_nodule_mask(
    scan_shape: Sequence[int],
    nodules: Sequence[dict],
    spacing: Sequence[float],
    origin: Sequence[float],
) -> np.ndarray:
    """
    Create a binary 3D mask from LUNA16 nodule annotations.

    For each nodule, converts world coordinates to voxel coordinates,
    then fills a spherical region around the center.

    Parameters
    ----------
    scan_shape : Sequence[int]
        Shape of the scan volume (Z, Y, X).
    nodules : Sequence[dict]
        List of nodule dictionaries with 'coordX', 'coordY', 'coordZ', 'diameter_mm'.
    spacing : Sequence[float]
        Voxel spacing (Z, Y, X).
    origin : Sequence[float]
        Scan origin (Z, Y, X).

    Returns
    -------
    np.ndarray
        Binary mask with 1s where nodules are located.
    """
    mask = np.zeros(scan_shape, dtype=np.uint8)
    spacing = np.asarray(spacing, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)

    for nodule in nodules:
        # Convert world coordinates to voxel indices
        center_voxel = world_to_voxel(
            (nodule["coordX"], nodule["coordY"], nodule["coordZ"]),
            origin,
            spacing,
        )
        radius_mm = float(nodule["diameter_mm"]) / 2.0

        vz, vy, vx = center_voxel
        # Calculate bounding box around the sphere
        rz = int(math.ceil(radius_mm / spacing[0]))
        ry = int(math.ceil(radius_mm / spacing[1]))
        rx = int(math.ceil(radius_mm / spacing[2]))

        z0 = max(vz - rz, 0)
        z1 = min(vz + rz + 1, scan_shape[0])
        y0 = max(vy - ry, 0)
        y1 = min(vy + ry + 1, scan_shape[1])
        x0 = max(vx - rx, 0)
        x1 = min(vx + rx + 1, scan_shape[2])

        # Create coordinate grids for the bounding box
        dz = np.arange(z0, z1, dtype=np.float64) - vz
        dy = np.arange(y0, y1, dtype=np.float64) - vy
        dx = np.arange(x0, x1, dtype=np.float64) - vx

        dz_mm = dz * spacing[0]
        dy_mm = dy * spacing[1]
        dx_mm = dx * spacing[2]

        # Check which voxels are inside the sphere
        zz, yy, xx = np.meshgrid(dz_mm, dy_mm, dx_mm, indexing="ij")
        sphere = (zz**2 + yy**2 + xx**2) <= (radius_mm**2)

        # Set mask values
        mask[z0:z1, y0:y1, x0:x1] |= sphere.astype(np.uint8)

    return mask
