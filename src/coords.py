import numpy as np


def world_to_voxel(
    world_coords: tuple[float, float, float],
    origin: tuple[float, float, float] | np.ndarray,
    spacing: tuple[float, float, float] | np.ndarray,
) -> tuple[int, int, int]:
    """
    Convert a 3D world coordinate (mm) to a voxel index in the NumPy array.

    How it works:
    - CT scan coordinates in annotations.csv are in *world space* (mm),
      using the (X, Y, Z) convention from DICOM/ITK.
    - The NumPy volume uses *voxel space* with (Z, Y, X) axis order.
    - Conversion formula for each axis:
          voxel_index = (world_coord - origin) / spacing
    - The result is rounded to the nearest integer because array indices
      must be whole numbers.
    - Axes are reordered from (X, Y, Z) → (Z, Y, X) to match the volume.

    Change from v1:
    - origin and spacing type hints broadened to accept np.ndarray as well
      as plain tuples, matching the updated return type of load_scan.

    Parameters
    ----------
    world_coords : tuple[float, float, float]
        World coordinates in mm, ordered (X, Y, Z) — as stored in the CSV.
    origin : tuple or np.ndarray, shape (3,)
        Scan origin in mm, ordered (Z, Y, X) — as returned by load_scan.
    spacing : tuple or np.ndarray, shape (3,)
        Voxel spacing in mm, ordered (Z, Y, X) — as returned by load_scan.

    Returns
    -------
    tuple[int, int, int]
        Voxel indices (z, y, x) for indexing into the volume array.

    Example
    -------
    >>> volume, spacing, origin = load_scan('path/to/scan.mhd')
    >>> nodules = get_patient_nodules(patient_id, annotations)
    >>> n = nodules[0]
    >>> vz, vy, vx = world_to_voxel(
    ...     (n['coordX'], n['coordY'], n['coordZ']),
    ...     origin, spacing
    ... )
    >>> print(volume[vz, vy, vx])   # HU value at nodule centre
    """
    # Unpack world coordinates (CSV uses X, Y, Z order)
    wx, wy, wz = world_coords

    # Unpack origin and spacing (both stored as Z, Y, X from load_scan)
    oz, oy, ox = origin
    sz, sy, sx = spacing

    # Apply the conversion and round to integer voxel indices
    vx = int(round((wx - ox) / sx))
    vy = int(round((wy - oy) / sy))
    vz = int(round((wz - oz) / sz))

    return vz, vy, vx


def voxel_to_world(
    voxel_coords: tuple[int, int, int],
    origin: tuple[float, float, float] | np.ndarray,
    spacing: tuple[float, float, float] | np.ndarray,
) -> tuple[float, float, float]:
    """
    Convert a voxel index (z, y, x) back to world coordinates (X, Y, Z) in mm.

    This is the exact inverse of world_to_voxel. It is needed whenever
    model predictions — which live in voxel space — must be reported,
    evaluated against annotations.csv, or visualised in the physical
    millimetre coordinate system used by DICOM-aware tools.

    Conversion formula for each axis:
        world_coord = voxel_index * spacing + origin

    Parameters
    ----------
    voxel_coords : tuple[int, int, int]
        Voxel indices ordered (z, y, x) — as used to index the NumPy volume.
    origin : tuple or np.ndarray, shape (3,)
        Scan origin in mm, ordered (Z, Y, X) — as returned by load_scan.
    spacing : tuple or np.ndarray, shape (3,)
        Voxel spacing in mm, ordered (Z, Y, X) — as returned by load_scan.

    Returns
    -------
    tuple[float, float, float]
        World coordinates (X, Y, Z) in mm — matching the convention used
        in annotations.csv.

    Example
    -------
    >>> volume, spacing, origin = load_scan('path/to/scan.mhd')
    >>> vz, vy, vx = world_to_voxel((wx, wy, wz), origin, spacing)
    >>> wx2, wy2, wz2 = voxel_to_world((vz, vy, vx), origin, spacing)
    >>> # wx2 ≈ wx, wy2 ≈ wy, wz2 ≈ wz  (within one voxel of rounding)
    """
    vz, vy, vx = voxel_coords

    oz, oy, ox = origin
    sz, sy, sx = spacing

    wx = vx * sx + ox
    wy = vy * sy + oy
    wz = vz * sz + oz

    return wx, wy, wz
