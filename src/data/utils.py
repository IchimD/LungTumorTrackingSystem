import argparse
from typing import Iterable, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def parse_spacing(value: str) -> Tuple[float, float, float]:
    """
    Parse a spacing string into a 3-tuple.

    Accepts '1' (isotropic) or '1,2,3' (anisotropic).

    Parameters
    ----------
    value : str
        Spacing string.

    Returns
    -------
    Tuple[float, float, float]
        Spacing values (Z, Y, X).

    Raises
    ------
    argparse.ArgumentTypeError
        If format is invalid.
    """
    parts = [float(p) for p in value.split(",")] if "," in value else [float(value)]
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    if len(parts) == 3:
        return tuple(parts)
    raise argparse.ArgumentTypeError(
        "--output_spacing must be one value or three comma-separated values, e.g. 1 or 1,1,1"
    )


def progress_iter(sequence: Sequence[str], desc: str):
    """
    Create a progress iterator, using tqdm if available.

    Parameters
    ----------
    sequence : Sequence[str]
        Sequence to iterate over.
    desc : str
        Description for progress bar.

    Returns
    -------
    Iterator
        Progress iterator.
    """
    if tqdm is not None:
        return tqdm(sequence, desc=desc)
    return sequence
