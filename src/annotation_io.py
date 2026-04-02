import os
import pandas as pd

# ---------------------------------------------------------------------------
# Expected CSV columns — validated by load_annotations / load_candidates so
# that a wrong file path surfaces a clear error rather than a mystery KeyError
# downstream.
# ---------------------------------------------------------------------------
_ANNOTATION_COLUMNS = {"seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"}
_CANDIDATE_COLUMNS  = {"seriesuid", "coordX", "coordY", "coordZ", "class"}


def load_annotations(csv_path: str) -> pd.DataFrame:
    """
    Load the LUNA16 annotations CSV into a pandas DataFrame.

    Expected columns: seriesuid, coordX, coordY, coordZ, diameter_mm.

    Changes from v1:
    - Raises ValueError if csv_path does not exist.
    - seriesuid is explicitly read as str to prevent pandas from silently
      casting long numeric UIDs to float64, which would corrupt equality
      comparisons in get_patient_nodules / build_patient_index.
    - Validates that all expected columns are present; raises a descriptive
      ValueError if any are missing (e.g. if candidates.csv was passed by
      mistake instead of annotations.csv).

    Parameters
    ----------
    csv_path : str
        Absolute path to annotations.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: seriesuid, coordX, coordY, coordZ,
        diameter_mm. One row per annotated nodule.

    Example
    -------
    >>> annotations = load_annotations(r'D:/LICENTA2/DATASET/annotations.csv')
    >>> print(annotations.shape)        # e.g. (1186, 5)
    >>> print(annotations.columns.tolist())
    ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm']
    """
    if not os.path.isfile(csv_path):
        raise ValueError(f"Annotations CSV not found: {csv_path!r}")

    df = pd.read_csv(csv_path, dtype={"seriesuid": str})

    missing = _ANNOTATION_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Annotations CSV is missing expected columns: {sorted(missing)}. "
            f"Got: {df.columns.tolist()}"
        )

    return df


def load_candidates(csv_path: str) -> pd.DataFrame:
    """
    Load the LUNA16 candidates CSV into a pandas DataFrame.

    The candidates file (candidates.csv) ships alongside annotations.csv
    and contains one row per candidate location with columns:
    seriesuid, coordX, coordY, coordZ, class.

    'class' is 1 for true nodule candidates (positives) and 0 for false
    positives, making this file essential for false-positive reduction
    training — a separate stage from the primary nodule detector.

    Parameters
    ----------
    csv_path : str
        Absolute path to candidates.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: seriesuid, coordX, coordY, coordZ, class.
        One row per candidate location.

    Raises
    ------
    ValueError
        If the file does not exist or expected columns are missing.

    Example
    -------
    >>> candidates = load_candidates(r'D:/LICENTA2/DATASET/candidates.csv')
    >>> positives = candidates[candidates['class'] == 1]
    """
    if not os.path.isfile(csv_path):
        raise ValueError(f"Candidates CSV not found: {csv_path!r}")

    df = pd.read_csv(csv_path, dtype={"seriesuid": str})

    missing = _CANDIDATE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Candidates CSV is missing expected columns: {sorted(missing)}. "
            f"Got: {df.columns.tolist()}"
        )

    return df


def build_patient_index(annotations: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Pre-index an annotations (or candidates) DataFrame by seriesuid for
    O(1) per-patient lookup.

    How it works:
    - Groups the DataFrame once by seriesuid using groupby.
    - Converts each group to a list of dicts — the same format returned
      by get_patient_nodules — so the two are interchangeable.
    - Returns a plain dict keyed by seriesuid.

    Why this matters:
    get_patient_nodules with a raw DataFrame does a full O(n) linear scan
    on every call. In a training loop over ~888 patients across 10 subsets
    this accumulates to millions of row comparisons per epoch. Building the
    index once before the loop and passing the resulting dict to
    get_patient_nodules reduces each lookup to O(1).

    Parameters
    ----------
    annotations : pd.DataFrame
        The full annotations or candidates DataFrame (from load_annotations
        or load_candidates).

    Returns
    -------
    dict[str, list[dict]]
        Maps seriesuid → list of nodule/candidate dicts.
        Patients with no entries are absent from the dict.

    Example
    -------
    >>> annotations = load_annotations(r'D:/LICENTA2/DATASET/annotations.csv')
    >>> index = build_patient_index(annotations)
    >>> index['1.3.6.1.4.1.14519...']   # instant O(1) lookup
    """
    index: dict[str, list[dict]] = {}
    for uid, group in annotations.groupby("seriesuid", sort=False):
        index[uid] = group.reset_index(drop=True).to_dict(orient="records")
    return index


def get_patient_nodules(
    patient_id: str,
    annotations: pd.DataFrame | dict[str, list[dict]],
) -> list[dict]:
    """
    Retrieve all annotated nodules that belong to a specific patient.

    Accepts either:
    - A pre-built index dict (from build_patient_index) — O(1) lookup,
      the preferred form inside training loops over many patients.
    - A raw pd.DataFrame (from load_annotations) — convenient for
      one-off lookups but performs an O(n) linear scan per call; avoid
      inside loops over all patients.

    Change from v1:
    - Now accepts a dict[str, list[dict]] in addition to a DataFrame.
      Pass the result of build_patient_index(annotations) for loops.

    Parameters
    ----------
    patient_id : str
        The seriesuid of the patient (as returned by list_all_patients).
    annotations : pd.DataFrame or dict[str, list[dict]]
        Either the full annotations DataFrame or a pre-built patient index.

    Returns
    -------
    list[dict]
        List of nodule dicts, each with keys:
        'seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm'.
        Empty list if no nodules are found for this patient.

    Example
    -------
    >>> # Efficient — build the index once, reuse per patient
    >>> index = build_patient_index(load_annotations(csv_path))
    >>> nodules = get_patient_nodules('1.3.6.1.4.1.14519...', index)
    """
    # O(1) path — preferred inside training loops
    if isinstance(annotations, dict):
        return annotations.get(patient_id, [])

    # DataFrame path — kept for backward compatibility / one-off use
    mask         = annotations["seriesuid"] == patient_id
    patient_rows = annotations[mask]
    return patient_rows.reset_index(drop=True).to_dict(orient="records")
