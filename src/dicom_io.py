"""
DICOM ingest and manifest construction.

Clinical imaging does not arrive as PNGs in a folder. It arrives as DICOM from a
PACS, with the patient identifier, modality, and acquisition parameters living in
the header rather than the filename. Any imaging model that cannot read DICOM is
a research demo, not a deployable component — and this is the single most common
gap between an academic imaging CV and a clinical-imaging job description.

This module reads a DICOM tree, pulls PatientID straight from the header (so
patient grouping is derived from the source of truth rather than guessed from a
filename), and emits the manifest consumed by src/splits.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import pydicom

    try:
        from pydicom.pixels import apply_modality_lut
    except ImportError:  # pydicom < 3
        from pydicom.pixel_data_handlers.util import apply_modality_lut

    HAVE_PYDICOM = True
except ImportError:  # keeps folder-based workflows importable without pydicom
    HAVE_PYDICOM = False


# Tags worth carrying forward. Thermography is stored under a mix of SC
# (Secondary Capture) and XC; do not assume a single modality string.
KEEP_TAGS = {
    "PatientID": "patient_id",
    "StudyInstanceUID": "study_uid",
    "SeriesInstanceUID": "series_uid",
    "SOPInstanceUID": "sop_uid",
    "Modality": "modality",
    "BodyPartExamined": "body_part",
    "Rows": "rows",
    "Columns": "cols",
    "Manufacturer": "manufacturer",
}


def read_dicom_pixels(path: str | Path) -> np.ndarray:
    """Read one DICOM file and return a float32 array in physical units.

    `apply_modality_lut` maps stored pixel values through RescaleSlope /
    RescaleIntercept. Skipping it is a routine and silent source of error: the
    model trains on raw stored integers whose meaning varies between scanners,
    then fails the moment it meets a device from a different vendor.
    """
    if not HAVE_PYDICOM:
        raise ImportError("pydicom is required for DICOM ingest: pip install pydicom")

    ds = pydicom.dcmread(str(path))
    arr = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() + arr.min() - arr

    return arr


def scan_dicom_tree(root: str | Path, label_map: dict[str, int] | None = None) -> pd.DataFrame:
    """Walk a DICOM directory and build a manifest.

    Args:
        root: directory to walk recursively.
        label_map: {PatientID -> 0/1}. Ground truth belongs with the clinical
            record, never inferred from a folder name.

    Returns:
        DataFrame with at minimum: path, patient_id, label.
    """
    if not HAVE_PYDICOM:
        raise ImportError("pydicom is required for DICOM ingest: pip install pydicom")

    root = Path(root)
    rows, skipped = [], 0

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception:
            skipped += 1
            continue

        rec = {"path": str(p)}
        for tag, col in KEEP_TAGS.items():
            rec[col] = getattr(ds, tag, None)

        if not rec.get("patient_id"):
            logger.warning("no PatientID in %s — cannot group safely, skipping", p)
            skipped += 1
            continue

        rows.append(rec)

    if skipped:
        logger.info("skipped %d unreadable or unidentifiable file(s)", skipped)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"no readable DICOM files found under {root}")

    if label_map is not None:
        df["label"] = df["patient_id"].map(label_map)
        missing = df["label"].isna().sum()
        if missing:
            raise ValueError(
                f"{missing} image(s) have no label in label_map. Refusing to "
                "guess or drop silently — supply a complete mapping."
            )
        df["label"] = df["label"].astype(int)

    logger.info("manifest: %d images, %d patients", len(df), df["patient_id"].nunique())
    return df


def manifest_from_folders(
    root: str | Path,
    classes: dict[str, int],
    *,
    validate_patient_labels: bool = True,
) -> pd.DataFrame:
    """Fallback for plain image datasets (e.g. the DMR-IR export).

    Expects <root>/<class_name>/<patient_id>/<image>.png. The patient identifier
    MUST be a real directory level — if you flatten patients into a single folder
    you cannot group correctly, and every metric downstream becomes unusable.
    """
    root = Path(root)
    rows = []
    for cls_name, label in classes.items():
        cls_dir = root / cls_name
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"expected class directory: {cls_dir}")
        for pat_dir in sorted(d for d in cls_dir.iterdir() if d.is_dir()):
            for img in sorted(pat_dir.glob("*")):
                if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                    continue
                rows.append(
                    {
                        "path": str(img),
                        "patient_id": pat_dir.name,
                        "label": int(label),
                        "image_id": str(img.relative_to(root)),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"no images found under {root}")

    if validate_patient_labels:
        duplicates = df.groupby("patient_id")["label"].nunique()
        if (duplicates > 1).any():
            bad = duplicates[duplicates > 1].index.tolist()
            raise ValueError(f"patient(s) {bad} carry conflicting labels. Resolve before training.")
    return df
