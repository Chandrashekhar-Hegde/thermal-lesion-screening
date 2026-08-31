"""Dataset integrity checks that run before patient-level splitting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass(frozen=True)
class IntegrityResult:
    """Clean manifest and a complete record of exclusions."""

    manifest: pd.DataFrame
    exclusions: pd.DataFrame
    report: dict


def decoded_pixel_hash(path: str | Path) -> str:
    """Hash decoded pixels without discarding high-bit-depth thermal values."""
    image_path = Path(path)
    try:
        with Image.open(image_path) as source:
            source.load()
            image = ImageOps.exif_transpose(source)
            if image.mode not in {"I", "F", "I;16", "I;16B", "I;16L"}:
                image = image.convert("RGBA")
            pixels = np.asarray(image)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"cannot decode image: {image_path}") from error

    dtype = pixels.dtype.newbyteorder("<")
    pixels = np.ascontiguousarray(pixels.astype(dtype, copy=False))
    digest = hashlib.sha256()
    digest.update(f"{pixels.shape}:{dtype.str}:".encode())
    digest.update(pixels.tobytes())
    return digest.hexdigest()


def _validate_columns(manifest: pd.DataFrame) -> None:
    required = {"path", "patient_id", "label"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing required column(s): {sorted(missing)}")
    if manifest.empty:
        raise ValueError("manifest cannot be empty")
    if manifest[list(required)].isna().any().any():
        raise ValueError("path, patient_id, and label cannot contain empty values")


def _full_record_aliases(eligible: pd.DataFrame) -> tuple[dict, set, int]:
    signatures = eligible.groupby("patient_id")["pixel_sha256"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    groups: dict[tuple[str, ...], list] = {}
    for patient_id, signature in signatures.items():
        groups.setdefault(signature, []).append(patient_id)

    aliases = {}
    ambiguous = set()
    duplicate_groups = 0
    for patients in groups.values():
        if len(patients) < 2:
            continue
        duplicate_groups += 1
        ordered = sorted(patients, key=str)
        labels = eligible.loc[eligible["patient_id"].isin(ordered), "label"].unique()
        if len(labels) == 1:
            canonical = ordered[0]
            aliases.update({patient_id: canonical for patient_id in ordered[1:]})
        else:
            ambiguous.update(ordered)
    return aliases, ambiguous, duplicate_groups


def clean_manifest_fingerprint(manifest: pd.DataFrame) -> str:
    """Fingerprint the retained patient labels and decoded images without paths."""
    required = {"patient_id", "label", "pixel_sha256"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"clean manifest is missing required column(s): {sorted(missing)}")
    digest = hashlib.sha256()
    records = manifest[list(required)].sort_values(
        ["patient_id", "label", "pixel_sha256"], key=lambda values: values.astype(str)
    )
    for row in records.itertuples(index=False):
        digest.update(f"{row.patient_id}\0{row.label}\0{row.pixel_sha256}\n".encode())
    return digest.hexdigest()


def audit_manifest(manifest: pd.DataFrame) -> IntegrityResult:
    """Remove unsafe records and report every decision.

    Same-label patient records with identical image sets are merged. Identical full
    records with conflicting labels are excluded. Partial cross-patient overlaps are
    removed from every implicated record. This avoids assigning an ambiguous image to
    a diagnosis while retaining unrelated images from otherwise usable patients.
    """
    _validate_columns(manifest)
    working = manifest.copy().reset_index(drop=True)
    working["source_index"] = working.index
    working["original_patient_id"] = working["patient_id"]

    hashes: list[str | None] = []
    decode_errors: dict[int, str] = {}
    for row in working.itertuples():
        try:
            hashes.append(decoded_pixel_hash(row.path))
        except ValueError as error:
            hashes.append(None)
            decode_errors[row.source_index] = str(error)
    working["pixel_sha256"] = hashes

    conflicting = set(
        working.groupby("patient_id")["label"].nunique().loc[lambda values: values > 1].index
    )
    reasons: dict[int, tuple[str, str]] = {}
    for row in working.itertuples():
        if row.patient_id in conflicting:
            reasons[row.source_index] = (
                "conflicting_diagnosis",
                "patient has more than one label",
            )
        elif row.source_index in decode_errors:
            reasons[row.source_index] = (
                "unreadable_image",
                decode_errors[row.source_index],
            )

    eligible = working[~working["source_index"].isin(reasons)]
    aliases, ambiguous_records, full_duplicate_groups = _full_record_aliases(eligible)
    for row in eligible.itertuples():
        if row.patient_id in ambiguous_records:
            reasons[row.source_index] = (
                "duplicate_record_conflicting_diagnosis",
                "identical patient image set has conflicting labels",
            )

    working["effective_patient_id"] = (
        working["patient_id"].map(aliases).fillna(working["patient_id"])
    )
    eligible = working[~working["source_index"].isin(reasons)]
    shared_hashes = (
        eligible.groupby("pixel_sha256")["effective_patient_id"]
        .agg(lambda values: sorted(set(values), key=str))
        .loc[lambda groups: groups.map(len) > 1]
        .to_dict()
    )
    for row in eligible.itertuples():
        if row.pixel_sha256 in shared_hashes:
            related = ", ".join(map(str, shared_hashes[row.pixel_sha256]))
            reasons[row.source_index] = (
                "cross_patient_shared_image",
                f"exact decoded pixels shared by patient IDs: {related}",
            )

    eligible = working[~working["source_index"].isin(reasons)].copy()
    eligible["is_alias"] = eligible["original_patient_id"].isin(aliases)
    duplicate_rows = set(
        eligible.sort_values(
            ["effective_patient_id", "pixel_sha256", "is_alias", "path", "source_index"]
        )
        .loc[
            lambda frame: frame.duplicated(
                subset=["effective_patient_id", "pixel_sha256"],
                keep="first",
            ),
            "source_index",
        ]
        .tolist()
    )
    for row in eligible.itertuples():
        if row.source_index not in duplicate_rows:
            continue
        if row.original_patient_id in aliases:
            reason = "merged_patient_record"
            detail = f"merged into canonical patient ID {row.effective_patient_id}"
        else:
            reason = "within_patient_duplicate"
            detail = "exact decoded pixels already represented for this patient"
        reasons[row.source_index] = (reason, detail)

    exclusion_rows = []
    for row in working.itertuples():
        if row.source_index not in reasons:
            continue
        reason, detail = reasons[row.source_index]
        exclusion_rows.append(
            {
                "source_index": int(row.source_index),
                "patient_id": row.original_patient_id,
                "path": row.path,
                "label": int(row.label),
                "pixel_sha256": row.pixel_sha256,
                "reason": reason,
                "detail": detail,
            }
        )

    cleaned = working[~working["source_index"].isin(reasons)].copy()
    cleaned["patient_id"] = cleaned["effective_patient_id"]
    cleaned = cleaned.drop(
        columns=["source_index", "original_patient_id", "effective_patient_id"]
    ).reset_index(drop=True)
    exclusion_columns = [
        "source_index",
        "patient_id",
        "path",
        "label",
        "pixel_sha256",
        "reason",
        "detail",
    ]
    exclusions = pd.DataFrame(exclusion_rows, columns=exclusion_columns)

    if cleaned.empty:
        raise ValueError("integrity policy excluded every image")
    if cleaned.groupby("patient_id")["label"].nunique().max() != 1:
        raise RuntimeError("integrity audit left a patient with conflicting labels")
    if (cleaned.groupby("pixel_sha256")["patient_id"].nunique() > 1).any():
        raise RuntimeError("integrity audit left a cross-patient duplicate")

    reason_counts = (
        exclusions["reason"].value_counts().sort_index().astype(int).to_dict()
        if not exclusions.empty
        else {}
    )
    retained_original = set(
        working.loc[~working["source_index"].isin(reasons), "original_patient_id"]
    )
    merged_aliases = set(aliases)
    excluded_patients = set(working["original_patient_id"]) - retained_original - merged_aliases
    report = {
        "policy": {
            "conflicting_diagnosis": "exclude patient",
            "same_label_identical_patient_record": "merge into canonical patient ID",
            "duplicate_record_conflicting_diagnosis": "exclude all linked patients",
            "cross_patient_shared_image": "exclude shared image from all records",
            "unreadable_image": "exclude image",
            "within_patient_duplicate": "keep one decoded image",
        },
        "before": {
            "images": int(len(working)),
            "patients": int(working["original_patient_id"].nunique()),
        },
        "after": {
            "images": int(len(cleaned)),
            "patients": int(cleaned["patient_id"].nunique()),
        },
        "excluded": {
            "images": int(len(exclusions)),
            "patients": int(len(excluded_patients)),
            "records_by_reason": reason_counts,
            "full_duplicate_record_groups": int(full_duplicate_groups),
            "partial_cross_patient_image_groups": int(len(shared_hashes)),
        },
        "merged_patient_aliases": int(len(merged_aliases)),
        "hash": "sha256 of decoded pixels with shape and normalized dtype",
        "clean_manifest_sha256": clean_manifest_fingerprint(cleaned),
    }
    return IntegrityResult(cleaned, exclusions, report)


def write_integrity_artifacts(result: IntegrityResult, output_dir: str | Path) -> None:
    """Write the clean manifest, exclusions, and summary for auditability."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result.manifest.to_csv(destination / "clean_manifest.csv", index=False)
    result.exclusions.to_csv(destination / "integrity_exclusions.csv", index=False)
    with (destination / "integrity_report.json").open("w", encoding="utf-8") as handle:
        json.dump(result.report, handle, indent=2, sort_keys=True)
        handle.write("\n")
