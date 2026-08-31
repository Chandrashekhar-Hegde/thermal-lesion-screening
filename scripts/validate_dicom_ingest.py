"""Create a path-free validation summary for a public DICOM series."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pydicom

from src.dicom_io import read_dicom_pixels, scan_dicom_tree


def validate(root: str | Path, source: str, series_uid: str | None = None) -> dict:
    manifest = scan_dicom_tree(root)
    if series_uid is not None:
        observed = set(manifest["series_uid"].astype(str))
        if observed != {series_uid}:
            raise ValueError(f"expected series {series_uid}, found {sorted(observed)}")

    pixel_arrays = [read_dicom_pixels(path) for path in manifest["path"]]
    return {
        "source": source,
        "pydicom_version": pydicom.__version__,
        "manifest_columns": sorted(manifest.columns.tolist()),
        "counts": {
            "instances": int(len(manifest)),
            "patients": int(manifest["patient_id"].nunique()),
            "studies": int(manifest["study_uid"].nunique()),
            "series": int(manifest["series_uid"].nunique()),
        },
        "modalities": sorted(manifest["modality"].dropna().astype(str).unique()),
        "body_parts": sorted(manifest["body_part"].dropna().astype(str).unique()),
        "manufacturers": sorted(manifest["manufacturer"].dropna().astype(str).unique()),
        "pixel_read": {
            "instances_succeeded": len(pixel_arrays),
            "shapes": sorted({"x".join(map(str, array.shape)) for array in pixel_arrays}),
            "dtypes": sorted({str(array.dtype) for array in pixel_arrays}),
        },
        "series_instance_uids": sorted(manifest["series_uid"].astype(str).unique()),
        "study_instance_uids": sorted(manifest["study_uid"].astype(str).unique()),
        "sop_instance_uids_unique": bool(manifest["sop_uid"].is_unique),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--source", required=True)
    parser.add_argument("--series-uid")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = validate(args.root, args.source, args.series_uid)
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
