from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.integrity import (
    audit_manifest,
    clean_manifest_fingerprint,
    decoded_pixel_hash,
    write_integrity_artifacts,
)


def _image(path: Path, value: int) -> None:
    Image.new("L", (4, 4), color=value).save(path)


def test_pixel_hash_compares_decoded_content_across_file_formats(tmp_path):
    png = tmp_path / "image.png"
    bmp = tmp_path / "image.bmp"
    _image(png, 30)
    _image(bmp, 30)

    assert decoded_pixel_hash(png) == decoded_pixel_hash(bmp)


def test_pixel_hash_preserves_high_bit_depth_values(tmp_path):
    first_png = tmp_path / "first.png"
    first_tiff = tmp_path / "first.tiff"
    second_png = tmp_path / "second.png"
    first = np.array([[0, 256], [1_000, 65_535]], dtype=np.uint16)
    second = np.array([[0, 257], [1_000, 65_535]], dtype=np.uint16)
    Image.fromarray(first).save(first_png)
    Image.fromarray(first).save(first_tiff)
    Image.fromarray(second).save(second_png)

    assert decoded_pixel_hash(first_png) == decoded_pixel_hash(first_tiff)
    assert decoded_pixel_hash(first_png) != decoded_pixel_hash(second_png)


def test_audit_merges_same_label_identical_patient_records(tmp_path):
    paths = [tmp_path / f"{name}.png" for name in ("a", "b", "c")]
    _image(paths[0], 10)
    _image(paths[1], 10)
    _image(paths[2], 30)
    manifest = pd.DataFrame(
        [
            {"path": str(paths[0]), "patient_id": "P1", "label": 0},
            {"path": str(paths[1]), "patient_id": "P2", "label": 0},
            {"path": str(paths[2]), "patient_id": "P3", "label": 1},
        ]
    )

    result = audit_manifest(manifest)

    assert result.manifest["patient_id"].tolist() == ["P1", "P3"]
    assert result.exclusions["patient_id"].tolist() == ["P2"]
    assert result.exclusions["reason"].tolist() == ["merged_patient_record"]
    assert result.report["merged_patient_aliases"] == 1


def test_audit_excludes_identical_records_with_conflicting_labels(tmp_path):
    paths = [tmp_path / f"{name}.png" for name in ("a", "b", "c")]
    _image(paths[0], 10)
    _image(paths[1], 10)
    _image(paths[2], 30)
    manifest = pd.DataFrame(
        [
            {"path": str(paths[0]), "patient_id": "P1", "label": 0},
            {"path": str(paths[1]), "patient_id": "P2", "label": 1},
            {"path": str(paths[2]), "patient_id": "P3", "label": 1},
        ]
    )

    result = audit_manifest(manifest)

    assert result.manifest["patient_id"].tolist() == ["P3"]
    assert set(result.exclusions["patient_id"]) == {"P1", "P2"}
    assert set(result.exclusions["reason"]) == {"duplicate_record_conflicting_diagnosis"}


def test_audit_removes_partial_cross_patient_overlap_only(tmp_path):
    paths = [tmp_path / f"{index}.png" for index in range(4)]
    for index, value in enumerate([10, 20, 10, 30]):
        _image(paths[index], value)
    manifest = pd.DataFrame(
        [
            {"path": str(paths[0]), "patient_id": "P1", "label": 0},
            {"path": str(paths[1]), "patient_id": "P1", "label": 0},
            {"path": str(paths[2]), "patient_id": "P2", "label": 1},
            {"path": str(paths[3]), "patient_id": "P2", "label": 1},
        ]
    )

    result = audit_manifest(manifest)

    assert set(result.manifest["patient_id"]) == {"P1", "P2"}
    assert len(result.manifest) == 2
    assert set(result.exclusions["reason"]) == {"cross_patient_shared_image"}


def test_audit_excludes_conflicting_patient_and_deduplicates_within_patient(tmp_path):
    paths = [tmp_path / f"{index}.png" for index in range(5)]
    values = [10, 20, 30, 30, 40]
    for index, path in enumerate(paths):
        _image(path, values[index])
    manifest = pd.DataFrame(
        [
            {"path": str(paths[0]), "patient_id": "P1", "label": 0},
            {"path": str(paths[1]), "patient_id": "P1", "label": 1},
            {"path": str(paths[2]), "patient_id": "P2", "label": 0},
            {"path": str(paths[3]), "patient_id": "P2", "label": 0},
            {"path": str(paths[4]), "patient_id": "P3", "label": 1},
        ]
    )

    result = audit_manifest(manifest)

    assert result.manifest["patient_id"].tolist() == ["P2", "P3"]
    assert set(result.exclusions["reason"]) == {
        "conflicting_diagnosis",
        "within_patient_duplicate",
    }


def test_audit_records_unreadable_images_and_writes_artifacts(tmp_path):
    valid = tmp_path / "valid.png"
    invalid = tmp_path / "broken.png"
    _image(valid, 10)
    invalid.write_text("not an image")
    manifest = pd.DataFrame(
        [
            {"path": str(valid), "patient_id": "P1", "label": 0},
            {"path": str(invalid), "patient_id": "P2", "label": 1},
        ]
    )

    result = audit_manifest(manifest)
    write_integrity_artifacts(result, tmp_path / "artifacts")

    assert result.exclusions.iloc[0]["reason"] == "unreadable_image"
    assert (tmp_path / "artifacts" / "clean_manifest.csv").is_file()
    assert (tmp_path / "artifacts" / "integrity_exclusions.csv").is_file()
    assert (tmp_path / "artifacts" / "integrity_report.json").is_file()
    assert result.report["clean_manifest_sha256"] == clean_manifest_fingerprint(result.manifest)
