import pytest

from src.dicom_io import manifest_from_folders


def test_manifest_from_folders_keeps_patient_groups(tmp_path):
    healthy = tmp_path / "healthy" / "P001"
    sick = tmp_path / "sick" / "P002"
    healthy.mkdir(parents=True)
    sick.mkdir(parents=True)
    (healthy / "a.png").touch()
    (healthy / "b.png").touch()
    (sick / "a.png").touch()

    manifest = manifest_from_folders(tmp_path, {"healthy": 0, "sick": 1})

    assert len(manifest) == 3
    assert manifest.groupby("patient_id")["label"].nunique().max() == 1


def test_manifest_from_folders_rejects_conflicting_patient_labels(tmp_path):
    healthy = tmp_path / "healthy" / "P001"
    sick = tmp_path / "sick" / "P001"
    healthy.mkdir(parents=True)
    sick.mkdir(parents=True)
    (healthy / "a.png").touch()
    (sick / "b.png").touch()

    with pytest.raises(ValueError, match="conflicting labels"):
        manifest_from_folders(tmp_path, {"healthy": 0, "sick": 1})
