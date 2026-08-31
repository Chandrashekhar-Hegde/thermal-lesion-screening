import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from src.dicom_io import manifest_from_folders, read_dicom_pixels, scan_dicom_tree


def _write_dicom(path, patient_id="P001", photometric="MONOCHROME2"):
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.PatientID = patient_id
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.Modality = "XC"
    dataset.BodyPartExamined = "BREAST"
    dataset.Rows = 2
    dataset.Columns = 2
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.RescaleSlope = 2
    dataset.RescaleIntercept = 10
    dataset.PixelData = np.arange(4, dtype=np.uint16).reshape(2, 2).tobytes()
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False
    dataset.save_as(path, write_like_original=False)


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


def test_scan_dicom_tree_reads_header_identifiers_and_labels(tmp_path):
    path = tmp_path / "image.dcm"
    _write_dicom(path)

    manifest = scan_dicom_tree(tmp_path, label_map={"P001": 1})

    assert len(manifest) == 1
    assert manifest.iloc[0]["patient_id"] == "P001"
    assert manifest.iloc[0]["modality"] == "XC"
    assert manifest.iloc[0]["body_part"] == "BREAST"
    assert manifest.iloc[0]["label"] == 1


def test_read_dicom_pixels_applies_modality_lut_and_monochrome_inversion(tmp_path):
    normal = tmp_path / "normal.dcm"
    inverted = tmp_path / "inverted.dcm"
    _write_dicom(normal)
    _write_dicom(inverted, photometric="MONOCHROME1")

    np.testing.assert_allclose(read_dicom_pixels(normal), [[10, 12], [14, 16]])
    np.testing.assert_allclose(read_dicom_pixels(inverted), [[16, 14], [12, 10]])
