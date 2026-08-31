"""Run dataset integrity checks without starting model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.dicom_io import manifest_from_folders
from src.integrity import audit_manifest, write_integrity_artifacts


def run(config_path: str) -> dict:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    manifest = manifest_from_folders(
        config["data"]["root"],
        config["data"]["classes"],
        validate_patient_labels=False,
    )
    result = audit_manifest(manifest)
    destination = Path(config.get("artifacts", {}).get("root", "artifacts"))
    write_integrity_artifacts(result, destination)
    print(json.dumps(result.report, indent=2, sort_keys=True))
    return result.report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    run(parser.parse_args().config)
