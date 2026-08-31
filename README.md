# Thermal lesion screening

[![CI](https://github.com/Chandrashekhar-Hegde/thermal-lesion-screening/actions/workflows/ci.yml/badge.svg)](https://github.com/Chandrashekhar-Hegde/thermal-lesion-screening/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A patient-level baseline for breast thermography screening. It audits the source data,
keeps patients isolated across folds, selects the operating rule on validation patients,
and reports uncertainty around patient-level metrics.

This is research software, not a medical device. It is not intended for diagnosis or
patient care.

## Evaluation status

No DMR-IR benchmark is published yet. The dataset is not redistributed here and was not
present when this repository revision was validated. Publishing placeholder numbers or a
model result from a single split would be misleading.

The configured protocol is five-fold, five-repeat patient-level nested cross-validation:

| Stage | Rule |
|---|---|
| Integrity | Decode and hash every image before splitting |
| Outer test | Every patient is tested once per repeat |
| Inner validation | Checkpoint, reducer, and threshold selection only |
| Operating point | Target 90% validation sensitivity; report resulting test specificity |
| Uncertainty | Spread across repeats and patient-cluster bootstrap intervals |
| Unit | Patients, never images |

A completed run generates `artifacts/RESULTS.md` and exactly three figures: ROC with a
patient-cluster bootstrap band, the sensitivity–specificity trade-off, and a patient-level
confusion matrix. Those files should be reviewed before any aggregate result is committed.

## Known DMR-IR issues

Pérez-Martín and Sánchez-Cauce reported 365 anomalies in the database snapshot they
reviewed, including exact thermal-image duplication across different patient records and
identical consecutive dynamic frames. Their findings were current to January 2021; this
pipeline does not assume that every current download is unchanged. See
[“Quality analysis of a breast thermal images database”](https://doi.org/10.1177/14604582231153779),
*Health Informatics Journal* 29(1), 2023.

Before splitting, the integrity pass:

- excludes patient IDs carrying conflicting labels;
- merges same-label patient records with identical complete image sets;
- excludes identical complete records when their labels conflict;
- removes exact partial cross-patient image overlaps from both records;
- keeps one copy of an exact within-patient duplicate; and
- records unreadable images instead of dropping them silently.

It hashes decoded pixels, shape, and normalized data type rather than file bytes, so
metadata-only differences do not hide a duplicate and 16-bit thermal values are preserved.
This is exact matching; it does not claim to detect cropped, mirrored, or lossy re-encoded
near-duplicates.

Run the audit without training:

```bash
python -m src.audit_data --config configs/default.yaml
```

Detailed exclusions contain paths and patient identifiers, so `artifacts/` remains
gitignored. Only aggregate counts belong in a public result.

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For repository checks:

```bash
pip install -r requirements-ci.txt
ruff check .
ruff format --check .
python -m pytest
```

CI also runs a synthetic end-to-end smoke test through evaluation, refit, ONNX export,
and ONNX Runtime inference. It verifies the workflow, not model performance.

## Data

Register for the [DMR-IR database](https://visual.ic.uff.br/dmi/) and download it from
the maintainer. Clinical data is not included in this repository.

Expected layout:

```text
data/dmr-ir/
├── healthy/
│   └── PAT_001/
│       ├── IR_0001.png
│       └── IR_0002.png
└── sick/
    └── PAT_002/
        ├── IR_0001.png
        └── IR_0002.png
```

The patient identifier must be a directory level. Do not flatten images into one folder.

## Repeated evaluation

Review `configs/default.yaml`, then run:

```bash
python -m src.train --config configs/default.yaml
```

This trains 25 outer-fold models. Within each fold, early stopping uses patient-level
validation AUROC. Mean, max, and top-three mean aggregation are compared on validation
patients; specificity at the fixed sensitivity target selects the reducer and threshold.
The untouched outer test patients are evaluated once.

Images are weighted inversely by their patient's image count in the loss, so a patient
with more views does not contribute more total training weight. Fixed-reducer estimates
are written as a paired exploratory comparison; they are not used to choose a headline
method after looking at outer-fold results.

Main outputs:

```text
artifacts/integrity_report.json
artifacts/integrity_exclusions.csv
artifacts/clean_manifest.csv
artifacts/cv_report.json
artifacts/cv_predictions.csv
artifacts/cv_reducer_predictions.csv
artifacts/RESULTS.md
artifacts/figures/roc_curve.png
artifacts/figures/sensitivity_specificity.png
artifacts/figures/confusion_matrix.png
```

## DICOM and PACS path

DMR-IR uses the folder loader. The DICOM loader is separately exercised against a fixed,
public three-instance breast-MR series from NCI IDC I-SPY1: one patient, one study, one
series, with all pixels decoded successfully. See [the reproducible check](docs/dicom.md)
and its [path-free validation summary](docs/validation/ispy1_dicom_ingest.json).

The same document contains an Orthanc upload/retrieval runbook. It is clearly marked as
not executed on the validation host because Docker and Orthanc were unavailable.

## Deployment artifact

Cross-validation does not nominate one fold model for deployment. After a completed run,
refit on all clean patients for the median selected epoch count:

```bash
python -m src.refit --config configs/default.yaml \
  --cv-report artifacts/cv_report.json
python -m src.benchmark_edge --onnx artifacts/model.onnx --runs 200
```

The refit model has no independent test estimate; its metadata says so. Benchmark the
ONNX file on the intended device, not the training machine.

## Repository layout

```text
configs/default.yaml       Evaluation settings
src/integrity.py           Pixel hashing, deduplication, and exclusions
src/dicom_io.py            DICOM and folder manifests
src/splits.py              Patient-level repeated nested folds
src/metrics.py             Aggregation, operating points, and intervals
src/reporting.py           Markdown results and three figures
src/train.py               Repeated cross-validation
src/refit.py               Unevaluated all-data deployment refit
src/benchmark_edge.py      ONNX Runtime latency benchmark
tests/                     Regression tests
```

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md). Report sensitive-data or security issues through
[SECURITY.md](SECURITY.md), not a public issue. Released under the [MIT License](LICENSE).
Citation metadata is in [CITATION.cff](CITATION.cff).
