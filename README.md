# Thermal lesion screening

[![CI](https://github.com/Chandrashekhar-Hegde/thermal-lesion-screening/actions/workflows/ci.yml/badge.svg)](https://github.com/Chandrashekhar-Hegde/thermal-lesion-screening/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A reproducible baseline for breast thermography screening. The pipeline keeps
patients isolated across splits, selects the operating threshold on validation data,
and reports patient-level performance with confidence intervals.

This is research software, not a medical device. It is not intended for diagnosis or
patient care.

## What is included

- Folder and DICOM manifest creation using a real patient identifier
- Stratified, patient-grouped train, validation, test, and cross-validation splits
- ResNet-18 and MobileNetV3-Small binary classifiers
- Validation-selected threshold for a target sensitivity
- Patient-level AUROC, AUPRC, sensitivity, specificity, PPV, NPV, and bootstrap intervals
- ONNX export and CPU latency benchmarking
- Tests that fail on patient leakage and invalid evaluation inputs

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the checks:

```bash
pip install -r requirements-ci.txt
ruff check .
ruff format --check .
pytest
```

## Data

The default configuration targets the
[DMR-IR database](https://visual.ic.uff.br/dmi/). Register and download it from the
maintainer; this repository does not redistribute clinical data.

Expected folder layout:

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

The patient identifier must be a directory level. Do not flatten images into a
single folder. For DICOM input, `src/dicom_io.py` reads `PatientID` from the header.

Never commit clinical images, manifests containing identifiers, or model artifacts.
The default `.gitignore` excludes `data/` and `artifacts/`.

## Train and evaluate

Review `configs/default.yaml`, then run:

```bash
python -m src.train --config configs/default.yaml
```

The run:

1. creates a patient-level split;
2. selects the best checkpoint on patient-level validation AUROC;
3. selects the operating threshold on validation patients;
4. evaluates the held-out test patients once; and
5. exports the report, checkpoint, split summary, and ONNX model to `artifacts/`.

Multiple image scores for one patient are averaged by default. Change
`eval.patient_score_reducer` to `max` only when that rule is defined before test-set
evaluation and is justified for the deployment workflow.

## Benchmark the exported model

Run this command on the intended deployment hardware:

```bash
python -m src.benchmark_edge --onnx artifacts/model.onnx --runs 200
```

The benchmark reports mean, median, p95, and p99 latency plus throughput and hardware
details.

## Repository layout

```text
configs/default.yaml       Training and evaluation settings
src/dicom_io.py            DICOM and folder manifest creation
src/splits.py              Patient-grouped splits and leakage checks
src/model.py               Models and ONNX export
src/metrics.py             Patient aggregation and diagnostic metrics
src/train.py               Training and held-out evaluation
src/benchmark_edge.py      Edge latency benchmark
tests/                     Regression tests
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please report security or sensitive-data
issues through the process in [SECURITY.md](SECURITY.md), not in a public issue.

## Citation

Citation metadata is available in [CITATION.cff](CITATION.cff).

## License

Released under the [MIT License](LICENSE).
