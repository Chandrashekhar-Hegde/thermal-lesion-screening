# Thermal Lesion Screening — a reproducible, patient-level pipeline

A screening classifier for medical thermography, built to the standard a clinical
imaging team would actually accept: **patient-level splits, sensitivity reported
with specificity, confidence intervals on every headline number, DICOM ingest, and
latency measured on the deployment hardware rather than the training machine.**

Nothing here is novel research. The point is that the methodology is not wrong —
which, in this corner of the literature, is unfortunately not the default.

---

## Why this repository exists

Most published thermography classifiers report a single, spectacular sensitivity
figure. A large share of them are not measuring what they claim to measure, for two
reasons that recur constantly:

**1. They split by image, not by patient.** Thermography datasets contain several
frames per patient. If frames from one patient land on both sides of the train/test
boundary, the network can identify the *person* instead of the *pathology*. The
reported number goes up. Clinical performance does not. The result is worthless — and
worse, it is worthless in a way that looks like success.

**2. They report sensitivity alone.** A model that returns "positive" for every
patient scores 100% sensitivity. Sensitivity without specificity is not a weak
result; it is not a result at all.

This repository is built so that neither failure is possible.
`src/splits.py` cannot emit a patient-leaking split — `assert_no_patient_leakage`
runs before any split is returned, and `tests/test_splits.py` deliberately corrupts a
split to prove the guard fires. `src/metrics.py` has no code path that reports
sensitivity without also reporting specificity, an operating threshold, and a
bootstrap interval.

---

## What it does

```
DICOM / image tree
        │
        ├─ src/dicom_io.py       PatientID read from the header, not guessed from a filename
        │
        ├─ src/splits.py         stratified + PATIENT-GROUPED train/val/test
        │                        └── assert_no_patient_leakage()  ← hard fail
        │
        ├─ src/train.py          ResNet-18, val-selected, test touched exactly once
        │
        ├─ src/metrics.py        AUROC · sens & spec at a named threshold · 95% CIs · calibration
        │
        └─ src/benchmark_edge.py ONNX Runtime latency on the actual target board
```

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/ -v                                    # prove the split guard works first
python -m src.train --config configs/default.yaml   # train + evaluate once
python -m src.benchmark_edge --onnx artifacts/model.onnx
```

## Data

Built against the **DMR-IR** breast thermography database (Federal Fluminense
University), which is the standard public benchmark and is organised by patient.
Register and download it separately; no clinical data is redistributed here.

Expected layout — the patient identifier **must** be a real directory level. If you
flatten patients into one folder, you cannot group correctly and every metric
downstream is unusable:

```
data/dmr-ir/
├── healthy/
│   ├── PAT_012/  IR_0001.png  IR_0002.png ...
│   └── PAT_013/  ...
└── sick/
    └── PAT_047/  ...
```

`manifest_from_folders` raises if one patient carries conflicting labels rather than
silently picking one.

---

## Design decisions worth arguing about

**ResNet-18, not something larger.** With a few hundred patients, a bigger backbone
buys overfitting and slower inference on a Pi. Choosing the smallest model that clears
the clinical bar is the engineering decision, and `mobilenet_v3_small` is wired in for
when latency becomes the binding constraint.

**Geometric augmentation only — never colour jitter.** In a thermogram the pixel
values *are* the temperatures. Randomising intensity destroys the physical signal. This
is a mistake that gets copied from natural-image pipelines constantly.

**Class imbalance handled in the loss, not by resampling.** Oversampling a patient's
frames scatters correlated images across batches — another quiet route to leakage.
`pos_weight` in `BCEWithLogitsLoss` does the job without touching the split.

**Threshold chosen for a target sensitivity, then specificity reported honestly.**
For a screening tool you fix the miss rate you can clinically tolerate and accept
whatever specificity that costs. Tuning the threshold to flatter the headline figure is
how screening tools get deployed, distrusted, and abandoned.

**Latency measured with p95, not just the mean.** A good mean with a heavy tail still
feels broken to a clinician standing over a patient in a rural camp.

---

## Reporting template

The pipeline emits `artifacts/test_report.json`. Report it whole. A bare sensitivity
figure is not a claim anyone should accept:

```
n = 240 images | 61 patients | positives = 65 (prevalence 27.1%)
AUROC  0.867   95% CI [0.811, 0.911]
AUPRC  0.710    Brier  0.163

At threshold 0.398 (chosen for >= 90% sensitivity):
  Sensitivity  0.954   95% CI [0.898, 1.000]
  Specificity  0.606   95% CI [0.530, 0.676]
  PPV 0.473    NPV 0.972
  TP 62   FP 69   TN 106   FN 3
```

Note what that example shows: sensitivity of 0.954 sounds excellent, and it comes with
a specificity of 0.606 and a PPV of 0.473 — meaning **more than half of everyone this
model flags does not have the disease.** That is a defensible tool for triage in a
low-resource setting where the alternative is no screening at all. It is not a
diagnostic. Saying so is the job.

---

## Limitations

- Single public dataset, single acquisition device. Thermography generalises poorly
  across cameras and ambient conditions; treat any number here as an upper bound on
  what a new site would see.
- No external validation cohort. Until a model is tested on data from a hospital that
  contributed nothing to training, its real-world performance is unknown.
- Trained on a curated benchmark, not a screening-camp distribution. Real prevalence is
  far lower, which will drag PPV down hard.
- Not a medical device. No CDSCO or CE clearance, no clinical claim, not for patient
  care.

---

## Licence

MIT.
