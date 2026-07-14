# Contributing

Bug reports, focused improvements, and reproducible evaluation work are welcome.

## Before opening a change

1. Create a branch from `main`.
2. Install the development checks with `pip install -r requirements-ci.txt`.
3. Keep patient-level separation intact. Any metric change must state its unit of
   analysis and where its operating threshold was selected.
4. Do not add clinical data, patient identifiers, trained weights, or generated
   artifacts.

Run the checks before submitting a pull request:

```bash
ruff check .
ruff format --check .
pytest
```

Keep pull requests small. Explain the behavior change, the reason for it, and how it
was tested.
