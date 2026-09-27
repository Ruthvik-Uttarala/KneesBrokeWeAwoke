# Execution Roadmap

The project advances only when each phase has a concrete exit condition.

## Phase 0 — Foundation

**Exit condition:** repository installs, tests pass, target schema is centralized.

## Phase 1 — Dataset preflight

- Load competition CSV files.
- Verify row counts and required columns.
- Map a StudyInstanceUID to its MRI series.
- Inspect sagittal, coronal, and axial metadata.
- Generate a schema-valid dummy submission.

**Exit condition:** one reproducible preflight script/notebook runs without manual edits.

## Phase 2 — DICOM ingestion

- Decode one full series.
- Sort slices using physical DICOM geometry.
- Handle missing or inconsistent metadata.
- Add tests around deterministic ordering.

**Exit condition:** study -> ordered volume works reliably.

## Phase 3 — Preprocessing and caching

- Plane-aware series selection.
- Intensity normalization.
- Slice sampling and resizing/cropping.
- Cache transformed inputs.
- Measure throughput and failure rate.

**Exit condition:** repeatable tensor generation across the training set.

## Phase 4 — Labels and validation

- Version weak labels derived from reports.
- Preserve positive / negative / undecided state where useful.
- Create study-level folds.
- Evaluate macro and per-target ROC-AUC.

**Exit condition:** label file + folds + metric code are versioned and reproducible.

## Phase 5 — Baseline model

- Implement an efficient 2.5D image baseline.
- Produce out-of-fold predictions.
- Track configs, seeds, runtimes, and checkpoints.

**Exit condition:** trustworthy CV score and one end-to-end trained checkpoint.

## Phase 6 — Kaggle inference

- Load test studies.
- Run deterministic offline inference.
- Generate final 12-column probabilities plus StudyInstanceUID.
- Validate submission schema and runtime.

**Exit condition:** valid Kaggle notebook producing submission.csv within competition limits.

## Phase 7 — Controlled improvement

One change at a time: labels, preprocessing, backbone, pooling/fusion, loss, ensemble.

**Exit condition:** retain only repeatable validation gains.
