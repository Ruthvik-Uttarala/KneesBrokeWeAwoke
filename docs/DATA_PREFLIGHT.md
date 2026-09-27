# Dataset Preflight

This is the first competition-data gate. It validates the cheap tabular contracts
before any DICOM decoding or model training.

## Official file contract

The competition provides:

- `train.csv`: one row per training study, with `StudyInstanceUID`, `Report`,
  and the 12 target columns.
- `train_series.csv`: one row per MRI series with `StudyInstanceUID`,
  `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`, and
  `Anatomical_Plane`.
- `test.csv`: test study IDs.
- `test_series.csv`: the same series metadata schema for test studies.
- `sample_submission.csv`: the required output schema.

MRI files are organized as:

```
train_series/
  <StudyInstanceUID>/
    <SeriesInstanceUID>/
      <SOPInstanceUID>.dcm
```

## Run on Kaggle

From a notebook terminal or shell cell:

```bash
pip install -e .
python scripts/preflight.py \
  --data-dir /kaggle/input/rsna-knee-abnormality-detection \
  --output-dir /kaggle/working/preflight
```

The command fails fast on schema mismatches and writes:

```
/kaggle/working/preflight/
  preflight_summary.json
  submission.csv
```

The generated `submission.csv` deliberately assigns `0.5` to every target.
It is a contract test, not a model.

## What must be checked in the output

1. CSV schemas pass validation.
2. No training-series StudyInstanceUID is orphaned from `train.csv`.
3. No test-series StudyInstanceUID is orphaned from `test.csv`.
4. Sample-submission study IDs match test study IDs exactly.
5. Anatomical planes contain only Sagittal, Coronal, and Axial.
6. The number of fully labeled studies is recorded.
7. Series-per-study and plane counts are recorded.
8. `submission.csv` has exactly one ID column plus the 12 target columns.

## Exit condition

This phase is complete only after the script runs against the official Kaggle
input and the resulting summary is recorded in the experiment log. Synthetic
unit tests prove our code contract, but they do not replace the official-data run.
