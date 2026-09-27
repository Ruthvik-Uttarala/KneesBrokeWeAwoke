"""Competition dataset preflight checks.

This module intentionally performs only cheap tabular validation. It proves the
study/series/submission contracts before we touch expensive DICOM decoding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from knee_ai.constants import STUDY_ID_COLUMN, TARGET_COLUMNS

TRAIN_REQUIRED_COLUMNS = (STUDY_ID_COLUMN, "Report", *TARGET_COLUMNS)
SERIES_REQUIRED_COLUMNS = (
    STUDY_ID_COLUMN,
    "SeriesInstanceUID",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
)
SUBMISSION_REQUIRED_COLUMNS = (STUDY_ID_COLUMN, *TARGET_COLUMNS)
EXPECTED_PLANES = {"Sagittal", "Coronal", "Axial"}


class PreflightError(ValueError):
    """Raised when competition files violate a required data contract."""


@dataclass(frozen=True)
class PreflightSummary:
    train_studies: int
    train_series: int
    fully_labeled_studies: int
    train_unique_studies_in_series: int
    test_studies: int
    test_series: int
    plane_counts_train: dict[str, int]
    plane_counts_test: dict[str, int]
    min_series_per_train_study: int
    median_series_per_train_study: float
    max_series_per_train_study: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise PreflightError(f"{name} is missing required columns: {missing}")


def _require_unique(frame: pd.DataFrame, column: str, name: str) -> None:
    duplicated = frame[column].duplicated(keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, column].astype(str).head(5).tolist()
        raise PreflightError(
            f"{name}.{column} must be unique; duplicate examples: {examples}"
        )


def _validate_planes(series: pd.DataFrame, name: str) -> None:
    observed = set(series["Anatomical_Plane"].dropna().astype(str).unique())
    unexpected = sorted(observed - EXPECTED_PLANES)
    if unexpected:
        raise PreflightError(
            f"{name}.Anatomical_Plane has unexpected values: {unexpected}"
        )


def validate_competition_frames(
    train: pd.DataFrame,
    train_series: pd.DataFrame,
    test: pd.DataFrame,
    test_series: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    """Validate the minimum contracts needed by every downstream pipeline stage."""

    _require_columns(train, TRAIN_REQUIRED_COLUMNS, "train.csv")
    _require_columns(train_series, SERIES_REQUIRED_COLUMNS, "train_series.csv")
    _require_columns(test, (STUDY_ID_COLUMN,), "test.csv")
    _require_columns(test_series, SERIES_REQUIRED_COLUMNS, "test_series.csv")
    _require_columns(
        sample_submission,
        SUBMISSION_REQUIRED_COLUMNS,
        "sample_submission.csv",
    )

    _require_unique(train, STUDY_ID_COLUMN, "train.csv")
    _require_unique(test, STUDY_ID_COLUMN, "test.csv")
    _require_unique(
        sample_submission,
        STUDY_ID_COLUMN,
        "sample_submission.csv",
    )

    _validate_planes(train_series, "train_series.csv")
    _validate_planes(test_series, "test_series.csv")

    train_ids = set(train[STUDY_ID_COLUMN].astype(str))
    train_series_ids = set(train_series[STUDY_ID_COLUMN].astype(str))
    orphan_train_series = sorted(train_series_ids - train_ids)
    if orphan_train_series:
        raise PreflightError(
            "train_series.csv contains studies absent from train.csv: "
            f"{orphan_train_series[:5]}"
        )

    test_ids = set(test[STUDY_ID_COLUMN].astype(str))
    test_series_ids = set(test_series[STUDY_ID_COLUMN].astype(str))
    orphan_test_series = sorted(test_series_ids - test_ids)
    if orphan_test_series:
        raise PreflightError(
            "test_series.csv contains studies absent from test.csv: "
            f"{orphan_test_series[:5]}"
        )

    submission_ids = set(sample_submission[STUDY_ID_COLUMN].astype(str))
    if submission_ids != test_ids:
        missing = sorted(test_ids - submission_ids)[:5]
        extra = sorted(submission_ids - test_ids)[:5]
        raise PreflightError(
            "sample_submission.csv study IDs must match test.csv exactly; "
            f"missing={missing}, extra={extra}"
        )


def build_dummy_submission(sample_submission: pd.DataFrame) -> pd.DataFrame:
    """Return a schema-preserving chance-level submission with all targets at 0.5."""

    _require_columns(
        sample_submission,
        SUBMISSION_REQUIRED_COLUMNS,
        "sample_submission.csv",
    )
    submission = sample_submission.loc[:, list(SUBMISSION_REQUIRED_COLUMNS)].copy()
    for target in TARGET_COLUMNS:
        submission[target] = 0.5
    return submission


def inspect_study(
    study_id: str,
    series_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return all series metadata for one study, sorted for human inspection."""

    _require_columns(series_frame, SERIES_REQUIRED_COLUMNS, "series metadata")
    rows = series_frame[
        series_frame[STUDY_ID_COLUMN].astype(str) == str(study_id)
    ].copy()
    if rows.empty:
        raise PreflightError(f"StudyInstanceUID not found in series metadata: {study_id}")

    return rows.sort_values(
        by=["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression", "SeriesInstanceUID"],
        kind="stable",
    ).reset_index(drop=True)


def summarize_frames(
    train: pd.DataFrame,
    train_series: pd.DataFrame,
    test: pd.DataFrame,
    test_series: pd.DataFrame,
) -> PreflightSummary:
    """Compute cheap dataset facts that should be logged before modeling."""

    target_frame = train.loc[:, list(TARGET_COLUMNS)]
    fully_labeled = int(target_frame.notna().all(axis=1).sum())

    series_counts = train_series.groupby(STUDY_ID_COLUMN).size()
    if series_counts.empty:
        raise PreflightError("train_series.csv contains no series")

    return PreflightSummary(
        train_studies=int(len(train)),
        train_series=int(len(train_series)),
        fully_labeled_studies=fully_labeled,
        train_unique_studies_in_series=int(train_series[STUDY_ID_COLUMN].nunique()),
        test_studies=int(len(test)),
        test_series=int(len(test_series)),
        plane_counts_train={
            str(k): int(v)
            for k, v in train_series["Anatomical_Plane"].value_counts().to_dict().items()
        },
        plane_counts_test={
            str(k): int(v)
            for k, v in test_series["Anatomical_Plane"].value_counts().to_dict().items()
        },
        min_series_per_train_study=int(series_counts.min()),
        median_series_per_train_study=float(series_counts.median()),
        max_series_per_train_study=int(series_counts.max()),
    )


def load_competition_frames(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the five CSV files required by the preflight."""

    root = Path(data_dir)
    paths = {
        "train": root / "train.csv",
        "train_series": root / "train_series.csv",
        "test": root / "test.csv",
        "test_series": root / "test_series.csv",
        "sample_submission": root / "sample_submission.csv",
    }

    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise PreflightError(f"Missing competition files: {missing}")

    return {name: pd.read_csv(path) for name, path in paths.items()}


def run_preflight(
    data_dir: str | Path,
    output_dir: str | Path,
) -> PreflightSummary:
    """Run validation, emit a summary JSON, and write a dummy submission."""

    frames = load_competition_frames(data_dir)
    validate_competition_frames(**frames)

    summary = summarize_frames(
        frames["train"],
        frames["train_series"],
        frames["test"],
        frames["test_series"],
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "preflight_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    dummy = build_dummy_submission(frames["sample_submission"])
    dummy.to_csv(output / "submission.csv", index=False)

    return summary
