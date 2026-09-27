import pandas as pd
import pytest

from knee_ai.constants import STUDY_ID_COLUMN, TARGET_COLUMNS
from knee_ai.data.preflight import (
    PreflightError,
    build_dummy_submission,
    inspect_study,
    summarize_frames,
    validate_competition_frames,
)


def _train() -> pd.DataFrame:
    rows = [
        {STUDY_ID_COLUMN: "study-1", "Report": "report one"},
        {STUDY_ID_COLUMN: "study-2", "Report": "report two"},
    ]
    frame = pd.DataFrame(rows)
    for target in TARGET_COLUMNS:
        frame[target] = [1.0, float("nan")]
    return frame


def _series(prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                STUDY_ID_COLUMN: f"{prefix}-1",
                "SeriesInstanceUID": f"{prefix}-sag",
                "Fluid_Sensitive": 1,
                "Fat_Suppression": 0,
                "Anatomical_Plane": "Sagittal",
            },
            {
                STUDY_ID_COLUMN: f"{prefix}-1",
                "SeriesInstanceUID": f"{prefix}-cor",
                "Fluid_Sensitive": 1,
                "Fat_Suppression": 1,
                "Anatomical_Plane": "Coronal",
            },
            {
                STUDY_ID_COLUMN: f"{prefix}-2",
                "SeriesInstanceUID": f"{prefix}-ax",
                "Fluid_Sensitive": 0,
                "Fat_Suppression": 0,
                "Anatomical_Plane": "Axial",
            },
        ]
    )


def _test() -> pd.DataFrame:
    return pd.DataFrame([{STUDY_ID_COLUMN: "test-1"}, {STUDY_ID_COLUMN: "test-2"}])


def _sample_submission() -> pd.DataFrame:
    frame = _test().copy()
    for target in TARGET_COLUMNS:
        frame[target] = 0.5
    return frame


def test_validate_competition_frames_accepts_valid_schema() -> None:
    validate_competition_frames(
        _train(),
        _series("study"),
        _test(),
        _series("test"),
        _sample_submission(),
    )


def test_dummy_submission_preserves_ids_and_sets_half_probability() -> None:
    submission = build_dummy_submission(_sample_submission())

    assert submission[STUDY_ID_COLUMN].tolist() == ["test-1", "test-2"]
    assert submission.columns.tolist() == [STUDY_ID_COLUMN, *TARGET_COLUMNS]
    assert (submission[list(TARGET_COLUMNS)] == 0.5).all().all()


def test_inspect_study_returns_only_requested_study() -> None:
    rows = inspect_study("study-1", _series("study"))

    assert len(rows) == 2
    assert set(rows[STUDY_ID_COLUMN]) == {"study-1"}
    assert set(rows["Anatomical_Plane"]) == {"Sagittal", "Coronal"}


def test_missing_required_column_fails_fast() -> None:
    broken = _series("study").drop(columns=["Anatomical_Plane"])

    with pytest.raises(PreflightError, match="missing required columns"):
        validate_competition_frames(
            _train(),
            broken,
            _test(),
            _series("test"),
            _sample_submission(),
        )


def test_orphan_series_study_fails_fast() -> None:
    broken = _series("study").copy()
    broken.loc[0, STUDY_ID_COLUMN] = "not-in-train"

    with pytest.raises(PreflightError, match="absent from train.csv"):
        validate_competition_frames(
            _train(),
            broken,
            _test(),
            _series("test"),
            _sample_submission(),
        )


def test_summary_counts_fully_labeled_studies() -> None:
    summary = summarize_frames(
        _train(),
        _series("study"),
        _test(),
        _series("test"),
    )

    assert summary.train_studies == 2
    assert summary.train_series == 3
    assert summary.fully_labeled_studies == 1
    assert summary.train_unique_studies_in_series == 2
