from knee_ai.constants import NUM_TARGETS, STUDY_ID_COLUMN, TARGET_COLUMNS


def test_target_schema_has_exactly_twelve_unique_targets() -> None:
    assert NUM_TARGETS == 12
    assert len(set(TARGET_COLUMNS)) == 12


def test_study_identifier_column() -> None:
    assert STUDY_ID_COLUMN == "StudyInstanceUID"
