"""Competition-wide constants.

Keep target names centralized so training, validation, inference, and submission
generation cannot silently disagree about column order.
"""

TARGET_COLUMNS: tuple[str, ...] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

NUM_TARGETS = len(TARGET_COLUMNS)
STUDY_ID_COLUMN = "StudyInstanceUID"
