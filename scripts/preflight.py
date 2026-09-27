#!/usr/bin/env python3
"""Run the RSNA competition dataset preflight.

Example on Kaggle:

python scripts/preflight.py \
  --data-dir /kaggle/input/rsna-knee-abnormality-detection \
  --output-dir /kaggle/working/preflight
"""

from __future__ import annotations

import argparse
import json

from knee_ai.data.preflight import inspect_study, load_competition_frames, run_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/kaggle/input/rsna-knee-abnormality-detection",
        help="Directory containing competition CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/preflight",
        help="Directory for summary JSON and dummy submission.csv.",
    )
    parser.add_argument(
        "--study-id",
        default=None,
        help="Optional StudyInstanceUID to print its series metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_preflight(args.data_dir, args.output_dir)

    print("DATASET PREFLIGHT PASSED")
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))

    frames = load_competition_frames(args.data_dir)
    study_id = args.study_id or str(frames["train"].iloc[0]["StudyInstanceUID"])
    print(f"\nSeries metadata for study {study_id}:")
    print(inspect_study(study_id, frames["train_series"]).to_string(index=False))
    print(f"\nArtifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
