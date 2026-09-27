# KneesBrokeWeAwoke

Engineering and ML-systems repository for the RSNA Knee Abnormality Detection competition.

## Goal

Build a reproducible pipeline that turns raw knee MRI DICOM studies into 12 abnormality probabilities and a valid `submission.csv`.

## Current milestone

**Foundation / preflight**

The immediate goal is to prove the complete data contract before model tuning:

```
competition CSVs -> study/series lookup -> DICOM decode -> preprocessing -> model input -> predictions -> submission.csv
```

## Repository layout

```
configs/        experiment configuration
docs/           engineering notes and roadmap
src/knee_ai/    reusable Python package
tests/          fast unit tests
notebooks/      exploratory and Kaggle notebooks (added as needed)
```

## Target abnormalities

1. ACL
2. MCL
3. Medial Meniscus
4. Lateral Meniscus
5. Medial OA
6. Lateral OA
7. PF OA
8. Effusion
9. Synovitis
10. Baker's
11. Contusion
12. Fracture

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

Install ML dependencies when needed:

```bash
pip install -e ".[ml,dev]"
```

## Engineering rules

- Do not commit competition data, DICOM files, model weights, Kaggle credentials, or generated caches.
- Every serious experiment must record its config, seed, preprocessing version, label version, validation metrics, runtime, and git commit.
- Optimize against reproducible cross-validation first; leaderboard score is supporting evidence, not the experiment log.
- Keep the final Kaggle inference path deterministic and internet-independent.

## Status

- [x] Repository initialized
- [x] Python project skeleton
- [x] Competition target schema
- [x] CI/test foundation
- [ ] Dataset preflight
- [ ] DICOM loader
- [ ] Physical slice ordering
- [ ] Preprocessing/cache pipeline
- [ ] Weak-label pipeline
- [ ] Baseline model + CV
- [ ] Kaggle inference notebook
- [ ] Final submission
