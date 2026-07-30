# GMS Reliability for Vision-Language Grounding

This repository contains the current MSc dissertation research prototype for
candidate-order stability in vision-language grounding models.

The project evaluates whether local perturbation geometry improves prediction
of held-out candidate-order stability beyond the clean score margin. The main
score is GMS, a bound-derived, scale-invariant stability indicator based on the
effective margin and the perturbation variance along the ranking-reversal
direction.

## Repository Layout

- `src/`: reusable reliability, geometry and perturbation code.
- `scripts/`: experiment, evaluation, figure-generation and report scripts.
- `tests/`: mathematical and implementation checks.
- `paper/`: theory notes and literature matrix.
- `docs/`: English methodology and experiment reports.
- `results/`: compact CSV, JSON and figure outputs from the two-week validation.
- `reports/`: presentation artifacts suitable for supervisor meetings.
- `data/`: lightweight split metadata only.

Chinese and bilingual notes are intentionally kept outside this repository in
`../doc/` so that the engineering repository remains English-only.

## Main Claims Currently Supported

1. Perturbation-aware candidate-order geometry predicts held-out stability
   better than the clean margin on GroundingDINO.
2. The same effect is initially replicated on YOLO-World.
3. Six to eight designed probes capture most of the full-probe signal.
4. GMS measures candidate-order stability, not semantic correctness.

## Quick Checks

```powershell
python -m pytest tests -q
```

## Reproducing The Main Pipeline

The original experiments used local model weights and COCO/RefCOCO-style data.
Large weights and raw datasets are not committed. Place them according to the
paths documented in `docs/` before running the full pipeline.

```powershell
python scripts/run_full_geometry_experiment.py --model groundingdino --limit 100 --tag full100
python scripts/evaluate_full_geometry.py --input outputs_refcoco/full_geometry/groundingdino_full100
python scripts/run_full_geometry_experiment.py --model yoloworld --limit 100 --tag full100
python scripts/evaluate_full_geometry.py --input outputs_refcoco/full_geometry/yoloworld_full100
python scripts/compare_models.py --groundingdino outputs_refcoco/full_geometry/groundingdino_full100 --yoloworld outputs_refcoco/full_geometry/yoloworld_full100
```

## Interpretation

GMS should be interpreted as a stability or deployment-risk indicator under a
registered perturbation family. It is not an exact correctness probability and
does not replace ground-truth semantic evaluation.
