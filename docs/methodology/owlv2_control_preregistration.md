# Frozen OWLv2 Control Extension

## Purpose

This extension adds `google/owlv2-base-patch16-ensemble` as a third
candidate-producing grounding model.  It is a model-control extension of the
completed RefCOCO, RefCOCO+, and Ref-L4 experiments.  It does not redefine the
benchmark estimand or retrospectively tune the protocol.

## Frozen model identity

- model: `google/owlv2-base-patch16-ensemble`;
- Hugging Face revision: `cfd3195ba4ea9592eec887ded089f4c08eff231d`;
- inference implementation: `transformers.Owlv2ForObjectDetection` and
  `transformers.Owlv2Processor`;
- model-specific batch size: four images, an execution parameter only.

## Inherited scientific parameters

For each dataset, the derived control configuration inherits the original
manifest and every scientific parameter without change:

- candidate discovery threshold: `0.03`;
- tracked clean candidates: `5`;
- exposed perturbed candidates: `20`;
- clean duplicate IoU: `0.70`;
- association IoU: `0.15`;
- threatening-birth duplicate IoU: `0.70`;
- five registered perturbation families and their frozen severity ranges;
- eight diagnostic and sixteen reference probes per family;
- balanced diagnostic prefixes `5`, `10`, `20`, and `40`;
- independent eighty-probe finite reference;
- base seed `20260825` and sample-derived seed construction;
- strict score inequality, no missing-score imputation, and the existing
  disjoint failure precedence;
- exact fixed-budget intervals and 2,000 bootstrap repetitions.

The GroundingDINO text threshold remains in the inherited configuration but is
not applicable to OWLv2.  OWLv2 uses the same numeric candidate-discovery
threshold as the two completed models; this is an equal registered operating
point, not a claim that raw confidence values are calibrated across models.

## Adapter boundary

The adapter passes one referring expression as one OWLv2 text query, converts
the model output to absolute `xyxy` boxes, clips coordinates to the image
plane, drops only degenerate boxes, sorts by the native OWLv2 score, and
returns at most the requested raw candidate-pool size.  Duplicate suppression,
tracked-candidate selection, association, coverage, ranking, threatening birth,
and aggregation remain in the shared benchmark code.

### Runtime compatibility amendment (2026-08-31)

The first execution exposed OWLv2's checkpoint-defined text-position limit on
long referring expressions.  The adapter therefore reads
`text_config.max_position_embeddings` from the pinned checkpoint and requests
deterministic tokenizer truncation at that exact limit.  This is an input-shape
compatibility requirement, not a fitted experimental parameter: manifests,
sample order, queries, model weights, candidate threshold, output contract,
probes, seeds, and estimators remain unchanged.  Expressions within the native
context are encoded identically.  The partial outputs written before the first
unsupported expression are consequently resume-compatible.

The pipeline was also amended to propagate every non-zero Python exit code, so
validation and analysis cannot run after an incomplete model execution and a
success message can only be emitted after all registered stages complete.

## Prohibited changes after freeze

- OWLv2-specific threshold tuning on target outputs;
- changes to the output contract or perturbation registry;
- sample removal based on OWLv2 eligibility or stability;
- changes to diagnostic budgets, reference probes, seeds, or failure rules;
- analysis changes motivated by the observed OWLv2 result.

The extension evaluates whether the existing measurement operation remains
executable and informative for an additional model family.  It does not by
itself identify a causal architectural effect.
