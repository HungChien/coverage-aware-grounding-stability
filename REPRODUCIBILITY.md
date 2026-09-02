# Reproducibility map

This file defines the smallest retained evidence chain for release 1.1.0.
Run `python scripts/verify_release.py` after cloning, and add
`--require-local-assets` on a machine prepared for full inference or
trace-based reanalysis.

## Frozen inputs

| Role | Path |
|---|---|
| probe distribution | `config/random_probe_distribution.json` |
| common output contract | `config/output_contract_preregistration_v2.freeze.json` |
| RefCOCO config | `config/operational_benchmark_owlv2_control_v1.json` |
| RefCOCO+ config | `config/operational_transfer_refcocoplus_owlv2_control_v1.json` |
| Ref-L4 config | `config/operational_transfer_refl4_owlv2_control_v1.json` |
| RefCOCO manifest | `data_operational/refcoco_unseen500/manifest.json` |
| RefCOCO+ manifest | `data_operational/refcocoplus_transfer1000/manifest.json` |
| Ref-L4 manifest | `data_operational/refl4_transfer1000/manifest.json` |

The manifest metadata records sampling seeds and source splits. Freeze JSON
files preserve the pre-run file hashes. They are historical audit records; a
later package refactor is expected to change live source paths and hashes.

## Execution chain

1. `scripts/run_operational_benchmark.py` generates resume-safe summaries and
   compressed candidate traces.
2. `scripts/validate_operational_artifacts.py` checks sample coverage, probe
   counts, shared probe identities, and result structure.
3. `scripts/analyse_operational_benchmark.py` produces dataset-level estimates,
   uncertainty, failure causes, and diagnostic-budget results.
4. The final analysis scripts produce complete-case, two-stage, reference-depth,
   contract, selection, severity, family-weight, and exposure-cap audits.
5. `scripts/build_operational_artifact_manifest.py` hashes the analysis output.

## Final trace-derived analyses

| Analysis | Script | Output directory |
|---|---|---|
| conditioning optimism | `scripts/analyse_complete_case_optimism.py` | `results/complete_case_optimism_analysis` |
| two-stage variance | `scripts/analyse_two_stage_sampling.py` | `results/two_stage_sampling_analysis` |
| 80-probe adequacy | `scripts/analyse_reference_80_adequacy.py` | `results/reference_80_adequacy` |
| output-contract grid | `scripts/analyse_output_contract_robustness.py` | `results/output_contract_robustness` |
| selection, severity, and mixture controls | `scripts/analyse_reviewer_risk_controls.py` | `results/reviewer_risk_controls` |
| exposure-50 confirmation | `scripts/summarise_cap50_confirmatory.py` | `results/cap50_confirmatory/summary` |

## Git-tracked versus local assets

Git tracks code, configurations, compact manifests, result tables, figures,
audit JSON, and artifact manifests. The following remain local because of
upstream licences or repository-size limits:

- dataset images and Ref-L4 parquet annotations;
- GroundingDINO, OWLv2, YOLO-World, and CLIP checkpoints/caches;
- `sample_traces.jsonl.gz` and cap-50 trace files;
- runtime logs and resume state.

This split supports two levels of reproduction:

- a normal clone can inspect the protocol, run unit tests, use the public API,
  and verify all tracked summaries;
- a prepared machine with the licensed datasets and local traces can rerun
  inference and every analysis exactly.

Point estimates in the analysis tables are deterministic. Bootstrap confidence
intervals are resampled independently on each analysis run, so CI boundaries
can differ from the dissertation tables by about `0.001` to `0.002` without
indicating a result mismatch.

If traces are distributed separately, preserve their paths under `results/`
and verify them before analysis. Do not commit them directly to ordinary Git.

## Integrity checks

```bash
python scripts/verify_release.py
python scripts/verify_release.py --require-local-assets
python -m pytest
python -m ruff check src scripts tests
```

The strict verifier checks all 2,500 manifest image paths, all nine primary
trace files, and all nine cap-50 trace files in addition to tracked release
assets. Trace sizes and SHA-256 hashes are recorded in
`LOCAL_ASSET_MANIFEST.json`; regenerate that record only when intentionally
creating a new release:

```bash
python scripts/build_local_asset_manifest.py
```
