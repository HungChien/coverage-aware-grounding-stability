# Post-inference analysis implementation amendment

The frozen inference, candidate contract, probe registry, estimands, and
statistical analysis rules were not changed.

After both models had completed inference and the primary artifact validation
had passed, the transfer-report script stopped while attaching the RefCOCO+
split label.  The summary table contains one row for each `(model, image_id,
ref_id)` combination, while the manifest contains one row for each `(image_id,
ref_id)` sample.  The merge had incorrectly declared a one-to-one relationship.

The implementation was corrected from `validate="one_to_one"` to
`validate="many_to_one"`.  This change only expresses the true table key
cardinality.  It does not alter samples, model outputs, probe outcomes,
estimators, bootstrap settings, or interpretation rules.  The complete test
suite subsequently passed (24 tests), and the transfer audit completed for all
1,000 target samples from each model.
