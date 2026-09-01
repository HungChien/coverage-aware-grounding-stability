from __future__ import annotations

import json
from pathlib import Path


OWLV2_REPOSITORY = "google/owlv2-base-patch16-ensemble"
OWLV2_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
ALLOWED_CONTROL_ONLY_KEYS = {
    "model_revisions",
    "owlv2_batch_size",
    "source_config",
}
IDENTITY_EXCEPTIONS = {"benchmark_name", "models", "protocol_version"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_control_identity(control_path: Path, repository_root: Path) -> dict:
    """Assert that an OWLv2 control config changes no registered estimator input."""

    control = load_json(control_path)
    source_path = repository_root / control["source_config"]
    source = load_json(source_path)

    for key, value in source.items():
        if key in IDENTITY_EXCEPTIONS:
            continue
        if control.get(key) != value:
            raise AssertionError(
                f"{control_path.name}: registered field {key!r} differs from "
                f"{source_path.name}"
            )

    source_models = source["models"]
    control_models = control["models"]
    if set(control_models) != {*source_models, "owlv2"}:
        raise AssertionError(f"{control_path.name}: model set is not source + OWLv2")
    for model, checkpoint in source_models.items():
        if control_models[model] != checkpoint:
            raise AssertionError(
                f"{control_path.name}: source checkpoint for {model} changed"
            )
    if control_models["owlv2"] != OWLV2_REPOSITORY:
        raise AssertionError(f"{control_path.name}: unexpected OWLv2 repository")
    if control.get("model_revisions", {}).get("owlv2") != OWLV2_REVISION:
        raise AssertionError(f"{control_path.name}: OWLv2 revision is not pinned")
    if control.get("owlv2_batch_size") != 4:
        raise AssertionError(f"{control_path.name}: OWLv2 batch size must be 4")

    extra_keys = set(control) - set(source)
    if extra_keys != ALLOWED_CONTROL_ONLY_KEYS:
        raise AssertionError(
            f"{control_path.name}: unexpected control-only keys {sorted(extra_keys)}"
        )
    return {
        "control": str(control_path.relative_to(repository_root)),
        "source": str(source_path.relative_to(repository_root)),
        "registered_fields_equal": True,
        "added_model": "owlv2",
        "owlv2_repository": OWLV2_REPOSITORY,
        "owlv2_revision": OWLV2_REVISION,
        "owlv2_batch_size": 4,
    }
