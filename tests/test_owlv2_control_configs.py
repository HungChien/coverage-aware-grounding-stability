from pathlib import Path

import pytest

from src.owlv2_control import assert_control_identity


ROOT = Path(__file__).resolve().parents[1]
CONTROL_CONFIGS = (
    "operational_benchmark_owlv2_control_v1.json",
    "operational_transfer_refcocoplus_owlv2_control_v1.json",
    "operational_transfer_refl4_owlv2_control_v1.json",
)


@pytest.mark.parametrize("config_name", CONTROL_CONFIGS)
def test_owlv2_control_changes_only_the_registered_model_extension(config_name):
    result = assert_control_identity(ROOT / "config" / config_name, ROOT)

    assert result["registered_fields_equal"] is True
    assert result["added_model"] == "owlv2"
