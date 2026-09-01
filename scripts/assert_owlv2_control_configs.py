from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coverage_aware_grounding_stability.owlv2_control import assert_control_identity


CONTROL_CONFIGS = (
    ROOT / "config" / "operational_benchmark_owlv2_control_v1.json",
    ROOT / "config" / "operational_transfer_refcocoplus_owlv2_control_v1.json",
    ROOT / "config" / "operational_transfer_refl4_owlv2_control_v1.json",
)


def main() -> None:
    audit = [assert_control_identity(path, ROOT) for path in CONTROL_CONFIGS]
    print(json.dumps({"status": "valid", "configs": audit}, indent=2))


if __name__ == "__main__":
    main()
