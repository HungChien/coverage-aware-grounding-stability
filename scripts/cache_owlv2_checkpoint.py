from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "operational_benchmark_owlv2_control_v1.json",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Verify the pinned checkpoint cache without network access.",
    )
    args = parser.parse_args()

    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    config = json.loads(args.config.read_text(encoding="utf-8"))
    repository = config["models"]["owlv2"]
    revision = config["model_revisions"]["owlv2"]
    loading = {"revision": revision, "local_files_only": args.local_only}
    Owlv2Processor.from_pretrained(repository, **loading)
    model = Owlv2ForObjectDetection.from_pretrained(repository, **loading)
    audit = {
        "status": "cached_and_loadable",
        "repository": repository,
        "requested_revision": revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "local_only": args.local_only,
    }
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
