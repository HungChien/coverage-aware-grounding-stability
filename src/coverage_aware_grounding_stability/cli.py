from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from . import __version__
from .api import BenchmarkEvaluator, candidate_from_dict
from .operational_stability import OutputContract


def _load_contract(path: Path | None) -> OutputContract:
    if path is None:
        return OutputContract()
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("output_contract", payload)
    if not isinstance(values, dict):
        raise ValueError(
            "config must be an output-contract object or contain output_contract"
        )
    return OutputContract(**values)


def _records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"line {line_number} is not a JSON object")
                    yield value
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        yield value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"item {index} is not a JSON object")
            yield item
    else:
        raise ValueError("input JSON must be an object or a list of objects")


def _evaluate(args: argparse.Namespace) -> int:
    evaluator = BenchmarkEvaluator(_load_contract(args.config))
    output_rows = []
    for record in _records(args.input):
        clean = [candidate_from_dict(item) for item in record["clean_candidates"]]
        perturbed = [
            candidate_from_dict(item) for item in record["perturbed_candidates"]
        ]
        result = evaluator.evaluate(clean, perturbed).to_dict()
        if "id" in record:
            result = {"id": record["id"], **result}
        output_rows.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {len(output_rows)} evaluation(s) to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cags",
        description="Coverage-aware candidate-order stability tools",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate model-independent candidate pairs from JSON or JSONL",
    )
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--config", type=Path)
    evaluate.set_defaults(handler=_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
