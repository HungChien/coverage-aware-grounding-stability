from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "results" / "operational_benchmark_v1",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "operational_benchmark_v1.json",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path

    required = [
        args.config,
        ROOT / "config" / "random_probe_distribution.json",
        manifest_path,
        ROOT / "src" / "operational_stability.py",
        ROOT / "src" / "random_probes.py",
        ROOT / "scripts" / "run_operational_benchmark.py",
        ROOT / "scripts" / "analyse_operational_benchmark.py",
    ]
    for model in ("groundingdino", "yoloworld"):
        required.extend(
            [
                args.result_root / model / "run_metadata.json",
                args.result_root / model / "sample_budget_summary.csv",
                args.result_root / model / "sample_traces.jsonl.gz",
            ]
        )
    required.extend(
        sorted(
            path
            for path in (args.result_root / "analysis").rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        )
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required artifacts:\n" + "\n".join(missing))

    files = []
    seen = set()
    for path in required:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            display = str(path.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            display = str(path)
        files.append(
            {
                "path": display,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    package_names = [
        "numpy",
        "pandas",
        "scipy",
        "Pillow",
        "torch",
        "transformers",
        "ultralytics",
        "matplotlib",
        "pytest",
    ]
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"

    manifest = {
        "benchmark": config["benchmark_name"],
        "file_count": len(files),
        "files": files,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
        },
        "storage_note": (
            "The compressed per-probe traces are hashed here but ignored by Git. "
            "Archive them as release or institutional-data assets when publishing."
        ),
    }
    output = args.result_root / "analysis" / "artifact_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {output} with {len(files)} hashed artifacts")


if __name__ == "__main__":
    main()
