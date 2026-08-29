import gzip
import json

import pandas as pd

from scripts.run_operational_benchmark import (
    completed_samples,
    load_rows,
    traced_samples,
)


def test_resume_requires_complete_summary_and_trace(tmp_path):
    summary = tmp_path / "summary.csv"
    trace = tmp_path / "trace.jsonl.gz"
    rows = [
        {"image_id": 1, "ref_id": 10, "diagnostic_budget": budget}
        for budget in (5, 10, 20, 40)
    ] + [
        {"image_id": 2, "ref_id": 20, "diagnostic_budget": budget}
        for budget in (5, 10, 20, 40)
    ]
    pd.DataFrame(rows).to_csv(summary, index=False)
    with gzip.open(trace, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"image_id": 1, "ref_id": 10}) + "\n")

    summary_done = completed_samples(summary, [5, 10, 20, 40])
    trace_done = traced_samples(trace)
    done = summary_done & trace_done
    resumed_rows = load_rows(summary, True, done)

    assert summary_done == {(1, 10), (2, 20)}
    assert trace_done == {(1, 10)}
    assert done == {(1, 10)}
    assert len(resumed_rows) == 4
    assert {(int(row["image_id"]), int(row["ref_id"])) for row in resumed_rows} == {
        (1, 10)
    }
