from __future__ import annotations

from pathlib import Path
import datetime as dt
import sys

import pandas as pd

# Ensure package imports work when running from tests folder
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from driftwatch.testing import run_self_tests  # noqa: E402


def main() -> int:
    sample_dir = BASE_DIR / "sample_data"
    out_dir = Path(__file__).resolve().parent

    baseline_path = str(sample_dir / "baseline.csv")
    current_path = str(sample_dir / "current.csv")

    results, artifacts = run_self_tests(baseline_path, current_path)

    df = pd.DataFrame([{
        "test_id": r.test_id,
        "description": r.description,
        "expected": r.expected,
        "outcome": r.outcome,
        "passed": r.passed,
    } for r in results])

    ts = dt.datetime.now().isoformat(timespec="seconds")
    md_lines = []
    md_lines.append("# DriftWatch Test Results")
    md_lines.append("")
    md_lines.append(f"Timestamp: {ts}")
    md_lines.append("")
    md_lines.append("## Summary")
    md_lines.append(f"- Baseline shape: {artifacts.get('baseline_shape')}")
    md_lines.append(f"- Current shape: {artifacts.get('current_shape')}")
    md_lines.append(f"- Health: {artifacts.get('health')}")
    md_lines.append(f"- Tier: {artifacts.get('tier')}")
    md_lines.append("")
    md_lines.append("## Test case outcomes")
    md_lines.append(df.to_markdown(index=False))
    md_lines.append("")

    out_path = out_dir / "TEST_RESULTS.md"
    out_path.write_text("\n".join(md_lines), encoding="utf-8")

    csv_path = out_dir / "TEST_RESULTS.csv"
    df.to_csv(csv_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
