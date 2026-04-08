from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _safe_slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "report"


@dataclass
class ReportBundle:
    report_markdown: str
    drift_table: pd.DataFrame
    artifacts: Dict[str, Path]


def build_markdown_report(
    *,
    run_id: str,
    timestamp_iso: str,
    analysis_methods: List[str],
    cleaning_steps: List[str],
    baseline_shape: Tuple[int, int],
    current_shape: Tuple[int, int],
    overall_health: str,
    severity_tier: str,
    drift_df: pd.DataFrame,
    operator_summary: str,
    recommended_action: str,
    latest_drift_signal: str,
    quick_compare: Optional[Dict[str, object]] = None,
    top_corr_df: Optional[pd.DataFrame] = None,
    baseline_schema: Optional[pd.DataFrame] = None,
    current_schema: Optional[pd.DataFrame] = None,
    compare_df: Optional[pd.DataFrame] = None,
    outlier_df: Optional[pd.DataFrame] = None,
) -> str:
    """Build a capstone-ready markdown report aligned to the current DriftWatch UI."""

    lines: List[str] = []

    lines.append("# DriftWatch Report")
    lines.append("")
    lines.append(f"Run ID: {run_id}")
    lines.append(f"Timestamp: {timestamp_iso}")
    lines.append("")

    lines.append("## Executive summary")
    lines.append(
        "This report summarizes a DriftWatch screening run comparing a Baseline dataset to a Current dataset "
        "to identify data-side drift signals, support operator triage, and document configuration, results, "
        "and governance notes."
    )
    lines.append("")

    lines.append("## Front door summary")
    lines.append(f"- System status: {overall_health}")
    lines.append(f"- Signal type: data-side (PSI screening)")
    lines.append(f"- Current severity: {severity_tier}")
    lines.append(f"- Latest drift signal: {latest_drift_signal}")
    lines.append(f"- Operator summary: {operator_summary}")
    lines.append(f"- Recommended action: {recommended_action}")
    lines.append("")

    if quick_compare:
        lines.append("## Quick comparison")
        lines.append(f"- Baseline rows: {quick_compare.get('baseline_rows', 'n/a')}")
        lines.append(f"- Current rows: {quick_compare.get('current_rows', 'n/a')}")
        lines.append(f"- Row delta: {quick_compare.get('row_delta', 'n/a')}")
        lines.append(f"- Baseline columns: {quick_compare.get('baseline_columns', 'n/a')}")
        lines.append(f"- Shared columns: {quick_compare.get('shared_columns', 'n/a')}")
        lines.append(f"- Features with drift: {quick_compare.get('features_with_drift', 'n/a')}")
        lines.append("")

    lines.append("## Configuration")
    lines.append(f"- Analysis methods selected: {', '.join(analysis_methods) if analysis_methods else 'none'}")
    lines.append(f"- Cleaning steps (applied in order): {', '.join(cleaning_steps) if cleaning_steps else 'none'}")
    lines.append(f"- Baseline shape (rows, cols): {baseline_shape[0]}, {baseline_shape[1]}")
    lines.append(f"- Current shape (rows, cols): {current_shape[0]}, {current_shape[1]}")
    lines.append("")

    lines.append("## Drift investigation")
    lines.append("Population Stability Index (PSI) is used here as a screening signal, not a final verdict.")
    lines.append("Rule of thumb: PSI < 0.10 none, 0.10 to 0.25 moderate, greater than 0.25 significant.")
    lines.append("")

    if drift_df.empty:
        lines.append("No drift results were produced. Ensure the datasets share columns and have enough non-missing data.")
    else:
        sev = drift_df["severity"].astype(str).str.lower() if "severity" in drift_df.columns else pd.Series(dtype=str)
        significant_count = int((sev == "significant").sum()) if not sev.empty else 0
        moderate_count = int((sev == "moderate").sum()) if not sev.empty else 0
        active_count = int((sev != "none").sum()) if not sev.empty else 0

        lines.append(f"- Features with active drift signals: {active_count}")
        lines.append(f"- Significant drift signals: {significant_count}")
        lines.append(f"- Moderate drift signals: {moderate_count}")
        lines.append("")
        lines.append("### Drift table")
        lines.append(drift_df.to_markdown(index=False))
    lines.append("")

    if baseline_schema is not None and not baseline_schema.empty:
        lines.append("## Baseline schema and missingness")
        lines.append(baseline_schema.to_markdown(index=False))
        lines.append("")

    if current_schema is not None and not current_schema.empty:
        lines.append("## Current schema and missingness")
        lines.append(current_schema.to_markdown(index=False))
        lines.append("")

    if compare_df is not None and not compare_df.empty:
        lines.append("## Comparison review")
        lines.append(compare_df.to_markdown(index=False))
        lines.append("")

    if outlier_df is not None and not outlier_df.empty:
        lines.append("## Outlier summary (IQR)")
        lines.append(outlier_df.to_markdown(index=False))
        lines.append("")

    lines.append("## Visual analysis summary")
    lines.append(
        "Visual investigation was available through numeric and categorical views in the application interface. "
        "Correlation analysis was also available for current data to support quick structural review."
    )
    lines.append("")

    if top_corr_df is not None and not top_corr_df.empty:
        lines.append("### Top correlation pairs")
        lines.append(top_corr_df.to_markdown(index=False))
        lines.append("")

    lines.append("## Security and governance notes")
    lines.append("This app enforces safe ingestion patterns (CSV only), timeouts for URL loads, and size limits for uploads and URL responses.")
    lines.append("Only load data from trusted sources when using URL ingestion.")
    lines.append("Reports are generated locally and are user initiated.")
    lines.append("PSI is a screening signal used to prioritize review and should be interpreted alongside data context and business impact.")
    lines.append("")

    lines.append("## Testing and product readiness")
    lines.append("The product includes an interactive dashboard, downloadable report artifacts, plot export capability, and a Testing tab for self-test execution.")
    lines.append("Self-tests can also be run from the command line with `python tests/run_tests.py`.")
    lines.append("This report structure is designed to support capstone documentation by pairing technical evidence with plain-language interpretation.")
    lines.append("")

    return '\n'.join(lines)


def save_report_bundle(
    *,
    output_dir: Path,
    base_filename: str,
    report_markdown: str,
    drift_df: pd.DataFrame,
    extra_tables: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Path]:
    """Save report and tables to disk with safe filenames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _safe_slug(base_filename)

    artifacts: Dict[str, Path] = {}
    report_path = output_dir / f"{slug}.md"
    report_path.write_text(report_markdown, encoding="utf-8")
    artifacts["report_md"] = report_path

    drift_path = output_dir / f"{slug}_drift.csv"
    drift_df.to_csv(drift_path, index=False)
    artifacts["drift_csv"] = drift_path

    if extra_tables:
        for name, df in extra_tables.items():
            p = output_dir / f"{slug}_{_safe_slug(name)}.csv"
            df.to_csv(p, index=False)
            artifacts[f"{name}_csv"] = p

    return artifacts
