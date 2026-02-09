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
    baseline_schema: Optional[pd.DataFrame] = None,
    current_schema: Optional[pd.DataFrame] = None,
    compare_df: Optional[pd.DataFrame] = None,
    outlier_df: Optional[pd.DataFrame] = None,
) -> str:
    """Build a markdown report. Topic 8 can extend this with additional sections."""

    lines: List[str] = []
    lines.append(f"# DriftWatch Report")
    lines.append("")
    lines.append(f"Run ID: {run_id}")
    lines.append(f"Timestamp: {timestamp_iso}")
    lines.append("")
    lines.append("## Front door summary")
    lines.append(f"- Overall system health: {overall_health}")
    lines.append(f"- Drift type: data-side (PSI screening)")
    lines.append(f"- Active severity tier: {severity_tier}")
    if not drift_df.empty:
        top = drift_df.iloc[0]
        lines.append(f"- Latest event summary: {top['feature']} shows {top['severity']} drift (PSI={top['psi']}).")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- Analysis methods selected: {', '.join(analysis_methods) if analysis_methods else 'none'}")
    lines.append(f"- Cleaning steps (applied in order): {', '.join(cleaning_steps) if cleaning_steps else 'none'}")
    lines.append(f"- Baseline shape (rows, cols): {baseline_shape[0]}, {baseline_shape[1]}")
    lines.append(f"- Current shape (rows, cols): {current_shape[0]}, {current_shape[1]}")
    lines.append("")
    lines.append("## Drift details (PSI screening)")
    lines.append("Rule of thumb: PSI < 0.10 none, 0.10 to 0.25 moderate, greater than 0.25 significant.")
    lines.append("")
    if drift_df.empty:
        lines.append("No drift results were produced. Ensure the datasets share columns and have enough non-missing data.")
    else:
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
        lines.append("## Baseline vs current comparison")
        lines.append(compare_df.to_markdown(index=False))
        lines.append("")
    if outlier_df is not None and not outlier_df.empty:
        lines.append("## Outlier summary (IQR)")
        lines.append(outlier_df.to_markdown(index=False))
        lines.append("")

    lines.append("## Security notes")
    lines.append("This app enforces safe ingestion patterns (CSV only), timeouts for URL loads, and size limits for uploads and URL responses.")
    lines.append("Only load data from trusted sources when using URL ingestion. Reports are generated locally and are user initiated.")
    lines.append("")
    lines.append("## Topic 8 report expansion")
    lines.append("This report capability is implemented now and is designed to be expanded in Topic 8 with any additional required sections and visuals.")
    lines.append("")
    return "\n".join(lines)


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
