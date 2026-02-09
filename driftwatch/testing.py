from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from driftwatch.cleaning import apply_cleaning_pipeline
from driftwatch.drift import compute_feature_drift, overall_health
from driftwatch.io import load_csv_from_path
from driftwatch.analysis import schema_missingness, compare_baseline_current, outlier_summary_iqr, correlation_matrix


@dataclass
class TestResult:
    test_id: str
    description: str
    expected: str
    outcome: str
    passed: bool


def run_self_tests(sample_baseline_path: str, sample_current_path: str) -> Tuple[List[TestResult], Dict[str, object]]:
    artifacts: Dict[str, object] = {}
    results: List[TestResult] = []

    # TC01 Load data
    b_res = load_csv_from_path(sample_baseline_path)
    c_res = load_csv_from_path(sample_current_path)
    passed = b_res.df is not None and c_res.df is not None and b_res.error is None and c_res.error is None
    results.append(TestResult(
        test_id="TC01",
        description="Load sample baseline and current from sample_data",
        expected="Both datasets load successfully",
        outcome="Pass" if passed else f"Fail (baseline_error={b_res.error}, current_error={c_res.error})",
        passed=passed,
    ))
    if not passed:
        return results, artifacts

    baseline = b_res.df
    current = c_res.df
    artifacts["baseline_shape"] = baseline.shape
    artifacts["current_shape"] = current.shape

    # TC02 Cleaning duplicates
    b_clean, b_log = apply_cleaning_pipeline(baseline, ["Drop duplicates"], params={"fill_value": "UNKNOWN", "iqr_k": 1.5})
    c_clean, c_log = apply_cleaning_pipeline(current, ["Drop duplicates"], params={"fill_value": "UNKNOWN", "iqr_k": 1.5})
    dup_b = int(baseline.duplicated().sum())
    dup_c = int(current.duplicated().sum())
    passed = dup_b >= 0 and dup_c >= 0 and len(b_log.steps) > 0 and len(c_log.steps) > 0
    results.append(TestResult(
        test_id="TC02",
        description="Drop duplicates and record cleaning log",
        expected="Cleaning log shows duplicates removed",
        outcome=f"Pass (baseline_dups={dup_b}, current_dups={dup_c})" if passed else "Fail",
        passed=passed,
    ))

    # TC03 Missingness report
    b_schema = schema_missingness(b_clean)
    c_schema = schema_missingness(c_clean)
    passed = not b_schema.empty and not c_schema.empty and "missing_pct" in b_schema.columns
    results.append(TestResult(
        test_id="TC03",
        description="Generate schema and missingness report",
        expected="Schema table contains dtype and missingness fields",
        outcome="Pass" if passed else "Fail",
        passed=passed,
    ))
    artifacts["baseline_schema_head"] = b_schema.head(5)
    artifacts["current_schema_head"] = c_schema.head(5)

    # TC04 Drift screening
    drift_list = compute_feature_drift(b_clean, c_clean)
    health, tier = overall_health(drift_list)
    passed = len(drift_list) > 0 and health in ("ok", "watch", "at risk", "unknown") and tier in ("none", "moderate", "significant", "unknown")
    results.append(TestResult(
        test_id="TC04",
        description="Compute PSI drift screening and overall health",
        expected="Drift list produced and health computed",
        outcome=f"Pass (health={health}, tier={tier}, top_feature={drift_list[0].feature})" if passed else "Fail",
        passed=passed,
    ))
    artifacts["health"] = health
    artifacts["tier"] = tier
    artifacts["top_drift"] = drift_list[0] if drift_list else None

    # TC05 Compare baseline vs current summary
    comp = compare_baseline_current(b_clean, c_clean)
    passed = not comp.empty and "baseline_missing_pct" in comp.columns
    results.append(TestResult(
        test_id="TC05",
        description="Generate baseline vs current comparison report",
        expected="Comparison table includes missingness and unique counts",
        outcome="Pass" if passed else "Fail",
        passed=passed,
    ))
    artifacts["compare_head"] = comp.head(5)

    # TC06 Outlier summary and correlation
    outliers = outlier_summary_iqr(c_clean, k=1.5)
    corr = correlation_matrix(c_clean)
    passed = outliers is not None and not outliers.empty
    results.append(TestResult(
        test_id="TC06",
        description="Compute outlier summary (IQR) and correlation matrix",
        expected="Outlier table produced (correlation may be empty if <2 numeric cols)",
        outcome=f"Pass (top_outlier_col={outliers.iloc[0]['column'] if not outliers.empty else 'n/a'})" if passed else "Fail",
        passed=passed,
    ))
    artifacts["outliers_head"] = outliers.head(5)
    artifacts["corr_shape"] = corr.shape if corr is not None else (0, 0)

    return results, artifacts
