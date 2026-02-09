from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def schema_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Return schema and missingness profile."""
    out = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(dropna=True),
    }).reset_index(names="column")
    return out


def numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric descriptive statistics."""
    return df.describe(include="number").T


def compare_baseline_current(baseline: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """
    Compare baseline and current at the column level.
    Provides basic evidence for what changed beyond PSI.
    """
    cols = sorted(set(baseline.columns).intersection(set(current.columns)))
    rows = []
    for c in cols:
        b = baseline[c]
        a = current[c]
        row = {
            "column": c,
            "baseline_dtype": str(b.dtype),
            "current_dtype": str(a.dtype),
            "baseline_missing_pct": round(float(b.isna().mean() * 100), 2),
            "current_missing_pct": round(float(a.isna().mean() * 100), 2),
            "baseline_n_unique": int(b.nunique(dropna=True)),
            "current_n_unique": int(a.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(b) and pd.api.types.is_numeric_dtype(a):
            row.update({
                "baseline_mean": float(np.nanmean(b)),
                "current_mean": float(np.nanmean(a)),
                "baseline_std": float(np.nanstd(b)),
                "current_std": float(np.nanstd(a)),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def outlier_summary_iqr(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Count IQR outliers per numeric column."""
    num = df.select_dtypes(include=["number"])
    rows = []
    for c in num.columns:
        x = num[c].dropna().astype(float)
        if len(x) < 4:
            rows.append({"column": c, "outliers": 0, "notes": "insufficient data"})
            continue
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or np.isnan(iqr):
            rows.append({"column": c, "outliers": 0, "notes": "zero IQR"})
            continue
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        n_out = int(((x < lo) | (x > hi)).sum())
        rows.append({"column": c, "outliers": n_out, "notes": ""})
    out = pd.DataFrame(rows).sort_values("outliers", ascending=False)
    return out


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix for numeric columns."""
    num = df.select_dtypes(include=["number"])
    if num.shape[1] < 2:
        return pd.DataFrame()
    return num.corr(numeric_only=True)
