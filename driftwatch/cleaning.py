
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class CleaningLog:
    steps: List[str]


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def drop_duplicates(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    before = len(df)
    out = df.drop_duplicates()
    after = len(out)
    log.steps.append(f"Drop duplicates: {before-after} rows removed.")
    return out


def drop_rows_with_missing(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    before = len(df)
    out = df.dropna()
    log.steps.append(f"Drop rows with missing: {before-len(out)} rows removed.")
    return out


def impute_numeric(df: pd.DataFrame, strategy: str, log: CleaningLog) -> pd.DataFrame:
    out = df.copy()
    cols = _numeric_cols(out)
    if not cols:
        log.steps.append("Impute numeric: no numeric columns found.")
        return out
    for c in cols:
        if out[c].isna().any():
            if strategy == "mean":
                val = out[c].mean()
            elif strategy == "median":
                val = out[c].median()
            else:
                raise ValueError("strategy must be 'mean' or 'median'")
            out[c] = out[c].fillna(val)
    log.steps.append(f"Impute numeric ({strategy}) for columns: {', '.join(cols) if cols else 'none'}.")
    return out


def fill_missing_categorical(df: pd.DataFrame, fill_value: str, log: CleaningLog) -> pd.DataFrame:
    out = df.copy()
    cols = _categorical_cols(out)
    for c in cols:
        if out[c].isna().any():
            out[c] = out[c].fillna(fill_value)
    log.steps.append(f"Fill categorical missing with '{fill_value}' for columns: {', '.join(cols) if cols else 'none'}.")
    return out


def standardize_numeric(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    out = df.copy()
    cols = _numeric_cols(out)
    for c in cols:
        x = out[c].astype(float)
        mu = x.mean()
        sd = x.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            continue
        out[c] = (x - mu) / sd
    log.steps.append("Standardize numeric (z-score).")
    return out


def normalize_numeric_minmax(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    out = df.copy()
    cols = _numeric_cols(out)
    for c in cols:
        x = out[c].astype(float)
        mn = x.min()
        mx = x.max()
        if mx == mn or np.isnan(mx) or np.isnan(mn):
            continue
        out[c] = (x - mn) / (mx - mn)
    log.steps.append("Normalize numeric (min-max).")
    return out


def cap_outliers_iqr(df: pd.DataFrame, k: float, log: CleaningLog) -> pd.DataFrame:
    out = df.copy()
    cols = _numeric_cols(out)
    for c in cols:
        x = out[c].astype(float)
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or np.isnan(iqr):
            continue
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        out[c] = x.clip(lo, hi)
    log.steps.append(f"Cap outliers using IQR (k={k}).")
    return out


def apply_cleaning_pipeline(
    df: pd.DataFrame,
    selected_steps: List[str],
    params: Dict[str, object] | None = None,
) -> Tuple[pd.DataFrame, CleaningLog]:
    """
    Apply a selected list of cleaning steps in order.
    """
    params = params or {}
    log = CleaningLog(steps=[])
    out = df.copy()

    for step in selected_steps:
        if step == "Drop duplicates":
            out = drop_duplicates(out, log)
        elif step == "Drop rows with missing":
            out = drop_rows_with_missing(out, log)
        elif step == "Impute numeric (mean)":
            out = impute_numeric(out, "mean", log)
        elif step == "Impute numeric (median)":
            out = impute_numeric(out, "median", log)
        elif step == "Fill categorical missing":
            fill_value = str(params.get("fill_value", "UNKNOWN"))
            out = fill_missing_categorical(out, fill_value, log)
        elif step == "Standardize numeric (z-score)":
            out = standardize_numeric(out, log)
        elif step == "Normalize numeric (min-max)":
            out = normalize_numeric_minmax(out, log)
        elif step == "Cap outliers (IQR)":
            k = float(params.get("iqr_k", 1.5))
            out = cap_outliers_iqr(out, k, log)
        else:
            log.steps.append(f"Unknown step ignored: {step}")

    return out, log
