
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    kind: str  # "numeric" or "categorical"
    severity: str  # "none", "moderate", "significant"


def _severity(psi: float) -> str:
    # Common rule of thumb thresholds for PSI
    if psi < 0.10:
        return "none"
    if psi < 0.25:
        return "moderate"
    return "significant"


def psi_numeric(expected: pd.Series, actual: pd.Series, bins: int = 10, eps: float = 1e-6) -> float:
    """
    Population Stability Index (PSI) for numeric variables.

    Notes:
    - Bin edges are derived from the expected (baseline) distribution.
    - Edges are extended to (-inf, +inf) so actual values outside the baseline range
      still land in a valid bin (avoids NaN bins from out-of-range values).
    - Distributions are aligned by index before computing PSI to prevent shape mismatches.
    """
    e = expected.dropna().astype(float)
    a = actual.dropna().astype(float)
    if len(e) < 2 or len(a) < 2:
        return float("nan")

    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(e, qs))

    if len(edges) <= 2:
        edges = np.unique(np.linspace(e.min(), e.max(), bins + 1))

    if len(edges) < 3:
        return float("nan")

    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    e_bins = pd.cut(e, bins=edges, include_lowest=True)
    a_bins = pd.cut(a, bins=edges, include_lowest=True)

    e_dist = e_bins.value_counts(normalize=True, dropna=False).sort_index()
    a_dist = a_bins.value_counts(normalize=True, dropna=False).sort_index()

    idx = e_dist.index.union(a_dist.index)
    e_dist = e_dist.reindex(idx, fill_value=0.0)
    a_dist = a_dist.reindex(idx, fill_value=0.0)

    e_p = np.clip(e_dist.values, eps, 1.0)
    a_p = np.clip(a_dist.values, eps, 1.0)

    psi = np.sum((a_p - e_p) * np.log(a_p / e_p))
    return float(psi)


def psi_categorical(expected: pd.Series, actual: pd.Series, eps: float = 1e-6) -> float:
    e = expected.fillna("<<MISSING>>").astype(str)
    a = actual.fillna("<<MISSING>>").astype(str)

    e_dist = e.value_counts(normalize=True, dropna=False)
    a_dist = a.value_counts(normalize=True, dropna=False)

    cats = sorted(set(e_dist.index).union(set(a_dist.index)))
    e_p = np.array([e_dist.get(c, 0.0) for c in cats])
    a_p = np.array([a_dist.get(c, 0.0) for c in cats])

    e_p = np.clip(e_p, eps, 1.0)
    a_p = np.clip(a_p, eps, 1.0)

    psi = np.sum((a_p - e_p) * np.log(a_p / e_p))
    return float(psi)


def compute_feature_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    max_features: int = 50,
) -> List[FeatureDrift]:
    """
    Compute PSI for shared columns between baseline and current.
    Returns a list sorted by PSI descending (worst drift first).
    """
    features = [c for c in baseline.columns if c in current.columns][:max_features]
    out: List[FeatureDrift] = []

    for c in features:
        b = baseline[c]
        k = "numeric" if pd.api.types.is_numeric_dtype(b) else "categorical"
        if k == "numeric":
            val = psi_numeric(b, current[c])
        else:
            val = psi_categorical(b, current[c])
        if np.isnan(val):
            continue
        out.append(FeatureDrift(feature=c, psi=val, kind=k, severity=_severity(val)))

    out.sort(key=lambda x: x.psi, reverse=True)
    return out


def overall_health(drift_list: List[FeatureDrift]) -> Tuple[str, str]:
    """
    Returns (health_indicator, tier) where tier is the worst severity present.
    """
    if not drift_list:
        return ("unknown", "unknown")

    worst = drift_list[0].severity
    if worst == "significant":
        return ("at risk", "significant")
    if worst == "moderate":
        return ("watch", "moderate")
    return ("ok", "none")
