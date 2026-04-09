from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    kind: str  # "numeric" or "categorical"
    severity: str  # "none", "moderate", "significant"
    test_used: str
    p_value: float
    validated_drift: str  # "yes", "no", "unavailable"


def _severity(psi: float) -> str:
    if psi < 0.10:
        return "none"
    if psi < 0.25:
        return "moderate"
    return "significant"


def _validated_label(p_value: float, alpha: float = 0.05) -> str:
    if pd.isna(p_value):
        return "unavailable"
    return "yes" if p_value < alpha else "no"


def psi_numeric(expected: pd.Series, actual: pd.Series, bins: int = 10, eps: float = 1e-6) -> float:
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


def validate_numeric_ks(expected: pd.Series, actual: pd.Series) -> Tuple[str, float]:
    e = expected.dropna().astype(float).to_numpy()
    a = actual.dropna().astype(float).to_numpy()

    if len(e) < 2 or len(a) < 2:
        return ("ks_2samp", float("nan"))

    try:
        from scipy.stats import ks_2samp
        res = ks_2samp(e, a, alternative="two-sided", method="auto")
        return ("ks_2samp", float(res.pvalue))
    except Exception:
        return ("ks_2samp", float("nan"))


def validate_categorical_chi2(expected: pd.Series, actual: pd.Series) -> Tuple[str, float]:
    e = expected.fillna("<<MISSING>>").astype(str)
    a = actual.fillna("<<MISSING>>").astype(str)

    cats = sorted(set(e.unique()).union(set(a.unique())))
    if len(cats) < 2:
        return ("chi2_contingency", float("nan"))

    e_counts = e.value_counts().reindex(cats, fill_value=0)
    a_counts = a.value_counts().reindex(cats, fill_value=0)
    observed = np.vstack([e_counts.values, a_counts.values])

    if observed.sum() == 0:
        return ("chi2_contingency", float("nan"))

    try:
        from scipy.stats.contingency import chi2_contingency
        _, p_value, _, _ = chi2_contingency(observed, correction=False)
        return ("chi2_contingency", float(p_value))
    except Exception:
        return ("chi2_contingency", float("nan"))


def compute_feature_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    max_features: int = 50,
    alpha: float = 0.05,
) -> List[FeatureDrift]:
    features = [c for c in baseline.columns if c in current.columns][:max_features]
    out: List[FeatureDrift] = []

    for c in features:
        b = baseline[c]
        k = "numeric" if pd.api.types.is_numeric_dtype(b) else "categorical"

        if k == "numeric":
            psi_val = psi_numeric(b, current[c])
            test_used, p_value = validate_numeric_ks(b, current[c])
        else:
            psi_val = psi_categorical(b, current[c])
            test_used, p_value = validate_categorical_chi2(b, current[c])

        if np.isnan(psi_val):
            continue

        out.append(
            FeatureDrift(
                feature=c,
                psi=psi_val,
                kind=k,
                severity=_severity(psi_val),
                test_used=test_used,
                p_value=p_value,
                validated_drift=_validated_label(p_value, alpha=alpha),
            )
        )

    out.sort(key=lambda x: x.psi, reverse=True)
    return out


def overall_health(drift_list: List[FeatureDrift]) -> Tuple[str, str]:
    if not drift_list:
        return ("unknown", "unknown")

    worst = drift_list[0].severity
    validated = [d for d in drift_list if d.validated_drift == "yes"]

    if any(d.severity == "significant" for d in validated):
        return ("at risk", "significant")

    if any(d.severity in {"moderate", "significant"} for d in validated):
        return ("monitor", "moderate")

    if worst in {"moderate", "significant"}:
        return ("monitor", worst)

    return ("stable", "none")