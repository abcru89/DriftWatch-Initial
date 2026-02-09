
from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_numeric_hist(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots()
    ax.hist(df[col].dropna(), bins=20)
    ax.set_title(f"Histogram: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    return fig


def plot_categorical_bar(df: pd.DataFrame, col: str, top_n: int = 20):
    counts = df[col].fillna("<<MISSING>>").astype(str).value_counts().head(top_n)
    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(f"Top categories: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    return fig
