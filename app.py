from __future__ import annotations

import datetime as dt
from pathlib import Path
from io import BytesIO
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from driftwatch.cleaning import apply_cleaning_pipeline
from driftwatch.drift import compute_feature_drift, overall_health
from driftwatch.io import (
    load_csv_from_upload,
    load_csv_from_url,
    load_csv_from_path,
    upload_text_to_blob,
    upload_bytes_to_blob,
)
from driftwatch.viz import plot_categorical_bar, plot_numeric_hist
from driftwatch.analysis import (
    schema_missingness,
    numeric_stats,
    compare_baseline_current,
    outlier_summary_iqr,
    correlation_matrix,
)
from driftwatch.reports import build_markdown_report, save_report_bundle
from driftwatch.testing import run_self_tests

# ------------------ States: Define Loaded persistent state ------------------
def init_state():
    defaults = {
        "datasets_loaded": False,
        "analysis_complete": False,
        "baseline_raw": None,
        "current_raw": None,
        "baseline_clean": None,
        "current_clean": None,
        "drift_table": None,
        "comparison_table": None,
        "outlier_table": None,
        "report_md": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def safe_df(df: Optional[pd.DataFrame]) -> bool:
    return df is not None and isinstance(df, pd.DataFrame) and df.shape[0] > 0 and df.shape[1] > 0

APP_TITLE = "DriftWatch"
st.set_page_config(page_title=f"{APP_TITLE} (v.2 ReprtPNGs)", layout="wide")

st.title(f"{APP_TITLE} (v.2 ReprtPNGs)")
st.caption("Data-side drift screening with PSI, selectable analysis methods, reports, and operator-style front door signals.")

# Paths
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_data"
OUTPUT_DIR = BASE_DIR / "outputs"
TEST_DIR = BASE_DIR / "tests"

# ------------------ Help content ------------------
HELP_TEXT = """### What DriftWatch does
DriftWatch compares a Baseline dataset to a Current dataset and screens for data-side drift using Population Stability Index (PSI).
PSI is used as a screening signal. It helps you decide where to investigate first.

### How to use the product
1) Load Baseline and Current in the sidebar (file upload or URL).
2) Choose optional cleaning and preprocessing steps.
3) Choose which analysis methods to run.
4) Click Run analysis.
5) Review the front door summary, then drill down into details and visuals.
6) Generate a report and download the report artifacts.

### How to interpret PSI
Rule of thumb for screening:
- PSI < 0.10: none
- PSI 0.10 to 0.25: moderate
- PSI > 0.25: significant

### Security guidance
Only load data from trusted sources when using URL ingestion.
Uploads and URL loads are size limited and URL loads enforce timeouts.
Reports are generated locally and are user initiated.
"""


# ------------------ Sidebar: Loaders ------------------
st.sidebar.header("1) Load datasets")

def dataset_loader(label: str):
    method = st.sidebar.radio(f"{label} source", ["File upload (CSV)", "URL (CSV)", "Use sample_data"], key=f"{label}_method")
    df = None
    err = None

    if method == "File upload (CSV)":
        up = st.sidebar.file_uploader(f"Upload {label} CSV", type=["csv"], key=f"{label}_file")
        res = load_csv_from_upload(up)
        df, err = res.df, res.error
    elif method == "URL (CSV)":
        url = st.sidebar.text_input(f"{label} CSV URL", key=f"{label}_url", placeholder="https://.../data.csv")
        if st.sidebar.button(f"Load {label} from URL", key=f"{label}_load_url"):
            res = load_csv_from_url(url)
            df, err = res.df, res.error
            st.session_state[f"{label}_df"] = df
            st.session_state[f"{label}_err"] = err
        df = st.session_state.get(f"{label}_df")
        err = st.session_state.get(f"{label}_err")
    else:
        sample_path = SAMPLE_DIR / ("baseline.csv" if label.lower() == "baseline" else "current.csv")
        res = load_csv_from_path(str(sample_path))
        df, err = res.df, res.error

    return df, err

baseline_df, baseline_err = dataset_loader("Baseline")
current_df, current_err = dataset_loader("Current")

# Persist raw datasets and loader state
if safe_df(baseline_df):
    st.session_state["baseline_raw"] = baseline_df.copy()
if safe_df(current_df):
    st.session_state["current_raw"] = current_df.copy()

st.session_state["datasets_loaded"] = safe_df(baseline_df) and safe_df(current_df)


st.sidebar.markdown("---")
st.sidebar.header("2) Cleaning and preprocessing")

CLEANING_OPTIONS = [
    "Drop duplicates",
    "Drop rows with missing",
    "Impute numeric (mean)",
    "Impute numeric (median)",
    "Fill categorical missing",
    "Cap outliers (IQR)",
    "Standardize numeric (z-score)",
    "Normalize numeric (min-max)",
]

selected_steps = st.sidebar.multiselect("Choose cleaning methods (applied in order)", CLEANING_OPTIONS, default=["Drop duplicates"])
fill_value = st.sidebar.text_input("Fill value (categorical missing)", value="UNKNOWN")
iqr_k = st.sidebar.slider("IQR cap multiplier (k)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
apply_to = st.sidebar.radio("Apply cleaning to", ["Baseline and Current", "Current only"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("3) Data analysis methods")

ANALYSIS_METHODS = [
    "Drift screening (PSI)",
    "Data quality and schema",
    "Descriptive statistics",
    "Visual investigation",
    "Baseline vs current comparison",
    "Outlier summary (IQR)",
    "Correlation analysis (numeric)",
]

selected_analyses = st.sidebar.multiselect(
    "Choose analysis methods to run",
    ANALYSIS_METHODS,
    default=["Drift screening (PSI)", "Data quality and schema", "Visual investigation"],
)

st.sidebar.markdown("---")
st.sidebar.header("4) Run")

run_btn = st.sidebar.button("Run analysis", type="primary")
st.sidebar.markdown("---")
with st.sidebar.expander("Help", expanded=False):
    st.markdown(HELP_TEXT)

# ------------------ Utilities ------------------

def show_exploration(df: pd.DataFrame, title: str):
    st.markdown(f"**{title}**")
    st.write(f"Rows: **{len(df):,}** | Columns: **{df.shape[1]}**")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("**Schema and missingness**")
    st.dataframe(schema_missingness(df), use_container_width=True, height=260)

    st.markdown("**Numeric profile**")
    st.dataframe(numeric_stats(df), use_container_width=True)



def fig_to_png_bytes(fig) -> bytes:
    """Convert a Matplotlib figure to PNG bytes for Streamlit downloads."""
    buf = BytesIO()
    # Use tight bounding box so labels are not clipped in exports
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    return buf.getvalue()

def show_visuals(df: pd.DataFrame, title: str):
    st.markdown(f"**{title}**")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Numeric view**")
        if num_cols:
            col = st.selectbox(f"{title} numeric column", num_cols, key=f"{title}_num_col")
            fig = plot_numeric_hist(df, col)
            run_id = st.session_state.get("run_id", "run")
            png_name = f"{run_id}_{title.replace(' ', '_')}_{col}_numeric.png"
            png = fig_to_png_bytes(fig)
            st.pyplot(fig, clear_figure=True)
            st.download_button(
                "Download numeric plot (PNG)",
                data=png,
                file_name=png_name,
                mime="image/png",
                key=f"dl_num_{title}_{col}_{run_id}",
            )
        else:
            st.info("No numeric columns found.")
    with c2:
        st.markdown("**Categorical view**")
        if cat_cols:
            col = st.selectbox(f"{title} categorical column", cat_cols, key=f"{title}_cat_col")
            fig = plot_categorical_bar(df, col)
            run_id = st.session_state.get("run_id", "run")
            png_name = f"{run_id}_{title.replace(' ', '_')}_{col}_categorical.png"
            png = fig_to_png_bytes(fig)
            st.pyplot(fig, clear_figure=True)
            st.download_button(
                "Download categorical plot (PNG)",
                data=png,
                file_name=png_name,
                mime="image/png",
                key=f"dl_cat_{title}_{col}_{run_id}",
            )
        else:
            st.info("No categorical columns found.")

def plot_corr_heatmap(corr: pd.DataFrame, title: str):
    import matplotlib.pyplot as plt

    if corr.empty:
        st.info("Correlation matrix is not available (need at least 2 numeric columns).")
        return

    n = len(corr.columns)

    # Keep the figure intentionally compact
    fig_width = min(6.5, max(4.5, n * 0.45))
    fig_height = min(4.0, max(3.0, n * 0.35))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(corr.values, aspect="equal", interpolation="nearest")

    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()

    run_id = st.session_state.get("run_id", "run")
    png_name = f"{run_id}_correlation_heatmap.png"
    png = fig_to_png_bytes(fig)

    # Render it in a narrower center column so it does not take over the page
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.pyplot(fig, clear_figure=True)
        st.download_button(
            "Download correlation heatmap (PNG)",
            data=png,
            file_name=png_name,
            mime="image/png",
            key=f"dl_corr_{run_id}",
        )

def top_correlation_pairs(corr: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if corr.empty or len(corr.columns) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "correlation"])

    pairs = []
    cols = list(corr.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append(
                {
                    "feature_a": cols[i],
                    "feature_b": cols[j],
                    "correlation": corr.iloc[i, j],
                    "abs_correlation": abs(corr.iloc[i, j]),
                }
            )

    out = pd.DataFrame(pairs).sort_values("abs_correlation", ascending=False).head(top_n)
    return out[["feature_a", "feature_b", "correlation"]]

def get_operator_recommendation(health: str, tier: str, drift_df: pd.DataFrame) -> tuple[str, str]:
    health_l = (health or "").lower()
    tier_l = (tier or "").lower()

    if not drift_df.empty and "feature" in drift_df.columns:
        top_feature = str(drift_df.iloc[0]["feature"])
    else:
        top_feature = "the leading feature"

    if health_l == "at risk" or tier_l == "significant":
        return (
            "high",
            f"Recommended action: Investigate {top_feature} first, validate whether the shift is expected, and consider retraining or threshold review if business impact is confirmed."
        )
    elif health_l == "monitor" or tier_l == "moderate":
        return (
            "medium",
            f"Recommended action: Monitor {top_feature} closely, review recent data changes, and watch for continued movement in the next run."
        )
    elif health_l == "stable" or tier_l == "none":
        return (
            "low",
            "Recommended action: No immediate intervention is needed. Continue routine monitoring and review only if new signals appear."
        )
    else:
        return (
            "info",
            "Recommended action: Review the selected analyses and inputs, then rerun if needed to confirm the current state."
        )
# ------------------ Main UI ------------------
tab_dashboard, tab_reports, tab_help, tab_testing = st.tabs(["Dashboard", "Reports", "Help", "Testing"])

with tab_help:
    st.markdown(HELP_TEXT)
    st.markdown("### Feature guide")
    st.markdown("- Load datasets: upload CSV or use URL or sample_data.")
    st.markdown("- Cleaning: applies steps in the order selected.")
    st.markdown("- Analysis methods: choose what to run for this session.")
    st.markdown("- Reports: generate and save report artifacts for submission.")
    st.markdown("- Testing: run self-tests and review outcomes.")

# Dashboard tab
with tab_dashboard:
    if not (safe_df(baseline_df) and safe_df(current_df)):
        st.info("Load Baseline and Current datasets in the sidebar. Tip: sample CSVs are available under sample_data.")
        if baseline_err:
            st.error(f"Baseline error: {baseline_err}")
        if current_err:
            st.error(f"Current error: {current_err}")
    else:
        # Only compute when user clicks Run analysis
        if run_btn:
            params = {"fill_value": fill_value, "iqr_k": iqr_k}

            b_clean = baseline_df.copy()
            c_clean = current_df.copy()

            b_log = None
            c_log = None
            if apply_to == "Baseline and Current":
                b_clean, b_log = apply_cleaning_pipeline(b_clean, selected_steps, params=params)
                c_clean, c_log = apply_cleaning_pipeline(c_clean, selected_steps, params=params)
            else:
                c_clean, c_log = apply_cleaning_pipeline(c_clean, selected_steps, params=params)

            # Save results FIRST
            st.session_state["b_clean"] = b_clean
            st.session_state["c_clean"] = c_clean
            st.session_state["b_log"] = b_log
            st.session_state["c_log"] = c_log
            st.session_state["selected_steps"] = selected_steps
            st.session_state["selected_analyses"] = selected_analyses
            st.session_state["run_id"] = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state["run_ts"] = dt.datetime.now().isoformat(timespec="seconds")

            drift_list = []
            health, tier = ("unknown", "unknown")
            drift_df = pd.DataFrame()

            if "Drift screening (PSI)" in selected_analyses:
                drift_list = compute_feature_drift(b_clean, c_clean)
                health, tier = overall_health(drift_list)
                if drift_list:
                    drift_df = pd.DataFrame([
                        {
                            "feature": d.feature,
                            "kind": d.kind,
                            "psi": round(d.psi, 6),
                            "severity": d.severity,
                        }
                        for d in drift_list
                    ])

            st.session_state["drift_df"] = drift_df
            st.session_state["health"] = health
            st.session_state["tier"] = tier

            if "Baseline vs current comparison" in selected_analyses:
                st.session_state["compare_df"] = compare_baseline_current(b_clean, c_clean)
            else:
                st.session_state["compare_df"] = None

            if "Outlier summary (IQR)" in selected_analyses:
                st.session_state["outlier_df"] = outlier_summary_iqr(c_clean, k=iqr_k)
            else:
                st.session_state["outlier_df"] = None

            if "Correlation analysis (numeric)" in selected_analyses:
                st.session_state["corr_df"] = correlation_matrix(c_clean)
            else:
                st.session_state["corr_df"] = None

            st.session_state["analysis_complete"] = True

        # Render SECOND, from saved state
        if not st.session_state.get("analysis_complete", False):
            st.warning("Datasets loaded. Choose cleaning and analysis methods, then click Run analysis.")
        else:
            b_clean = st.session_state["b_clean"]
            c_clean = st.session_state["c_clean"]
            b_log = st.session_state.get("b_log")
            c_log = st.session_state.get("c_log")
            drift_df = st.session_state.get("drift_df", pd.DataFrame())
            health = st.session_state.get("health", "unknown")
            tier = st.session_state.get("tier", "unknown")

            # All the rest of the dashboard renderings must come after this comment or break visualizaitons

            # Cleaning log
            saved_steps = st.session_state.get("selected_steps", [])
            saved_analyses = st.session_state.get("selected_analyses", [])

            # Front door summary (always visible)
            st.subheader("Front door")

            if health.lower() == "at risk":
                status_label = "AT RISK"
                status_note = "Drift signals require operator review."
            elif health.lower() == "monitor":
                status_label = "MONITOR"
                status_note = "Some drift signals are present. Watch for further movement."
            elif health.lower() == "stable":
                status_label = "STABLE"
                status_note = "No notable drift signals detected in this run."
            else:
                status_label = health.upper()
                status_note = "Review the selected analyses and underlying inputs."

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("System status", status_label)
            kpi2.metric("Signal type", "DATA-SIDE")
            kpi3.metric("Current severity", tier.upper())

            if not drift_df.empty:
                top = drift_df.iloc[0]
                summary = f"{top['feature']} is the leading signal with {top['severity']} drift (PSI={top['psi']})."
            else:
                summary = "No drift signal was produced in the last completed run."

            kpi4.metric("Latest drift signal", summary)

            st.info(f"Operator summary: {status_note}")

            action_level, action_text = get_operator_recommendation(health, tier, drift_df)

            if action_level == "high":
                st.error(action_text)
            elif action_level == "medium":
                st.warning(action_text)
            elif action_level == "low":
                st.success(action_text)
            else:
                st.info(action_text)

            st.caption("Review the front door first, then open only the sections you need.")

            st.markdown("### Quick comparison")

            shared_cols = len(set(b_clean.columns).intersection(set(c_clean.columns)))
            row_delta = len(c_clean) - len(b_clean)

            if not drift_df.empty and "severity" in drift_df.columns:
                active_drift_count = int((drift_df["severity"].str.lower() != "none").sum())
            else:
                active_drift_count = 0

            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Baseline rows", f"{len(b_clean):,}")
            q2.metric("Current rows", f"{len(c_clean):,}")
            q3.metric("Row delta", f"{row_delta:+,}")
            q4.metric("Features with drift", f"{active_drift_count:,}")

            q5, q6 = st.columns(2)
            with q5:
                st.metric("Baseline columns", f"{b_clean.shape[1]:,}")
            with q6:
                st.metric("Shared columns", f"{shared_cols:,}")

            # Cleaning log
            with st.expander("Processing log", expanded=False):
                st.write("Selected steps (in order):")
                st.write(saved_steps if saved_steps else ["(none)"])

                if b_log and b_log.steps:
                    st.markdown("**Baseline log**")
                    for s in b_log.steps:
                        st.write("- " + s)

                if c_log and c_log.steps:
                    st.markdown("**Current log**")
                    for s in c_log.steps:
                        st.write("- " + s)

            # Drift investigation
            with st.expander("Drift investigation", expanded=False):
                if "Drift screening (PSI)" in saved_analyses:
                    if not drift_df.empty:
                        st.dataframe(drift_df, use_container_width=True, height=320)
                        st.caption(
                            "Rule of thumb: PSI < 0.10 none, 0.10 to 0.25 moderate, greater than 0.25 significant (screening only)."
                        )
                    else:
                        st.info("No drift computed. Ensure shared columns exist and data is sufficient.")
                else:
                    st.info("Drift screening was not selected for this run.")

            # Data exploration
            with st.expander("Data exploration", expanded=False):
                if "Data quality and schema" in saved_analyses or "Descriptive statistics" in saved_analyses:
                    colA, colB = st.columns(2)
                    with colA:
                        show_exploration(b_clean, "Baseline data")
                    with colB:
                        show_exploration(c_clean, "Current data")
                else:
                    st.info("Data exploration was not selected for this run.")

            # Visual analysis
            with st.expander("Visual analysis", expanded=False):
                rendered_visuals = False

                if "Visual investigation" in saved_analyses:
                    visA, visB = st.columns(2)
                    with visA:
                        show_visuals(b_clean, "Baseline data")
                    with visB:
                        show_visuals(c_clean, "Current data")
                    rendered_visuals = True

                if "Correlation analysis (numeric)" in saved_analyses:
                    st.markdown("**Current data correlation**")
                    corr = st.session_state.get("corr_df")
                    if corr is None or not isinstance(corr, pd.DataFrame):
                        corr = correlation_matrix(c_clean)
                        st.session_state["corr_df"] = corr
                    
                    top_corr = top_correlation_pairs(corr, top_n=5)

                    if not top_corr.empty:
                        st.markdown("**Top correlation pairs**")
                        st.dataframe(top_corr, use_container_width=True, height=210)

                    plot_corr_heatmap(corr, "Correlation heatmap")
                    rendered_visuals = True

                if not rendered_visuals:
                    st.info("Visual analysis was not selected for this run.")

            # Comparison and outliers
            with st.expander("Comparison review", expanded=False):
                rendered_detail = False

                if "Baseline vs current comparison" in saved_analyses:
                    st.markdown("**Baseline vs current comparison**")
                    comp = st.session_state.get("compare_df")
                    if comp is None or not isinstance(comp, pd.DataFrame):
                        comp = compare_baseline_current(b_clean, c_clean)
                        st.session_state["compare_df"] = comp
                    st.dataframe(comp, use_container_width=True, height=320)
                    rendered_detail = True

                if "Outlier summary (IQR)" in saved_analyses:
                    st.markdown("**Outlier summary (IQR)**")
                    out_df = st.session_state.get("outlier_df")
                    if out_df is None or not isinstance(out_df, pd.DataFrame):
                        out_df = outlier_summary_iqr(c_clean, k=iqr_k)
                        st.session_state["outlier_df"] = out_df
                    st.dataframe(out_df, use_container_width=True, height=260)
                    rendered_detail = True

                if not rendered_detail:
                    st.info("Comparison and outlier review were not selected for this run.")

            st.caption(f"Run timestamp: {st.session_state.get('run_ts', '')}")

# Reports tab
with tab_reports:
    st.markdown("### Generate and save reports")
    st.write("This tab develops report capability now. You can expand report content in Topic 8 without changing the core workflow.")
    if not st.session_state.get("analysis_complete", False):
        st.info("Run analysis first to generate report content.")
    else:
        b_clean = st.session_state["b_clean"]
        c_clean = st.session_state["c_clean"]
        drift_df = st.session_state.get("drift_df", pd.DataFrame())
        health = st.session_state.get("health", "unknown")
        tier = st.session_state.get("tier", "unknown")
        compare_df = st.session_state.get("compare_df")
        outlier_df = st.session_state.get("outlier_df")

        run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp_iso = st.session_state.get("run_ts", dt.datetime.now().isoformat(timespec="seconds"))
        analysis_methods = st.session_state.get("selected_analyses", [])
        cleaning_steps = st.session_state.get("selected_steps", [])
        b_schema = schema_missingness(b_clean)
        c_schema = schema_missingness(c_clean)

        corr_df = st.session_state.get("corr_df")
        top_corr_df = (
            top_correlation_pairs(corr_df, top_n=5)
            if isinstance(corr_df, pd.DataFrame) and not corr_df.empty
            else None
        )

        if health.lower() == "at risk":
            operator_summary = "Drift signals require operator review."
        elif health.lower() == "monitor":
            operator_summary = "Some drift signals are present. Watch for further movement."
        elif health.lower() == "stable":
            operator_summary = "No notable drift signals detected in this run."
        else:
            operator_summary = "Review the selected analyses and underlying inputs."

        _, recommended_action = get_operator_recommendation(health, tier, drift_df)

        if not drift_df.empty:
            top = drift_df.iloc[0]
            latest_drift_signal = f"{top['feature']} is the leading signal with {top['severity']} drift (PSI={top['psi']})."
        else:
            latest_drift_signal = "No drift signal was produced in the last completed run."

        shared_cols = len(set(b_clean.columns).intersection(set(c_clean.columns)))
        row_delta = len(c_clean) - len(b_clean)

        if not drift_df.empty and "severity" in drift_df.columns:
            active_drift_count = int((drift_df["severity"].astype(str).str.lower() != "none").sum())
        else:
            active_drift_count = 0

        quick_compare = {
            "baseline_rows": len(b_clean),
            "current_rows": len(c_clean),
            "row_delta": row_delta,
            "baseline_columns": b_clean.shape[1],
            "shared_columns": shared_cols,
            "features_with_drift": active_drift_count,
        }

        report_md = build_markdown_report(
            run_id=run_id,
            timestamp_iso=timestamp_iso,
            analysis_methods=analysis_methods,
            cleaning_steps=cleaning_steps,
            baseline_shape=b_clean.shape,
            current_shape=c_clean.shape,
            overall_health=health,
            severity_tier=tier,
            drift_df=drift_df,
            operator_summary=operator_summary,
            recommended_action=recommended_action,
            latest_drift_signal=latest_drift_signal,
            quick_compare=quick_compare,
            top_corr_df=top_corr_df,
            baseline_schema=b_schema,
            current_schema=c_schema,
            compare_df=compare_df,
            outlier_df=outlier_df,
        )

        st.markdown("#### Report preview")
        st.markdown(report_md)

        st.markdown("#### Save artifacts to file")

        base_name = f"driftwatch_report_{run_id}"
        archive_to_azure = st.checkbox("Also archive report artifacts to Azure Blob Storage", value=False)

        if archive_to_azure:
            st.caption("Azure archival will run after the local outputs folder save completes.")

        if st.button("Save report bundle to outputs folder"):
            extras = {}

            if compare_df is not None and isinstance(compare_df, pd.DataFrame) and not compare_df.empty:
                extras["comparison"] = compare_df
            if outlier_df is not None and isinstance(outlier_df, pd.DataFrame) and not outlier_df.empty:
                extras["outliers"] = outlier_df

            quick_compare_df = pd.DataFrame([quick_compare])
            if not quick_compare_df.empty:
                extras["quick_comparison"] = quick_compare_df

            if top_corr_df is not None and isinstance(top_corr_df, pd.DataFrame) and not top_corr_df.empty:
                extras["top_correlations"] = top_corr_df

            artifacts = save_report_bundle(
                output_dir=OUTPUT_DIR,
                base_filename=base_name,
                report_markdown=report_md,
                drift_df=drift_df if isinstance(drift_df, pd.DataFrame) else pd.DataFrame(),
                extra_tables=extras if extras else None,
            )

            st.success("Saved report artifacts locally.")
            st.session_state["saved_artifacts"] = artifacts

            if archive_to_azure:
                try:
                    prefix = f"runs/{run_id}"
                    azure_urls = {}

                    azure_urls["report_md"] = upload_text_to_blob(
                        report_md,
                        f"{prefix}/{base_name}.md",
                        content_type="text/markdown; charset=utf-8",
                        metadata={"run_id": run_id, "artifact": "report_markdown"},
                    )

                    if isinstance(drift_df, pd.DataFrame) and not drift_df.empty:
                        azure_urls["drift_csv"] = upload_bytes_to_blob(
                            drift_df.to_csv(index=False).encode("utf-8"),
                            f"{prefix}/{base_name}_drift.csv",
                            content_type="text/csv; charset=utf-8",
                            metadata={"run_id": run_id, "artifact": "drift_csv"},
                        )

                    if compare_df is not None and isinstance(compare_df, pd.DataFrame) and not compare_df.empty:
                        azure_urls["comparison_csv"] = upload_bytes_to_blob(
                            compare_df.to_csv(index=False).encode("utf-8"),
                            f"{prefix}/{base_name}_comparison.csv",
                            content_type="text/csv; charset=utf-8",
                            metadata={"run_id": run_id, "artifact": "comparison_csv"},
                        )

                    if outlier_df is not None and isinstance(outlier_df, pd.DataFrame) and not outlier_df.empty:
                        azure_urls["outliers_csv"] = upload_bytes_to_blob(
                            outlier_df.to_csv(index=False).encode("utf-8"),
                            f"{prefix}/{base_name}_outliers.csv",
                            content_type="text/csv; charset=utf-8",
                            metadata={"run_id": run_id, "artifact": "outliers_csv"},
                        )

                    if top_corr_df is not None and isinstance(top_corr_df, pd.DataFrame) and not top_corr_df.empty:
                        azure_urls["top_correlations_csv"] = upload_bytes_to_blob(
                            top_corr_df.to_csv(index=False).encode("utf-8"),
                            f"{prefix}/{base_name}_top_correlations.csv",
                            content_type="text/csv; charset=utf-8",
                            metadata={"run_id": run_id, "artifact": "top_correlations_csv"},
                        )

                    if not quick_compare_df.empty:
                        azure_urls["quick_comparison_csv"] = upload_bytes_to_blob(
                            quick_compare_df.to_csv(index=False).encode("utf-8"),
                            f"{prefix}/{base_name}_quick_comparison.csv",
                            content_type="text/csv; charset=utf-8",
                            metadata={"run_id": run_id, "artifact": "quick_comparison_csv"},
                        )

                    st.session_state["azure_artifacts"] = azure_urls
                    st.success("Azure Blob archival completed.")

                    with st.expander("Azure artifact locations", expanded=False):
                        for name, url in azure_urls.items():
                            st.write(f"{name}: {url}")

                except Exception as e:
                    st.error(f"Azure archival failed: {e}")

        # Downloads
        st.download_button(
            "Download report markdown",
            data=report_md.encode("utf-8"),
            file_name=f"{base_name}.md",
            mime="text/markdown",
        )
        if isinstance(drift_df, pd.DataFrame) and not drift_df.empty:
            st.download_button(
                "Download drift table CSV",
                data=drift_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{base_name}_drift.csv",
                mime="text/csv",
            )

# Testing tab
with tab_testing:
    st.markdown("### Test plan and outcomes")
    st.write("This section tests calculations, data processing, user engagement workflow, results, and visual report capability.")
    st.markdown("Run the full test script from the command line:")
    st.code("python tests/run_tests.py", language="bash")

    sample_baseline = str(SAMPLE_DIR / "baseline.csv")
    sample_current = str(SAMPLE_DIR / "current.csv")

    if st.button("Run self-tests now"):
        results, artifacts = run_self_tests(sample_baseline, sample_current)
        df_results = pd.DataFrame([{
            "test_id": r.test_id,
            "description": r.description,
            "expected": r.expected,
            "outcome": r.outcome,
            "passed": r.passed,
        } for r in results])
        st.dataframe(df_results, use_container_width=True)

        st.markdown("#### Key outcomes")
        st.write(f"Baseline shape: {artifacts.get('baseline_shape')}")
        st.write(f"Current shape: {artifacts.get('current_shape')}")
        st.write(f"Health: {artifacts.get('health')}")
        st.write(f"Tier: {artifacts.get('tier')}")

        if "outliers_head" in artifacts:
            st.markdown("Outlier summary (head):")
            st.dataframe(artifacts["outliers_head"], use_container_width=True)

        st.success("Self-tests completed. See table above for outcomes.")
