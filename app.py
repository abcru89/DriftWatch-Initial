from __future__ import annotations

import datetime as dt
from pathlib import Path
from io import BytesIO
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from driftwatch.cleaning import apply_cleaning_pipeline
from driftwatch.drift import compute_feature_drift, overall_health
from driftwatch.io import load_csv_from_upload, load_csv_from_url, load_csv_from_path
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


APP_TITLE = "DriftWatch"
st.set_page_config(page_title=f"{APP_TITLE} (Final Iteration)", layout="wide")

st.title(f"{APP_TITLE} (Final Iteration)")
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
def safe_df(df: Optional[pd.DataFrame]) -> bool:
    return df is not None and isinstance(df, pd.DataFrame) and df.shape[0] > 0 and df.shape[1] > 0

def show_exploration(df: pd.DataFrame, title: str):
    st.subheader(f"{title}: Exploration")
    st.write(f"Rows: **{len(df):,}** | Columns: **{df.shape[1]}**")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("**Schema and missingness**")
    st.dataframe(schema_missingness(df), use_container_width=True, height=260)

    st.markdown("**Descriptive statistics (numeric)**")
    st.dataframe(numeric_stats(df), use_container_width=True)

def show_visuals(df: pd.DataFrame, title: str):
    st.subheader(f"{title}: Visuals")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Numeric**")
        if num_cols:
            col = st.selectbox(f"{title} numeric column", num_cols, key=f"{title}_num_col")
            fig = plot_numeric_hist(df, col)
            run_id = st.session_state.get("run_id", "run")
            png_name = f"{run_id}_{title.replace(' ', '_')}_{col}_numeric.png"
            # Render before display because clear_figure may clear the canvas
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
        st.markdown("**Categorical**")
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
    fig, ax = plt.subplots()
    im = ax.imshow(corr.values)
    ax.set_title(title)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    fig.tight_layout()
    run_id = st.session_state.get("run_id", "run")
    png_name = f"{run_id}_correlation_heatmap.png"
    png = fig_to_png_bytes(fig)
    st.pyplot(fig, clear_figure=True)
    st.download_button(
        "Download correlation heatmap (PNG)",
        data=png,
        file_name=png_name,
        mime="image/png",
        key=f"dl_corr_{run_id}",
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
        if not run_btn:
            st.warning("Datasets loaded. Choose cleaning and analysis methods, then click Run analysis.")
        else:
            params = {"fill_value": fill_value, "iqr_k": iqr_k}

            # Apply cleaning
            b_clean = baseline_df.copy()
            c_clean = current_df.copy()

            b_log = None
            c_log = None
            if apply_to == "Baseline and Current":
                b_clean, b_log = apply_cleaning_pipeline(b_clean, selected_steps, params=params)
                c_clean, c_log = apply_cleaning_pipeline(c_clean, selected_steps, params=params)
            else:
                c_clean, c_log = apply_cleaning_pipeline(c_clean, selected_steps, params=params)

            # Store results in session state for Reports tab
            st.session_state["b_clean"] = b_clean
            st.session_state["c_clean"] = c_clean
            st.session_state["b_log"] = b_log
            st.session_state["c_log"] = c_log
            st.session_state["selected_steps"] = selected_steps
            st.session_state["selected_analyses"] = selected_analyses
            st.session_state["run_id"] = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state["run_ts"] = dt.datetime.now().isoformat(timespec="seconds")

            # Always compute drift list if selected
            drift_list = []
            health, tier = ("unknown", "unknown")
            drift_df = pd.DataFrame()
            if "Drift screening (PSI)" in selected_analyses:
                drift_list = compute_feature_drift(b_clean, c_clean)
                health, tier = overall_health(drift_list)
                if drift_list:
                    drift_df = pd.DataFrame([{
                        "feature": d.feature,
                        "kind": d.kind,
                        "psi": round(d.psi, 6),
                        "severity": d.severity
                    } for d in drift_list])
                st.session_state["drift_df"] = drift_df
                st.session_state["health"] = health
                st.session_state["tier"] = tier

            # Front door summary (always visible)
            st.subheader("Front door")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Overall system health", health.upper())
            kpi2.metric("Drift type", "DATA-SIDE")
            kpi3.metric("Active severity tier", tier.upper())
            if not drift_df.empty:
                top = drift_df.iloc[0]
                summary = f"Most recent drift: {top['feature']} shows {top['severity']} drift (PSI={top['psi']})."
            else:
                summary = "Most recent drift: none computed (check selections and inputs)."
            kpi4.metric("Latest event summary", summary)
            st.caption("Tip: Use the analysis methods selector in the sidebar to show only what you need.")

            # Cleaning log
            with st.expander("Cleaning and preprocessing log"):
                st.write("Selected steps (in order):")
                st.write(selected_steps if selected_steps else ["(none)"])
                if b_log and b_log.steps:
                    st.markdown("**Baseline log**")
                    for s in b_log.steps:
                        st.write("- " + s)
                if c_log and c_log.steps:
                    st.markdown("**Current log**")
                    for s in c_log.steps:
                        st.write("- " + s)

            # Analysis sections based on selection
            if "Drift screening (PSI)" in selected_analyses:
                st.subheader("Drift details (PSI screening)")
                if not drift_df.empty:
                    st.dataframe(drift_df, use_container_width=True, height=320)
                    st.caption("Rule of thumb: PSI < 0.10 none, 0.10 to 0.25 moderate, greater than 0.25 significant (screening only).")
                else:
                    st.info("No drift computed. Ensure shared columns exist and data is sufficient.")

            colA, colB = st.columns(2)

            if "Data quality and schema" in selected_analyses or "Descriptive statistics" in selected_analyses:
                with colA:
                    if "Data quality and schema" in selected_analyses or "Descriptive statistics" in selected_analyses:
                        show_exploration(b_clean, "Baseline (post-cleaning)")
                with colB:
                    if "Data quality and schema" in selected_analyses or "Descriptive statistics" in selected_analyses:
                        show_exploration(c_clean, "Current (post-cleaning)")

            if "Visual investigation" in selected_analyses:
                with colA:
                    show_visuals(b_clean, "Baseline (post-cleaning)")
                with colB:
                    show_visuals(c_clean, "Current (post-cleaning)")

            if "Baseline vs current comparison" in selected_analyses:
                st.subheader("Baseline vs current comparison")
                comp = compare_baseline_current(b_clean, c_clean)
                st.session_state["compare_df"] = comp
                st.dataframe(comp, use_container_width=True, height=320)

            if "Outlier summary (IQR)" in selected_analyses:
                st.subheader("Outlier summary (IQR)")
                out_df = outlier_summary_iqr(c_clean, k=iqr_k)
                st.session_state["outlier_df"] = out_df
                st.dataframe(out_df, use_container_width=True, height=260)

            if "Correlation analysis (numeric)" in selected_analyses:
                st.subheader("Correlation analysis (numeric)")
                corr = correlation_matrix(c_clean)
                st.session_state["corr_df"] = corr
                plot_corr_heatmap(corr, "Current data correlation matrix")

            st.caption(f"Run timestamp: {st.session_state.get('run_ts', '')}")

# Reports tab
with tab_reports:
    st.markdown("### Generate and save reports")
    st.write("This tab develops report capability now. You can expand report content in Topic 8 without changing the core workflow.")
    if "b_clean" not in st.session_state or "c_clean" not in st.session_state:
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
            baseline_schema=b_schema,
            current_schema=c_schema,
            compare_df=compare_df,
            outlier_df=outlier_df,
        )

        st.markdown("#### Report preview")
        st.markdown(report_md)

        st.markdown("#### Save artifacts to file")
        base_name = f"driftwatch_report_{run_id}"
        if st.button("Save report bundle to outputs folder"):
            extras = {}
            if compare_df is not None and isinstance(compare_df, pd.DataFrame) and not compare_df.empty:
                extras["comparison"] = compare_df
            if outlier_df is not None and isinstance(outlier_df, pd.DataFrame) and not outlier_df.empty:
                extras["outliers"] = outlier_df
            artifacts = save_report_bundle(
                output_dir=OUTPUT_DIR,
                base_filename=base_name,
                report_markdown=report_md,
                drift_df=drift_df if isinstance(drift_df, pd.DataFrame) else pd.DataFrame(),
                extra_tables=extras if extras else None,
            )
            st.success("Saved report artifacts.")
            st.session_state["saved_artifacts"] = artifacts

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
