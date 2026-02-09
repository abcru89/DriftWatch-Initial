# DriftWatch Test Plan

## Objective
Validate every function of the data product, including calculations, data processing, user engagement workflow, results, and visual report capability.

## Test approach
Testing is executed in two ways:
1) Automated self-tests using the sample datasets:
   - Run: python tests/run_tests.py
   - Output: tests/TEST_RESULTS.md and tests/TEST_RESULTS.csv
2) Interactive verification in the Streamlit app:
   - Use the Testing tab and click Run self-tests now
   - Verify the UI renders results tables and does not crash

## Functions validated
- Loading data
  - Sample data load through local paths
  - Upload and URL paths are validated interactively in the app
- Cleaning pipeline
  - Duplicate removal
  - Missing value handling (available as selectable methods)
  - Outlier capping and numeric transforms (available as selectable methods)
- Drift screening calculations
  - PSI for numeric and categorical columns
  - Severity tiers (none, moderate, significant)
  - Overall health indicator based on worst tier
- Analysis methods
  - Schema and missingness reporting
  - Descriptive statistics
  - Baseline vs current comparison
  - Outlier summary (IQR)
  - Correlation analysis (numeric)
  - Visual investigation plots
- Report generation and saving
  - Markdown report generation
  - Saving report and tables to outputs folder
  - Download buttons for report and drift table

## Evidence
- Automated test outcomes are captured in tests/TEST_RESULTS.md and tests/TEST_RESULTS.csv
- Report artifacts are saved under outputs/ with a timestamped filename
