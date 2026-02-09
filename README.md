DriftWatch Submission Package

What this includes
- Streamlit app: app.py
- Core package: driftwatch/
- Sample datasets: sample_data/baseline.csv and sample_data/current.csv
- Outputs folder: outputs/ (reports saved here)
- Tests: tests/run_tests.py and driftwatch/testing.py
- Help is available inside the app in the sidebar Help section and in the Help tab.

How to run
1) Install dependencies:
   pip install -r requirements.txt
2) Start the app:
   streamlit run app.py
3) Load data (upload, URL, or sample_data)
4) Choose cleaning and analysis methods
5) Run analysis
6) Generate and save reports in the Reports tab
7) Run tests:
   python tests/run_tests.py

Report capability
Report generation and saving are implemented now. You can expand report content in Topic 8 without changing the core workflow.

Submission documents
- Milestone3_Submission.md
- Milestone3_Submission.docx
