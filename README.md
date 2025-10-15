# Conversion KPI Analyzer

This Streamlit application helps you identify which parameters and parameter combinations are most associated with lead conversion. Upload a CSV that contains the mandatory columns `lead_code` and `conversion_flag` (0 or 1). After validation, you can explore single-parameter performance, multi-parameter combinations, and the correlation of numeric parameters with the conversion flag.

## Getting started

```bash
pip install -r requirements.txt
streamlit run app.py
```

1. Upload your CSV via the sidebar.
2. Click **Validate File** to run the data sanity checks.
3. Use the controls on the main page to explore parameter-level performance and correlations.

## Smoke test

After installing the dependencies you can confirm the page boots without runtime errors by running:

```bash
streamlit run app.py --server.headless true --server.port 8501
```

Press `Ctrl+C` in the terminal to stop the development server once you've confirmed it starts successfully.
