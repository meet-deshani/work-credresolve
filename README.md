# Conversion KPI Analyzer

This Streamlit application helps you identify which parameters and parameter combinations are most associated with lead conversion. Upload a CSV that contains the mandatory columns `lead_code` and `conversion_flag` (0 or 1). After validation, you can explore single-parameter performance, multi-parameter combinations, and the correlation of numeric parameters with the conversion flag.

## Getting started

```bash
pip install -r requirements.txt
streamlit run app.py
```

1. Open the **Data Upload & Validation** section and upload your CSV.
2. Click **Validate Data** to run the sanity checks and view the automatic data health summary.
3. Navigate with the sidebar to explore single parameters, parameter combinations, and focus recommendations that highlight high- and low-performing attributes.

## Smoke test

After installing the dependencies you can confirm the page boots without runtime errors by running:

```bash
streamlit run app.py --server.headless true --server.port 8501
```

Press `Ctrl+C` in the terminal to stop the development server once you've confirmed it starts successfully.
