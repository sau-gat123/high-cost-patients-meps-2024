# High-Cost Patient Dashboard (MEPS 2024)

Streamlit dashboard for the project "Predictors of High-Cost Patient Status Among U.S. Adults:
Logistic Regression vs. Naive Bayes Using MEPS 2024 Data".

## Run it
    pip install -r requirements.txt
    streamlit run app.py

## Rebuild the numbers (optional)
The raw MEPS file is NOT included. Download HC-256 (2024 Full-Year Consolidated) from
https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPUFNumber=HC-256,
save it as `h256.csv` here, then:

    pip install pandas numpy statsmodels scikit-learn matplotlib
    python export_dashboard_data.py

This runs `analysis.py` (the full analysis) and writes aggregate results to `data/`.

## Files
- `app.py` dashboard
- `analysis.py` full analysis (cleaning, EDA, assumption checks, regression, models)
- `export_dashboard_data.py` saves aggregate-only results for the dashboard
- `data/` aggregate results only, no person-level records

## Notes
Associations from one year of cross-sectional data, not causal. Only 9 high-cost adults were uninsured, so that estimate is imprecise.
