# Probabilistic Revenue Forecasting for E-commerce Marketing

Python 3.11 deterministic forecasting pipeline for NetElixir hackathon submissions.

## Run

```bash
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

The entrypoint reads all CSVs in `data/`, generates engineered features, loads the committed pickled forecast engine, and writes `output/predictions.csv` with the required 30d / 60d / 90d probabilistic forecasts.
