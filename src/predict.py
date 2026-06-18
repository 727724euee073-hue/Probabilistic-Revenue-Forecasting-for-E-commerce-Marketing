from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from .forecast_model import RevenueForecastModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the pickled model and write probabilistic revenue predictions.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.model, "rb") as handle:
        model = pickle.load(handle)

    if not isinstance(model, RevenueForecastModel):
        raise TypeError(f"Unexpected model type: {type(model)!r}")

    features = pd.read_csv(args.features)
    predictions = model.predict(features)

    required_columns = ["forecast_period", "predicted_revenue", "p10", "p50", "p90", "predicted_roas", "confidence_score"]
    predictions = predictions.loc[:, required_columns].copy()
    predictions["forecast_period"] = predictions["forecast_period"].astype(str)
    for column in ["predicted_revenue", "p10", "p50", "p90"]:
        predictions[column] = predictions[column].round(0).astype(int)
    predictions["predicted_roas"] = predictions["predicted_roas"].round(4)
    predictions["confidence_score"] = predictions["confidence_score"].round(4)
    predictions.to_csv(Path(args.output), index=False)


if __name__ == "__main__":
    main()
