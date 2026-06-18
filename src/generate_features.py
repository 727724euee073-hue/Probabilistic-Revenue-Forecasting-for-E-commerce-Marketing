from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_processing import build_daily_features, estimate_missing_revenue, load_all_data, summarize_feature_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate normalized features from campaign data.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_all_data(args.data_dir)
    raw = estimate_missing_revenue(raw)
    features = build_daily_features(raw)
    metadata = summarize_feature_metadata(raw)

    features_path = output_dir / "features.csv"
    metadata_path = output_dir / "feature_metadata.json"
    raw_path = output_dir / "normalized_data.csv"

    features.to_csv(features_path, index=False)
    raw.to_csv(raw_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
