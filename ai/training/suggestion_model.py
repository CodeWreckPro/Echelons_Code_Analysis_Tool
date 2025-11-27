"""
Training script for the Actionable Suggestions model.

This trains a small supervised model on tabular features extracted from:
- Complexity metrics (radon)
- Duplication ratio
- Volatility (churn/commit frequency)
- Maintainability index
- Ownership concentration

Target outputs:
- priority_score (0–1)
- roi_score (0–100)

The actual suggestions text is produced by rule/heuristic layers. The model focuses on scoring.
"""

import argparse
import os
from typing import List
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor




def train_suggestion_model(features_csv: str, output_path: str):
    df = pd.read_csv(features_csv)

    required = [
        "avg_function_complexity",
        "duplication_ratio",
        "file_change_frequency",
        "maintainability_index",
        "ownership_concentration",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = 0.0

    X = df[required].values
    # For demonstration, we synthesize targets if not present
    if "priority_score" not in df.columns:
        df["priority_score"] = np.clip(0.5 * (df["avg_function_complexity"] / 10.0) + 0.3 * df["duplication_ratio"] + 0.2 * df["file_change_frequency"], 0.0, 1.0)
    if "roi_score" not in df.columns:
        df["roi_score"] = np.clip(60 + 30 * df["duplication_ratio"] + 20 * (df["avg_function_complexity"] / 10.0), 0.0, 100.0)

    y_priority = df["priority_score"].values
    y_roi = df["roi_score"].values

    # Two regressors; pack together
    model_priority = GradientBoostingRegressor(random_state=42)
    model_roi = GradientBoostingRegressor(random_state=42)

    model_priority.fit(X, y_priority)
    model_roi.fit(X, y_roi)

    bundle = {
        "model_priority": model_priority,
        "model_roi": model_roi,
        "features": required,
        "version": "v1",
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(bundle, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train Suggestion Scoring Model")
    parser.add_argument("--features-csv", required=True, help="Path to CSV with features")
    parser.add_argument("--output", default="ai/models/suggestion_model.joblib", help="Output joblib path")
    args = parser.parse_args()
    out = train_suggestion_model(args.features_csv, args.output)
    print(f"Saved suggestion model to {out}")


if __name__ == "__main__":
    main()