
from pathlib import Path  # noqa: I001

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

BASELINE_FILE = ROOT / "reports" / "baseline" / "baseline_metrics.csv"
XGBOOST_FILE = ROOT / "reports" / "xgboost" / "metrics.csv"

OUTPUT_DIR = ROOT / "reports" / "model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    baseline = pd.read_csv(BASELINE_FILE)
    xgboost = pd.read_csv(XGBOOST_FILE)

    # Add model name to XGBoost metrics
    xgboost["Model"] = "XGBoost"

    # Combine results
    comparison = pd.concat(
        [baseline, xgboost],
        ignore_index=True
    )

    # Standardize column order
    comparison = comparison[["Model", "MAE", "RMSE", "RMSLE"]]

    comparison = comparison.sort_values("MAE").reset_index(drop=True)

    # Save comparison CSV
    output_file = OUTPUT_DIR / "model_comparison.csv"
    comparison.to_csv(output_file, index=False)

    print("\nFinal Model Comparison:")
    print(comparison.to_string(index=False))

    print("\nBest Model by MAE:")
    print(comparison.iloc[0]["Model"])

    # Generate metric comparison charts
    metrics = ["MAE", "RMSE", "RMSLE"]

    for metric in metrics:
        plt.figure(figsize=(10, 6))

        plt.bar(
            comparison["Model"],
            comparison[metric]
        )

        plt.title(f"Model Comparison - {metric}")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.xticks(rotation=20)
        plt.tight_layout()

        chart_path = OUTPUT_DIR / f"{metric.lower()}_comparison.png"
        plt.savefig(chart_path, dpi=300)
        plt.close()

        print("Saved:", chart_path)

    print("\nModel comparison completed successfully.")


if __name__ == "__main__":
    main()
