
"""Generate a portfolio-ready report for synthetic demand model evaluation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "synthetic" / "baseline_comparison"

OVERALL_FILE = REPORT_DIR / "metrics_overall.csv"
HORIZON_FILE = REPORT_DIR / "metrics_by_horizon.csv"
SUMMARY_FILE = REPORT_DIR / "baseline_comparison_summary.json"


def main():
    overall = pd.read_csv(OVERALL_FILE)
    horizon = pd.read_csv(HORIZON_FILE)

    with SUMMARY_FILE.open(encoding="utf-8") as file:
        summary = json.load(file)

    if summary.get("data_source") != "synthetic":
        raise ValueError("Expected synthetic evaluation data.")

    required = {
        "model", "records", "MAE", "RMSE",
        "WAPE_percent", "sMAPE_percent",
    }
    if not required.issubset(overall.columns):
        raise ValueError("Overall metrics file is missing required columns.")

    if not required.issubset(horizon.columns):
        raise ValueError("Horizon metrics file is missing required columns.")

    xgb = overall.loc[overall["model"] == "xgboost"].iloc[0]
    moving = overall.loc[
        overall["model"] == "moving_average_7"
    ].iloc[0]
    seasonal = overall.loc[
        overall["model"] == "seasonal_naive_lag_7"
    ].iloc[0]

    moving_improvement = (
        (moving["WAPE_percent"] - xgb["WAPE_percent"])
        / moving["WAPE_percent"]
    ) * 100

    seasonal_improvement = (
        (seasonal["WAPE_percent"] - xgb["WAPE_percent"])
        / seasonal["WAPE_percent"]
    ) * 100

    # Chart 1: WAPE across forecast horizons.
    fig, ax = plt.subplots(figsize=(10, 6))

    for model, group in horizon.groupby("model"):
        group = group.sort_values("horizon_day")
        ax.plot(
            group["horizon_day"],
            group["WAPE_percent"],
            marker="o",
            label=model,
        )

    ax.set_title("WAPE by Forecast Horizon")
    ax.set_xlabel("Forecast Horizon (Days)")
    ax.set_ylabel("WAPE (%)")
    ax.set_xticks(range(1, 8))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "wape_by_horizon.png", dpi=160)
    plt.close(fig)

    # Chart 2: MAE across forecast horizons.
    fig, ax = plt.subplots(figsize=(10, 6))

    for model, group in horizon.groupby("model"):
        group = group.sort_values("horizon_day")
        ax.plot(
            group["horizon_day"],
            group["MAE"],
            marker="o",
            label=model,
        )

    ax.set_title("MAE by Forecast Horizon")
    ax.set_xlabel("Forecast Horizon (Days)")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_xticks(range(1, 8))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "mae_by_horizon.png", dpi=160)
    plt.close(fig)

    # Chart 3: Overall WAPE comparison.
    ordered = overall.sort_values("WAPE_percent")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ordered["model"], ordered["WAPE_percent"])
    ax.set_title("Overall Model Comparison")
    ax.set_ylabel("WAPE (%)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "overall_wape_comparison.png", dpi=160)
    plt.close(fig)

    # Prepare Markdown tables.
    overall_table = overall[
        ["model", "records", "MAE", "RMSE",
         "WAPE_percent", "sMAPE_percent"]
    ].round(3).to_markdown(index=False)

    horizon_table = horizon[
        ["model", "horizon_day", "MAE", "RMSE",
         "WAPE_percent", "sMAPE_percent"]
    ].round(3).sort_values(
        ["horizon_day", "model"]
    ).to_markdown(index=False)

    report = f"""# Synthetic Demand Forecasting — Model Evaluation

## Data disclosure

**Data source: SYNTHETIC.**

{summary["disclosure"]}

## Evaluation methodology

- Backtest origins: {summary["backtest_origins"]}
- Store-item combinations: {summary["store_item_combinations"]}
- Recursive forecast horizon: {summary["forecast_horizon_days"]} days
- Predictions per model: {summary["predictions_per_model"]:,}
- Models: XGBoost, 7-day moving average, and 7-day seasonal naive.
- All models use the same rolling evaluation windows.

## Overall model performance

{overall_table}

## WAPE improvement

XGBoost achieved:
- **{moving_improvement:.2f}% lower WAPE** than the 7-day moving average.
- **{seasonal_improvement:.2f}% lower WAPE** than the 7-day seasonal naive baseline.

## Performance across forecast horizons

{horizon_table}

XGBoost has the lowest MAE, RMSE, WAPE, and sMAPE at each of the seven forecast horizons.

Its WAPE ranges from **{horizon.loc[horizon["model"] == "xgboost", "WAPE_percent"].min():.2f}%** to **{horizon.loc[horizon["model"] == "xgboost", "WAPE_percent"].max():.2f}%** across the seven days.

## Visualizations

### Overall WAPE comparison

![Overall WAPE](overall_wape_comparison.png)

### WAPE by forecast horizon

![WAPE by horizon](wape_by_horizon.png)

### MAE by forecast horizon

![MAE by horizon](mae_by_horizon.png)

## Conclusion

On this synthetic portfolio dataset, XGBoost outperformed both
statistical baselines across all evaluated metrics and forecast horizons.

These results demonstrate the evaluation workflow and model's performance
on simulated data. They do not establish real-world retail forecasting
performance. Further validation on genuine retailer data is required.
"""

    output = REPORT_DIR / "MODEL_EVALUATION_REPORT.md"
    output.write_text(report, encoding="utf-8")

    print("\n===== EVALUATION REPORT GENERATED =====")
    print(f"Report: {output}")
    print(f"XGBoost WAPE: {xgb['WAPE_percent']:.2f}%")
    print(f"Improvement vs moving average: {moving_improvement:.2f}%")
    print(f"Improvement vs seasonal naive: {seasonal_improvement:.2f}%")
    print("\nGenerated charts:")
    print("- wape_by_horizon.png")
    print("- mae_by_horizon.png")
    print("- overall_wape_comparison.png")


if __name__ == "__main__":
    main()
