
from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

COMPARISON_FILE = ROOT / "reports" / "model_comparison" / "model_comparison.csv"
MODEL_FILE = ROOT / "models" / "xgboost_demand_model.joblib"

XGBOOST_REPORT_DIR = ROOT / "reports" / "xgboost"
COMPARISON_DIR = ROOT / "reports" / "model_comparison"

TRACKING_DB = ROOT / "mlflow.db"


def main():
    # Configure local SQLite tracking
    mlflow.set_tracking_uri(
        f"sqlite:///{TRACKING_DB.as_posix()}"
    )

    mlflow.set_experiment("SmartDemand-AI-Demand-Forecasting")

    comparison = pd.read_csv(COMPARISON_FILE)

    print("Starting MLflow experiment tracking...")

    for _, row in comparison.iterrows():
        model_name = str(row["Model"])

        with mlflow.start_run(run_name=model_name):

            mlflow.log_param("model_type", model_name)
            mlflow.log_param("validation_period_days", 90)
            mlflow.log_param(
                "dataset",
                "Store Item Demand Forecasting"
            )

            mlflow.log_metrics({
                "mae": float(row["MAE"]),
                "rmse": float(row["RMSE"]),
                "rmsle": float(row["RMSLE"])
            })

            # Log XGBoost configuration and artifacts
            if model_name == "XGBoost":
                mlflow.log_params({
                    "n_estimators": 300,
                    "max_depth": 8,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "objective": "reg:squarederror",
                    "tree_method": "hist",
                    "random_state": 42
                })

                mlflow.log_artifact(str(MODEL_FILE))
                mlflow.log_artifact(
                    str(XGBOOST_REPORT_DIR / "metrics.csv")
                )
                mlflow.log_artifact(
                    str(XGBOOST_REPORT_DIR / "validation_predictions.csv")
                )
                mlflow.log_artifact(
                    str(XGBOOST_REPORT_DIR / "feature_importance.csv")
                )

            print(
                f"Logged {model_name}: "
                f"MAE={row['MAE']:.4f}, "
                f"RMSE={row['RMSE']:.4f}, "
                f"RMSLE={row['RMSLE']:.4f}"
            )

       # Log the overall model comparison report
    with mlflow.start_run(run_name="Model-Comparison-Report"):
        mlflow.log_artifact(str(COMPARISON_FILE))

        for metric in ["mae", "rmse", "rmsle"]:
            chart = COMPARISON_DIR / f"{metric}_comparison.png"
            mlflow.log_artifact(str(chart))

    print("\nMLflow tracking completed successfully.")
    print("Tracking database:", TRACKING_DB)
    print("Experiment:", "SmartDemand-AI-Demand-Forecasting")


if __name__ == "__main__":
    main()
