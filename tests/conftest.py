import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _build_synthetic_artifacts(model_path: Path, history_path: Path):
    from xgboost import XGBRegressor

    from src.inference.predict import FEATURES, create_features

    rng = np.random.default_rng(0)
    n_days = 400
    dates = pd.date_range("2016-11-01", periods=n_days)

    sales = np.clip(
        20
        + 5 * np.sin(np.arange(n_days) * 2 * np.pi / 7)  # pyright: ignore[reportOperatorIssue]
        + rng.normal(0, 1, n_days),
        0,
        None,
    )

    pd.DataFrame({"date": dates, "store": 1, "item": 1, "sales": sales}).to_csv(
        history_path, index=False
    )

    rows, targets = [], []

    for i in range(365, n_days):
        rows.append(create_features(dates[i], sales[:i].tolist()))
        targets.append(sales[i])

    model = XGBRegressor(
        n_estimators=20,
        max_depth=3,
        random_state=0,
    )
    model.fit(
        pd.DataFrame(rows, columns=FEATURES),
        np.array(targets),
    )

    joblib.dump(model, model_path)


# Keep tests independent of production artifacts and AWS.
os.environ.pop("S3_BUCKET", None)

_test_dir = Path(tempfile.mkdtemp(prefix="smartdemand-test-"))
_model = _test_dir / "model.joblib"
_history = _test_dir / "history.csv"

os.environ["MODEL_PATH"] = str(_model)
os.environ["HISTORY_FILE"] = str(_history)

_build_synthetic_artifacts(_model, _history)
