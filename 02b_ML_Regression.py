#!/usr/bin/env python3
"""
train_catboost_regression.py

Train and evaluate one CatBoostRegressor per regression target dataset.

Assumptions
-----------
- You have already created one CSV per target in a directory (default: ml_datasets/).
  Example: ml_datasets/cellulase_rep1.csv
- Each per-target CSV contains:
    - the selected feature columns
    - exactly ONE target column (the filename target)
- All features are numeric (0/1 binaries and numeric counts/scores).

What this script does (per target)
----------------------------------
1) Loads <data_dir>/<target>.csv
2) Splits into train/test
3) Trains CatBoostRegressor using a validation split (for evaluation only; no early stopping, no best-iteration selection)
4) Runs K-fold CV on the training set for additional validation
5) Evaluates on test set with regression metrics
6) Saves:
   - model:           <outdir>/<target>/model.cbm
   - report:          <outdir>/<target>/report.txt
   - predictions:     <outdir>/<target>/predictions.csv
   - feature import.: <outdir>/<target>/feature_importance.csv

Usage
-----
python train_catboost_regression.py

Dependencies
------------
pip install pandas numpy scikit-learn catboost
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =========================
# USER SETTINGS (EDIT HERE)
# =========================

DATA_DIR = Path("ml_datasets")          # Directory containing per-target CSVs (e.g., cellulase_rep1.csv)
OUTDIR = Path("results_regression")     # Base output directory (one subfolder per target)

RANDOM_STATE = 42
TEST_SIZE = 0.2                         # Fraction of rows used for test set
VAL_SIZE = 0.2                          # Fraction of training used for validation (evaluation only)
CV_FOLDS = 5                            # KFold on training-full for additional validation


REGRESSION_TARGETS: List[str] = [
    "cellulase_rep1",
    "cellulase_rep2",
    "cellulase_rep3",
    "cellulase_mean",
    "cellulase_sd",
    "xylanase_rep1",
    "xylanase_rep2",
    "xylanase_rep3",
    "xylanase_mean",
    "xylanase_sd",
]


def load_target_csv(data_dir: Path, target: str) -> pd.DataFrame:
    path = data_dir / f"{target}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataset columns: {list(df.columns)}")

    # Drop rows without target
    df = df.dropna(subset=[target]).copy()

    y = df[target]
    X = df.drop(columns=[target])

    return X, y


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE is optional; protect against division by zero
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

    return {
        "rmse": rmse,
        "mae": float(mae),
        "r2": float(r2),
        "mape_percent": mape,
    }


def train_one_target(
    df: pd.DataFrame,
    target: str,
    out_base: Path,
    random_state: int,
    test_size: float,
    val_size: float,
    cv_folds: int,
    params: Dict,
) -> None:
    outdir = out_base / target
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Prepare data
    X, y = split_features_target(df, target)

    # CatBoost can handle NaN; still, enforce numeric where possible
    X = X.apply(pd.to_numeric, errors="coerce")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Validation split (evaluation only; no early stopping, no best model selection)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state
    )

    train_pool = Pool(X_train, y_train, feature_names=list(X.columns))
    val_pool = Pool(X_val, y_val, feature_names=list(X.columns))
    test_pool = Pool(X_test, y_test, feature_names=list(X.columns))

    # 2) Train model (CatBoost "out of the box": no algorithm parameters)
    model = CatBoostRegressor(**params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=False,
        verbose=False,
    )

    # 3) Cross-validation on training-full (additional validation)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_metrics = []

    X_train_full_np = X_train_full.to_numpy()

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_train_full_np), start=1):
        X_tr = X_train_full.iloc[tr_idx]
        y_tr = y_train_full.iloc[tr_idx]
        X_va = X_train_full.iloc[va_idx]
        y_va = y_train_full.iloc[va_idx]

        m = CatBoostRegressor(**params)
        m.fit(
            Pool(X_tr, y_tr, feature_names=list(X.columns)),
            eval_set=Pool(X_va, y_va, feature_names=list(X.columns)),
            use_best_model=False,
            verbose=False,
        )

        pred_va = m.predict(X_va)
        fold_metrics = compute_regression_metrics(y_va.to_numpy(), np.asarray(pred_va))
        fold_metrics["fold"] = fold_idx
        cv_metrics.append(fold_metrics)

    cv_df = pd.DataFrame(cv_metrics)

    # 4) Test evaluation
    y_pred_test = model.predict(test_pool)
    test_metrics = compute_regression_metrics(y_test.to_numpy(), np.asarray(y_pred_test))

    # 5) Feature importance
    importances = model.get_feature_importance(type="PredictionValuesChange")
    fi_df = pd.DataFrame({"feature": list(X.columns), "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False)

    # 6) Save artifacts
    model_path = outdir / "model.cbm"
    model.save_model(model_path)

    pred_path = outdir / "predictions.csv"
    pred_df = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_pred": np.asarray(y_pred_test),
        },
        index=X_test.index,
    )
    pred_df.to_csv(pred_path, index=True)

    fi_path = outdir / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)

    cv_path = outdir / "cv_metrics.csv"
    cv_df.to_csv(cv_path, index=False)

    # 7) Human-readable report
    report_path = outdir / "report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Target: {target}\n")
        f.write(f"Rows total (after dropping missing target): {len(X)}\n")
        f.write(f"Train full: {len(X_train_full)} | Test: {len(X_test)}\n")
        f.write(f"Train: {len(X_train)} | Val: {len(X_val)}\n\n")

        f.write("CatBoostRegressor parameters\n")
        f.write(json.dumps(params, indent=2))
        f.write("\n\n")

        f.write("Test metrics\n")
        for k, v in test_metrics.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

        f.write(f"CV metrics (KFold={cv_folds}) on training-full\n")
        f.write(cv_df.drop(columns=["fold"]).mean(numeric_only=True).to_string())
        f.write("\n\n")

        f.write("Top 20 feature importances (PredictionValuesChange)\n")
        f.write(fi_df.head(20).to_string(index=False))
        f.write("\n")

    print(f"[OK] {target}: saved to {outdir}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # CatBoost "out of the box": pass NO algorithm parameters.
    # allow_writing_files is only I/O behavior (prevents CatBoost from dumping extra files).
    params: Dict = {
        "allow_writing_files": False,
    }

    for target in REGRESSION_TARGETS:
        try:
            df = load_target_csv(DATA_DIR, target)
            train_one_target(
                df=df,
                target=target,
                out_base=OUTDIR,
                random_state=RANDOM_STATE,
                test_size=TEST_SIZE,
                val_size=VAL_SIZE,
                cv_folds=CV_FOLDS,
                params=params,
            )
        except FileNotFoundError:
            print(f"[SKIP] Missing file for target: {target}")
        except Exception as e:
            print(f"[FAIL] {target}: {e}")


if __name__ == "__main__":
    main()