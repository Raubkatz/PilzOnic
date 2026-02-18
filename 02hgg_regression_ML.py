#!/usr/bin/env python3
"""
train_catboost_classification.py
(unchanged header)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import matplotlib.pyplot as plt

# =========================
# USER SETTINGS (EDIT HERE)
# =========================

DATA_DIR = Path("ml_datasets")
OUTDIR = Path("results_regression")

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2

EARLY_STOPPING_ROUNDS = 200

CAT_MISSING_TOKEN = "__MISSING__"

MAX_CAT_UNIQUE = 200
MAX_CAT_UNIQUE_FRACTION = 0.05

# Subsample behavior (regression replacement for "undersampling")
# We create *subsampled* versions of train/val/test:
# - take 0.9 * n_rows of each split (random, no replacement)
SUBSAMPLE_FRACTION = 0.9
SUBSAMPLE_SHUFFLE = True

# Define regression targets (filenames + column names)
REGRESSION_TARGETS: List[str] = [
    # Example placeholders — set these to your actual regression targets
    # "combined_score",
    # "cellulase_activity",
    # "xylanase_activity",
]

REGRESSION_TARGETS: List[str] = [
    "cellulase_mean",
    "xylanase_mean",
    "cellulase_rep1",
    "cellulase_rep2",
    "cellulase_rep3",
    "cellulase_sd",
    "xylanase_rep1",
    "xylanase_rep2",
    "xylanase_rep3",
    "xylanase_sd",
]


def load_target_csv(data_dir: Path, target: str) -> pd.DataFrame:
    path = data_dir / f"{target}.csv"
    print(f"[LOAD] Target={target} | path={path}")
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    # Robust read: try fast C engine, fallback to python engine if tokenization/read fails
    try:
        df = pd.read_csv(path)
        print(f"[LOAD] OK (engine=c) | shape={df.shape[0]}x{df.shape[1]}")
        return df
    except Exception as e:
        print(f"[LOAD] FAIL (engine=c) -> fallback engine=python | err={e}")
        df = pd.read_csv(path, engine="python")
        print(f"[LOAD] OK (engine=python) | shape={df.shape[0]}x{df.shape[1]}")
        return df


def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    print(f"[SPLIT] Target={target} | input_shape={df.shape[0]}x{df.shape[1]}")
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataset columns: {list(df.columns)}")

    n0 = len(df)
    df = df.dropna(subset=[target]).copy()
    n1 = len(df)
    print(f"[SPLIT] Dropped missing target rows: {n0 - n1} | remaining={n1}")

    y = df[target]
    X = df.drop(columns=[target])

    print(f"[SPLIT] X_shape={X.shape[0]}x{X.shape[1]} | y_len={len(y)}")
    return X, y


def coerce_targets(y: pd.Series) -> pd.Series:
    print(f"[TARGET] dtype={y.dtype} | n={len(y)}")
    y_num = pd.to_numeric(y, errors="coerce")
    frac_num = float(y_num.notna().mean()) if len(y_num) else 0.0
    print(f"[TARGET] numeric_coerce | frac_numeric={frac_num:.3f}")

    # Drop-in safety: if lots of non-numeric, warn (do not drop silently)
    if frac_num < 0.99:
        n_bad = int(y_num.isna().sum())
        print(f"[TARGET] WARNING: non-numeric target values -> NaN after coercion: {n_bad}")

    return y_num.astype(float)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE with guard for zeros
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape_percent": mape,
    }


def _is_integer_like_numeric(s_num: pd.Series) -> bool:
    s2 = s_num.dropna()
    if len(s2) == 0:
        return False
    return np.allclose(s2.to_numpy(), np.round(s2.to_numpy()))


def preprocess_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
    """
    Prepare X for CatBoost (robust to mixed feature types + neighbor-expanded columns).

    - Any categorical column is forced to STRING (CatBoost rejects float categorical values).
    - Numeric columns are float.
    - Binary columns are treated as numeric floats 0/1.
    """
    print(f"[PREP] start | X_shape={X.shape[0]}x{X.shape[1]}")
    X_out = X.copy()

    # Replace pandas.NA with np.nan early to avoid NAType issues
    X_out = X_out.replace({pd.NA: np.nan})

    n_rows = len(X_out)
    cat_cols: List[str] = []
    bin_cols: List[str] = []

    # 1) dtype-based categoricals
    for c in X_out.columns:
        s = X_out[c]
        dtype = s.dtype
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(dtype, pd.CategoricalDtype):
            cat_cols.append(c)

    cat_set = set(cat_cols)
    print(f"[PREP] dtype_categoricals={len(cat_cols)}")

    # 2) detect binary + encoded categoricals among remaining columns
    n_bin_by_bool = 0
    n_bin_by_values = 0
    n_cat_by_encoded = 0

    for c in X_out.columns:
        if c in cat_set:
            continue

        s = X_out[c]

        if pd.api.types.is_bool_dtype(s):
            bin_cols.append(c)
            n_bin_by_bool += 1
            continue

        s_num = pd.to_numeric(s, errors="coerce")
        uniq = pd.unique(s_num.dropna())
        uniq_set = set(uniq.tolist()) if len(uniq) else set()

        if len(uniq_set) > 0 and uniq_set.issubset({0, 1}):
            bin_cols.append(c)
            n_bin_by_values += 1
            continue

        if _is_integer_like_numeric(s_num):
            nunique = pd.Series(uniq).nunique()
            if nunique > 1 and (nunique <= MAX_CAT_UNIQUE) and (nunique / max(n_rows, 1) <= MAX_CAT_UNIQUE_FRACTION):
                cat_cols.append(c)
                cat_set.add(c)
                n_cat_by_encoded += 1

    bin_set = set(bin_cols)
    print(f"[PREP] detected_binaries={len(bin_cols)} | by_bool={n_bin_by_bool} | by_values={n_bin_by_values}")
    print(f"[PREP] detected_encoded_categoricals={n_cat_by_encoded} | total_categoricals={len(cat_cols)}")

    # 3) Categorical: FORCE TO STRING + sentinel for missing
    for c in cat_cols:
        s = X_out[c]
        s = s.where(~pd.isna(s), other=CAT_MISSING_TOKEN)
        X_out[c] = s.astype(str)

    # 4) Binary: numeric float 0/1 with np.nan for missing
    for c in bin_cols:
        s = X_out[c]
        if pd.api.types.is_bool_dtype(s):
            s_num = s.astype(float)
        else:
            s2 = s
            dtype2 = s2.dtype
            if pd.api.types.is_object_dtype(s2) or pd.api.types.is_string_dtype(s2) or isinstance(dtype2, pd.CategoricalDtype):
                s2 = (
                    s2.astype(str)
                    .str.strip()
                    .str.lower()
                    .replace({"true": "1", "false": "0", "nan": np.nan, "none": np.nan, "": np.nan})
                )
            s_num = pd.to_numeric(s2, errors="coerce").astype(float)

        s_num = s_num.where(np.isnan(s_num) | np.isin(s_num, [0.0, 1.0]), other=np.nan)
        X_out[c] = s_num

    # 5) Everything else: numeric float
    n_numeric_converted = 0
    for c in X_out.columns:
        if c in cat_set:
            continue
        if c in bin_set:
            continue
        if pd.api.types.is_bool_dtype(X_out[c]):
            X_out[c] = X_out[c].astype(float)
            n_numeric_converted += 1
        else:
            X_out[c] = pd.to_numeric(X_out[c], errors="coerce").astype(float)
            n_numeric_converted += 1
    print(f"[PREP] numeric_converted={n_numeric_converted}")

    # Diagnostics
    all_missing_cols = [c for c in X_out.columns if pd.isna(X_out[c]).all()]
    constant_cols = []
    for c in X_out.columns:
        if c in cat_set:
            if pd.Series(X_out[c]).nunique(dropna=False) <= 1:
                constant_cols.append(c)
        else:
            if pd.Series(X_out[c]).dropna().nunique() <= 1:
                constant_cols.append(c)

    diagnostics = {
        "binary_cols": bin_cols,
        "categorical_cols": cat_cols,
        "numeric_cols": [c for c in X_out.columns if c not in cat_set],
        "all_missing_cols": all_missing_cols,
        "constant_cols": constant_cols,
    }

    miss_total = int(pd.isna(X_out).sum().sum())
    print(f"[PREP] done | cat={len(cat_cols)} | bin={len(bin_cols)} | numeric_incl_bin={len(diagnostics['numeric_cols'])}")
    print(f"[PREP] missing_cells_total={miss_total} | all_missing_cols={len(all_missing_cols)} | constant_cols={len(constant_cols)}")

    if all_missing_cols:
        print(f"[PREP] all_missing_cols (first 20)={all_missing_cols[:20]}")
    if constant_cols:
        print(f"[PREP] constant_cols (first 20)={constant_cols[:20]}")
    if cat_cols:
        print(f"[PREP] categorical_cols (first 25)={cat_cols[:25]}")
    if bin_cols:
        print(f"[PREP] binary_cols (first 25)={bin_cols[:25]}")

    return X_out, cat_cols, diagnostics


def _make_subsample_indices(n: int, frac: float, random_state: int) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    k = int(np.floor(frac * n))
    k = max(k, 1)
    idx = np.arange(n)
    take = rng.choice(idx, size=min(k, n), replace=False)
    if SUBSAMPLE_SHUFFLE:
        rng.shuffle(take)
    return take


def _save_regression_plots(y_true: np.ndarray, y_pred: np.ndarray, outdir: Path, split_name: str) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred

    # 1) Regression grade plot: y_true vs y_pred
    fig = plt.figure()
    plt.scatter(y_true, y_pred, s=6)
    lo = np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)])
    hi = np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)])
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title(f"Regression grade plot: {split_name}")
    fig.savefig(outdir / f"plot_regression_grade_{split_name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2) Residuals histogram
    fig = plt.figure()
    plt.hist(resid[~np.isnan(resid)], bins=60)
    plt.xlabel("residual (y_true - y_pred)")
    plt.ylabel("count")
    plt.title(f"Residual histogram: {split_name}")
    fig.savefig(outdir / f"plot_residual_hist_{split_name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3) Residuals vs prediction
    fig = plt.figure()
    plt.scatter(y_pred, resid, s=6)
    plt.axhline(0.0)
    plt.xlabel("y_pred")
    plt.ylabel("residual (y_true - y_pred)")
    plt.title(f"Residuals vs y_pred: {split_name}")
    fig.savefig(outdir / f"plot_residual_vs_pred_{split_name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _evaluate_split(
    model: CatBoostRegressor,
    X_split: pd.DataFrame,
    y_split: pd.Series,
    cat_feature_names: List[str],
    split_name: str,
    outdir: Path,
    feature_names: List[str],
) -> Dict[str, float]:
    print(f"[EVAL] {split_name}: start | n={len(X_split)}")
    pool = Pool(X_split, y_split, feature_names=feature_names, cat_features=cat_feature_names)

    y_pred = np.asarray(model.predict(pool)).ravel().astype(float)
    y_true = np.asarray(y_split, dtype=float)

    metrics = compute_regression_metrics(y_true, y_pred)

    pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=X_split.index)
    pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]
    pred_df.to_csv(outdir / f"predictions_{split_name}.csv", index=True)

    _save_regression_plots(y_true, y_pred, outdir, split_name)

    print(f"[EVAL] {split_name}: done | metrics={metrics}")
    return {"split": split_name, **metrics}


def train_one_target(
        df: pd.DataFrame,
        target: str,
        out_base: Path,
        random_state: int,
        test_size: float,
        val_size: float,
        params: Dict,
) -> None:
    outdir = out_base / target
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== TRAIN TARGET (REGRESSION): {target} ===")
    print(f"[INFO] outdir={outdir}")

    X_raw, y_raw = split_features_target(df, target)
    y = coerce_targets(y_raw)

    # target summary
    try:
        y_s = pd.Series(y)
        print(f"[INFO] target_summary: n={len(y_s)} | missing={int(y_s.isna().sum())} | mean={float(np.nanmean(y_s)):.6g} | std={float(np.nanstd(y_s)):.6g}")
        qs = np.nanpercentile(y_s.to_numpy(dtype=float), [0, 1, 5, 25, 50, 75, 95, 99, 100])
        print(f"[INFO] target_quantiles: 0/1/5/25/50/75/95/99/100 = {qs}")
    except Exception:
        pass

    X, cat_feature_names, diag = preprocess_features(X_raw)

    print(
        f"[INFO] feature_inventory | total={X.shape[1]} | cat={len(diag['categorical_cols'])} | bin={len(diag['binary_cols'])} | numeric_incl_bin={len(diag['numeric_cols'])}"
    )
    print(f"[INFO] cat_features_used_count={len(cat_feature_names)}")

    # Remove rows where target is NaN (regression cannot train on NaN targets)
    mask = ~pd.isna(y)
    dropped = int((~mask).sum())
    if dropped:
        print(f"[TARGET] Dropping rows with NaN target after coercion: {dropped}")
    X = X.loc[mask]
    y = pd.Series(y).loc[mask]

    print(f"[SPLIT] test_size={test_size} | val_size={val_size}")

    # Regression: no stratify
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state
    )

    print(f"[SPLIT] train_full={len(X_train_full)} | test={len(X_test)} | train={len(X_train)} | val={len(X_val)}")

    train_pool = Pool(X_train, y_train, feature_names=list(X.columns), cat_features=cat_feature_names)
    val_pool = Pool(X_val, y_val, feature_names=list(X.columns), cat_features=cat_feature_names)

    print(f"[CATBOOST] fit_start | early_stopping_rounds={params.get('early_stopping_rounds', None)}")
    model = CatBoostRegressor(**params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        verbose=False,
    )
    try:
        print(f"[CATBOOST] fit_done | best_iteration={model.get_best_iteration()} | best_score={model.get_best_score()}")
    except Exception:
        print(f"[CATBOOST] fit_done")

    # 6 evaluations (train/val/test + subsampled train/val/test)
    eval_rows: List[Dict[str, float]] = []

    eval_rows.append(_evaluate_split(model, X_train, y_train, cat_feature_names, "train_full", outdir, list(X.columns)))
    eval_rows.append(_evaluate_split(model, X_val, y_val, cat_feature_names, "val_full", outdir, list(X.columns)))
    eval_rows.append(_evaluate_split(model, X_test, y_test, cat_feature_names, "test_full", outdir, list(X.columns)))

    print(f"[SS] subsample_fraction={SUBSAMPLE_FRACTION}")
    tr_take = _make_subsample_indices(len(X_train), SUBSAMPLE_FRACTION, random_state + 101)
    va_take = _make_subsample_indices(len(X_val), SUBSAMPLE_FRACTION, random_state + 202)
    te_take = _make_subsample_indices(len(X_test), SUBSAMPLE_FRACTION, random_state + 303)

    X_train_ss, y_train_ss = X_train.iloc[tr_take], y_train.iloc[tr_take]
    X_val_ss, y_val_ss = X_val.iloc[va_take], y_val.iloc[va_take]
    X_test_ss, y_test_ss = X_test.iloc[te_take], y_test.iloc[te_take]

    print(f"[SS] train_ss={len(X_train_ss)} | val_ss={len(X_val_ss)} | test_ss={len(X_test_ss)}")

    eval_rows.append(_evaluate_split(model, X_train_ss, y_train_ss, cat_feature_names, "train_subsample", outdir, list(X.columns)))
    eval_rows.append(_evaluate_split(model, X_val_ss, y_val_ss, cat_feature_names, "val_subsample", outdir, list(X.columns)))
    eval_rows.append(_evaluate_split(model, X_test_ss, y_test_ss, cat_feature_names, "test_subsample", outdir, list(X.columns)))

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(outdir / "metrics_by_split.csv", index=False)
    print(f"[EVAL] wrote metrics_by_split.csv")

    # Feature importance
    print(f"[FI] computing_feature_importance")
    importances = model.get_feature_importance(type="PredictionValuesChange", data=train_pool)
    fi_df = pd.DataFrame({"feature": list(X.columns), "importance": importances}).sort_values("importance", ascending=False)
    print(f"[FI] done | top10:\n{fi_df.head(10).to_string(index=False)}")

    print(f"[SAVE] model/fi/report -> {outdir}")
    model_path = outdir / "model.cbm"
    model.save_model(model_path)

    fi_path = outdir / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)

    # Human-readable report
    report_path = outdir / "report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Target (regression): {target}\n")
        f.write(f"Rows total (after dropping missing target): {len(X)}\n")
        f.write(f"Train full: {len(X_train_full)} | Test: {len(X_test)}\n")
        f.write(f"Train: {len(X_train)} | Val: {len(X_val)}\n\n")

        f.write("Feature inventory (from dataset)\n")
        f.write(f"- total_features: {X.shape[1]}\n")
        f.write(f"- categorical_features: {len(diag['categorical_cols'])}\n")
        f.write(f"- binary_features: {len(diag['binary_cols'])}\n")
        f.write(f"- numeric_features (incl. binary): {len(diag['numeric_cols'])}\n")
        f.write(f"- all_missing_features: {len(diag['all_missing_cols'])}\n")
        f.write(f"- constant_features: {len(diag['constant_cols'])}\n\n")

        f.write("Categorical features used\n")
        f.write(str(cat_feature_names))
        f.write("\n\n")

        f.write("CatBoostRegressor parameters\n")
        f.write(json.dumps(params, indent=2))
        f.write("\n\n")

        f.write("Metrics by split (6 evaluations)\n")
        f.write(eval_df.to_string(index=False))
        f.write("\n\n")

        f.write("Plot files written per split\n")
        f.write("- plot_regression_grade_<split>.png\n")
        f.write("- plot_residual_hist_<split>.png\n")
        f.write("- plot_residual_vs_pred_<split>.png\n\n")

        f.write("Top 50 feature importances (PredictionValuesChange)\n")
        f.write(fi_df.head(50).to_string(index=False))
        f.write("\n")

    print(f"[OK] {target}: saved to {outdir}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    params: Dict = {
        "random_seed": RANDOM_STATE,
        "allow_writing_files": False,
        # Sensible default for regression; you can remove if you want pure defaults
        "loss_function": "RMSE",
    }
    if EARLY_STOPPING_ROUNDS is not None:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS

    print(f"[MAIN] DATA_DIR={DATA_DIR.resolve()}")
    print(f"[MAIN] OUTDIR={OUTDIR.resolve()}")
    print(f"[MAIN] targets={len(REGRESSION_TARGETS)} | {REGRESSION_TARGETS}")
    print(f"[MAIN] split: test={TEST_SIZE} val={VAL_SIZE} | seed={RANDOM_STATE}")
    print(f"[MAIN] subsample: frac={SUBSAMPLE_FRACTION}")

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
                params=params,
            )
        except FileNotFoundError:
            print(f"[SKIP] Missing file for target: {target}")
        except Exception as e:
            print(f"[FAIL] {target}: {e}")


if __name__ == "__main__":
    main()