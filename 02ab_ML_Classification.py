#!/usr/bin/env python3
"""
train_catboost_classification.py

Train and evaluate one CatBoostClassifier per classification target dataset.

                _________
           _.-""         ""-._
        .-"   _..---.._      "-.
      .'    .'  .-.-.  '.       '.
     /     /   ( o o )   \        \
    ;     ;     \_-_/      ;       ;
    |     |   .-.___.-.    |       |
    ;     ;  /  .---.  \   ;       ;
     \     \  \ '---' /   /       /
      '.     '._\___/_.-'      .'
        "-._      _      _.-"
             ""--( )--""
                 / \
                /___\

Assumptions
-----------
- You have one CSV per target in a directory (default: ml_datasets/).
  Example: ml_datasets/cellulase_q50.csv
- Each per-target CSV contains:
    - the selected feature columns (possibly including neighbor-expanded columns)
    - exactly ONE target column (the filename target)
- Classification targets are expected to be class labels (commonly 0/1).
  If they are numeric floats, this script will attempt to coerce.

What this script does (per target)
----------------------------------
1) Loads <data_dir>/<target>.csv
2) Splits into train/test with stratification when possible
3) Trains CatBoostClassifier with early stopping using a validation split
4) Runs Stratified K-fold CV on the training set
5) Evaluates on test set:
   - confusion matrix
   - accuracy, balanced accuracy
   - precision/recall/F1 (macro + weighted)
   - ROC AUC (binary only, if probabilities are available)
6) Saves:
   - model:            <outdir>/<target>/model.cbm
   - report:           <outdir>/<target>/report.txt
   - predictions:      <outdir>/<target>/predictions.csv
   - confusion matrix: <outdir>/<target>/confusion_matrix.csv
   - feature import.:  <outdir>/<target>/feature_importance.csv

Usage
-----
python train_catboost_classification.py

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
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =========================
# USER SETTINGS (EDIT HERE)
# =========================

DATA_DIR = Path("ml_datasets")              # Directory containing per-target CSVs (e.g., cellulase_q50.csv)
OUTDIR = Path("results_classification")     # Base output directory (one subfolder per target)

RANDOM_STATE = 42
TEST_SIZE = 0.2                             # Fraction of rows used for test set
VAL_SIZE = 0.2                              # Fraction of training used for validation (early stopping)
CV_FOLDS = 5                                # StratifiedKFold on training-full

# CatBoost "out of the box" (keep minimal; early stopping retained as before)
EARLY_STOPPING_ROUNDS = 200

# Categorical feature base names (neighbor-expanded columns will be detected automatically)
# Example neighbor columns: unit_kind_p, unit_kind_pp, generation_f, mating_type_ff, ...
CATEGORICAL_BASE_COLUMNS = ["unit_kind", "generation", "mating_type"]

# Optional: known binary feature base names (neighbor-expanded variants will be handled as numeric)
BINARY_BASE_COLUMNS = ["has_frameshift", "has_stop_gained", "has_start_lost", "has_splice_disrupt"]

# Categorical missing value placeholder (string)
CAT_MISSING_TOKEN = "__MISSING__"


CLASSIFICATION_TARGETS_old: List[str] = [
    "combined_q50",
    "cellulase_q50",
    "xylanase_q50",
    "cellulase_q33",
    "cellulase_q25",
    "cellulase_q20",
    "xylanase_q33",
    "xylanase_q25",
    "xylanase_q20",
]

CLASSIFICATION_TARGETS: List[str] = [
    "cellulase_q33",
    "cellulase_q25",
    "cellulase_q20",
    "xylanase_q33",
    "xylanase_q25",
    "xylanase_q20",
    "combined_q50",
    "cellulase_q50",
    "xylanase_q50",
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


def coerce_labels(y: pd.Series) -> pd.Series:
    """
    Coerce classification labels into a sensible form.

    - If labels are strings, keep them.
    - If labels look numeric, convert to int when possible.
    """
    if y.dtype == object:
        y_str = y.astype(str).str.strip()
        y_num = pd.to_numeric(y_str, errors="coerce")
        if y_num.notna().mean() > 0.95:
            y_int = y_num.round().astype("Int64")
            return y_int.astype(str) if y_int.isna().any() else y_int.astype(int)
        return y_str

    y_num = pd.to_numeric(y, errors="coerce")
    if np.allclose(y_num, np.round(y_num), equal_nan=True):
        return pd.Series(np.round(y_num).astype(int), index=y.index)
    return y_num


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _is_neighbor_variant(col: str, base: str) -> bool:
    # Matches:
    #   base
    #   base_p, base_pp, base_ppp...
    #   base_f, base_ff, base_fff...
    if col == base:
        return True
    if col.startswith(base + "_"):
        suf = col[len(base) + 1 :]
        if suf and set(suf).issubset({"p", "f"}):
            return True
    return False


def detect_categorical_columns(X: pd.DataFrame) -> List[str]:
    """
    Detect categorical columns, including neighbor-expanded variants.

    Rules:
    - Any column with dtype object or category is categorical.
    - Any column whose name matches a configured categorical base name, including neighbor variants, is categorical.
    """
    cat_cols = set()

    # dtype-based
    for c in X.columns:
        if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_categorical_dtype(X[c]):
            cat_cols.add(c)

    # name-based (neighbor variants)
    for base in CATEGORICAL_BASE_COLUMNS:
        for c in X.columns:
            if _is_neighbor_variant(c, base):
                cat_cols.add(c)

    return [c for c in X.columns if c in cat_cols]  # preserve column order


def preprocess_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare X for CatBoost:
    - Keep categorical columns as strings, fill missing with CAT_MISSING_TOKEN.
    - Convert boolean columns to 0/1.
    - Convert all remaining non-categorical columns to numeric (coerce errors to NaN).

    Returns
    -------
    X_out, cat_feature_names
    """
    X_out = X.copy()

    cat_cols = detect_categorical_columns(X_out)

    # Categorical: force to string, fill missing with sentinel
    for c in cat_cols:
        X_out[c] = X_out[c].astype("string")
        X_out[c] = X_out[c].fillna(CAT_MISSING_TOKEN)

    # Boolean -> int
    for c in X_out.columns:
        if pd.api.types.is_bool_dtype(X_out[c]):
            X_out[c] = X_out[c].astype(int)

    # Non-categorical: numeric coercion
    non_cat_cols = [c for c in X_out.columns if c not in cat_cols]
    for c in non_cat_cols:
        X_out[c] = pd.to_numeric(X_out[c], errors="coerce")

    return X_out, cat_cols


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
    X_raw, y_raw = split_features_target(df, target)
    y = coerce_labels(y_raw)

    # Preprocess features (categorical/numeric/binary + neighbor-expanded columns)
    X, cat_feature_names = preprocess_features(X_raw)

    # Stratify only if there is more than one class
    stratify = y if y.nunique() > 1 else None

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Validation split for early stopping (within training)
    stratify_train = y_train_full if y_train_full.nunique() > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state, stratify=stratify_train
    )

    # CatBoost Pool with categorical features specified (names are allowed with DataFrame feature names). :contentReference[oaicite:2]{index=2}
    train_pool = Pool(X_train, y_train, feature_names=list(X.columns), cat_features=cat_feature_names)
    val_pool = Pool(X_val, y_val, feature_names=list(X.columns), cat_features=cat_feature_names)
    test_pool = Pool(X_test, y_test, feature_names=list(X.columns), cat_features=cat_feature_names)

    # 2) Train model
    model = CatBoostClassifier(**params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        verbose=False,
    )

    # 3) Cross-validation on training-full
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_rows = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train_full, y_train_full), start=1):
        X_tr = X_train_full.iloc[tr_idx]
        y_tr = y_train_full.iloc[tr_idx]
        X_va = X_train_full.iloc[va_idx]
        y_va = y_train_full.iloc[va_idx]

        m = CatBoostClassifier(**params)
        m.fit(
            Pool(X_tr, y_tr, feature_names=list(X.columns), cat_features=cat_feature_names),
            eval_set=Pool(X_va, y_va, feature_names=list(X.columns), cat_features=cat_feature_names),
            use_best_model=True,
            verbose=False,
        )

        pred_va = m.predict(X_va)
        fold_metrics = compute_classification_metrics(y_va.to_numpy(), np.asarray(pred_va).ravel())
        fold_metrics["fold"] = fold_idx
        cv_rows.append(fold_metrics)

    cv_df = pd.DataFrame(cv_rows)

    # 4) Test evaluation
    y_pred_test = np.asarray(model.predict(test_pool)).ravel()
    test_metrics = compute_classification_metrics(y_test.to_numpy(), y_pred_test)

    # Confusion matrix
    labels_sorted = sorted(pd.unique(y_test))
    cm = confusion_matrix(y_test, y_pred_test, labels=labels_sorted)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in labels_sorted],
        columns=[f"pred_{l}" for l in labels_sorted],
    )

    # ROC AUC (binary only)
    roc_auc = None
    if len(labels_sorted) == 2:
        try:
            # Use probabilities for AUC. :contentReference[oaicite:3]{index=3}
            proba = model.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score((y_test == labels_sorted[1]).astype(int), proba))
        except Exception:
            roc_auc = None

    # 5) Feature importance
    importances = model.get_feature_importance(type="PredictionValuesChange")
    fi_df = pd.DataFrame({"feature": list(X.columns), "importance": importances}).sort_values("importance", ascending=False)

    # 6) Save artifacts
    model_path = outdir / "model.cbm"
    model.save_model(model_path)

    pred_path = outdir / "predictions.csv"
    pred_df = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_pred": y_pred_test,
        },
        index=X_test.index,
    )
    try:
        proba_all = model.predict_proba(X_test)
        for i, cls in enumerate(model.classes_):
            pred_df[f"proba_{cls}"] = proba_all[:, i]
    except Exception:
        pass
    pred_df.to_csv(pred_path, index=True)

    cm_path = outdir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path, index=True)

    fi_path = outdir / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)

    cv_path = outdir / "cv_metrics.csv"
    cv_df.to_csv(cv_path, index=False)

    # 7) Human-readable report
    report_path = outdir / "report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Target: {target}\n")
        f.write(f"Rows total (after dropping missing target): {len(X)}\n")
        f.write(f"Classes (test): {labels_sorted}\n")
        f.write(f"Train full: {len(X_train_full)} | Test: {len(X_test)}\n")
        f.write(f"Train: {len(X_train)} | Val: {len(X_val)}\n\n")

        f.write("Categorical features used\n")
        f.write(str(cat_feature_names))
        f.write("\n\n")

        f.write("CatBoostClassifier parameters\n")
        f.write(json.dumps(params, indent=2))
        f.write("\n\n")

        f.write("Test metrics\n")
        for k, v in test_metrics.items():
            f.write(f"- {k}: {v}\n")
        if roc_auc is not None:
            f.write(f"- roc_auc (binary): {roc_auc}\n")
        f.write("\n")

        f.write("Classification report (test)\n")
        f.write(classification_report(y_test, y_pred_test, zero_division=0))
        f.write("\n\n")

        f.write("Confusion matrix (test)\n")
        f.write(cm_df.to_string())
        f.write("\n\n")

        f.write(f"CV metrics (StratifiedKFold={cv_folds}) on training-full\n")
        f.write(cv_df.drop(columns=["fold"]).mean(numeric_only=True).to_string())
        f.write("\n\n")

        f.write("Top 20 feature importances (PredictionValuesChange)\n")
        f.write(fi_df.head(20).to_string(index=False))
        f.write("\n")

    print(f"[OK] {target}: saved to {outdir}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Minimal params; cat_features are supplied via Pool (recommended). :contentReference[oaicite:4]{index=4}
    params: Dict = {
        "random_seed": RANDOM_STATE,
        "allow_writing_files": False,
    }
    if EARLY_STOPPING_ROUNDS is not None:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS

    for target in CLASSIFICATION_TARGETS:
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