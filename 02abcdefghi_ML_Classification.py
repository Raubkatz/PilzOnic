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
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
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

DATA_DIR = Path("ml_datasets")
OUTDIR = Path("results_classification")

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2

EARLY_STOPPING_ROUNDS = 200

CAT_MISSING_TOKEN = "__MISSING__"

MAX_CAT_UNIQUE = 200
MAX_CAT_UNIQUE_FRACTION = 0.05

# Undersampling behavior
UNDERSAMPLE_FRACTION_MINORITY = 0.9  # take 0.9 * minority_count (or min class count for multiclass)
UNDERSAMPLE_SHUFFLE = True

CLASSIFICATION_TARGETS: List[str] = [
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

CLASSIFICATION_TARGETS_add: List[str] = [
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


def coerce_labels(y: pd.Series) -> pd.Series:
    print(f"[LABELS] dtype={y.dtype} | n={len(y)} | nunique={y.nunique(dropna=True)}")
    if y.dtype == object:
        y_str = y.astype(str).str.strip()
        y_num = pd.to_numeric(y_str, errors="coerce")
        frac_num = float(y_num.notna().mean()) if len(y_num) else 0.0
        print(f"[LABELS] object->numeric_coerce | frac_numeric={frac_num:.3f}")
        if y_num.notna().mean() > 0.95:
            y_int = y_num.round().astype("Int64")
            out = y_int.astype(str) if y_int.isna().any() else y_int.astype(int)
            print(
                f"[LABELS] coerced_to={'str' if y_int.isna().any() else 'int'} | nunique={pd.Series(out).nunique(dropna=True)}"
            )
            return out
        print(f"[LABELS] kept_as_str | nunique={y_str.nunique(dropna=True)}")
        return y_str

    y_num = pd.to_numeric(y, errors="coerce")
    if np.allclose(y_num, np.round(y_num), equal_nan=True):
        out = pd.Series(np.round(y_num).astype(int), index=y.index)
        print(f"[LABELS] numeric->int | nunique={out.nunique(dropna=True)}")
        return out
    out = y_num
    print(f"[LABELS] numeric_kept_float | nunique={pd.Series(out).nunique(dropna=True)}")
    return out


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


def _is_integer_like_numeric(s_num: pd.Series) -> bool:
    s2 = s_num.dropna()
    if len(s2) == 0:
        return False
    return np.allclose(s2.to_numpy(), np.round(s2.to_numpy()))


def preprocess_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
    """
    Prepare X for CatBoost (robust to mixed feature types + neighbor-expanded columns).

    FIXES vs previous version:
    - Any categorical column is forced to STRING (not object), so values like 9.0 become "9.0"
      (CatBoost rejects float categorical values).
    - Avoid deprecated is_categorical_dtype usage (handled implicitly).
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

    # extra prints so you know what's happening
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


def _make_balanced_subset_indices(y: pd.Series, frac_minority: float, random_state: int) -> np.ndarray:
    """
    Create a balanced subset:
    - Binary case: take floor(frac_minority * minority_count) from minority and same from majority.
    - Multiclass fallback: take floor(frac_minority * min_class_count) from EACH class.
    Returns indices (original index values from y).
    """
    y_s = pd.Series(y)
    vc = y_s.value_counts(dropna=False)
    classes = vc.index.tolist()

    if len(classes) < 2:
        # nothing to balance
        return y_s.index.to_numpy()

    rng = np.random.RandomState(random_state)

    if len(classes) == 2:
        cls_min = vc.idxmin()
        cls_maj = vc.idxmax()
        n_min = int(vc.loc[cls_min])
        n_take = int(np.floor(frac_minority * n_min))
        n_take = max(n_take, 1)

        idx_min = y_s[y_s == cls_min].index.to_numpy()
        idx_maj = y_s[y_s == cls_maj].index.to_numpy()

        take_min = rng.choice(idx_min, size=min(n_take, len(idx_min)), replace=False)
        take_maj = rng.choice(idx_maj, size=min(n_take, len(idx_maj)), replace=False)

        idx = np.concatenate([take_min, take_maj])
        if UNDERSAMPLE_SHUFFLE:
            rng.shuffle(idx)
        return idx

    # multiclass: balanced across all classes
    min_count = int(vc.min())
    n_take = int(np.floor(frac_minority * min_count))
    n_take = max(n_take, 1)

    parts = []
    for cls in classes:
        idx_cls = y_s[y_s == cls].index.to_numpy()
        parts.append(rng.choice(idx_cls, size=min(n_take, len(idx_cls)), replace=False))

    idx = np.concatenate(parts)
    if UNDERSAMPLE_SHUFFLE:
        rng.shuffle(idx)
    return idx


def _evaluate_split(
    model: CatBoostClassifier,
    X_split: pd.DataFrame,
    y_split: pd.Series,
    cat_feature_names: List[str],
    split_name: str,
    outdir: Path,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Evaluate metrics + confusion matrix (+ ROC AUC when binary).
    Writes confusion_matrix_<split>.csv and predictions_<split>.csv.
    Returns metrics dict with split_name.
    """
    print(f"[EVAL] {split_name}: start | n={len(X_split)}")
    pool = Pool(X_split, y_split, feature_names=feature_names, cat_features=cat_feature_names)

    y_pred = np.asarray(model.predict(pool)).ravel()
    metrics = compute_classification_metrics(np.asarray(y_split), y_pred)

    labels_sorted = sorted(pd.unique(y_split))
    cm = confusion_matrix(y_split, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in labels_sorted],
        columns=[f"pred_{l}" for l in labels_sorted],
    )
    cm_df.to_csv(outdir / f"confusion_matrix_{split_name}.csv", index=True)

    roc_auc = None
    if len(labels_sorted) == 2:
        try:
            proba = model.predict_proba(X_split)[:, 1]
            roc_auc = float(roc_auc_score((pd.Series(y_split) == labels_sorted[1]).astype(int), proba))
            metrics["roc_auc"] = roc_auc
        except Exception as e:
            print(f"[EVAL] {split_name}: roc_auc_failed | err={e}")

    pred_df = pd.DataFrame({"y_true": np.asarray(y_split), "y_pred": y_pred}, index=X_split.index)
    try:
        proba_all = model.predict_proba(X_split)
        for i, cls in enumerate(model.classes_):
            pred_df[f"proba_{cls}"] = proba_all[:, i]
    except Exception as e:
        print(f"[EVAL] {split_name}: predict_proba_failed | err={e}")
    pred_df.to_csv(outdir / f"predictions_{split_name}.csv", index=True)

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

    print(f"\n=== TRAIN TARGET: {target} ===")
    print(f"[INFO] outdir={outdir}")

    X_raw, y_raw = split_features_target(df, target)
    y = coerce_labels(y_raw)

    # class distribution print
    try:
        vc = pd.Series(y).value_counts(dropna=False)
        print(f"[INFO] class_counts:\n{vc.to_string()}")
    except Exception:
        pass

    X, cat_feature_names, diag = preprocess_features(X_raw)

    print(
        f"[INFO] feature_inventory | total={X.shape[1]} | cat={len(diag['categorical_cols'])} | bin={len(diag['binary_cols'])} | numeric_incl_bin={len(diag['numeric_cols'])}"
    )
    print(f"[INFO] cat_features_used_count={len(cat_feature_names)}")

    stratify = y if pd.Series(y).nunique() > 1 else None
    print(f"[SPLIT] stratify={'yes' if stratify is not None else 'no'} | test_size={test_size} | val_size={val_size}")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    stratify_train = y_train_full if pd.Series(y_train_full).nunique() > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state, stratify=stratify_train
    )

    print(f"[SPLIT] train_full={len(X_train_full)} | test={len(X_test)} | train={len(X_train)} | val={len(X_val)}")

    # build pools for training
    train_pool = Pool(X_train, y_train, feature_names=list(X.columns), cat_features=cat_feature_names)
    val_pool = Pool(X_val, y_val, feature_names=list(X.columns), cat_features=cat_feature_names)

    print(f"[CATBOOST] fit_start | early_stopping_rounds={params.get('early_stopping_rounds', None)}")
    model = CatBoostClassifier(**params)
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

    # 6 evaluations (train/val/test + undersampled train/val/test)
    eval_rows: List[Dict[str, float]] = []

    # full splits
    eval_rows.append(_evaluate_split(model, X_train, y_train, cat_feature_names, "train_full", outdir, list(X.columns)))
    eval_rows.append(_evaluate_split(model, X_val, y_val, cat_feature_names, "val_full", outdir, list(X.columns)))
    eval_rows.append(_evaluate_split(model, X_test, y_test, cat_feature_names, "test_full", outdir, list(X.columns)))

    # undersampled splits (balanced)
    print(f"[US] undersampling_fraction_minority={UNDERSAMPLE_FRACTION_MINORITY}")
    idx_train_us = _make_balanced_subset_indices(pd.Series(y_train, index=X_train.index), UNDERSAMPLE_FRACTION_MINORITY, random_state + 101)
    idx_val_us = _make_balanced_subset_indices(pd.Series(y_val, index=X_val.index), UNDERSAMPLE_FRACTION_MINORITY, random_state + 202)
    idx_test_us = _make_balanced_subset_indices(pd.Series(y_test, index=X_test.index), UNDERSAMPLE_FRACTION_MINORITY, random_state + 303)

    X_train_us, y_train_us = X_train.loc[idx_train_us], pd.Series(y_train, index=X_train.index).loc[idx_train_us]
    X_val_us, y_val_us = X_val.loc[idx_val_us], pd.Series(y_val, index=X_val.index).loc[idx_val_us]
    X_test_us, y_test_us = X_test.loc[idx_test_us], pd.Series(y_test, index=X_test.index).loc[idx_test_us]

    print(f"[US] train_us={len(X_train_us)} | val_us={len(X_val_us)} | test_us={len(X_test_us)}")
    try:
        print(f"[US] train_us_class_counts:\n{pd.Series(y_train_us).value_counts().to_string()}")
        print(f"[US] val_us_class_counts:\n{pd.Series(y_val_us).value_counts().to_string()}")
        print(f"[US] test_us_class_counts:\n{pd.Series(y_test_us).value_counts().to_string()}")
    except Exception:
        pass

    eval_rows.append(_evaluate_split(model, X_train_us, y_train_us, cat_feature_names, "train_under", outdir, list(X.columns)))
    eval_rows.append(_evaluate_split(model, X_val_us, y_val_us, cat_feature_names, "val_under", outdir, list(X.columns)))
    eval_rows.append(_evaluate_split(model, X_test_us, y_test_us, cat_feature_names, "test_under", outdir, list(X.columns)))

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(outdir / "metrics_by_split.csv", index=False)
    print(f"[EVAL] wrote metrics_by_split.csv")

    # Feature importance (aligned to train_pool)
    print(f"[FI] computing_feature_importance")
    importances = model.get_feature_importance(type="PredictionValuesChange", data=train_pool)
    fi_df = pd.DataFrame({"feature": list(X.columns), "importance": importances}).sort_values("importance", ascending=False)
    print(f"[FI] done | top10:\n{fi_df.head(10).to_string(index=False)}")

    # Save artifacts
    print(f"[SAVE] model/fi/report -> {outdir}")
    model_path = outdir / "model.cbm"
    model.save_model(model_path)

    fi_path = outdir / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)

    # Human-readable report
    report_path = outdir / "report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Target: {target}\n")
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

        f.write("CatBoostClassifier parameters\n")
        f.write(json.dumps(params, indent=2))
        f.write("\n\n")

        f.write("Metrics by split (6 evaluations)\n")
        f.write(eval_df.to_string(index=False))
        f.write("\n\n")

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
    }
    if EARLY_STOPPING_ROUNDS is not None:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS

    print(f"[MAIN] DATA_DIR={DATA_DIR.resolve()}")
    print(f"[MAIN] OUTDIR={OUTDIR.resolve()}")
    print(f"[MAIN] targets={len(CLASSIFICATION_TARGETS)} | {CLASSIFICATION_TARGETS}")
    print(f"[MAIN] split: test={TEST_SIZE} val={VAL_SIZE} | seed={RANDOM_STATE}")
    print(f"[MAIN] undersample: frac_minority={UNDERSAMPLE_FRACTION_MINORITY}")

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
                params=params,
            )
        except FileNotFoundError:
            print(f"[SKIP] Missing file for target: {target}")
        except Exception as e:
            print(f"[FAIL] {target}: {e}")


if __name__ == "__main__":
    main()