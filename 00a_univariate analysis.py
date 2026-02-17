#!/usr/bin/env python3
"""
analyze_feature_distributions_classification.py

Analyze and plot feature distributions per CLASS for classification targets
using the per-target ML datasets created in ml_datasets/<target>.csv.

Inputs
------
- One CSV per classification target in DATA_DIR, e.g.:
    ml_datasets/cellulase_q50.csv
  Each file contains:
    - selected features (possibly including neighbor-expanded features like *_p, *_pp, *_f, *_ff, ...)
    - exactly one target column (same name as the file stem)

Outputs (per target)
--------------------
OUTDIR/<target>/
  overview/
    numeric/        # per-feature plots across classes (box + violin) + stats
    categorical/    # per-feature stacked bars across classes + tables
  per_class/
    <class_label>/
      numeric/      # per-feature histograms within the class
      categorical/  # per-feature value counts within the class
  summaries/
    feature_types.csv
    numeric_summary_by_class.csv
    categorical_counts_by_class/ <feature>.csv

Feature type handling
---------------------
- Numeric features:
    * Boxplot + violin across classes (overview)
    * Histogram per class (per_class)
- Binary features (0/1):
    * Treated as categorical (stacked bar across classes, counts per class)
- Categorical features:
    * Stacked bar (proportions) across classes (overview)
    * Bar of value counts per class (per_class)

Color scheme / style
--------------------
Uses the same palette idea as in your provided scripts.

Dependencies
------------
pip install pandas numpy matplotlib seaborn
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# USER SETTINGS (EDIT HERE)
# =========================

DATA_DIR = Path("ml_datasets")
OUTDIR = Path("feature_analysis_classification")

CLASSIFICATION_TARGETS: List[str] = [
    "combined_q50",
    "cellulase_q50",
    "cellulase_q33",
    "cellulase_q25",
    "cellulase_q20",
    "xylanase_q50",
    "xylanase_q33",
    "xylanase_q25",
    "xylanase_q20",
]

ENCODING = "utf-8"
SEP = ","

# Plot configuration
DPI = 300
FONT_SCALE = 1.05

# Palette consistent with your evaluation scripts
PALETTE = ["#769FB6", "#D5D6AA", "#9DBBAE", "#E2DBBE", "#188FA7"]

# When a numeric column has only these values (ignoring NaN), treat it as binary categorical
BINARY_VALUE_SET = {0, 1}

# Optional: force some columns to be categorical even if they look numeric
FORCE_CATEGORICAL: List[str] = [
    "unit_kind",
    "mating_type",
]

# Optional: maximum number of category levels to plot (avoid unreadable plots)
MAX_CATEGORY_LEVELS = 40

# If True: drop rows with missing target
DROP_MISSING_TARGET = True


# =========================
# Helpers
# =========================

def _apply_style() -> None:
    sns.set_context("notebook", font_scale=FONT_SCALE)
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = DPI


def _safe_filename(s: str) -> str:
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _load_target_csv(target: str) -> pd.DataFrame:
    path = DATA_DIR / f"{target}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return pd.read_csv(path, sep=SEP, encoding=ENCODING)


def _split_X_y(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in columns: {list(df.columns)}")
    if DROP_MISSING_TARGET:
        df = df.dropna(subset=[target]).copy()
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def _coerce_target_labels(y: pd.Series) -> pd.Series:
    # keep strings as-is, numeric as int if close to int
    if y.dtype == object:
        return y.astype(str).str.strip()
    y_num = pd.to_numeric(y, errors="coerce")
    if np.allclose(y_num, np.round(y_num), equal_nan=True):
        return pd.Series(np.round(y_num).astype(int), index=y.index).astype(str)
    return y_num.astype(str)


def _infer_feature_types(X: pd.DataFrame) -> pd.DataFrame:
    """
    Return a table with columns:
      feature, dtype, inferred_type in {"numeric","categorical","binary"}
    """
    rows = []
    for c in X.columns:
        s = X[c]

        if c in FORCE_CATEGORICAL:
            rows.append((c, str(s.dtype), "categorical"))
            continue

        # object/category -> categorical
        if s.dtype == object or str(s.dtype).startswith("category"):
            rows.append((c, str(s.dtype), "categorical"))
            continue

        # try numeric coercion
        s_num = pd.to_numeric(s, errors="coerce")
        uniq = set(pd.unique(s_num.dropna()))
        if len(uniq) > 0 and uniq.issubset(BINARY_VALUE_SET):
            rows.append((c, str(s.dtype), "binary"))
        else:
            rows.append((c, str(s.dtype), "numeric"))

    return pd.DataFrame(rows, columns=["feature", "dtype", "inferred_type"])


def _prepare_features(X: pd.DataFrame, types_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    - Numeric: coerce to numeric
    - Binary: coerce to 0/1 numeric (nullable), treat as categorical in plotting
    - Categorical: convert to category
    """
    Xp = X.copy()
    tmap = dict(zip(types_df["feature"], types_df["inferred_type"]))

    for c, t in tmap.items():
        if t == "numeric":
            Xp[c] = pd.to_numeric(Xp[c], errors="coerce")
        elif t == "binary":
            Xp[c] = pd.to_numeric(Xp[c], errors="coerce")
            # keep only 0/1/NA
            Xp.loc[~(Xp[c].isna() | Xp[c].isin([0, 1])), c] = pd.NA
            Xp[c] = Xp[c].astype("Int64").astype("category")
        else:
            Xp[c] = Xp[c].astype("category")

    return Xp, tmap


def _numeric_summary_by_class(df: pd.DataFrame, feature: str, y_col: str) -> pd.DataFrame:
    g = df.groupby(y_col)[feature]
    out = g.agg(
        mean="mean",
        std="std",
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        n="count",
    )
    out["sem"] = out["std"] / np.sqrt(np.maximum(out["n"], 1))
    out.insert(0, "feature", feature)
    return out.reset_index().rename(columns={y_col: "class"})


def _plot_numeric_overview(df: pd.DataFrame, feature: str, y_col: str, outdir: Path) -> None:
    # box
    plt.figure(figsize=(8, 5))
    ax = sns.boxplot(data=df, x=y_col, y=feature, palette=PALETTE, showfliers=False)
    ax.set_xlabel("class")
    ax.set_ylabel(feature)
    ax.set_title(f"{feature} (box)")
    plt.tight_layout()
    plt.savefig(outdir / f"{_safe_filename(feature)}_box.png", dpi=DPI)
    plt.close()

    # violin
    plt.figure(figsize=(8, 5))
    ax = sns.violinplot(data=df, x=y_col, y=feature, palette=PALETTE, inner="quartile", linewidth=1)
    ax.set_xlabel("class")
    ax.set_ylabel(feature)
    ax.set_title(f"{feature} (violin)")
    plt.tight_layout()
    plt.savefig(outdir / f"{_safe_filename(feature)}_violin.png", dpi=DPI)
    plt.close()


def _plot_numeric_per_class(df: pd.DataFrame, feature: str, y_col: str, outdir_per_class: Path) -> None:
    for cls, sub in df[[y_col, feature]].dropna().groupby(y_col):
        cls_dir = outdir_per_class / _safe_filename(str(cls)) / "numeric"
        cls_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))
        ax = sns.histplot(sub[feature], kde=True)
        ax.set_xlabel(feature)
        ax.set_ylabel("count")
        ax.set_title(f"{feature} (class={cls})")
        plt.tight_layout()
        plt.savefig(cls_dir / f"{_safe_filename(feature)}_hist.png", dpi=DPI)
        plt.close()


def _plot_categorical_overview(df: pd.DataFrame, feature: str, y_col: str, outdir: Path) -> None:
    # Limit levels for readability
    vc = df[feature].value_counts(dropna=False)
    levels = vc.index.tolist()
    if len(levels) > MAX_CATEGORY_LEVELS:
        keep = set(levels[:MAX_CATEGORY_LEVELS - 1])
        df2 = df.copy()
        df2[feature] = df2[feature].apply(lambda v: v if v in keep else "OTHER").astype("category")
    else:
        df2 = df

    # REVERSED orientation:
    # x-axis = class (y_col)
    # stacked colors = feature categories
    ct = pd.crosstab(df2[y_col], df2[feature], normalize="index")
    ct = ct.fillna(0.0)

    ax = ct.plot(kind="bar", stacked=True, color=PALETTE, figsize=(10, 5))
    ax.set_xlabel("class")
    ax.set_ylabel("proportion within class")
    ax.set_title(f"{feature} (stacked proportions by class)")
    plt.tight_layout()
    plt.savefig(outdir / f"{_safe_filename(feature)}_stacked_prop.png", dpi=DPI)
    plt.close()

    # also save counts table (reversed orientation, rows=class, cols=feature levels)
    ct_counts = pd.crosstab(df2[y_col], df2[feature])
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    ct_counts.to_csv(outdir / "tables" / f"{_safe_filename(feature)}_counts.csv", encoding=ENCODING)


def _plot_categorical_per_class(df: pd.DataFrame, feature: str, y_col: str, outdir_per_class: Path) -> None:
    for cls, sub in df[[y_col, feature]].dropna().groupby(y_col):
        cls_dir = outdir_per_class / _safe_filename(str(cls)) / "categorical"
        cls_dir.mkdir(parents=True, exist_ok=True)

        counts = sub[feature].value_counts()
        # Limit levels for readability
        if len(counts) > MAX_CATEGORY_LEVELS:
            counts = counts.iloc[:MAX_CATEGORY_LEVELS]

        plt.figure(figsize=(10, 5))
        ax = counts.plot(kind="bar")
        ax.set_xlabel(feature)
        ax.set_ylabel("count")
        ax.set_title(f"{feature} (class={cls})")
        plt.tight_layout()
        plt.savefig(cls_dir / f"{_safe_filename(feature)}_counts.png", dpi=DPI)
        plt.close()


def analyze_one_target(target: str) -> None:
    print(f"\n=== Target: {target} ===")
    df = _load_target_csv(target)
    X, y_raw = _split_X_y(df, target)
    y = _coerce_target_labels(y_raw)

    # Build analysis dataframe
    types_df = _infer_feature_types(X)
    Xp, type_map = _prepare_features(X, types_df)
    dfA = Xp.copy()
    dfA["__class__"] = y

    # Output folders
    out_base = OUTDIR / target
    out_over_num = out_base / "overview" / "numeric"
    out_over_cat = out_base / "overview" / "categorical"
    out_per_class = out_base / "per_class"
    out_summ = out_base / "summaries"

    out_over_num.mkdir(parents=True, exist_ok=True)
    out_over_cat.mkdir(parents=True, exist_ok=True)
    out_per_class.mkdir(parents=True, exist_ok=True)
    out_summ.mkdir(parents=True, exist_ok=True)

    # Save inferred types
    types_df.sort_values(["inferred_type", "feature"]).to_csv(out_summ / "feature_types.csv", index=False, encoding=ENCODING)

    # Numeric processing
    numeric_feats = [f for f, t in type_map.items() if t == "numeric"]
    cat_feats = [f for f, t in type_map.items() if t in ("categorical", "binary")]

    print(f"  Rows: {len(dfA)} | Classes: {sorted(dfA['__class__'].unique().tolist())}")
    print(f"  Numeric features: {len(numeric_feats)}")
    print(f"  Categorical/binary features: {len(cat_feats)}")

    # Numeric overview + stats
    numeric_summaries = []
    for i, feat in enumerate(numeric_feats, start=1):
        print(f"  [numeric {i}/{len(numeric_feats)}] {feat}")
        _plot_numeric_overview(dfA, feat, "__class__", out_over_num)
        _plot_numeric_per_class(dfA, feat, "__class__", out_per_class)
        numeric_summaries.append(_numeric_summary_by_class(dfA, feat, "__class__"))

    if numeric_summaries:
        pd.concat(numeric_summaries, ignore_index=True).to_csv(
            out_summ / "numeric_summary_by_class.csv", index=False, encoding=ENCODING
        )

    # Categorical overview + per-class
    for i, feat in enumerate(cat_feats, start=1):
        print(f"  [categorical {i}/{len(cat_feats)}] {feat}")
        _plot_categorical_overview(dfA, feat, "__class__", out_over_cat)
        _plot_categorical_per_class(dfA, feat, "__class__", out_per_class)

    # Write a quick README-ish note
    with (out_base / "README.txt").open("w", encoding=ENCODING) as f:
        f.write(f"Target: {target}\n")
        f.write(f"Rows analyzed: {len(dfA)}\n")
        f.write(f"Classes: {sorted(dfA['__class__'].unique().tolist())}\n")
        f.write(f"Numeric features: {len(numeric_feats)}\n")
        f.write(f"Categorical/binary features: {len(cat_feats)}\n")
        f.write("\nFolder structure:\n")
        f.write("  overview/numeric: box+violin across classes\n")
        f.write("  overview/categorical: stacked bars across classes + tables/\n")
        f.write("  per_class/<class>/(numeric|categorical): within-class plots\n")
        f.write("  summaries/: feature typing + numeric stats\n")

    print(f"  [OK] Wrote: {out_base}")


def main() -> None:
    _apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    for t in CLASSIFICATION_TARGETS:
        try:
            analyze_one_target(t)
        except FileNotFoundError:
            print(f"[SKIP] Missing file: {DATA_DIR / (t + '.csv')}")
        except Exception as e:
            print(f"[FAIL] {t}: {e}")


if __name__ == "__main__":
    main()