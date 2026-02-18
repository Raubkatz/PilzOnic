#!/usr/bin/env python3
"""
report_ml_datasets.py

Create a per-file TXT report for each ML dataset CSV in a folder (e.g., ml_datasets/).

What it does (per CSV)
----------------------
- Detects the target column (by filename stem if present; otherwise tries to infer)
- Reports:
  * rows/cols, feature count
  * feature names (full list)
  * missingness (overall + top columns)
  * duplicates
  * constant / all-missing columns
  * inferred feature types (numeric / binary / categorical)
  * for classification targets: class distribution + imbalance metrics
  * neighbor-feature inventory (counts by suffix order like _p/_pp/_f/_ff/... if present)

Output
------
Writes: <DATA_DIR>/<target>__DATASET_REPORT.txt
(i.e., same folder as the dataset CSVs)

Dependencies
------------
pip install pandas numpy
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =========================
# USER SETTINGS (EDIT HERE)
# =========================

DATA_DIR = Path("ml_datasets")
ENCODING = "utf-8"
SEP = ","

# If True: also write one folder-level summary CSV in DATA_DIR
WRITE_FOLDER_SUMMARY_CSV = True

# Heuristic: treat numeric columns with values subset of {0,1} as binary
BINARY_VALUE_SET = {0, 1}

# For huge datasets, avoid printing excessively long tables
TOP_MISSING_COLS_TO_SHOW = 30
TOP_LEVELS_TO_SHOW = 30


# =========================
# Helpers
# =========================

def _safe_filename(s: str) -> str:
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _guess_target_column(df: pd.DataFrame, stem: str) -> str:
    # Primary rule: file stem is the target column (your pipeline convention)
    if stem in df.columns:
        return stem

    # Fallback: if exactly one column looks like a target (not perfect, but helpful)
    # Prefer columns with low-ish cardinality (classification-like) or known target patterns.
    candidates = [c for c in df.columns if c.lower().endswith(("_q50", "_q33", "_q25", "_q20", "_mean", "_sd", "_rep1", "_rep2", "_rep3"))]
    if len(candidates) == 1:
        return candidates[0]

    # If none, last column fallback (common convention)
    return df.columns[-1]


def _infer_series_type(s: pd.Series) -> str:
    """
    Returns one of: "binary", "numeric", "categorical".
    """
    # Categorical/object -> categorical
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s) or pd.api.types.is_string_dtype(s):
        return "categorical"

    # Try numeric
    s_num = pd.to_numeric(s, errors="coerce")
    uniq = set(pd.unique(s_num.dropna()))
    if len(uniq) > 0 and uniq.issubset(BINARY_VALUE_SET):
        return "binary"
    return "numeric"


def _infer_feature_types(X: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, int]]:
    tmap: Dict[str, str] = {}
    counts = {"binary": 0, "numeric": 0, "categorical": 0}
    for c in X.columns:
        t = _infer_series_type(X[c])
        tmap[c] = t
        counts[t] += 1
    return tmap, counts


def _neighbor_suffix_order(col: str) -> int:
    """
    Returns neighbor order k if col ends with _p... or _f... (only p/f chars),
    e.g. unit_kind_ppp -> 3, n_snps_f -> 1.
    Returns 0 if not a neighbor-expanded column.
    """
    if "_" not in col:
        return 0
    base, suf = col.rsplit("_", 1)
    if suf and set(suf).issubset({"p", "f"}):
        return len(suf)
    return 0


def _summarize_classes(y: pd.Series) -> Tuple[str, pd.DataFrame]:
    """
    Returns:
      - short text summary line
      - table with counts, proportions
    """
    y_str = y.astype(str).fillna("__MISSING__")
    counts = y_str.value_counts(dropna=False)
    total = int(counts.sum())

    dfc = pd.DataFrame(
        {
            "class": counts.index.astype(str),
            "count": counts.values.astype(int),
            "proportion": (counts.values / max(total, 1)).astype(float),
        }
    )

    # imbalance stats
    if len(dfc) >= 2:
        maj = int(dfc["count"].max())
        mino = int(dfc["count"].min())
        ratio = (maj / max(mino, 1))
        summary = f"classes={len(dfc)} | majority={maj} | minority={mino} | imbalance_ratio={ratio:.4f}"
    else:
        summary = f"classes={len(dfc)} | single-class (cannot stratify)"

    return summary, dfc


def _write_report(path: Path, text: str) -> None:
    path.write_text(text, encoding=ENCODING)


# =========================
# Core
# =========================

def analyze_one_file(csv_path: Path) -> Dict[str, object]:
    df = pd.read_csv(csv_path, sep=SEP, encoding=ENCODING)
    stem = csv_path.stem
    target = _guess_target_column(df, stem)

    if target not in df.columns:
        # should not happen, but guard
        target = df.columns[-1]

    y = df[target]
    X = df.drop(columns=[target])

    n_rows, n_cols = df.shape
    n_features = X.shape[1]

    # Missingness
    missing_per_col = df.isna().sum()
    missing_total = int(missing_per_col.sum())
    missing_any_rows = int(df.isna().any(axis=1).sum())
    missing_rate = float(missing_total / max(n_rows * n_cols, 1))

    # Duplicates
    n_dup_rows = int(df.duplicated().sum())

    # Constant / all-missing columns
    all_missing_cols = [c for c in df.columns if df[c].isna().all()]
    constant_cols = []
    for c in df.columns:
        s = df[c]
        # nunique(dropna=True)=1 means constant ignoring NA;
        # treat also as constant if all values are NA (captured above)
        if not s.isna().all() and s.nunique(dropna=True) <= 1:
            constant_cols.append(c)

    # Feature types
    tmap, tcounts = _infer_feature_types(X)

    # Neighbor columns inventory
    neighbor_orders = [(_neighbor_suffix_order(c), c) for c in X.columns]
    neighbor_only = [c for k, c in neighbor_orders if k > 0]
    neighbor_max_order = max([k for k, _ in neighbor_orders], default=0)
    neighbor_order_counts: Dict[int, int] = {}
    for k, c in neighbor_orders:
        if k <= 0:
            continue
        neighbor_order_counts[k] = neighbor_order_counts.get(k, 0) + 1

    # Class summary (works for both classification/regression, but only meaningful for classification)
    class_summary_line, class_table = _summarize_classes(y)

    # Build report text
    lines: List[str] = []
    lines.append(f"Dataset file: {csv_path.name}")
    lines.append(f"Detected target column: {target}")
    lines.append("")
    lines.append("Shape")
    lines.append(f"- rows: {n_rows}")
    lines.append(f"- total columns (features + target): {n_cols}")
    lines.append(f"- features: {n_features}")
    lines.append("")
    lines.append("Target distribution")
    lines.append(f"- {class_summary_line}")
    lines.append(class_table.to_string(index=False))
    lines.append("")
    lines.append("Missingness")
    lines.append(f"- total missing cells: {missing_total}")
    lines.append(f"- rows with any missing: {missing_any_rows} / {n_rows}")
    lines.append(f"- missing rate (cells): {missing_rate:.6f}")
    lines.append("")
    top_missing = missing_per_col.sort_values(ascending=False)
    top_missing = top_missing[top_missing > 0].head(TOP_MISSING_COLS_TO_SHOW)
    if len(top_missing) == 0:
        lines.append("- no missing values detected")
    else:
        lines.append(f"- top missing columns (top {TOP_MISSING_COLS_TO_SHOW}):")
        for c, v in top_missing.items():
            lines.append(f"  * {c}: {int(v)}")
    lines.append("")
    lines.append("Duplicates / constants")
    lines.append(f"- duplicated rows: {n_dup_rows}")
    lines.append(f"- all-missing columns: {len(all_missing_cols)}")
    if all_missing_cols:
        lines.append("  " + ", ".join(all_missing_cols))
    lines.append(f"- constant columns (ignoring NA): {len(constant_cols)}")
    if constant_cols:
        lines.append("  " + ", ".join(constant_cols))
    lines.append("")
    lines.append("Feature types (inferred)")
    lines.append(f"- numeric: {tcounts['numeric']}")
    lines.append(f"- binary: {tcounts['binary']}")
    lines.append(f"- categorical: {tcounts['categorical']}")
    lines.append("")
    lines.append("Neighbor-expanded features")
    lines.append(f"- neighbor columns present: {len(neighbor_only)}")
    lines.append(f"- max neighbor order detected: {neighbor_max_order}")
    if neighbor_order_counts:
        lines.append("- columns per order:")
        for k in sorted(neighbor_order_counts):
            lines.append(f"  * order {k}: {neighbor_order_counts[k]}")
    lines.append("")
    lines.append("Feature list (in dataset order)")
    for c in X.columns:
        lines.append(f"- {c}   [{tmap[c]}]")
    lines.append("")

    report_text = "\n".join(lines)

    # Write report in same folder as CSV
    report_path = csv_path.parent / f"{stem}__DATASET_REPORT.txt"
    _write_report(report_path, report_text)

    # Return row for folder summary
    return {
        "file": csv_path.name,
        "target": target,
        "rows": n_rows,
        "features": n_features,
        "binary_features": tcounts["binary"],
        "categorical_features": tcounts["categorical"],
        "numeric_features": tcounts["numeric"],
        "missing_cells": missing_total,
        "rows_with_missing": missing_any_rows,
        "missing_rate_cells": missing_rate,
        "duplicated_rows": n_dup_rows,
        "all_missing_cols": len(all_missing_cols),
        "constant_cols": len(constant_cols),
        "neighbor_cols": len(neighbor_only),
        "neighbor_max_order": neighbor_max_order,
        "n_classes": len(class_table),
        "majority_class_count": int(class_table["count"].max()) if len(class_table) else 0,
        "minority_class_count": int(class_table["count"].min()) if len(class_table) else 0,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"[FAIL] No CSV files found in: {DATA_DIR.resolve()}")
        return

    print(f"[1/2] Found {len(csv_files)} dataset CSV files in {DATA_DIR.resolve()}")
    rows = []
    for i, p in enumerate(csv_files, start=1):
        print(f"  ({i}/{len(csv_files)}) Reporting: {p.name}")
        try:
            rows.append(analyze_one_file(p))
        except Exception as e:
            # Still write a minimal error report
            err_path = p.parent / f"{p.stem}__DATASET_REPORT.txt"
            _write_report(err_path, f"Dataset file: {p.name}\n\n[FAIL] Could not analyze file:\n{e}\n")
            print(f"      [FAIL] {p.name}: {e}")

    if WRITE_FOLDER_SUMMARY_CSV and rows:
        summ = pd.DataFrame(rows).sort_values(["target", "file"])
        out_csv = DATA_DIR / "__DATASET_FOLDER_SUMMARY.csv"
        summ.to_csv(out_csv, index=False, encoding=ENCODING)
        print(f"[2/2] Wrote folder summary: {out_csv.resolve()}")
    else:
        print("[2/2] Done.")


if __name__ == "__main__":
    main()