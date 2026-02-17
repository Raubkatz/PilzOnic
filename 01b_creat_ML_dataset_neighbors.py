#!/usr/bin/env python3
"""
Create reduced ML-ready CSV datasets from a wide "features.csv".

What this script does
---------------------
Given an input CSV (default: features.csv) containing many columns, this script:
1) Selects a fixed set of FEATURE columns (defined below).
2) Expands the feature set by adding features from neighboring units (previous/next) within the same sample_id,
   based on row order (neighbors are the rows directly above/below within the same sample_id).
3) For EACH target column (also defined below), creates a separate output CSV that contains:
      [selected + neighbor-expanded features] + [that single target]
4) Writes one CSV per target into an output directory.

Key behaviors
-------------
- Column validation:
    * If a required feature/target column is missing, the script can either:
        - fail (default), OR
        - skip that dataset (use --skip-missing-target / --skip-missing-features).
- Missing values:
    * By default, rows with missing target values are dropped for each target-dataset.
    * Optionally drop rows with missing feature values (use --drop-missing-features).
- Binary columns:
    * The listed binary columns are coerced to 0/1 (if possible). If coercion fails, the script errors.
- Neighbor features:
    * Neighbors are defined ONLY within the same sample_id.
    * Immediate neighbors are based on row adjacency:
        previous row (same sample_id) -> "_p"
        next row (same sample_id)     -> "_f"
    * Higher-order neighbors are chained by row adjacency:
        "_pp" = 2 rows above (same sample_id chain), "_ff" = 2 rows below, etc.

Outputs
-------
For each target (e.g., cellulase_rep1), you get a file like:
    <outdir>/<target>.csv

Notes
-----
- The script does not “learn” anything; it only prepares files.
- It is written to be readable and maintainable (not spaghetti code).

Author
------
Generated for Sebastian Raubitzek.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


# =========================
# USER SETTINGS (EDIT HERE)
# =========================

# Folder that contains features.csv
INPUT_DIR = Path("MyPilz")

# Input file name (inside INPUT_DIR)
INPUT_FILENAME = "features.csv"

# Output directory for per-target ML datasets
OUTDIR = Path("ml_datasets")

# Column handling (these replace the CLI flags)
SKIP_MISSING_TARGET = False     # if True: skip targets missing in input CSV
SKIP_MISSING_FEATURES = False   # if True: drop missing feature columns instead of failing

# Missing data handling (these replace the CLI flags)
DROP_MISSING_TARGET = True      # drop rows with missing target (recommended)
DROP_MISSING_FEATURES = False   # drop rows with missing feature values

# CSV reading/writing (these replace the CLI flags)
SEP = ","
ENCODING = "utf-8"

# Neighbor feature expansion
NEIGHBOR_ORDER = 1              # 0 = no neighbor features; 1 = p/f; 2 = p,pp and f,ff; etc.


# ----------------------------
# Configuration: columns to use
# ----------------------------

FEATURE_COLUMNS: List[str] = [
    #"n_variants_total",
    "n_snps",
    "n_indels",
    "n_HIGH",
    "n_MODERATE",
    "n_LOW",
    "n_MODIFIER",
    #"impact_score",
    "has_frameshift",      # Binary (0/1)
    "has_stop_gained",     # Binary (0/1)
    "has_start_lost",      # Binary (0/1)
    "has_splice_disrupt",  # Binary (0/1)
    "unit_kind",
    "unit_len_bp",
    #"generation",
    #"mating_type",
    "has_LOF",
    "has_NMD",
    "max_indel_bp",
    "total_indel_bp"
]

BINARY_FEATURE_COLUMNS: List[str] = [
    "has_frameshift",
    "has_stop_gained",
    "has_start_lost",
    "has_splice_disrupt",
    "has_LOF",
    "has_NMD",
]

TARGET_COLUMNS: List[str] = [
    "combined_q50",
    "cellulase_q50",
    "cellulase_q33",
    "cellulase_q25",
    "cellulase_q20",
    "xylanase_q50",
    "xylanase_q33",
    "xylanase_q25",
    "xylanase_q20",
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

# Required column for neighbor detection
NEIGHBOR_ID_COLUMN = "sample_id"


# ----------------------------
# Helper functions
# ----------------------------

def validate_columns(
    df: pd.DataFrame,
    features: List[str],
    targets: List[str],
    skip_missing_features: bool,
    skip_missing_target: bool,
) -> Tuple[List[str], List[str]]:
    """
    Validate requested feature and target columns.

    Returns
    -------
    (final_features, final_targets)
        final_features: features that exist in df (or all requested if strict)
        final_targets: targets that exist in df (or all requested if strict)

    Raises
    ------
    KeyError
        If strict mode is enabled and required columns are missing.
    """
    missing_features = [c for c in features if c not in df.columns]
    missing_targets = [t for t in targets if t not in df.columns]

    # Features
    if missing_features and not skip_missing_features:
        raise KeyError(
            f"Missing feature columns in input CSV: {missing_features}\n"
            f"Either fix the input CSV or set SKIP_MISSING_FEATURES = True."
        )
    final_features = [c for c in features if c in df.columns]

    # Targets
    if missing_targets and not skip_missing_target:
        raise KeyError(
            f"Missing target columns in input CSV: {missing_targets}\n"
            f"Either fix the input CSV or set SKIP_MISSING_TARGET = True."
        )
    final_targets = [t for t in targets if t in df.columns]

    return final_features, final_targets


def coerce_binary_columns(df: pd.DataFrame, binary_cols: List[str]) -> pd.DataFrame:
    """
    Coerce binary columns to integers 0/1.

    Accepts values like:
    - 0/1
    - True/False
    - "0"/"1"
    - "true"/"false" (case-insensitive)

    Raises
    ------
    ValueError
        If coercion fails for any binary column.
    """
    out = df.copy()

    for col in binary_cols:
        if col not in out.columns:
            continue

        # Normalize common string representations
        s = out[col]
        if s.dtype == object:
            s = s.astype(str).str.strip().str.lower().replace(
                {"true": "1", "false": "0", "nan": pd.NA, "none": pd.NA, "": pd.NA}
            )

        # Convert to numeric, then validate values are in {0,1} or NA
        s_num = pd.to_numeric(s, errors="coerce")
        invalid_mask = ~(s_num.isna() | s_num.isin([0, 1]))
        if invalid_mask.any():
            bad_examples = out.loc[invalid_mask, col].head(10).tolist()
            raise ValueError(
                f"Binary coercion failed for column '{col}'. "
                f"Found non-binary values (examples): {bad_examples}"
            )

        out[col] = s_num.astype("Int64")  # nullable integer type

    return out


def make_one_dataset(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    drop_missing_target: bool,
    drop_missing_features: bool,
) -> pd.DataFrame:
    """
    Build a single ML dataset for one target: features + that target.

    Steps:
    1) Select relevant columns.
    2) Drop rows with missing target (optional).
    3) Drop rows with missing features (optional).
    """
    cols = features + [target]
    ds = df.loc[:, cols].copy()

    if drop_missing_target:
        ds = ds.dropna(subset=[target])

    if drop_missing_features:
        ds = ds.dropna(subset=features)

    return ds


def safe_filename(name: str) -> str:
    """Make a target name safe as a filename (minimal sanitization)."""
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _neighbor_indices_for_group_by_row(n: int, order: int) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Compute neighbor indices within a single group by row adjacency.

    For a group of length n (already in the correct row order):
      prev1[i] = i-1 (if exists)
      next1[i] = i+1 (if exists)

    Higher-order neighbors are chained:
      prev(k)[i] = prev1(prev(k-1)[i])
      next(k)[i] = next1(next(k-1)[i])
    """
    prev1 = np.full(n, -1, dtype=int)
    next1 = np.full(n, -1, dtype=int)

    if n == 0 or order <= 0:
        return {}, {}

    for i in range(1, n):
        prev1[i] = i - 1
    for i in range(0, n - 1):
        next1[i] = i + 1

    prev_map: Dict[int, np.ndarray] = {1: prev1}
    next_map: Dict[int, np.ndarray] = {1: next1}

    for k in range(2, order + 1):
        prev_k = np.full(n, -1, dtype=int)
        next_k = np.full(n, -1, dtype=int)

        prev_prev = prev_map[k - 1]
        next_prev = next_map[k - 1]

        for i in range(n):
            j = prev_prev[i]
            if j != -1:
                prev_k[i] = prev1[j]

        for i in range(n):
            j = next_prev[i]
            if j != -1:
                next_k[i] = next1[j]

        prev_map[k] = prev_k
        next_map[k] = next_k

    return prev_map, next_map


def add_neighbor_features(
    df: pd.DataFrame,
    base_features: List[str],
    sample_id_col: str,
    order: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add neighbor-expanded feature columns to df using ROW ORDER adjacency.

    For each base feature 'x', this creates:
      x_p, x_pp, ... (previous neighbors in the CSV row order, within the same sample_id)
      x_f, x_ff, ... (following neighbors in the CSV row order, within the same sample_id)
    up to the specified order.

    Missing neighbors produce NA values.

    Returns
    -------
    (df_out, expanded_feature_list)
    """
    if order <= 0:
        return df, base_features

    required = [sample_id_col]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise KeyError(
            f"Neighbor expansion requires columns {required}, but missing: {missing_required}"
        )

    # Work on a stable copy, preserve original row order
    df_out = df.copy()
    df_out["_orig_row_idx__"] = np.arange(len(df_out), dtype=int)

    # Keep the ORIGINAL row order, but group by sample_id preserving order.
    df_sorted = df_out.sort_values("_orig_row_idx__", kind="mergesort").reset_index(drop=True)

    expanded_features = list(base_features)

    # Preallocate new columns
    for k in range(1, order + 1):
        p_suffix = "p" * k
        f_suffix = "f" * k
        for feat in base_features:
            df_sorted[f"{feat}_{p_suffix}"] = pd.NA
            df_sorted[f"{feat}_{f_suffix}"] = pd.NA

    # Process each sample_id group IN CURRENT ROW ORDER
    for _, g_idx in df_sorted.groupby(sample_id_col, sort=False).groups.items():
        idx = np.asarray(list(g_idx), dtype=int)
        n = len(idx)
        if n <= 1:
            continue

        prev_map, next_map = _neighbor_indices_for_group_by_row(n=n, order=order)

        for k in range(1, order + 1):
            p_suffix = "p" * k
            f_suffix = "f" * k

            prev_idx_local = prev_map[k]
            next_idx_local = next_map[k]

            prev_idx_global = np.where(prev_idx_local == -1, -1, idx[prev_idx_local])
            next_idx_global = np.where(next_idx_local == -1, -1, idx[next_idx_local])

            for feat in base_features:
                vals_prev = np.full(n, np.nan, dtype=object)
                mask_prev = prev_idx_global != -1
                if mask_prev.any():
                    vals_prev[mask_prev] = df_sorted.loc[prev_idx_global[mask_prev], feat].to_numpy()
                df_sorted.loc[idx, f"{feat}_{p_suffix}"] = vals_prev

                vals_next = np.full(n, np.nan, dtype=object)
                mask_next = next_idx_global != -1
                if mask_next.any():
                    vals_next[mask_next] = df_sorted.loc[next_idx_global[mask_next], feat].to_numpy()
                df_sorted.loc[idx, f"{feat}_{f_suffix}"] = vals_next

    # Restore original row order (already same), drop helper col
    df_sorted = df_sorted.sort_values("_orig_row_idx__", kind="mergesort").drop(columns=["_orig_row_idx__"])
    df_sorted.reset_index(drop=True, inplace=True)

    # Build expanded feature list
    for k in range(1, order + 1):
        p_suffix = "p" * k
        f_suffix = "f" * k
        for feat in base_features:
            expanded_features.append(f"{feat}_{p_suffix}")
        for feat in base_features:
            expanded_features.append(f"{feat}_{f_suffix}")

    return df_sorted, expanded_features


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    print("[1/8] Initializing paths and output directory...")
    input_path = INPUT_DIR / INPUT_FILENAME
    outdir = OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"      Input CSV: {input_path}")
    print(f"      Output dir: {outdir}")
    print(f"      Neighbor order: {NEIGHBOR_ORDER}")

    print("[2/8] Loading input CSV...")
    df = pd.read_csv(input_path, sep=SEP, encoding=ENCODING)
    print(f"      Loaded shape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("[3/8] Validating requested feature/target columns...")
    final_features, final_targets = validate_columns(
        df=df,
        features=FEATURE_COLUMNS,
        targets=TARGET_COLUMNS,
        skip_missing_features=SKIP_MISSING_FEATURES,
        skip_missing_target=SKIP_MISSING_TARGET,
    )
    print(f"      Base features kept: {len(final_features)}")
    print(f"      Targets found: {len(final_targets)}")

    print("[4/8] Coercing binary feature columns to 0/1 where present...")
    df = coerce_binary_columns(df, BINARY_FEATURE_COLUMNS)
    present_bin = [c for c in BINARY_FEATURE_COLUMNS if c in df.columns]
    print(f"      Binary columns processed: {present_bin if present_bin else 'none'}")

    print("[5/8] Expanding feature set with neighboring units (row-adjacency)...")
    df, expanded_features = add_neighbor_features(
        df=df,
        base_features=final_features,
        sample_id_col=NEIGHBOR_ID_COLUMN,
        order=NEIGHBOR_ORDER,
    )
    print(f"      Expanded features total: {len(expanded_features)}")

    print("[6/8] Creating one dataset per target and writing CSVs...")
    written = 0
    total = len(final_targets)
    for i, target in enumerate(final_targets, start=1):
        print(f"      ({i}/{total}) Processing target: {target}")
        ds = make_one_dataset(
            df=df,
            features=expanded_features,
            target=target,
            drop_missing_target=DROP_MISSING_TARGET,
            drop_missing_features=DROP_MISSING_FEATURES,
        )

        out_path = outdir / f"{safe_filename(target)}.csv"
        ds.to_csv(out_path, index=False, encoding=ENCODING)
        written += 1
        print(f"            Wrote: {out_path} (rows={len(ds)})")

    print("[7/8] Done. Summary:")
    print(f"Input:   {input_path.resolve()}")
    print(f"Outdir:  {outdir.resolve()}")
    print(f"Base features used ({len(final_features)}): {final_features}")
    print(f"Expanded features used ({len(expanded_features)}): {len(expanded_features)} total")
    print(f"Targets written ({written}): {final_targets}")

    print("[8/8] Finished.")


if __name__ == "__main__":
    main()