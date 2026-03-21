"""
DATA LOADER
===========
This module handles loading the Duolingo notification dataset from parquet files.

The dataset is split into 6 parquet files:
  - 3 training files (~87.7M rows, 15 days)
  - 3 test files (~114.5M rows, 19 days)

Each row represents ONE notification event: one user received one push notification
at one point in time. The columns are:
  - datetime: float (days since dataset start)
  - ui_language: str (user's language)
  - eligible_templates: list of str (which templates A-L could be sent)
  - history: list of tuples (previously sent templates and when)
  - selected_template: str (which template was actually sent)
  - session_end_completed: int 0 or 1 (did the user complete a lesson within 2 hours?)

WHY WE NEED THIS:
  The raw data is split across multiple parquet files. We need a clean way to:
  1. Load just a sample (for development — 200M rows is too much for testing)
  2. Load full train or test sets (for final evaluation)
  3. Parse the history column (which may be encoded as strings)
"""

import pandas as pd
import glob
import os
import ast


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
# We use relative paths from the project root. If you run scripts from
# a different directory, adjust DATA_DIR accordingly.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


def get_parquet_files(split="train"):
    """
    Find all parquet files for a given split (train or test).

    Args:
        split: "train" or "test"

    Returns:
        list of file paths, sorted alphabetically

    Example:
        >>> files = get_parquet_files("train")
        >>> print(files)
        ['data/raw/train-part-1/xxx.parquet', 'data/raw/train-part-2/xxx.parquet', ...]
    """
    pattern = os.path.join(DATA_DIR, f"{split}-part-*", "*.parquet")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No parquet files found for split='{split}' in {DATA_DIR}. "
            f"Make sure you extracted the tar.gz files into data/raw/"
        )

    print(f"[data_loader] Found {len(files)} parquet files for '{split}' split")
    return files


def parse_history(history_value):
    """
    Parse the history column into a list of (template, days_ago) tuples.

    The history column in the parquet files might be stored as:
      - A Python list of tuples (already parsed)
      - A string representation of a list (needs ast.literal_eval)
      - None or empty (no history for this user)

    Args:
        history_value: the raw value from the history column

    Returns:
        list of (template_str, days_ago_float) tuples, or empty list

    Example:
        >>> parse_history([("A", 1.2), ("F", 3.5)])
        [("A", 1.2), ("F", 3.5)]

        >>> parse_history(None)
        []
    """
    # If it's None or NaN, return empty
    if history_value is None:
        return []

    # If it's already a list, return as-is
    if isinstance(history_value, list):
        return history_value

    # If it's a string, try to parse it
    if isinstance(history_value, str):
        if history_value.strip() == "" or history_value.strip() == "[]":
            return []
        try:
            return ast.literal_eval(history_value)
        except (ValueError, SyntaxError):
            return []

    return []


def parse_eligible_templates(eligible_value):
    """
    Parse the eligible_templates column into a list of template strings.

    Similar to history, this might be stored in different formats depending
    on the parquet serialization.

    Args:
        eligible_value: raw value from eligible_templates column

    Returns:
        list of template strings (e.g., ["A", "C", "F"])
    """
    if eligible_value is None:
        return []

    if isinstance(eligible_value, list):
        return eligible_value

    if isinstance(eligible_value, str):
        if eligible_value.strip() == "" or eligible_value.strip() == "[]":
            return []
        try:
            return ast.literal_eval(eligible_value)
        except (ValueError, SyntaxError):
            return []

    return []


def load_sample(n_rows=100_000, split="train"):
    """
    Load a small sample of the data for development and testing.

    WHY: The full dataset is ~200M rows. You don't want to wait 10 minutes
    every time you test a small code change. Start with 100K rows, get your
    code working, then scale up.

    Args:
        n_rows: how many rows to load (default 100K)
        split: "train" or "test"

    Returns:
        pd.DataFrame with parsed columns
    """
    files = get_parquet_files(split)

    # Read only from the first file, and only n_rows
    print(f"[data_loader] Loading {n_rows:,} sample rows from {os.path.basename(files[0])}...")
    df = pd.read_parquet(files[0])

    # Take the first n_rows (or all if the file has fewer)
    df = df.head(n_rows).copy()

    print(f"[data_loader] Loaded {len(df):,} rows")
    print(f"[data_loader] Columns: {list(df.columns)}")
    print(f"[data_loader] Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    return df


def load_full(split="train"):
    """
    Load the full train or test dataset by reading all parquet files.

    WARNING: This loads ~88M rows (train) or ~115M rows (test) into memory.
    Make sure you have enough RAM (expect 4-8 GB for the full dataset).

    Args:
        split: "train" or "test"

    Returns:
        pd.DataFrame
    """
    files = get_parquet_files(split)

    print(f"[data_loader] Loading full '{split}' dataset from {len(files)} files...")

    # Read each parquet file and concatenate
    dfs = []
    for i, f in enumerate(files):
        print(f"  Reading file {i + 1}/{len(files)}: {os.path.basename(f)}...")
        dfs.append(pd.read_parquet(f))

    df = pd.concat(dfs, ignore_index=True)

    print(f"[data_loader] Loaded {len(df):,} rows total")
    print(f"[data_loader] Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    return df


def describe_dataset(df):
    """
    Print a comprehensive summary of the dataset.
    Call this after loading to understand what you're working with.

    Args:
        df: DataFrame loaded by load_sample() or load_full()
    """
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumn types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")

    print(f"\n--- datetime ---")
    print(f"  Range: {df['datetime'].min():.2f} to {df['datetime'].max():.2f} days")
    print(f"  Span: {df['datetime'].max() - df['datetime'].min():.1f} days")

    print(f"\n--- ui_language ---")
    print(f"  Unique languages: {df['ui_language'].nunique()}")
    print(f"  Top 5: {df['ui_language'].value_counts().head().to_dict()}")

    print(f"\n--- selected_template ---")
    print(f"  Unique templates: {df['selected_template'].nunique()}")
    print(f"  Templates: {sorted(df['selected_template'].unique())}")

    print(f"\n--- session_end_completed (REWARD) ---")
    print(f"  Mean reward: {df['session_end_completed'].mean():.4f}")
    print(f"  Total engaged: {df['session_end_completed'].sum():,} / {len(df):,}")

    # Check eligible_templates format
    print(f"\n--- eligible_templates ---")
    sample_val = df['eligible_templates'].iloc[0]
    print(f"  Type of first value: {type(sample_val).__name__}")
    print(f"  First value: {sample_val}")

    # Check history format
    print(f"\n--- history ---")
    sample_hist = df['history'].iloc[0]
    print(f"  Type of first value: {type(sample_hist).__name__}")
    print(f"  First value: {sample_hist}")

    # Find a row with non-empty history for better understanding
    for i in range(min(100, len(df))):
        h = df['history'].iloc[i]
        if h is not None and (isinstance(h, list) and len(h) > 0) or (isinstance(h, str) and h != "[]"):
            print(f"  Example non-empty history (row {i}): {h}")
            break

    print("=" * 60)
