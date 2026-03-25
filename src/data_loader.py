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
  - eligible_templates: numpy array of str (which templates A-L could be sent)
  - history: numpy array of dicts with keys 'template' and 'n_days'
  - selected_template: str (which template was actually sent)
  - session_end_completed: bool (did the user complete a lesson within 2 hours?)

WHY WE NEED THIS:
  The raw data is split across multiple parquet files. We need a clean way to:
  1. Load just a sample (for development — 200M rows is too much for testing)
  2. Load full train or test sets (for final evaluation)
  3. Parse the history and eligible_templates columns (stored as numpy arrays)
"""

import pandas as pd
import numpy as np
import glob
import os
import ast
import time


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

# When True, load_sample() ignores n_rows and loads the ENTIRE dataset.
# Set to False for quick development with smaller samples.
USE_FULL_DATA = True


def get_parquet_files(split="train"):
    """
    Find all parquet files for a given split (train or test).

    Args:
        split: "train" or "test"

    Returns:
        list of file paths, sorted alphabetically
    """
    # Support both folder patterns:
    # 1. data/train-part-*.parquet (direct files)
    # 2. data/train-part-*/*.parquet (subfolders)
    pattern1 = os.path.join(DATA_DIR, f"{split}-part-*.parquet")
    pattern2 = os.path.join(DATA_DIR, f"{split}-part-*", "*.parquet")
    
    files = sorted(list(set(glob.glob(pattern1) + glob.glob(pattern2))))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No parquet files found for split='{split}' in {DATA_DIR}. "
            f"Expected files matching {split}-part-*.parquet"
        )

    print(f"[data_loader] Found {len(files)} parquet files for '{split}' split")
    return files


def parse_history(history_value):
    """
    Parse the history column into a list of (template, days_ago) tuples.

    The history column in the parquet files can be stored as:
      - A numpy ndarray of dicts: [{'template': 'A', 'n_days': 1.2}, ...]
      - A Python list of dicts or tuples
      - A string representation (needs ast.literal_eval)
      - None or empty

    Args:
        history_value: the raw value from the history column

    Returns:
        list of (template_str, days_ago_float) tuples, or empty list
    """
    # None or NaN
    if history_value is None:
        return []

    # Handle numpy ndarray (most common format from parquet)
    if isinstance(history_value, np.ndarray):
        result = []
        for entry in history_value:
            if isinstance(entry, dict):
                t = entry.get("template", "")
                d = entry.get("n_days", 0.0)
                result.append((t, float(d)))
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                result.append((str(entry[0]), float(entry[1])))
        return result

    # Handle Python list
    if isinstance(history_value, list):
        result = []
        for entry in history_value:
            if isinstance(entry, dict):
                t = entry.get("template", "")
                d = entry.get("n_days", 0.0)
                result.append((t, float(d)))
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                result.append((str(entry[0]), float(entry[1])))
        return result

    # Handle string representation
    if isinstance(history_value, str):
        if history_value.strip() == "" or history_value.strip() == "[]":
            return []
        try:
            parsed = ast.literal_eval(history_value)
            return parse_history(parsed)  # recurse to handle the parsed result
        except (ValueError, SyntaxError):
            return []

    return []


def parse_eligible_templates(eligible_value):
    """
    Parse the eligible_templates column into a list of template strings.

    The column can be stored as:
      - A numpy ndarray of strings: array(['A', 'C', 'F'], dtype=object)
      - A Python list of strings
      - A string representation

    Args:
        eligible_value: raw value from eligible_templates column

    Returns:
        list of template strings (e.g., ["A", "C", "F"])
    """
    if eligible_value is None:
        return []

    # Handle numpy ndarray (most common from parquet)
    if isinstance(eligible_value, np.ndarray):
        return eligible_value.tolist()

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


def _postprocess(df, batch_size=1_000_000):
    """
    Fix column types after loading from parquet.
    Processes data in batches for better performance with large datasets.

    - session_end_completed: convert bool → int (0/1) for numeric operations
    - eligible_templates: convert ndarray/str/list → standard list of strings
    - history: convert ndarray/str/list → list of (template, days_ago) tuples
    """
    total = len(df)
    if total == 0:
        return df

    # Convert bool reward to int so .mean(), .sum() etc. work as expected
    if "session_end_completed" in df.columns:
        if df["session_end_completed"].dtype == bool:
            df["session_end_completed"] = df["session_end_completed"].astype(int)

    # Process eligible_templates and history in batches
    n_batches = (total + batch_size - 1) // batch_size
    print(f"[postprocess] Processing {total:,} rows in {n_batches} batch(es) of {batch_size:,}...")

    pp_start = time.time()
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch_start = time.time()

        if "eligible_templates" in df.columns:
            df.iloc[start:end, df.columns.get_loc("eligible_templates")] = (
                df.iloc[start:end]["eligible_templates"].apply(parse_eligible_templates)
            )

        if "history" in df.columns:
            df.iloc[start:end, df.columns.get_loc("history")] = (
                df.iloc[start:end]["history"].apply(parse_history)
            )

        batch_elapsed = time.time() - batch_start
        total_elapsed = time.time() - pp_start
        rows_done = end
        rows_per_sec = rows_done / total_elapsed if total_elapsed > 0 else 0
        remaining = (total - rows_done) / rows_per_sec if rows_per_sec > 0 else 0

        print(f"  Batch {batch_idx + 1}/{n_batches}: "
              f"rows {start:,}-{end:,} done in {batch_elapsed:.1f}s | "
              f"Total: {rows_done:,}/{total:,} ({rows_done/total:.0%}) | "
              f"ETA: {remaining:.0f}s")

    print(f"[postprocess] All {total:,} rows processed in {time.time() - pp_start:.1f}s")
    return df


def load_sample(n_rows=None, split="train"):
    """
    Load training or test data from the full raw parquet files.

    When USE_FULL_DATA is True (default), ignores n_rows and loads the
    ENTIRE dataset from all parquet files. When USE_FULL_DATA is False
    and n_rows is specified, loads only that many rows for faster dev/testing.

    Args:
        n_rows: how many rows to load, or None for all rows
        split: "train" or "test"

    Returns:
        pd.DataFrame with parsed columns
    """
    # Try to find split-specific parquet files; fall back to sample files
    try:
        files = get_parquet_files(split)
    except FileNotFoundError:
        # Prefer larger sample first, then fall back to smaller
        sample_500k = os.path.join(DATA_DIR, "..", "sample_500k.parquet")
        sample_10k = os.path.join(DATA_DIR, "..", "sample_10k.parquet")
        if os.path.exists(sample_500k):
            print(f"[data_loader] Full '{split}' parquet files not found. "
                  f"Falling back to sample_500k.parquet")
            files = [sample_500k]
        elif os.path.exists(sample_10k):
            print(f"[data_loader] Full '{split}' parquet files not found. "
                  f"Falling back to sample_10k.parquet")
            files = [sample_10k]
        else:
            raise

    # When USE_FULL_DATA is True, always load everything
    if USE_FULL_DATA:
        if n_rows is not None:
            print(f"[data_loader] USE_FULL_DATA=True → ignoring n_rows={n_rows:,}, loading ALL data")
        n_rows = None

    load_start = time.time()

    if n_rows is None:
        # Load ALL data from ALL files (full dataset mode)
        print(f"[data_loader] === FULL DATA MODE ===")
        print(f"[data_loader] Loading full '{split}' dataset from {len(files)} files...")

        dfs = []
        total_rows = 0
        for i, f in enumerate(files):
            file_start = time.time()
            print(f"  [{i + 1}/{len(files)}] Reading {os.path.basename(f)}...", end=" ", flush=True)
            part = pd.read_parquet(f)
            file_elapsed = time.time() - file_start
            total_rows += len(part)
            print(f"{len(part):,} rows in {file_elapsed:.1f}s (total so far: {total_rows:,})")
            dfs.append(part)

        print(f"[data_loader] Concatenating {len(dfs)} DataFrames...", end=" ", flush=True)
        concat_start = time.time()
        df = pd.concat(dfs, ignore_index=True)
        print(f"done in {time.time() - concat_start:.1f}s")
    else:
        # Load only n_rows from files (sample mode)
        print(f"[data_loader] === SAMPLE MODE ===")
        print(f"[data_loader] Loading {n_rows:,} rows from {len(files)} file(s)...")

        try:
            import pyarrow.parquet as pq
            import pyarrow as pa

            batches = []
            rows_so_far = 0
            for f in files:
                if rows_so_far >= n_rows:
                    break
                pf = pq.ParquetFile(f)
                for i in range(pf.metadata.num_row_groups):
                    if rows_so_far >= n_rows:
                        break
                    batch = pf.read_row_group(i)
                    batches.append(batch)
                    rows_so_far += batch.num_rows
                print(f"  Read {os.path.basename(f)}: {rows_so_far:,} rows so far")

            table = pa.concat_tables(batches)
            df = table.to_pandas()
            df = df.head(n_rows).copy()
        except ImportError:
            dfs = []
            rows_so_far = 0
            for f in files:
                if rows_so_far >= n_rows:
                    break
                part = pd.read_parquet(f)
                dfs.append(part)
                rows_so_far += len(part)
                print(f"  Read {os.path.basename(f)}: {rows_so_far:,} rows so far")
            df = pd.concat(dfs, ignore_index=True)
            df = df.head(n_rows).copy()

    load_elapsed = time.time() - load_start
    print(f"[data_loader] Raw data loaded: {len(df):,} rows in {load_elapsed:.1f}s")
    print(f"[data_loader] Memory (raw): {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Post-process column types
    print(f"[data_loader] Post-processing columns (parsing history & templates)...", flush=True)
    pp_start = time.time()
    df = _postprocess(df)
    pp_elapsed = time.time() - pp_start
    print(f"[data_loader] Post-processing done in {pp_elapsed:.1f}s")

    total_elapsed = time.time() - load_start
    print(f"[data_loader] ✓ Final: {len(df):,} rows loaded in {total_elapsed:.1f}s total")
    print(f"[data_loader] Columns: {list(df.columns)}")
    print(f"[data_loader] Memory (final): {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

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

    dfs = []
    for i, f in enumerate(files):
        print(f"  Reading file {i + 1}/{len(files)}: {os.path.basename(f)}...")
        dfs.append(pd.read_parquet(f))

    df = pd.concat(dfs, ignore_index=True)
    df = _postprocess(df)

    print(f"[data_loader] Loaded {len(df):,} rows total")
    print(f"[data_loader] Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    return df


def iter_parquet_chunks(split="train", chunk_size=1_000_000, columns=None,
                        parse_eligible=True, parse_hist=True):
    """
    Yield DataFrames of at most `chunk_size` rows from the raw parquet files.

    This lets you process the full 88M-row dataset on a 16 GB machine by
    never holding more than one chunk in memory at a time.

    Args:
        split: "train" or "test"
        chunk_size: max rows per yielded DataFrame (default 1M ≈ 150-200 MB)
        columns: list of column names to read, or None for all columns.
                 Reading fewer columns = faster + less memory.
        parse_eligible: if True, parse eligible_templates column (default True)
        parse_hist: if True, parse history column (default True).
                    Set False when history isn't needed (e.g. for reward rates,
                    RDS computation) — saves significant time.

    Yields:
        pd.DataFrame with (optionally) parsed columns
    """
    import pyarrow.parquet as pq

    files = get_parquet_files(split)
    total_yielded = 0

    for file_idx, fpath in enumerate(files):
        pf = pq.ParquetFile(fpath)
        print(f"[iter_chunks] File {file_idx + 1}/{len(files)}: "
              f"{os.path.basename(fpath)} ({pf.metadata.num_rows:,} rows, "
              f"{pf.metadata.num_row_groups} row groups)")

        buffer_batches = []
        buffer_rows = 0

        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=columns)
            batch_df = table.to_pandas()

            buffer_batches.append(batch_df)
            buffer_rows += len(batch_df)

            # Flush when buffer reaches chunk_size
            while buffer_rows >= chunk_size:
                combined = pd.concat(buffer_batches, ignore_index=True)
                chunk = combined.iloc[:chunk_size].copy()
                remainder = combined.iloc[chunk_size:].copy()

                chunk = _postprocess_selective(chunk, parse_eligible, parse_hist)
                total_yielded += len(chunk)
                print(f"[iter_chunks] Yielding chunk: {len(chunk):,} rows "
                      f"(total yielded: {total_yielded:,})")
                yield chunk

                buffer_batches = [remainder] if len(remainder) > 0 else []
                buffer_rows = len(remainder)

        # Flush remaining rows from this file
        if buffer_rows > 0:
            combined = pd.concat(buffer_batches, ignore_index=True)
            combined = _postprocess_selective(combined, parse_eligible, parse_hist)
            total_yielded += len(combined)
            print(f"[iter_chunks] Yielding chunk: {len(combined):,} rows "
                  f"(total yielded: {total_yielded:,})")
            yield combined
            buffer_batches = []
            buffer_rows = 0

    print(f"[iter_chunks] Done. Total rows yielded: {total_yielded:,}")


def _postprocess_selective(df, parse_elig=True, parse_hist=True):
    """
    Lighter version of _postprocess — only parse the columns you need.
    Skipping history parsing saves ~50% of postprocessing time.
    """
    if len(df) == 0:
        return df

    if "session_end_completed" in df.columns:
        if df["session_end_completed"].dtype == bool:
            df["session_end_completed"] = df["session_end_completed"].astype(int)

    if parse_elig and "eligible_templates" in df.columns:
        df["eligible_templates"] = df["eligible_templates"].apply(parse_eligible_templates)

    if parse_hist and "history" in df.columns:
        df["history"] = df["history"].apply(parse_history)

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

    # Find a row with non-empty history
    for i in range(min(100, len(df))):
        h = df['history'].iloc[i]
        if isinstance(h, list) and len(h) > 0:
            print(f"  Example non-empty history (row {i}): {h[:3]}")
            break

    print("=" * 60)
