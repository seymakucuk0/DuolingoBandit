"""
RELATIVE DIFFERENCE SCORING (Layer 2) — PAPER-ACCURATE VERSION
===============================================================
Implements the RDS formula from Yancey & Settles (KDD 2020), Equations 1-2.

For each template (arm) 'a':

  mu_plus_a  = importance-weighted avg reward WHEN a is chosen
  mu_minus_a = importance-weighted avg reward WHEN a is eligible but NOT chosen

  score_a = (mu_plus_a - mu_minus_a) / mu_minus_a

This is a RELATIVE difference (percentage lift), not absolute.
The importance weights correct for varying eligible set sizes.
"""

import numpy as np
import pandas as pd
from collections import defaultdict


def compute_template_reward_rates(df):
    """
    Compute raw reward rate per template (unchanged — used for reference).
    """
    grouped = df.groupby("selected_template")["session_end_completed"]
    reward_rates = grouped.mean().to_dict()
    counts = grouped.count().to_dict()

    print(f"[difference_score] Computed reward rates for {len(reward_rates)} templates:")
    for t in sorted(reward_rates.keys()):
        print(f"  Template {t}: reward_rate = {reward_rates[t]:.4f}, count = {counts[t]:,}")

    return reward_rates, counts


def compute_rds_paper(df):
    """
    Compute RDS scores using the EXACT paper formula (Equations 1-2).

    For each arm a:
      mu_plus_a  = sum(w_t * r_t for events where a chosen) / sum(w_t ...)
      mu_minus_a = sum(w_t * r_t for events where a eligible but NOT chosen) / sum(w_t ...)

      w_t (when chosen)     = |eligible_t|       (= 1 / logging_prob)
      w_t (when not chosen) = |eligible_t| / (|eligible_t| - 1)

      score_a = (mu_plus_a - mu_minus_a) / mu_minus_a

    Args:
        df: DataFrame with columns:
            - selected_template
            - session_end_completed (0 or 1)
            - eligible_templates (list of strings)

    Returns:
        rds_scores: dict {template: score}
        counts: dict {template: number_of_times_sent}
    """
    print("[difference_score] Computing RDS scores (paper formula)...")

    # Accumulators for each template
    # mu_plus: weighted reward sum / weight sum (when template IS chosen)
    plus_wr_sum = defaultdict(float)   # sum of w * r
    plus_w_sum = defaultdict(float)    # sum of w
    plus_count = defaultdict(int)

    # mu_minus: weighted reward sum / weight sum (when template is eligible but NOT chosen)
    minus_wr_sum = defaultdict(float)
    minus_w_sum = defaultdict(float)

    eligible_col = df["eligible_templates"].values
    selected_col = df["selected_template"].values
    reward_col = df["session_end_completed"].values

    for i in range(len(df)):
        eligible = eligible_col[i]
        if isinstance(eligible, str):
            import ast
            eligible = ast.literal_eval(eligible)
        elif hasattr(eligible, 'tolist'):
            eligible = eligible.tolist()

        selected = selected_col[i]
        reward = float(reward_col[i])
        n_elig = len(eligible)

        if n_elig == 0:
            continue

        # For the CHOSEN template: w = |eligible| (inverse of logging prob 1/|eligible|)
        w_plus = float(n_elig)
        plus_wr_sum[selected] += w_plus * reward
        plus_w_sum[selected] += w_plus
        plus_count[selected] += 1

        # For all OTHER eligible templates: w = |eligible| / (|eligible| - 1)
        if n_elig > 1:
            w_minus = float(n_elig) / float(n_elig - 1)
            for t in eligible:
                if t != selected:
                    minus_wr_sum[t] += w_minus * reward
                    minus_w_sum[t] += w_minus

    # Compute scores
    all_templates = sorted(set(list(plus_w_sum.keys()) + list(minus_w_sum.keys())))
    rds_scores = {}
    counts = {}

    for t in all_templates:
        mu_plus = plus_wr_sum[t] / plus_w_sum[t] if plus_w_sum[t] > 0 else 0.0
        mu_minus = minus_wr_sum[t] / minus_w_sum[t] if minus_w_sum[t] > 0 else 0.0
        counts[t] = plus_count[t]

        if mu_minus > 0:
            score = (mu_plus - mu_minus) / mu_minus
        else:
            score = 0.0

        rds_scores[t] = score
        print(f"  Template {t}: score = {score:+.6f} "
              f"(mu+ = {mu_plus:.6f}, mu- = {mu_minus:.6f}, n = {plus_count[t]:,})")

    return rds_scores, counts


def compute_rds_paper_chunked_pass1(chunk):
    """
    Process one chunk for RDS computation. Returns per-template accumulators.
    Used by fit_chunked methods for streaming over full dataset.

    Returns:
        dict with keys: plus_wr_sum, plus_w_sum, plus_count, minus_wr_sum, minus_w_sum
    """
    plus_wr_sum = defaultdict(float)
    plus_w_sum = defaultdict(float)
    plus_count = defaultdict(int)
    minus_wr_sum = defaultdict(float)
    minus_w_sum = defaultdict(float)

    eligible_col = chunk["eligible_templates"].values
    selected_col = chunk["selected_template"].values
    reward_col = chunk["session_end_completed"].values

    for i in range(len(chunk)):
        eligible = eligible_col[i]
        if isinstance(eligible, str):
            import ast
            eligible = ast.literal_eval(eligible)
        elif hasattr(eligible, 'tolist'):
            eligible = eligible.tolist()

        selected = selected_col[i]
        reward = float(reward_col[i])
        n_elig = len(eligible)

        if n_elig == 0:
            continue

        w_plus = float(n_elig)
        plus_wr_sum[selected] += w_plus * reward
        plus_w_sum[selected] += w_plus
        plus_count[selected] += 1

        if n_elig > 1:
            w_minus = float(n_elig) / float(n_elig - 1)
            for t in eligible:
                if t != selected:
                    minus_wr_sum[t] += w_minus * reward
                    minus_w_sum[t] += w_minus

    return {
        "plus_wr_sum": dict(plus_wr_sum),
        "plus_w_sum": dict(plus_w_sum),
        "plus_count": dict(plus_count),
        "minus_wr_sum": dict(minus_wr_sum),
        "minus_w_sum": dict(minus_w_sum),
    }


def merge_rds_accumulators(acc_list):
    """
    Merge multiple chunk accumulators into one, then compute final scores.

    Args:
        acc_list: list of dicts from compute_rds_paper_chunked_pass1

    Returns:
        rds_scores: dict {template: score}
        counts: dict {template: count}
    """
    # Merge all accumulators
    plus_wr = defaultdict(float)
    plus_w = defaultdict(float)
    plus_c = defaultdict(int)
    minus_wr = defaultdict(float)
    minus_w = defaultdict(float)

    for acc in acc_list:
        for t, v in acc["plus_wr_sum"].items():
            plus_wr[t] += v
        for t, v in acc["plus_w_sum"].items():
            plus_w[t] += v
        for t, v in acc["plus_count"].items():
            plus_c[t] += v
        for t, v in acc["minus_wr_sum"].items():
            minus_wr[t] += v
        for t, v in acc["minus_w_sum"].items():
            minus_w[t] += v

    # Compute final scores
    all_templates = sorted(set(list(plus_w.keys()) + list(minus_w.keys())))
    rds_scores = {}
    counts = {}

    print("[difference_score] Final RDS scores (paper formula):")
    for t in all_templates:
        mu_plus = plus_wr[t] / plus_w[t] if plus_w[t] > 0 else 0.0
        mu_minus = minus_wr[t] / minus_w[t] if minus_w[t] > 0 else 0.0
        counts[t] = plus_c[t]

        if mu_minus > 0:
            score = (mu_plus - mu_minus) / mu_minus
        else:
            score = 0.0

        rds_scores[t] = score
        print(f"  Template {t}: score = {score:+.4%} "
              f"(mu+ = {mu_plus:.6f}, mu- = {mu_minus:.6f}, n = {plus_c[t]:,})")

    return rds_scores, counts


# Keep old functions for backward compatibility (notebooks 02-03 that import them)
def compute_counterfactual_baseline(eligible_templates, reward_rates):
    """Legacy function — kept for backward compatibility."""
    if len(eligible_templates) == 0:
        return 0.0
    total = 0.0
    valid_count = 0
    for t in eligible_templates:
        if t in reward_rates:
            total += reward_rates[t]
            valid_count += 1
    if valid_count == 0:
        return 0.0
    return total / valid_count


def compute_relative_difference_scores_fast(df, reward_rates):
    """Legacy function — kept for backward compatibility. Use compute_rds_paper instead."""
    return compute_rds_paper(df)[0]
