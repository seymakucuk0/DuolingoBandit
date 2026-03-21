"""
RELATIVE DIFFERENCE SCORING (Layer 2)
=====================================
This is the CORE of the algorithm. Instead of using raw click rates to judge
how good a template is, we measure how much better it is compared to what
would have happened with a random choice.

WHY THIS MATTERS:
  Imagine template A has a 30% click rate and template B has a 20% click rate.
  Naively, A looks better. But what if A is only ever shown to super-active users
  who would click on anything? And B is shown to dormant users who rarely click?

  The Relative Difference Score fixes this by asking:
    "Given the OTHER templates that were eligible at the same time,
     did this template perform BETTER or WORSE than average?"

  This removes the confounding effect of user quality.

THE MATH:
  For each notification event where template 'a' was sent:

    baseline(event) = average reward rate across all eligible templates in that event

    diff(event) = actual_reward - baseline(event)

  The Relative Difference Score for template 'a' is:

    RDS(a) = average of diff(event) for all events where template a was sent

  A positive RDS means the template consistently outperforms its peers.
  A negative RDS means it consistently underperforms.
  Near-zero means it's about average.
"""

import numpy as np
import pandas as pd
from collections import defaultdict


def compute_template_reward_rates(df):
    """
    STEP 1: For each template, compute its raw reward rate.

    This is simply: (# of times template was sent AND user engaged) / (# times template was sent)

    Args:
        df: DataFrame with columns 'selected_template' and 'session_end_completed'

    Returns:
        reward_rates: dict {template_name: reward_rate}
        counts: dict {template_name: number_of_times_sent}

    Example output:
        reward_rates = {"A": 0.25, "B": 0.18, "C": 0.31, ...}
        counts = {"A": 150000, "B": 80000, "C": 200000, ...}
    """
    # Group by template and compute mean reward and count
    grouped = df.groupby("selected_template")["session_end_completed"]

    reward_rates = grouped.mean().to_dict()
    counts = grouped.count().to_dict()

    print(f"[difference_score] Computed reward rates for {len(reward_rates)} templates:")
    for t in sorted(reward_rates.keys()):
        print(f"  Template {t}: reward_rate = {reward_rates[t]:.4f}, count = {counts[t]:,}")

    return reward_rates, counts


def compute_counterfactual_baseline(eligible_templates, reward_rates):
    """
    STEP 2: For a SINGLE event, compute the counterfactual baseline.

    The idea: "If we had chosen randomly from the eligible templates,
    what reward would we expect on average?"

    Args:
        eligible_templates: list of templates eligible for this event (e.g., ["A", "C", "F"])
        reward_rates: dict mapping each template to its overall reward rate

    Returns:
        float: the expected reward under random selection from eligible set

    Example:
        eligible = ["A", "C", "F"]
        reward_rates = {"A": 0.25, "C": 0.31, "F": 0.15}
        baseline = (0.25 + 0.31 + 0.15) / 3 = 0.2367

    WHY NOT JUST USE THE GLOBAL AVERAGE?
        Because different events have different eligible sets. An event where
        only high-performing templates are eligible has a higher baseline than
        one where only low-performing templates are eligible. The difference
        score accounts for this by comparing each template against its actual
        competition in that specific event.
    """
    if len(eligible_templates) == 0:
        return 0.0

    # Sum up the reward rates of all eligible templates, then divide by count
    total = 0.0
    valid_count = 0
    for t in eligible_templates:
        if t in reward_rates:
            total += reward_rates[t]
            valid_count += 1

    if valid_count == 0:
        return 0.0

    return total / valid_count


def compute_relative_difference_scores(df, reward_rates):
    """
    STEP 3: Compute the Relative Difference Score for every template.

    For each template 'a':
      1. Filter to all events where selected_template == a
      2. For each such event:
         - Compute the counterfactual baseline from its eligible_templates
         - diff = actual_reward (0 or 1) - baseline
      3. RDS(a) = average of all diffs

    Args:
        df: DataFrame with columns:
            - selected_template
            - session_end_completed
            - eligible_templates (list of strings)
        reward_rates: dict from compute_template_reward_rates()

    Returns:
        rds_scores: dict {template: relative_difference_score}

    PERFORMANCE NOTE:
        This function iterates over rows, which is slow for 200M rows.
        For development, use a sample (100K-1M rows). For production,
        we can vectorize this, but clarity comes first.
    """
    print("[difference_score] Computing Relative Difference Scores...")

    # We'll accumulate diffs for each template
    template_diffs = defaultdict(list)

    # Iterate over each row in the dataframe
    for idx, row in df.iterrows():
        template = row["selected_template"]
        reward = row["session_end_completed"]
        eligible = row["eligible_templates"]

        # Make sure eligible is a list
        if isinstance(eligible, str):
            import ast
            eligible = ast.literal_eval(eligible)

        # Compute counterfactual baseline for this event
        baseline = compute_counterfactual_baseline(eligible, reward_rates)

        # The difference: how much better (or worse) was the actual reward
        # compared to what we'd expect from random selection?
        diff = reward - baseline

        template_diffs[template].append(diff)

    # Average the diffs for each template → that's the RDS
    rds_scores = {}
    for t in sorted(template_diffs.keys()):
        diffs = template_diffs[t]
        rds_scores[t] = np.mean(diffs)
        print(f"  Template {t}: RDS = {rds_scores[t]:+.6f} (from {len(diffs):,} events)")

    return rds_scores


def compute_relative_difference_scores_fast(df, reward_rates):
    """
    FAST VERSION of compute_relative_difference_scores using vectorized operations.

    Same logic as above, but uses pandas/numpy operations instead of row-by-row
    iteration. This is 10-100x faster and necessary for the full dataset.

    The key insight: we can compute the baseline for each row using .apply(),
    then do the subtraction in one vectorized step.

    Args:
        df: DataFrame (same as above)
        reward_rates: dict from compute_template_reward_rates()

    Returns:
        rds_scores: dict {template: relative_difference_score}
    """
    print("[difference_score] Computing Relative Difference Scores (fast version)...")

    # Step 1: Compute the counterfactual baseline for every row
    # This is the only part that can't be fully vectorized (because eligible_templates
    # is a variable-length list), so we use .apply()
    def row_baseline(eligible):
        if isinstance(eligible, str):
            import ast
            eligible = ast.literal_eval(eligible)
        elif hasattr(eligible, 'tolist'):
            eligible = eligible.tolist()
        return compute_counterfactual_baseline(eligible, reward_rates)

    baselines = df["eligible_templates"].apply(row_baseline)

    # Step 2: Compute diff = reward - baseline for every row (fully vectorized!)
    diffs = df["session_end_completed"].values - baselines.values

    # Step 3: Group by template and average the diffs
    diff_series = pd.Series(diffs, index=df.index)
    rds_scores = df.groupby("selected_template").apply(
        lambda grp: diff_series.loc[grp.index].mean()
    ).to_dict()

    for t in sorted(rds_scores.keys()):
        count = (df["selected_template"] == t).sum()
        print(f"  Template {t}: RDS = {rds_scores[t]:+.6f} (from {count:,} events)")

    return rds_scores
