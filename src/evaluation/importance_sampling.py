"""
OFFLINE POLICY EVALUATION — WEIGHTED IMPORTANCE SAMPLING
========================================================
This is the TRICKIEST part of the project. Read carefully.

THE PROBLEM:
  We built an algorithm (RDS) that selects templates. But we can't deploy it
  live to see if it works — we only have historical data. How do we estimate
  the algorithm's performance from data collected by a DIFFERENT policy?

  This is called "off-policy evaluation" or "counterfactual evaluation."

THE KEY INSIGHT:
  The historical data was collected under a RANDOM policy (every eligible
  template equally likely). Our new RDS policy assigns DIFFERENT probabilities
  to each template. If our policy would have chosen the same template that
  was actually sent (and the user engaged), that's evidence our policy is good.

  But we can't just look at cases where our policy agrees with the data —
  we need to account for HOW MUCH MORE/LESS likely our policy was to make
  each choice.

THE SOLUTION — IMPORTANCE SAMPLING:
  For each event in the data:
    - The logging (random) policy chose template 'a' with probability:
        π_log(a) = 1 / |eligible_templates|

    - Our target (RDS) policy would choose template 'a' with probability:
        π_target(a) = softmax probability from our algorithm

    - The importance weight is:
        w = π_target(a) / π_log(a)

    - If w > 1: our policy is MORE likely to choose this template than random
    - If w < 1: our policy is LESS likely to choose this template than random

  The estimated reward of our policy is:

    V̂(π_target) = Σ(w_i × r_i) / Σ(w_i)

  This is called WEIGHTED Importance Sampling (WIS). It's more stable than
  the unweighted version because it normalizes by the sum of weights.

WHY THIS WORKS:
  By re-weighting the observed rewards, we're essentially creating a
  "virtual dataset" where templates are selected according to our policy
  instead of randomly. Events that our policy would strongly prefer get
  upweighted; events it would avoid get downweighted.

VARIANCE AND CLIPPING:
  If our policy assigns very high probability to a template that was rarely
  chosen by the logging policy, the weight can be enormous (e.g., w = 100).
  A few extreme weights can make the estimate very noisy.

  Solution: clip the weights to a maximum value (e.g., 20). This adds a
  small amount of bias but greatly reduces variance.
"""

import numpy as np


def compute_logging_probability(n_eligible):
    """
    Compute the probability that the logging (random) policy selected
    any specific template.

    Under random selection: P(template) = 1 / |eligible_templates|

    Args:
        n_eligible: int, number of eligible templates for this event

    Returns:
        float: probability (e.g., 0.25 if 4 templates were eligible)
    """
    if n_eligible <= 0:
        return 1.0  # safety fallback
    return 1.0 / n_eligible


def compute_importance_weights(df, target_probs_list, max_weight=None):
    """
    Compute importance sampling weights for each event.

    Args:
        df: DataFrame with columns:
            - eligible_templates (list)
            - selected_template (str)
        target_probs_list: list of dicts, one per row.
            Each dict maps {template: probability} under the RDS policy.
            target_probs_list[i] gives the RDS probability distribution
            for the i-th event.
        max_weight: float or None. If set, clip weights to this maximum.
                    Typical value: 10-50. Reduces variance at cost of small bias.

    Returns:
        np.array of importance weights, one per event

    EXAMPLE:
        Event: eligible=["A","B","C","D"], selected="B", reward=1

        Logging policy:  P_log("B") = 1/4 = 0.25
        RDS policy:      P_rds("B") = 0.40

        weight = 0.40 / 0.25 = 1.6

        Interpretation: our policy is 1.6x more likely to choose B than random.
        So this positive reward gets upweighted in our estimate.
    """
    n = len(df)
    weights = np.zeros(n)

    eligible_col = df["eligible_templates"].values
    selected_col = df["selected_template"].values

    for i in range(n):
        eligible = eligible_col[i]

        # Parse eligible if it's a string
        if isinstance(eligible, str):
            import ast
            eligible = ast.literal_eval(eligible)

        # Logging probability (random policy)
        p_log = compute_logging_probability(len(eligible))

        # Target probability (RDS policy)
        selected = selected_col[i]
        target_probs = target_probs_list[i]
        p_target = target_probs.get(selected, 0.0)

        # Importance weight
        if p_log > 0:
            w = p_target / p_log
        else:
            w = 0.0

        weights[i] = w

    # Optional: clip extreme weights to reduce variance
    if max_weight is not None:
        n_clipped = np.sum(weights > max_weight)
        if n_clipped > 0:
            print(f"[importance_sampling] Clipping {n_clipped:,} weights "
                  f"above {max_weight} ({n_clipped / n:.2%} of events)")
        weights = np.clip(weights, 0, max_weight)

    # Report statistics
    print(f"[importance_sampling] Weight statistics:")
    print(f"  Mean:   {weights.mean():.4f}")
    print(f"  Std:    {weights.std():.4f}")
    print(f"  Min:    {weights.min():.4f}")
    print(f"  Max:    {weights.max():.4f}")
    print(f"  Median: {np.median(weights):.4f}")

    return weights


def weighted_importance_sampling(rewards, weights):
    """
    Compute the Weighted Importance Sampling (WIS) estimate of policy value.

    This estimates: "what would the average reward be if we used our policy?"

    Args:
        rewards: np.array of observed rewards (0 or 1 for each event)
        weights: np.array of importance weights (from compute_importance_weights)

    Returns:
        float: estimated expected reward under the target policy

    THE FORMULA:
        V̂ = Σ(w_i × r_i) / Σ(w_i)

    WHY WEIGHTED (not simple)?
        Simple IS:  V̂ = (1/n) × Σ(w_i × r_i)
        Weighted IS: V̂ = Σ(w_i × r_i) / Σ(w_i)

        Weighted IS is "self-normalizing" — it divides by the sum of weights
        instead of n. This makes it more stable because:
        - If weights are all ~1 (policy similar to random): same as simple average
        - If some weights are extreme: the denominator adjusts, preventing
          a few big weights from dominating the estimate

    EXAMPLE:
        rewards = [1, 0, 1, 0, 1]
        weights = [1.5, 0.8, 2.0, 0.3, 1.2]

        numerator   = 1.5×1 + 0.8×0 + 2.0×1 + 0.3×0 + 1.2×1 = 4.7
        denominator = 1.5 + 0.8 + 2.0 + 0.3 + 1.2 = 5.8

        V̂ = 4.7 / 5.8 = 0.810

        Compare to simple average of rewards: 3/5 = 0.600

        The WIS estimate is higher because our policy upweights the events
        where the user engaged (weights 1.5, 2.0, 1.2 are bigger than
        weights 0.8, 0.3 for non-engagement events).
    """
    rewards = np.asarray(rewards, dtype=float)
    weights = np.asarray(weights, dtype=float)

    numerator = np.sum(weights * rewards)
    denominator = np.sum(weights)

    if denominator == 0:
        print("[importance_sampling] WARNING: sum of weights is 0!")
        return 0.0

    estimate = numerator / denominator

    # Also compute the "effective sample size" — how many independent
    # samples our weighted estimate is equivalent to.
    # ESS = (Σ w_i)² / Σ (w_i²)
    # If ESS is close to n, the weights are uniform and estimate is reliable.
    # If ESS is much smaller than n, a few weights dominate and estimate is noisy.
    ess = (np.sum(weights) ** 2) / np.sum(weights ** 2)

    print(f"[importance_sampling] WIS estimate: {estimate:.6f}")
    print(f"[importance_sampling] Effective sample size: {ess:,.0f} / {len(rewards):,} "
          f"({ess / len(rewards):.1%})")

    return estimate
