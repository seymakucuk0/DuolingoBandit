"""
BAYESIAN SMOOTHING (Layer 3)
============================
This module regularizes the Relative Difference Scores using Bayesian shrinkage.

THE PROBLEM:
  Imagine template K was only sent 50 times (because it's rarely eligible).
  Its RDS might be +0.15, which looks amazing. But with only 50 data points,
  that score is very noisy — it could easily be +0.15 by pure luck.

  Meanwhile, template A was sent 10,000,000 times and has an RDS of +0.02.
  That score is very reliable — 10M data points is a lot of evidence.

  If we trust the raw scores, we'd heavily favor template K. But we shouldn't,
  because we're not confident in K's score.

THE SOLUTION — BAYESIAN SHRINKAGE:
  We "pull" every template's score toward the global average. Templates with
  lots of data barely move. Templates with little data get pulled strongly.

  Formula:
    smoothed(a) = (n_a × RDS(a) + κ × μ) / (n_a + κ)

  Where:
    n_a = number of observations for template a
    RDS(a) = raw relative difference score
    κ (kappa) = shrinkage strength (hyperparameter, e.g., 1000)
    μ = global weighted mean of all RDS scores

  INTUITION:
    - If n_a is HUGE (like 10M): smoothed(a) ≈ RDS(a)  [data dominates]
    - If n_a is TINY (like 50):  smoothed(a) ≈ μ        [prior dominates]
    - κ controls the crossover point: when n_a = κ, the score is exactly
      halfway between the data and the prior.
"""

import numpy as np


def compute_global_mean(rds_scores, counts):
    """
    Compute the weighted global mean of RDS scores.

    We weight by count because templates with more observations should
    contribute more to the global average.

    Args:
        rds_scores: dict {template: rds_score}
        counts: dict {template: count}

    Returns:
        float: weighted average of RDS scores

    Example:
        scores = {"A": 0.02, "B": -0.01}
        counts = {"A": 1000000, "B": 500000}
        global_mean = (1000000 * 0.02 + 500000 * -0.01) / (1000000 + 500000)
                    = (20000 - 5000) / 1500000
                    = 0.01
    """
    total_weighted = 0.0
    total_count = 0

    for t in rds_scores:
        if t in counts:
            total_weighted += counts[t] * rds_scores[t]
            total_count += counts[t]

    if total_count == 0:
        return 0.0

    return total_weighted / total_count


def bayesian_smooth(rds_scores, counts, kappa):
    """
    Apply Bayesian shrinkage to RDS scores.

    Args:
        rds_scores: dict {template: rds_score}
        counts: dict {template: number_of_times_sent}
        kappa: float, shrinkage strength
               - Higher kappa = more conservative (scores pulled more toward mean)
               - Lower kappa = more aggressive (trust the data more)
               - Typical range: 100 to 10000

    Returns:
        smoothed_scores: dict {template: smoothed_score}

    Example:
        rds_scores = {"A": 0.05, "B": -0.02, "K": 0.15}
        counts = {"A": 5000000, "B": 3000000, "K": 50}
        kappa = 1000

        global_mean ≈ 0.02 (weighted average)

        smoothed("A") = (5000000 * 0.05 + 1000 * 0.02) / (5000000 + 1000)
                      ≈ 0.05   [barely changed — lots of data]

        smoothed("K") = (50 * 0.15 + 1000 * 0.02) / (50 + 1000)
                      ≈ 0.026  [pulled strongly toward mean — very little data]
    """
    # Step 1: compute the global mean (the "prior")
    global_mean = compute_global_mean(rds_scores, counts)
    print(f"[bayesian_smoothing] Global weighted mean RDS: {global_mean:+.6f}")
    print(f"[bayesian_smoothing] Kappa (shrinkage strength): {kappa}")

    # Step 2: apply shrinkage to each template
    smoothed_scores = {}
    for t in sorted(rds_scores.keys()):
        n = counts.get(t, 0)
        raw = rds_scores[t]

        # The Bayesian shrinkage formula
        smoothed = (n * raw + kappa * global_mean) / (n + kappa)

        smoothed_scores[t] = smoothed

        # Show how much each score was adjusted
        change = smoothed - raw
        direction = "←shrunk" if abs(change) > 0.0001 else "≈same"
        print(f"  Template {t}: raw={raw:+.6f} → smoothed={smoothed:+.6f} "
              f"(n={n:,}, {direction})")

    return smoothed_scores
