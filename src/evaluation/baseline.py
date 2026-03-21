"""
BASELINE EVALUATION
===================
This module computes the baseline performance of the random policy.

WHY WE NEED A BASELINE:
  To know if our algorithm is any good, we need something to compare against.
  The simplest baseline is the RANDOM POLICY — picking a template uniformly
  at random from the eligible set.

  Lucky for us, the Duolingo dataset WAS collected under a random policy.
  That means the average reward in the dataset IS the random baseline.

  If our RDS algorithm achieves a higher estimated reward (via importance
  sampling), then we know it's better than random.

  The key metric is LIFT:
    lift = (algorithm_reward - baseline_reward) / baseline_reward

  Duolingo reported ~1.9% lift in their paper. We're aiming for similar.
"""


def compute_random_baseline(df):
    """
    Compute the expected reward under a random selection policy.

    Since the data was collected under random selection, the observed
    average reward IS the random policy's performance.

    Args:
        df: DataFrame with 'session_end_completed' column

    Returns:
        float: average reward rate

    Example:
        If 25% of notifications led to session completion:
        baseline = 0.25
    """
    baseline = df["session_end_completed"].mean()
    total = len(df)
    engaged = df["session_end_completed"].sum()

    print(f"[baseline] Random policy performance:")
    print(f"  Total notifications: {total:,}")
    print(f"  User engaged: {engaged:,}")
    print(f"  Baseline reward rate: {baseline:.6f} ({baseline:.2%})")

    return baseline


def compute_lift(target_value, baseline_value):
    """
    Compute the relative lift of the target policy over baseline.

    Args:
        target_value: float, estimated reward of the target (RDS) policy
        baseline_value: float, reward of the baseline (random) policy

    Returns:
        float: relative lift (e.g., 0.019 means 1.9% improvement)

    Example:
        baseline = 0.2500
        target   = 0.2548
        lift     = (0.2548 - 0.2500) / 0.2500 = 0.0192 = 1.92%
    """
    if baseline_value == 0:
        return 0.0

    lift = (target_value - baseline_value) / baseline_value

    print(f"\n[baseline] RESULTS:")
    print(f"  Random baseline:  {baseline_value:.6f}")
    print(f"  RDS policy:       {target_value:.6f}")
    print(f"  Absolute gain:    {target_value - baseline_value:+.6f}")
    print(f"  Relative lift:    {lift:+.2%}")

    if lift > 0:
        print(f"  → Algorithm OUTPERFORMS random by {lift:.2%}")
    elif lift < 0:
        print(f"  → Algorithm UNDERPERFORMS random by {abs(lift):.2%}")
    else:
        print(f"  → Algorithm performs SAME as random")

    return lift
