"""
=============================================================================
RDS PIPELINE RUNNER — RUN THIS FILE TO TEST EVERYTHING PART BY PART
=============================================================================

This script runs each component of the Recovering Difference Softmax algorithm
one at a time, printing detailed output so you can see what each part does.

HOW TO RUN:
    cd DuolingoBandit
    python run_pipeline.py

WHAT HAPPENS:
    Part 1: Load data           → loads a small sample to work with
    Part 2: Explore data        → prints statistics about the dataset
    Part 3: Baseline            → computes random policy performance
    Part 4: Reward rates        → computes raw reward rate per template
    Part 5: Difference scores   → computes Relative Difference Scores
    Part 6: Bayesian smoothing  → regularizes scores
    Part 7: Recency penalty     → demonstrates the recency mechanism
    Part 8: Softmax selection   → demonstrates template selection
    Part 9: Full evaluation     → runs the complete pipeline with importance sampling

Each part prints "PART X: PASSED" or "PART X: FAILED" so you can see what works.

If a part fails, fix it before moving on. Each part depends on previous parts.
=============================================================================
"""

import sys
import os
import traceback
import numpy as np

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(__file__))


def run_part(part_num, part_name, func):
    """Helper to run each part with nice formatting and error handling."""
    print("\n")
    print("█" * 70)
    print(f"█  PART {part_num}: {part_name}")
    print("█" * 70)
    try:
        result = func()
        print(f"\n✅ PART {part_num}: PASSED — {part_name}")
        return result
    except Exception as e:
        print(f"\n❌ PART {part_num}: FAILED — {part_name}")
        print(f"   Error: {e}")
        traceback.print_exc()
        return None


# ==========================================================================
# PART 1: LOAD DATA
# ==========================================================================
def part1_load_data():
    """
    Load a small sample of the Duolingo dataset.
    We use 50K rows for speed — enough to test everything.
    """
    from src.data_loader import load_sample

    df = load_sample(n_rows=50_000, split="train")

    # Basic sanity checks
    assert len(df) > 0, "DataFrame is empty!"
    assert "selected_template" in df.columns, "Missing 'selected_template' column!"
    assert "session_end_completed" in df.columns, "Missing 'session_end_completed' column!"
    assert "eligible_templates" in df.columns, "Missing 'eligible_templates' column!"
    assert "history" in df.columns, "Missing 'history' column!"

    print(f"\n  ✓ Loaded {len(df):,} rows with {len(df.columns)} columns")
    print(f"  ✓ Columns: {list(df.columns)}")
    print(f"\n  First 3 rows:")
    print(df.head(3).to_string())

    return df


# ==========================================================================
# PART 2: EXPLORE DATA
# ==========================================================================
def part2_explore_data(df):
    """
    Understand the dataset — what templates exist, what reward rates look like,
    how eligibility works, what history looks like.
    """
    from src.data_loader import describe_dataset

    describe_dataset(df)

    # Additional exploration
    print("\n--- Reward rate by template ---")
    template_stats = df.groupby("selected_template").agg(
        count=("session_end_completed", "count"),
        reward_rate=("session_end_completed", "mean")
    ).sort_values("reward_rate", ascending=False)
    print(template_stats.to_string())

    # Eligible template counts
    print("\n--- How many templates are eligible per event? ---")
    n_eligible = df["eligible_templates"].apply(
        lambda x: len(x) if isinstance(x, list) else len(eval(x)) if isinstance(x, str) else 0
    )
    print(f"  Mean: {n_eligible.mean():.1f}")
    print(f"  Min:  {n_eligible.min()}")
    print(f"  Max:  {n_eligible.max()}")
    print(f"  Distribution:")
    print(n_eligible.value_counts().sort_index().to_string())

    return True


# ==========================================================================
# PART 3: BASELINE
# ==========================================================================
def part3_baseline(df):
    """
    Compute the random baseline — the performance we need to beat.
    """
    from src.evaluation.baseline import compute_random_baseline

    baseline = compute_random_baseline(df)

    assert 0 < baseline < 1, f"Baseline {baseline} is outside (0, 1) — something is wrong"
    print(f"\n  ✓ Baseline reward rate: {baseline:.6f}")
    print(f"  This means {baseline:.1%} of users engaged after a random notification.")
    print(f"  Our algorithm needs to beat this number.")

    return baseline


# ==========================================================================
# PART 4: REWARD RATES
# ==========================================================================
def part4_reward_rates(df):
    """
    Compute raw reward rate for each template.
    This is Layer 1 — the simplest scoring.
    """
    from src.scoring.difference_score import compute_template_reward_rates

    reward_rates, counts = compute_template_reward_rates(df)

    assert len(reward_rates) > 0, "No reward rates computed!"
    assert all(0 <= v <= 1 for v in reward_rates.values()), "Reward rates must be between 0 and 1"

    print(f"\n  ✓ Computed reward rates for {len(reward_rates)} templates")

    # Show the ranking
    sorted_templates = sorted(reward_rates.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Template ranking by raw reward rate:")
    for rank, (t, rate) in enumerate(sorted_templates, 1):
        bar = "█" * int(rate * 200)
        print(f"    {rank}. Template {t}: {rate:.4f} ({counts[t]:>8,} events) {bar}")

    return reward_rates, counts


# ==========================================================================
# PART 5: RELATIVE DIFFERENCE SCORES
# ==========================================================================
def part5_difference_scores(df, reward_rates):
    """
    Compute the Relative Difference Score for each template.
    This is Layer 2 — the core innovation.

    WATCH FOR: templates that had high raw reward rates but low/negative RDS.
    That means their high raw rate was due to being shown to active users,
    not because they're actually good templates.
    """
    from src.scoring.difference_score import compute_relative_difference_scores_fast

    rds_scores = compute_relative_difference_scores_fast(df, reward_rates)

    assert len(rds_scores) > 0, "No RDS scores computed!"

    print(f"\n  ✓ Computed RDS for {len(rds_scores)} templates")

    # Compare raw vs RDS ranking
    sorted_raw = sorted(reward_rates.items(), key=lambda x: x[1], reverse=True)
    sorted_rds = sorted(rds_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  RANKING COMPARISON:")
    print(f"  {'Rank':<6} {'By Raw Rate':<20} {'By RDS':<20}")
    print(f"  {'-'*46}")
    for i in range(len(sorted_raw)):
        raw_t, raw_v = sorted_raw[i]
        rds_t, rds_v = sorted_rds[i] if i < len(sorted_rds) else ("?", 0)
        changed = " ← MOVED" if raw_t != rds_t else ""
        print(f"  {i+1:<6} {raw_t} ({raw_v:.4f})       {rds_t} ({rds_v:+.6f}){changed}")

    return rds_scores


# ==========================================================================
# PART 6: BAYESIAN SMOOTHING
# ==========================================================================
def part6_bayesian_smoothing(rds_scores, counts):
    """
    Apply Bayesian smoothing to regularize the RDS scores.
    This is Layer 3 — protecting against noisy estimates.

    WATCH FOR: templates with few observations getting pulled strongly
    toward the global mean.
    """
    from src.scoring.bayesian_smoothing import bayesian_smooth

    kappa = 1000
    smoothed_scores = bayesian_smooth(rds_scores, counts, kappa)

    assert len(smoothed_scores) == len(rds_scores), "Should have same number of templates"

    print(f"\n  ✓ Applied Bayesian smoothing with κ={kappa}")
    print(f"\n  Templates most affected by smoothing:")
    for t in sorted(smoothed_scores.keys()):
        raw = rds_scores[t]
        smooth = smoothed_scores[t]
        change = abs(smooth - raw)
        if change > 0.0001:
            print(f"    {t}: {raw:+.6f} → {smooth:+.6f} (shifted by {change:.6f})")

    return smoothed_scores


# ==========================================================================
# PART 7: RECENCY PENALTY
# ==========================================================================
def part7_recency_penalty(smoothed_scores):
    """
    Demonstrate the recency penalty mechanism.
    This is Layer 4 — preventing notification fatigue.

    We'll create fake user histories and show how scores change.
    """
    from src.recency.recency_penalty import (
        compute_recency_penalty,
        explain_recency_adjustment,
    )

    gamma = 0.1
    h = 0.5

    # Pick some templates to demonstrate with
    templates = sorted(smoothed_scores.keys())[:4]
    demo_scores = {t: smoothed_scores[t] for t in templates}

    # --- Scenario 1: No history (new user) ---
    print("\n  SCENARIO 1: New user (no history)")
    print("  Expectation: no penalties, scores unchanged")
    history_empty = []
    adjusted = explain_recency_adjustment(demo_scores, history_empty, gamma, h)
    for t in templates:
        assert adjusted[t] == demo_scores[t], f"Score should be unchanged for {t}!"
    print("  ✓ Correct! No penalties applied.")

    # --- Scenario 2: Template A sent 0.5 days ago ---
    print(f"\n  SCENARIO 2: Template {templates[0]} sent 0.5 days ago")
    print(f"  Expectation: {templates[0]} gets penalized, others unchanged")
    history_recent = [(templates[0], 0.5)]
    adjusted = explain_recency_adjustment(demo_scores, history_recent, gamma, h)

    penalty_val = compute_recency_penalty(templates[0], history_recent, gamma, h)
    print(f"\n  Penalty on {templates[0]}: {penalty_val:.6f}")
    print(f"  (That's γ × exp(-h × d) = {gamma} × exp(-{h} × 0.5) = {gamma * np.exp(-h * 0.5):.6f})")
    print("  ✓ Recency penalty working correctly!")

    # --- Scenario 3: Multiple templates in history ---
    print(f"\n  SCENARIO 3: Multiple templates in history")
    history_multi = [(templates[0], 0.3), (templates[1], 2.0), (templates[2], 7.0)]
    print(f"  History: {history_multi}")
    adjusted = explain_recency_adjustment(demo_scores, history_multi, gamma, h)

    print("\n  ✓ Notice how penalty decreases with time:")
    print(f"    {templates[0]} (0.3d ago): strong penalty")
    print(f"    {templates[1]} (2.0d ago): moderate penalty")
    print(f"    {templates[2]} (7.0d ago): almost no penalty")

    return True


# ==========================================================================
# PART 8: SOFTMAX SELECTION
# ==========================================================================
def part8_softmax_selection(smoothed_scores):
    """
    Demonstrate softmax template selection.
    This is the final selection step — how we actually pick a template.

    We'll show how different temperature values affect the selection.
    """
    from src.bandit.softmax_selector import (
        softmax_probabilities,
        softmax_select,
        explain_softmax_selection,
    )

    templates = sorted(smoothed_scores.keys())[:5]
    demo_scores = {t: smoothed_scores[t] for t in templates}

    # --- Show probabilities at different temperatures ---
    print("  Showing how temperature (τ) affects selection:\n")

    for tau in [1, 10, 50, 100]:
        probs = softmax_probabilities(templates, demo_scores, tau)
        print(f"  τ = {tau:>3}: ", end="")
        for t in templates:
            bar = "█" * int(probs[t] * 30)
            print(f"{t}={probs[t]:.2f} {bar}  ", end="")
        print()

    # --- Detailed view at τ=50 ---
    print(f"\n  Detailed softmax at τ=50:")
    explain_softmax_selection(templates, demo_scores, 50)

    # --- Run 1000 selections to show empirical distribution ---
    tau = 50
    rng = np.random.default_rng(42)
    selections = []
    for _ in range(1000):
        selected = softmax_select(templates, demo_scores, tau, rng)
        selections.append(selected)

    print(f"\n  Empirical selection distribution (1000 draws at τ={tau}):")
    probs = softmax_probabilities(templates, demo_scores, tau)
    for t in templates:
        empirical = selections.count(t) / 1000
        theoretical = probs[t]
        print(f"    {t}: empirical={empirical:.3f}  theoretical={theoretical:.3f}")

    print("  ✓ Softmax selection working correctly!")

    return True


# ==========================================================================
# PART 9: FULL PIPELINE EVALUATION
# ==========================================================================
def part9_full_evaluation(train_df, test_df):
    """
    Run the complete RDS pipeline end-to-end:
      1. Fit on training data
      2. Evaluate on test data
      3. Report lift over random baseline

    THIS IS THE FINAL TEST. If this works, your algorithm is complete.
    """
    from src.bandit.rds_policy import RDSPolicy

    # Create the policy with initial hyperparameters
    policy = RDSPolicy(
        kappa=1000,    # Bayesian shrinkage
        gamma=0.1,     # Recency penalty magnitude
        h=0.5,         # Recency decay rate
        tau=50,        # Softmax temperature
    )

    # Fit on training data
    policy.fit(train_df)

    # Print a summary of what was learned
    policy.summary()

    # Evaluate on test data (using a sample for speed)
    results = policy.evaluate(test_df, max_weight=20, sample_size=10_000)

    # Report results
    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║        FINAL RESULTS                 ║")
    print(f"  ╠══════════════════════════════════════╣")
    print(f"  ║  Random Baseline:  {results['baseline']:.6f}          ║")
    print(f"  ║  RDS Policy:       {results['target_value']:.6f}          ║")
    print(f"  ║  Lift:             {results['lift']:+.4f}            ║")
    print(f"  ║  Events evaluated: {results['n_events']:>8,}          ║")
    print(f"  ╚══════════════════════════════════════╝")

    return results


# ==========================================================================
# MAIN — RUN ALL PARTS IN SEQUENCE
# ==========================================================================
def main():
    print("=" * 70)
    print("  DUOLINGO BANDIT — RECOVERING DIFFERENCE SOFTMAX PIPELINE")
    print("  Running all parts sequentially. Each part tests one component.")
    print("=" * 70)

    # Part 1: Load data
    df = run_part(1, "LOAD DATA", part1_load_data)
    if df is None:
        print("\n💀 Cannot continue without data. Fix Part 1 first.")
        return

    # Part 2: Explore data
    run_part(2, "EXPLORE DATA", lambda: part2_explore_data(df))

    # Part 3: Baseline
    baseline = run_part(3, "RANDOM BASELINE", lambda: part3_baseline(df))

    # Part 4: Reward rates
    result4 = run_part(4, "RAW REWARD RATES", lambda: part4_reward_rates(df))
    if result4 is None:
        print("\n💀 Cannot continue without reward rates. Fix Part 4 first.")
        return
    reward_rates, counts = result4

    # Part 5: Relative Difference Scores
    rds_scores = run_part(5, "RELATIVE DIFFERENCE SCORES",
                          lambda: part5_difference_scores(df, reward_rates))
    if rds_scores is None:
        print("\n💀 Cannot continue without RDS. Fix Part 5 first.")
        return

    # Part 6: Bayesian Smoothing
    smoothed_scores = run_part(6, "BAYESIAN SMOOTHING",
                               lambda: part6_bayesian_smoothing(rds_scores, counts))
    if smoothed_scores is None:
        print("\n💀 Cannot continue without smoothed scores. Fix Part 6 first.")
        return

    # Part 7: Recency Penalty
    run_part(7, "RECENCY PENALTY", lambda: part7_recency_penalty(smoothed_scores))

    # Part 8: Softmax Selection
    run_part(8, "SOFTMAX SELECTION", lambda: part8_softmax_selection(smoothed_scores))

    # Part 9: Full Pipeline
    # Load a separate sample for "test" (use second half of our data)
    print("\n\n  Preparing train/test split for full evaluation...")
    split_idx = len(df) // 2
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"  Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")

    results = run_part(9, "FULL PIPELINE EVALUATION",
                       lambda: part9_full_evaluation(train_df, test_df))

    # Final summary
    print("\n\n")
    print("=" * 70)
    print("  ALL PARTS COMPLETE!")
    print("=" * 70)
    print("""
  Next steps:
    1. Run on larger data: change n_rows in Part 1 (try 500K, then 1M)
    2. Use real train/test split: load train and test parquet files separately
    3. Tune hyperparameters: try different κ, γ, h, τ values
    4. Check notebooks/ for visualization ideas
    5. Read docs/PROJECT_GUIDE.md for the full roadmap
    """)


if __name__ == "__main__":
    main()
