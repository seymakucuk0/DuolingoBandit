"""
RDS POLICY — THE FULL PIPELINE
===============================
This module wires together all the components into one coherent policy.

The pipeline is:
  1. FIT (on training data):
     - Compute per-template reward rates
     - Compute Relative Difference Scores
     - Apply Bayesian smoothing
     → Result: a smoothed score for each template (stored in self.smoothed_scores)

  2. SELECT (for each event):
     - Start with smoothed scores
     - Apply recency penalty based on user's history
     - Apply softmax selection over eligible templates
     → Result: a selected template (or probability distribution)

  3. EVALUATE (on test data):
     - For each test event, compute what probability our policy assigns
       to the template that was actually sent
     - Compute importance weights
     - Use Weighted Importance Sampling to estimate our policy's reward
     - Compare against random baseline
     → Result: estimated lift over random

This class is the main thing you'll interact with.
"""

import numpy as np
from tqdm import tqdm

# Import all our components
from src.scoring.difference_score import (
    compute_template_reward_rates,
    compute_relative_difference_scores_fast,
)
from src.scoring.bayesian_smoothing import bayesian_smooth
from src.recency.recency_penalty import adjust_scores_with_recency
from src.bandit.softmax_selector import softmax_probabilities, softmax_select
from src.evaluation.baseline import compute_random_baseline, compute_lift
from src.evaluation.importance_sampling import (
    compute_importance_weights,
    weighted_importance_sampling,
)


class RDSPolicy:
    """
    The Recovering Difference Softmax policy.

    USAGE:
        # Step 1: Create with hyperparameters
        policy = RDSPolicy(kappa=1000, gamma=0.1, h=0.5, tau=50)

        # Step 2: Train on training data
        policy.fit(train_df)

        # Step 3: Evaluate on test data
        results = policy.evaluate(test_df)

        # Step 4: Look at results
        print(f"Lift over random: {results['lift']:.2%}")

    HYPERPARAMETERS:
        kappa (float): Bayesian shrinkage strength. Higher = more conservative.
                       Range: 100-10000. Start with 1000.

        gamma (float): Recency penalty magnitude. Higher = stronger "don't repeat" pressure.
                       Range: 0.01-0.3. Start with 0.1.

        h (float):     Recency decay rate. Higher = penalty fades faster.
                       Range: 0.1-2.0. Start with 0.5.

        tau (float):   Softmax temperature. Higher = more exploitation.
                       Range: 1-100. Start with 50.
    """

    def __init__(self, kappa=1000, gamma=0.1, h=0.5, tau=50):
        self.kappa = kappa
        self.gamma = gamma
        self.h = h
        self.tau = tau

        # These are set during fit()
        self.reward_rates = None
        self.rds_scores = None
        self.counts = None
        self.smoothed_scores = None
        self.is_fitted = False

    def fit(self, df):
        """
        Learn template scores from training data.

        This runs three steps:
          1. Compute raw reward rates per template
          2. Compute Relative Difference Scores
          3. Apply Bayesian smoothing

        The result is self.smoothed_scores — a dict mapping each template
        to its smoothed score. These scores are GLOBAL (same for all users).
        The per-user personalization happens in select_template() via
        the recency penalty.

        Args:
            df: training DataFrame
        """
        print("=" * 60)
        print("FITTING RDS POLICY")
        print("=" * 60)

        # Step 1: Raw reward rates
        print("\n--- Step 1: Computing per-template reward rates ---")
        self.reward_rates, self.counts = compute_template_reward_rates(df)

        # Step 2: Relative Difference Scores
        print("\n--- Step 2: Computing Relative Difference Scores ---")
        self.rds_scores = compute_relative_difference_scores_fast(df, self.reward_rates)

        # Step 3: Bayesian Smoothing
        print("\n--- Step 3: Applying Bayesian Smoothing ---")
        self.smoothed_scores = bayesian_smooth(self.rds_scores, self.counts, self.kappa)

        self.is_fitted = True
        print("\n✓ Policy fitted successfully!")
        print(f"  Templates learned: {sorted(self.smoothed_scores.keys())}")

    def get_probabilities(self, eligible_templates, history):
        """
        Compute the probability distribution over eligible templates
        for a specific event.

        This is the core decision function:
          1. Start with smoothed scores (from training)
          2. Adjust for recency (based on this user's history)
          3. Apply softmax over eligible templates only

        Args:
            eligible_templates: list of eligible template names
            history: list of (template, days_ago) tuples

        Returns:
            dict {template: probability}
        """
        if not self.is_fitted:
            raise RuntimeError("Policy not fitted! Call fit() first.")

        # Step 1: Get smoothed scores for eligible templates only
        eligible_scores = {
            t: self.smoothed_scores.get(t, 0.0)
            for t in eligible_templates
        }

        # Step 2: Apply recency penalty
        adjusted_scores = adjust_scores_with_recency(
            eligible_scores, history, self.gamma, self.h
        )

        # Step 3: Softmax selection probabilities
        probs = softmax_probabilities(eligible_templates, adjusted_scores, self.tau)

        return probs

    def select_template(self, eligible_templates, history, rng=None):
        """
        Select a template for this specific event.

        Args:
            eligible_templates: list of eligible template names
            history: list of (template, days_ago) tuples
            rng: random number generator (for reproducibility)

        Returns:
            str: selected template name
        """
        if not self.is_fitted:
            raise RuntimeError("Policy not fitted! Call fit() first.")

        eligible_scores = {
            t: self.smoothed_scores.get(t, 0.0)
            for t in eligible_templates
        }

        adjusted_scores = adjust_scores_with_recency(
            eligible_scores, history, self.gamma, self.h
        )

        return softmax_select(eligible_templates, adjusted_scores, self.tau, rng)

    def evaluate(self, df, max_weight=20, sample_size=None):
        """
        Evaluate the policy on test data using Weighted Importance Sampling.

        This is the final assessment: "How much better is our algorithm
        than random template selection?"

        Args:
            df: test DataFrame
            max_weight: clip importance weights to this maximum (reduces variance)
            sample_size: if set, evaluate on a random sample (for speed)

        Returns:
            dict with keys:
                'baseline': float, random policy reward rate
                'target_value': float, estimated RDS policy reward rate
                'lift': float, relative improvement
                'n_events': int, number of events evaluated
        """
        if not self.is_fitted:
            raise RuntimeError("Policy not fitted! Call fit() first.")

        print("=" * 60)
        print("EVALUATING RDS POLICY")
        print("=" * 60)

        # Optionally sample for speed
        if sample_size is not None and sample_size < len(df):
            print(f"\n[evaluate] Sampling {sample_size:,} events from {len(df):,}...")
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        # Step 1: Compute baseline
        print("\n--- Step 1: Computing random baseline ---")
        baseline = compute_random_baseline(df)

        # Step 2: For each event, compute our policy's probabilities
        print(f"\n--- Step 2: Computing RDS probabilities for {len(df):,} events ---")
        target_probs_list = []

        eligible_col = df["eligible_templates"].values
        history_col = df["history"].values

        for i in tqdm(range(len(df)), desc="Computing probabilities"):
            eligible = eligible_col[i]
            history = history_col[i]

            # Parse eligible_templates if needed
            if isinstance(eligible, str):
                import ast
                eligible = ast.literal_eval(eligible)
            elif hasattr(eligible, 'tolist'):
                eligible = eligible.tolist()  # convert numpy array to list

            # Parse history if needed
            if history is None:
                history = []
            elif isinstance(history, str):
                import ast
                history = ast.literal_eval(history)
            elif hasattr(history, 'tolist'):
                history = history.tolist()  # convert numpy array to list

            probs = self.get_probabilities(eligible, history)
            target_probs_list.append(probs)

        # Step 3: Compute importance weights
        print("\n--- Step 3: Computing importance weights ---")
        weights = compute_importance_weights(df, target_probs_list, max_weight=max_weight)

        # Step 4: Weighted Importance Sampling estimate
        print("\n--- Step 4: Computing WIS estimate ---")
        rewards = df["session_end_completed"].values
        target_value = weighted_importance_sampling(rewards, weights)

        # Step 5: Compute lift
        print("\n--- Step 5: Computing lift ---")
        lift = compute_lift(target_value, baseline)

        results = {
            "baseline": baseline,
            "target_value": target_value,
            "lift": lift,
            "n_events": len(df),
            "hyperparameters": {
                "kappa": self.kappa,
                "gamma": self.gamma,
                "h": self.h,
                "tau": self.tau,
            },
        }

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)

        return results

    def summary(self):
        """Print a summary of the fitted policy."""
        if not self.is_fitted:
            print("Policy not yet fitted. Call fit() with training data.")
            return

        print("\nRDS Policy Summary")
        print("-" * 40)
        print(f"Hyperparameters:")
        print(f"  κ (kappa) = {self.kappa}  [shrinkage]")
        print(f"  γ (gamma) = {self.gamma}  [recency penalty magnitude]")
        print(f"  h         = {self.h}  [recency decay rate]")
        print(f"  τ (tau)   = {self.tau}  [softmax temperature]")
        print(f"\nLearned template scores:")
        for t in sorted(self.smoothed_scores.keys()):
            raw = self.rds_scores.get(t, 0)
            smooth = self.smoothed_scores[t]
            count = self.counts.get(t, 0)
            print(f"  {t}: smoothed={smooth:+.6f}  (raw={raw:+.6f}, n={count:,})")
