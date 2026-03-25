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
from collections import defaultdict
from tqdm import tqdm

# Import all our components
from src.scoring.difference_score import (
    compute_template_reward_rates,
    compute_counterfactual_baseline,
    compute_relative_difference_scores_fast,
    compute_rds_paper,
    compute_rds_paper_chunked_pass1,
    merge_rds_accumulators,
)
from src.scoring.bayesian_smoothing import bayesian_smooth
from src.recency.recency_penalty import adjust_scores_with_recency
from src.bandit.softmax_selector import softmax_probabilities, softmax_select
from src.evaluation.baseline import compute_random_baseline, compute_lift
from src.evaluation.importance_sampling import (
    compute_logging_probability,
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

    def __init__(self, kappa=1000, gamma=0.017, h=15, tau=0.0025,
                 use_smoothing=False, use_argmax=True):
        """
        Args:
            kappa: Bayesian shrinkage strength (only used if use_smoothing=True)
            gamma: Recency penalty magnitude (paper default: 0.017)
            h: Recency half-life in days (paper default: 15)
            tau: Softmax temperature (paper convention: score/tau, smaller=greedier)
                 Set use_argmax=True to ignore tau and use greedy selection.
            use_smoothing: If False (default for offline eval), skip Bayesian smoothing.
                           Paper Section 3.1: "we did not use empirical Bayes (sigma=0)"
            use_argmax: If True (default), use greedy argmax for evaluation.
                        Paper Section 3.1: "we used argmax instead of softmax"
        """
        self.kappa = kappa
        self.gamma = gamma
        self.h = h
        self.tau = tau
        self.use_smoothing = use_smoothing
        self.use_argmax = use_argmax

        # These are set during fit()
        self.reward_rates = None
        self.rds_scores = None
        self.counts = None
        self.smoothed_scores = None
        self.is_fitted = False

        # Language-specific scores (set by fit_chunked_by_language)
        self.lang_smoothed_scores = {}  # {lang: {template: score}}
        self.use_language = False

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

    def get_probabilities(self, eligible_templates, history, language=None):
        """
        Compute the probability distribution over eligible templates
        for a specific event.

        Args:
            eligible_templates: list of eligible template names
            history: list of (template, days_ago) tuples
            language: optional ui_language string. If provided and language-
                      specific scores exist, uses those instead of global.

        Returns:
            dict {template: probability}
        """
        if not self.is_fitted:
            raise RuntimeError("Policy not fitted! Call fit() first.")

        # Pick language-specific or global scores
        if self.use_language and language and language in self.lang_smoothed_scores:
            scores = self.lang_smoothed_scores[language]
        else:
            scores = self.smoothed_scores

        eligible_scores = {t: scores.get(t, 0.0) for t in eligible_templates}

        adjusted_scores = adjust_scores_with_recency(
            eligible_scores, history, self.gamma, self.h
        )

        if self.use_argmax:
            # Greedy: probability 1.0 for best template, 0.0 for others
            best = max(eligible_templates, key=lambda t: adjusted_scores.get(t, 0.0))
            return {t: (1.0 if t == best else 0.0) for t in eligible_templates}
        else:
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

    def fit_chunked(self, split="train", chunk_size=1_000_000):
        """
        Fit using paper-accurate RDS formula via chunked streaming.

        Single pass: reads eligible_templates + selected_template + reward,
        accumulates importance-weighted mu_plus/mu_minus per template.

        Args:
            split: "train" or "test"
            chunk_size: rows per chunk (default 1M)
        """
        from src.data_loader import iter_parquet_chunks

        print("=" * 60)
        print("FITTING RDS POLICY (paper formula, chunked)")
        print("=" * 60)

        # ---- Single pass: compute paper RDS via chunked accumulators ----
        print("\n--- Computing RDS scores (paper formula) ---")
        accumulators = []

        for chunk in iter_parquet_chunks(split, chunk_size,
                                          columns=["eligible_templates", "selected_template",
                                                   "session_end_completed"],
                                          parse_eligible=True, parse_hist=False):
            acc = compute_rds_paper_chunked_pass1(chunk)
            accumulators.append(acc)
            del chunk

        self.rds_scores, self.counts = merge_rds_accumulators(accumulators)

        # Compute reward rates from the accumulators
        self.reward_rates = {}
        total_plus_wr = defaultdict(float)
        total_plus_w = defaultdict(float)
        for acc in accumulators:
            for t, v in acc["plus_wr_sum"].items():
                total_plus_wr[t] += v
            for t, v in acc["plus_w_sum"].items():
                total_plus_w[t] += v
        for t in total_plus_w:
            self.reward_rates[t] = total_plus_wr[t] / total_plus_w[t] if total_plus_w[t] > 0 else 0.0

        # ---- Bayesian Smoothing (optional — paper disables for offline eval) ----
        if self.use_smoothing:
            print("\n--- Applying Bayesian Smoothing ---")
            self.smoothed_scores = bayesian_smooth(
                self.rds_scores, self.counts, self.kappa
            )
        else:
            print("\n--- Smoothing DISABLED (paper offline mode: sigma=0) ---")
            self.smoothed_scores = dict(self.rds_scores)

        self.is_fitted = True
        print("\n✓ Policy fitted successfully!")
        print(f"  Templates learned: {sorted(self.smoothed_scores.keys())}")

    def fit_chunked_by_language(self, split="train", chunk_size=1_000_000,
                                 min_lang_count=50_000):
        """
        Fit language-specific RDS scores using paper-accurate formula.

        Single pass: for each chunk, split by language and accumulate
        importance-weighted mu_plus/mu_minus per (language, template).
        """
        from src.data_loader import iter_parquet_chunks

        print("=" * 60)
        print("FITTING RDS POLICY BY LANGUAGE (paper formula, chunked)")
        print("=" * 60)

        # Accumulators per language and global
        lang_accs = defaultdict(list)  # lang → list of chunk accumulators
        global_accs = []

        print("\n--- Computing per-language RDS (paper formula) ---")
        for chunk in iter_parquet_chunks(split, chunk_size,
                columns=["ui_language", "eligible_templates",
                         "selected_template", "session_end_completed"],
                parse_eligible=True, parse_hist=False):

            # Global accumulator
            global_accs.append(compute_rds_paper_chunked_pass1(chunk))

            # Per-language accumulators
            for lang, sub in chunk.groupby("ui_language"):
                sub_reset = sub.reset_index(drop=True)
                lang_accs[lang].append(compute_rds_paper_chunked_pass1(sub_reset))

            del chunk

        # Merge global
        print("\nGlobal scores:")
        self.rds_scores, self.counts = merge_rds_accumulators(global_accs)

        # Determine qualified languages
        lang_totals = {}
        for lang, accs in lang_accs.items():
            total = sum(sum(a["plus_count"].values()) for a in accs)
            lang_totals[lang] = total

        qualified = {l for l, n in lang_totals.items() if n >= min_lang_count}
        total_q = sum(lang_totals[l] for l in qualified)
        total_all = sum(lang_totals.values())
        print(f"\n[fit] {len(qualified)} languages qualify (>= {min_lang_count:,} events)")
        print(f"[fit] Coverage: {total_q:,} / {total_all:,} ({total_q/total_all:.1%})")

        # Merge per-language
        lang_rds = {}
        lang_counts = {}
        for lang in sorted(qualified):
            n = lang_totals[lang]
            print(f"\nLanguage '{lang}' ({n:,} events):")
            scores, cnts = merge_rds_accumulators(lang_accs[lang])
            lang_rds[lang] = scores
            lang_counts[lang] = cnts

        # Compute reward rates from global accumulators
        self.reward_rates = {}
        total_plus_wr = defaultdict(float)
        total_plus_w = defaultdict(float)
        for acc in global_accs:
            for t, v in acc["plus_wr_sum"].items():
                total_plus_wr[t] += v
            for t, v in acc["plus_w_sum"].items():
                total_plus_w[t] += v
        for t in total_plus_w:
            self.reward_rates[t] = total_plus_wr[t] / total_plus_w[t] if total_plus_w[t] > 0 else 0.0

        # ---- Bayesian smoothing (optional) ----
        if self.use_smoothing:
            print("\n--- Applying Bayesian Smoothing ---")
            self.smoothed_scores = bayesian_smooth(
                self.rds_scores, self.counts, self.kappa)
            self.lang_smoothed_scores = {}
            for lang in sorted(qualified):
                self.lang_smoothed_scores[lang] = bayesian_smooth(
                    lang_rds[lang], lang_counts[lang], self.kappa)
        else:
            print("\n--- Smoothing DISABLED (paper offline mode) ---")
            self.smoothed_scores = dict(self.rds_scores)
            self.lang_smoothed_scores = {lang: dict(lang_rds[lang]) for lang in qualified}

        self.use_language = True
        self.is_fitted = True

        print(f"\n✓ Policy fitted by language!")
        print(f"  Languages with own scores: {len(self.lang_smoothed_scores)}")
        print(f"  Global fallback for {len(lang_totals) - len(qualified)} rare languages")

    def evaluate_chunked(self, split="test", chunk_size=500_000,
                         max_weight=20, sample_size=None):
        """
        Evaluate the policy on test data by streaming chunks — low memory.

        Accumulates the WIS numerator/denominator and baseline stats across
        chunks so the full test set never needs to fit in memory at once.

        Args:
            split: "train" or "test"
            chunk_size: rows per chunk
            max_weight: clip importance weights
            sample_size: if set, stop after this many rows (approximate)

        Returns:
            dict with baseline, target_value, lift, n_events
        """
        from src.data_loader import iter_parquet_chunks
        import ast as _ast

        if not self.is_fitted:
            raise RuntimeError("Policy not fitted! Call fit() or fit_chunked() first.")

        print("=" * 60)
        print("EVALUATING RDS POLICY (chunked — low-memory mode)")
        print("=" * 60)

        # Accumulators
        wis_numerator = 0.0
        wis_denominator = 0.0
        baseline_sum = 0.0
        total_events = 0

        for chunk in iter_parquet_chunks(split, chunk_size):
            eligible_col = chunk["eligible_templates"].values
            history_col = chunk["history"].values
            selected_col = chunk["selected_template"].values
            reward_col = chunk["session_end_completed"].values
            lang_col = chunk["ui_language"].values if "ui_language" in chunk.columns else None

            for i in range(len(chunk)):
                eligible = eligible_col[i]
                history = history_col[i]
                selected = selected_col[i]
                reward = float(reward_col[i])
                language = lang_col[i] if lang_col is not None else None

                if isinstance(eligible, str):
                    eligible = _ast.literal_eval(eligible)
                elif hasattr(eligible, 'tolist'):
                    eligible = eligible.tolist()

                if history is None:
                    history = []
                elif isinstance(history, str):
                    history = _ast.literal_eval(history)
                elif hasattr(history, 'tolist'):
                    history = history.tolist()

                # Logging probability (random)
                p_log = 1.0 / max(len(eligible), 1)

                # Target probability (RDS) — passes language for lang-specific scores
                probs = self.get_probabilities(eligible, history, language=language)
                p_target = probs.get(selected, 0.0)

                w = p_target / p_log if p_log > 0 else 0.0
                w = min(w, max_weight)

                wis_numerator += w * reward
                wis_denominator += w
                baseline_sum += reward
                total_events += 1

            del chunk

            print(f"[evaluate_chunked] Processed {total_events:,} events so far")

            if sample_size is not None and total_events >= sample_size:
                print(f"[evaluate_chunked] Reached sample_size={sample_size:,}, stopping.")
                break

        baseline = baseline_sum / total_events if total_events > 0 else 0.0
        target_value = wis_numerator / wis_denominator if wis_denominator > 0 else 0.0
        lift = (target_value - baseline) / baseline if baseline > 0 else 0.0

        print(f"\n{'=' * 60}")
        print("EVALUATION COMPLETE (chunked)")
        print(f"{'=' * 60}")
        print(f"  Random baseline:  {baseline:.6f}")
        print(f"  RDS policy:       {target_value:.6f}")
        print(f"  Lift:             {lift:+.4f} ({lift:+.2%})")
        print(f"  Events evaluated: {total_events:,}")

        return {
            "baseline": baseline,
            "target_value": target_value,
            "lift": lift,
            "n_events": total_events,
            "hyperparameters": {
                "kappa": self.kappa,
                "gamma": self.gamma,
                "h": self.h,
                "tau": self.tau,
            },
        }

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
