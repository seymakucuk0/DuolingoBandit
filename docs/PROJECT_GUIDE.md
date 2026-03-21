# DuolingoBandit — Complete Project Guide

> A step-by-step guide for implementing the Recovering Difference Softmax (RDS) algorithm.
> Read this document fully before writing any code.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Understanding the Data](#2-understanding-the-data)
3. [The Algorithm — Piece by Piece](#3-the-algorithm--piece-by-piece)
4. [Implementation Roadmap](#4-implementation-roadmap)
5. [Step 1: Data Loading & EDA](#step-1-data-loading--eda)
6. [Step 2: Baseline Random Policy](#step-2-baseline-random-policy)
7. [Step 3: Relative Difference Scoring](#step-3-relative-difference-scoring)
8. [Step 4: Bayesian Smoothing](#step-4-bayesian-smoothing)
9. [Step 5: Recency Penalty](#step-5-recency-penalty)
10. [Step 6: Softmax Selection](#step-6-softmax-selection)
11. [Step 7: Offline Policy Evaluation](#step-7-offline-policy-evaluation)
12. [Step 8: Full RDS Pipeline](#step-8-full-rds-pipeline)
13. [Step 9: Send-Time Optimization (Extension)](#step-9-send-time-optimization-extension)
14. [Step 10: Kariyer.net Adaptation](#step-10-kariyernet-adaptation)
15. [File-by-File Map](#file-by-file-map)

---

## 1. The Big Picture

### What problem are we solving?

Duolingo (and Kariyer.net) sends push notifications to users to bring them back to the app. There are multiple **notification templates** (e.g., "You're on a streak!", "New lesson available!", etc.). The question is:

> **Which template should we send to each user, right now, to maximize the chance they open the app?**

### Why not just pick the best template?

Three reasons:

1. **Sleeping arms**: Not every template is eligible for every user. A "You're on a 5-day streak!" notification can't go to someone with no streak. The set of available templates changes per user per moment.

2. **Notification fatigue**: If you keep sending the same top-performing template, users get bored and stop clicking. The effectiveness of a template **decays** when sent repeatedly.

3. **Exploration vs. exploitation**: If you only send the template you *think* is best, you never learn whether other templates might be better. You need to occasionally try others.

### What is the solution?

The **Recovering Difference Softmax (RDS)** algorithm, which:
- Scores each template by how much it **lifts** engagement above a baseline (not raw click rate)
- Applies a **recency penalty** so recently-sent templates get demoted
- Uses **softmax** to probabilistically select templates (balancing explore vs exploit)
- Handles **sleeping arms** by only considering eligible templates

---

## 2. Understanding the Data

### The Duolingo Dataset

Location: `data/raw/` (6 parquet files — 3 train, 3 test)

| Column | Type | Description |
|--------|------|-------------|
| `datetime` | float | Time of notification, measured in **days** from the start of the dataset (e.g., 0.0 = start, 1.5 = noon on day 2) |
| `ui_language` | string | User's interface language (ISO 639-1, e.g., "en", "es", "pt") |
| `eligible_templates` | list of strings | Which templates (A through L) this user could receive **at this moment** |
| `history` | list of tuples | Previous notifications this user received: each entry is `(template, days_ago)` |
| `selected_template` | string | Which template was actually sent (A through L) |
| `session_end_completed` | int (0 or 1) | **THE REWARD** — did the user complete a lesson within 2 hours of the notification? |

### Key observations about the data

- There are **12 templates** labeled A through L
- Each row is one notification event — one user, one moment in time
- The dataset has **no user ID** column — each row is independent (you cannot track users across rows directly)
- `eligible_templates` is the "sleeping arms" mechanism — it varies per row
- `history` encodes what the user saw recently — this feeds the recency penalty
- The data was collected under a **random policy** (Duolingo sent templates randomly during the logging period), which is critical for offline evaluation
- **Train**: ~87.7M rows over 15 days
- **Test**: ~114.5M rows over 19 days

### What a single row looks like (conceptual example)

```
datetime:               3.75          (day 3, 6:00 PM)
ui_language:            "en"
eligible_templates:     ["A", "C", "F", "H"]
history:                [("A", 1.2), ("F", 3.5)]   ← template A sent 1.2 days ago, F sent 3.5 days ago
selected_template:      "C"
session_end_completed:  1             (user opened the app and completed a lesson)
```

---

## 3. The Algorithm — Piece by Piece

The RDS algorithm has **four layers** that stack on top of each other. Each layer adds one concept. Here they are from bottom to top:

### Layer 1: Raw Reward Rate

The simplest approach — for each template, calculate:

```
raw_rate(template) = (# times template was sent AND user engaged) / (# times template was sent)
```

**Problem**: This is misleading. Template A might have a high click rate not because it's a good template, but because it happens to be sent to highly active users who would have engaged anyway.

### Layer 2: Relative Difference Score

Instead of raw rates, measure each template's **lift above a baseline**.

For each notification event where template `a` was sent:
- The **actual reward** is what happened: `r_a` (0 or 1)
- The **counterfactual baseline** is: "what would the expected reward have been if we picked randomly from the eligible set?"

```
baseline(event) = (1 / |eligible_templates|) × Σ reward_rate(t)   for all t in eligible_templates
```

The **difference** for this event is:

```
diff(event) = r_a - baseline(event)
```

The **Relative Difference Score** for template `a` is the average of these differences across all events where `a` was sent:

```
RDS(a) = mean of diff(event) for all events where selected_template == a
```

**Why this is better**: If a template is only eligible alongside other high-performing templates, its baseline is high, so it needs to actually be *better than its peers* to score well. This removes the confounding effect of user quality.

### Layer 3: Bayesian Smoothing (Regularization)

Templates with very few observations have noisy scores. A template sent only 10 times might look amazing or terrible by chance.

We apply Bayesian shrinkage to pull low-data scores toward the global mean:

```
smoothed_score(a) = (n_a × RDS(a) + κ × global_mean) / (n_a + κ)
```

Where:
- `n_a` = number of times template `a` was sent
- `κ` (kappa) = a hyperparameter controlling how much we shrink toward the mean (higher κ = more conservative)
- `global_mean` = the average RDS across all templates

With low data, the score approaches `global_mean`. With lots of data, it approaches the observed `RDS(a)`.

### Layer 4: Recency Penalty

A template that was just sent to a user yesterday is less effective than one they haven't seen in weeks. We model this with an **exponential decay**:

```
recency_penalty(a, user_history) = γ × exp(-h × days_since_last_sent(a))
```

Where:
- `γ` (gamma) = maximum penalty magnitude (how much to penalize a just-sent template)
- `h` = decay rate (how quickly the penalty fades — higher h = faster recovery)
- `days_since_last_sent(a)` = from the `history` column, how many days ago template `a` was last sent to this user
- If template `a` was **never sent** to this user, the penalty is **0**

The **final adjusted score** is:

```
adjusted_score(a) = smoothed_score(a) - recency_penalty(a, user_history)
```

### The Selection Step: Softmax

Given the adjusted scores for all eligible templates, we select one using the **softmax** (Boltzmann) distribution:

```
P(a) = exp(τ × adjusted_score(a)) / Σ exp(τ × adjusted_score(t))   for all t in eligible_templates
```

Where:
- `τ` (tau) = temperature parameter
  - High τ → almost always pick the best-scoring template (exploitation)
  - Low τ → nearly uniform random selection (exploration)
  - τ = 0 → perfectly uniform (equivalent to random policy)

The template is then **sampled** from this probability distribution.

### Putting it all together — one decision

```
For a notification event:
  1. Look at eligible_templates → ["A", "C", "F", "H"]
  2. Look up smoothed_score for each: A=0.03, C=0.05, F=0.01, H=0.04
  3. Look at history → A was sent 1.2 days ago
  4. Compute recency penalty: A gets penalized (recently sent), others get 0
     adjusted: A=0.03-0.02=0.01, C=0.05, F=0.01, H=0.04
  5. Apply softmax → P(A)=0.12, P(C)=0.42, P(F)=0.12, P(H)=0.34
  6. Sample from distribution → send template C
```

### Hyperparameters to tune

| Parameter | Symbol | What it controls | Typical range |
|-----------|--------|-----------------|---------------|
| Shrinkage strength | κ | How much to regularize low-data templates | 100–10000 |
| Recency penalty magnitude | γ | How much to penalize recently-sent templates | 0.01–0.2 |
| Recency decay rate | h | How fast the penalty fades | 0.1–2.0 |
| Softmax temperature | τ | Exploration vs exploitation | 1–100 |

---

## 4. Implementation Roadmap

Here is the order you should implement things. Each step builds on the previous one.

```
Step 1: Data Loading & EDA          → notebooks/01_eda.ipynb
Step 2: Baseline Random Policy      → src/evaluation/baseline.py
Step 3: Relative Difference Scoring → src/scoring/difference_score.py
Step 4: Bayesian Smoothing          → src/scoring/bayesian_smoothing.py
Step 5: Recency Penalty             → src/recency/recency_penalty.py
Step 6: Softmax Selection           → src/bandit/softmax_selector.py
Step 7: Offline Policy Evaluation   → src/evaluation/importance_sampling.py
Step 8: Full RDS Pipeline           → src/bandit/rds_policy.py + notebooks/02_evaluation.ipynb
Step 9: Send-Time Optimization      → src/send_time/time_optimizer.py
Step 10: Kariyer.net Adaptation     → src/kariyernet/ (new module)
```

---

## Step 1: Data Loading & EDA

**Goal**: Understand the data before touching any algorithm.

**File**: `notebooks/01_eda.ipynb`

**What to do**:

```python
import pandas as pd
import glob

# Load a SAMPLE first (the full dataset is ~200M rows — start small)
train_files = sorted(glob.glob("data/raw/train-part-*/*.parquet"))
df_sample = pd.read_parquet(train_files[0])  # just the first file

# Questions to answer:
# 1. How many rows? How many columns? What are the dtypes?
df_sample.info()
df_sample.head(10)

# 2. What templates exist? (should be A through L)
# 3. What is the overall reward rate (session_end_completed)?
# 4. What is the reward rate PER template?
# 5. How many templates are typically eligible per event?
# 6. What does the history column look like? How is it encoded?
# 7. Distribution of ui_language
# 8. How does reward rate vary by language?
# 9. How does reward rate vary by time of day (from datetime)?
# 10. How often is each template selected?
```

**Key visualizations to make**:
- Bar chart: reward rate per template
- Histogram: number of eligible templates per event
- Histogram: history length per event
- Line chart: reward rate over time (by day)
- Heatmap: template selection frequency vs eligibility frequency

**Why this matters**: You need to see the data before coding the algorithm. The EDA will reveal whether some templates are rarely eligible, whether reward rates are very different across templates, and how the history column is structured (which you need for the recency penalty).

---

## Step 2: Baseline Random Policy

**Goal**: Establish the baseline performance of random template selection.

**File**: `src/evaluation/baseline.py`

**What to implement**:

```python
def compute_random_baseline(df):
    """
    Under a random policy, each eligible template is equally likely.
    Since the DATA was collected under a random policy, the observed
    average reward IS the random baseline.

    Returns:
        float: average reward rate under random selection
    """
    return df["session_end_completed"].mean()
```

This is simple but important — every other policy will be compared against this number.

---

## Step 3: Relative Difference Scoring

**Goal**: Compute the Relative Difference Score for each template.

**File**: `src/scoring/difference_score.py`

**What to implement**:

```python
def compute_template_reward_rates(df):
    """
    For each template, compute:
      reward_rate(a) = mean(session_end_completed) where selected_template == a

    Returns:
        dict: {template: reward_rate}
    """

def compute_counterfactual_baseline(eligible_templates, reward_rates):
    """
    For a single event, the counterfactual baseline is the average reward
    rate across all eligible templates.

    Args:
        eligible_templates: list of templates eligible for this event
        reward_rates: dict mapping template -> reward rate

    Returns:
        float: baseline expected reward if we chose randomly from eligible set
    """

def compute_relative_difference_scores(df, reward_rates):
    """
    For each template a:
      1. Filter to events where selected_template == a
      2. For each such event, compute:
         diff = session_end_completed - counterfactual_baseline(eligible_templates)
      3. RDS(a) = mean(diff) across all such events

    Returns:
        dict: {template: relative_difference_score}
    """
```

**Key insight**: The counterfactual baseline uses the reward rates of the *eligible* templates, not all templates. This is what makes the score "relative" — it accounts for the fact that eligibility sets vary.

---

## Step 4: Bayesian Smoothing

**Goal**: Regularize scores for templates with few observations.

**File**: `src/scoring/bayesian_smoothing.py`

**What to implement**:

```python
def bayesian_smooth(scores, counts, kappa):
    """
    Apply Bayesian shrinkage to pull low-data scores toward the global mean.

    Args:
        scores: dict {template: relative_difference_score}
        counts: dict {template: number_of_times_sent}
        kappa: float, shrinkage strength

    Returns:
        dict: {template: smoothed_score}

    Formula:
        global_mean = weighted average of scores (weighted by counts)
        smoothed(a) = (n_a * score_a + kappa * global_mean) / (n_a + kappa)
    """
```

**Why this matters**: In the Duolingo dataset some templates are much rarer than others. Without smoothing, a template sent 50 times could have a wildly misleading score.

---

## Step 5: Recency Penalty

**Goal**: Penalize templates that were recently sent to the user.

**File**: `src/recency/recency_penalty.py`

**What to implement**:

```python
import math

def compute_recency_penalty(template, history, gamma, h):
    """
    Compute the recency penalty for a specific template given user history.

    Args:
        template: str, the template to compute penalty for (e.g., "A")
        history: list of (template, days_ago) tuples
        gamma: float, maximum penalty magnitude
        h: float, decay rate

    Returns:
        float: the penalty value (0 if template not in history)

    Formula:
        If template was last sent `d` days ago:
            penalty = gamma * exp(-h * d)
        If template was never sent:
            penalty = 0
    """

def adjust_scores_with_recency(scores, history, gamma, h):
    """
    For each template, subtract its recency penalty from its score.

    Args:
        scores: dict {template: smoothed_score}
        history: list of (template, days_ago) tuples for this user
        gamma: float
        h: float

    Returns:
        dict: {template: adjusted_score}
    """
```

**Important detail about `history`**: You need to parse this column from the dataset. Check in your EDA (Step 1) how it's encoded — it might be a string that needs parsing, or it might already be a list. Each entry should give you (template_name, days_since_it_was_sent).

**Intuition for the parameters**:
- `gamma = 0.1, h = 0.5`: A template sent yesterday gets penalized by `0.1 * exp(-0.5) ≈ 0.06`. A template sent 5 days ago gets penalized by `0.1 * exp(-2.5) ≈ 0.008` (almost nothing).
- Higher `gamma` = stronger "don't repeat" pressure
- Higher `h` = penalty fades faster (templates "recover" sooner)

---

## Step 6: Softmax Selection

**Goal**: Select a template probabilistically based on adjusted scores.

**File**: `src/bandit/softmax_selector.py`

**What to implement**:

```python
import numpy as np

def softmax_select(eligible_templates, adjusted_scores, tau):
    """
    Select a template using the softmax (Boltzmann) distribution.

    Args:
        eligible_templates: list of template names that are eligible right now
        adjusted_scores: dict {template: adjusted_score}
        tau: float, temperature parameter (higher = more greedy)

    Returns:
        str: selected template name

    Steps:
        1. Get scores for eligible templates only
        2. Multiply by tau
        3. Subtract max for numerical stability
        4. Compute exp(tau * score) for each
        5. Normalize to get probabilities
        6. Sample from the distribution
    """

def softmax_probabilities(eligible_templates, adjusted_scores, tau):
    """
    Return the probability distribution without sampling.
    Useful for evaluation (importance sampling needs these probabilities).

    Returns:
        dict: {template: probability}
    """
```

**Numerical stability tip**: Before computing `exp(tau * score)`, subtract the maximum score. This prevents overflow:

```python
scores = np.array([adjusted_scores[t] for t in eligible_templates])
shifted = tau * scores - np.max(tau * scores)
probs = np.exp(shifted) / np.sum(np.exp(shifted))
```

---

## Step 7: Offline Policy Evaluation

**Goal**: Estimate how well the RDS policy would perform **without actually deploying it**.

**File**: `src/evaluation/importance_sampling.py`

This is the trickiest and most important evaluation piece.

### The Problem

We have data collected under a **random policy** (the "logging policy"). We want to estimate the expected reward of our **new RDS policy** (the "target policy"). We can't just run the new policy live — we need to estimate from historical data.

### The Solution: Weighted Importance Sampling

For each event in the test data:
- The logging policy chose template `a` with probability `π_log(a)` = `1 / |eligible_templates|` (random)
- Our target RDS policy would choose template `a` with probability `π_target(a)` (from softmax)
- The actual reward was `r`

The importance weight for this event is:

```
w = π_target(a) / π_log(a)
```

The estimated reward of the target policy is:

```
V(π_target) = Σ(w_i × r_i) / Σ(w_i)
```

This is called **weighted importance sampling** (WIS). It re-weights the observed rewards by how much more (or less) likely the target policy would have been to make the same choice.

**What to implement**:

```python
def compute_importance_weights(df, target_policy_probs):
    """
    For each event:
      - logging_prob = 1 / len(eligible_templates)  [random policy]
      - target_prob = target_policy_probs for the selected_template
      - weight = target_prob / logging_prob

    Args:
        df: DataFrame with columns eligible_templates, selected_template
        target_policy_probs: dict or function that returns P(selected_template)
                             under the target policy for each event

    Returns:
        np.array of importance weights
    """

def weighted_importance_sampling(rewards, weights):
    """
    Compute the WIS estimate of policy value.

    Args:
        rewards: array of observed rewards (0 or 1)
        weights: array of importance weights

    Returns:
        float: estimated expected reward under target policy
    """
    return np.sum(weights * rewards) / np.sum(weights)

def compute_lift(target_value, baseline_value):
    """
    Compute relative lift over baseline.

    Returns:
        float: (target - baseline) / baseline
    """
```

### Variance truncation

Importance weights can be extreme (e.g., a weight of 50 means the target policy is 50x more likely to pick that template than random). Extreme weights make the estimate noisy. Common practice:

```python
# Clip weights to reduce variance
weights = np.clip(weights, 0, max_weight)  # max_weight is a hyperparameter, e.g., 20
```

---

## Step 8: Full RDS Pipeline

**Goal**: Wire all components together into one end-to-end pipeline.

**File**: `src/bandit/rds_policy.py`

**What to implement**:

```python
class RDSPolicy:
    """
    The full Recovering Difference Softmax policy.

    Usage:
        # Training phase (on train data)
        policy = RDSPolicy(kappa=1000, gamma=0.1, h=0.5, tau=50)
        policy.fit(train_df)

        # Evaluation phase (on test data)
        results = policy.evaluate(test_df)
    """

    def __init__(self, kappa, gamma, h, tau):
        self.kappa = kappa
        self.gamma = gamma
        self.h = h
        self.tau = tau
        self.smoothed_scores = None

    def fit(self, df):
        """
        Learn template scores from training data.
        1. Compute per-template reward rates
        2. Compute relative difference scores
        3. Apply Bayesian smoothing
        Stores self.smoothed_scores
        """

    def select_template(self, eligible_templates, history):
        """
        Given eligible templates and user history, select a template.
        1. Get smoothed scores for eligible templates
        2. Apply recency penalty using history
        3. Apply softmax selection
        Returns: selected template
        """

    def get_probabilities(self, eligible_templates, history):
        """
        Same as select_template but returns the full probability distribution.
        Needed for importance sampling evaluation.
        """

    def evaluate(self, df):
        """
        Evaluate the policy on test data using weighted importance sampling.
        1. For each row in df, compute P(selected_template) under this policy
        2. Compute importance weights
        3. Return WIS estimate and lift over random baseline
        """
```

**Evaluation notebook**: `notebooks/02_evaluation.ipynb`

```python
# Train on train set
policy = RDSPolicy(kappa=1000, gamma=0.1, h=0.5, tau=50)
policy.fit(train_df)

# Evaluate on test set
results = policy.evaluate(test_df)
print(f"Random baseline: {results['baseline']:.4f}")
print(f"RDS policy:      {results['target_value']:.4f}")
print(f"Lift:            {results['lift']:.2%}")

# Compare with Duolingo paper results:
# They reported ~1.9% lift in 2-hour conversion rate
```

**Hyperparameter tuning notebook**: `notebooks/03_hyperparameter_tuning.ipynb`

```python
# Grid search over hyperparameters
# Evaluate each combination on a validation split of the train data
# Pick the best, then evaluate once on test data
```

---

## Step 9: Send-Time Optimization (Extension)

**Goal**: Optimize *when* to send notifications, not just *which* template.

**File**: `src/send_time/time_optimizer.py`

This is YOUR extension beyond the original Duolingo paper.

**Approach**:
- Use the `datetime` column to extract hour-of-day
- Group events by time buckets (e.g., 2-hour windows)
- Compute reward rates per time bucket
- Extend the arm definition from just `template` to `(template, time_bucket)`
- Or: use time as a separate multiplicative factor on the score

**What to implement**:

```python
def extract_time_features(df):
    """
    From the datetime column (float in days), extract:
    - hour_of_day (0-23)
    - time_bucket (e.g., 0-3, 3-6, 6-9, ..., 21-24)
    - day_of_week (if enough days in dataset)
    """

def compute_time_reward_rates(df):
    """
    Compute reward rate per time bucket.
    Returns dict: {time_bucket: reward_rate}
    """

def compute_template_time_scores(df):
    """
    Compute reward rate per (template, time_bucket) pair.
    This is the interaction effect.
    """
```

---

## Step 10: Kariyer.net Adaptation

**Goal**: Redesign the algorithm for Kariyer.net's domain.

**Directory**: `src/kariyernet/`

**Key differences from Duolingo**:

| Aspect | Duolingo | Kariyer.net |
|--------|----------|-------------|
| Reward signal | Lesson completed within 2 hours | App open or job listing click within defined window |
| Arm definition | Template only | (Template, User Segment) pair |
| User segments | By ui_language | By CV completeness, job search activity, recency of visit |
| Eligibility rules | From data | Domain-specific: "CV completion" only for incomplete profiles, "new listings" only for active searchers |
| Timing | Ignored | Send-time is a real variable |

**This step depends on receiving Kariyer.net data.** Until then, design the interfaces and document the adaptation plan.

---

## File-by-File Map

Here is exactly what every file in the project does:

```
DuolingoBandit/
│
├── README.md                              # Project overview (already created)
├── requirements.txt                       # Python dependencies (already created)
├── .gitignore                             # Git ignore rules (already created)
│
├── data/
│   └── raw/                               # Extracted parquet files (DO NOT commit these)
│       ├── train-part-1/                   # ~29M rows
│       ├── train-part-2/                   # ~29M rows
│       ├── train-part-3/                   # ~29M rows
│       ├── test-part-1/                    # ~38M rows
│       ├── test-part-2/                    # ~38M rows
│       └── test-part-3/                    # ~38M rows
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # Functions to load & merge parquet files
│   │                                        load_train(), load_test(), load_sample()
│   │
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── difference_score.py            # Layer 2: Relative Difference Scoring
│   │   │                                    compute_template_reward_rates()
│   │   │                                    compute_relative_difference_scores()
│   │   └── bayesian_smoothing.py          # Layer 3: Bayesian regularization
│   │                                        bayesian_smooth()
│   │
│   ├── recency/
│   │   ├── __init__.py
│   │   └── recency_penalty.py             # Layer 4: Exponential decay penalty
│   │                                        compute_recency_penalty()
│   │                                        adjust_scores_with_recency()
│   │
│   ├── bandit/
│   │   ├── __init__.py
│   │   ├── softmax_selector.py            # Softmax (Boltzmann) template selection
│   │   │                                    softmax_select()
│   │   │                                    softmax_probabilities()
│   │   └── rds_policy.py                  # Full RDS pipeline (fit + select + evaluate)
│   │                                        class RDSPolicy
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── baseline.py                    # Random policy baseline
│   │   │                                    compute_random_baseline()
│   │   └── importance_sampling.py         # Offline policy evaluation via WIS
│   │                                        compute_importance_weights()
│   │                                        weighted_importance_sampling()
│   │
│   └── send_time/
│       ├── __init__.py
│       └── time_optimizer.py              # Send-time optimization (your extension)
│                                            extract_time_features()
│                                            compute_template_time_scores()
│
├── notebooks/
│   ├── 01_eda.ipynb                       # Exploratory data analysis
│   ├── 02_evaluation.ipynb                # Full evaluation: RDS vs random
│   ├── 03_hyperparameter_tuning.ipynb     # Grid search for κ, γ, h, τ
│   └── 04_send_time_analysis.ipynb        # Send-time optimization experiments
│
├── experiments/
│   └── results/                           # Saved evaluation results (JSON/CSV)
│
├── tests/
│   ├── __init__.py
│   ├── test_difference_score.py           # Unit tests for scoring
│   ├── test_recency_penalty.py            # Unit tests for recency
│   ├── test_softmax.py                    # Unit tests for softmax
│   └── test_importance_sampling.py        # Unit tests for evaluation
│
└── docs/
    ├── PROJECT_GUIDE.md                   # THIS FILE — the complete guide
    └── system_design.md                   # System design doc for Kariyer.net (Deliverable 5)
```

---

## Quick Reference: The Math

### Relative Difference Score for template a

```
r̄_a = mean reward when template a is sent

baseline(event) = (1/|E|) × Σ r̄_t    for t ∈ eligible_templates(event)

diff(event) = reward(event) - baseline(event)

RDS(a) = mean of diff(event)  for events where selected_template = a
```

### Bayesian Smoothing

```
μ = global weighted mean of RDS scores

smoothed(a) = (n_a × RDS(a) + κ × μ) / (n_a + κ)
```

### Recency Penalty

```
penalty(a) = γ × exp(-h × days_since_last_sent(a))

adjusted(a) = smoothed(a) - penalty(a)
```

### Softmax Selection

```
P(a) = exp(τ × adjusted(a)) / Σ exp(τ × adjusted(t))    for t ∈ eligible set
```

### Weighted Importance Sampling

```
w_i = π_target(selected_template_i) / π_logging(selected_template_i)

π_logging(a) = 1 / |eligible_templates|

V̂(π_target) = Σ(w_i × r_i) / Σ(w_i)
```

---

## Getting Started Checklist

- [ ] Open `DuolingoBandit/` in your IDE
- [ ] Create and activate a virtual environment
- [ ] `pip install -r requirements.txt`
- [ ] Create `notebooks/01_eda.ipynb` and run through Step 1
- [ ] Implement `src/data_loader.py`
- [ ] Implement Steps 2-6 one at a time with unit tests
- [ ] Wire everything together in Step 8
- [ ] Run offline evaluation in `notebooks/02_evaluation.ipynb`
- [ ] Tune hyperparameters in `notebooks/03_hyperparameter_tuning.ipynb`
- [ ] Extend with send-time optimization (Step 9)
