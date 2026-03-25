"""
RECENCY PENALTY (Layer 4)
=========================
This module applies a penalty to templates that were recently sent to a user.

THE PROBLEM — NOTIFICATION FATIGUE:
  Imagine we figure out that template C is the best performer. If we always
  send template C, users will:
    1. Get bored of seeing the same message
    2. Start ignoring it
    3. Eventually get annoyed and disable notifications entirely

  This is called "notification fatigue" or "habituation." The effectiveness
  of any template DECREASES the more recently it was sent.

THE SOLUTION — EXPONENTIAL DECAY PENALTY:
  We subtract a penalty from a template's score based on how recently it
  was sent to this specific user.

  penalty(a) = γ × 0.5^(d / h)

  Where:
    γ (gamma) = maximum penalty (applied when d=0, i.e., just sent)
    h = half-life in days (when d=h, penalty is halved)
    d = days since template 'a' was last sent to this user

  Paper defaults: γ = 0.017, h = 15 (days)

  EXAMPLES (with γ=0.017, h=15):
    Just sent (d=0):    penalty = 0.017 × 0.5^0     = 0.0170  [maximum]
    Sent 1 day ago:     penalty = 0.017 × 0.5^(1/15) = 0.0162
    Sent 7 days ago:    penalty = 0.017 × 0.5^(7/15) = 0.0119
    Sent 15 days ago:   penalty = 0.017 × 0.5^1      = 0.0085  [halved]
    Sent 30 days ago:   penalty = 0.017 × 0.5^2      = 0.0043
    Never sent:         penalty = 0.0

  The template "recovers" over time — hence the name "Recovering Bandit."

WHY EXPONENTIAL DECAY?
  - It's fast at first (the penalty drops quickly in the first few days)
  - It approaches zero but never quite reaches it
  - It's smooth and differentiable (nice mathematical properties)
  - It matches the psychological pattern of habituation and recovery
"""

import math


def compute_recency_penalty(template, history, gamma, h):
    """
    Compute the recency penalty for ONE template given the user's history.

    Args:
        template: str, the template to compute penalty for (e.g., "C")
        history: list of (template_name, days_ago) tuples
                 Example: [("A", 1.2), ("C", 0.5), ("F", 3.0)]
                 This means: template A was sent 1.2 days ago,
                             template C was sent 0.5 days ago,
                             template F was sent 3.0 days ago.
        gamma: float, maximum penalty magnitude (e.g., 0.1)
        h: float, decay rate (e.g., 0.5)

    Returns:
        float: the penalty value (always >= 0)
               0.0 if the template was never sent to this user

    IMPORTANT: We only care about the MOST RECENT time the template was sent.
    If template C appears multiple times in history, we use the smallest days_ago
    (i.e., the most recent occurrence).
    """
    if history is None or (hasattr(history, '__len__') and len(history) == 0):
        return 0.0

    # Find the most recent time this template was sent
    # (smallest days_ago value for this template)
    #
    # IMPORTANT: The history column in the Duolingo dataset is stored as an array
    # of dicts like: [{"template": "A", "n_days": 1.2}, {"template": "C", "n_days": 3.5}]
    # But we also support the (template, days_ago) tuple format for flexibility.
    most_recent = None
    for entry in history:
        # Handle dict format: {"template": "A", "n_days": 1.2}
        if isinstance(entry, dict):
            hist_template = entry.get("template", "")
            days_ago = entry.get("n_days", 0)
        # Handle tuple format: ("A", 1.2)
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            hist_template, days_ago = entry
        else:
            continue

        if hist_template == template:
            if most_recent is None or days_ago < most_recent:
                most_recent = days_ago

    # If this template was never sent to this user, no penalty
    if most_recent is None:
        return 0.0

    # Apply the half-life decay formula (paper Equation 3)
    # penalty = gamma * 0.5^(d / h)
    penalty = gamma * math.pow(0.5, most_recent / h) if h > 0 else 0.0
    return penalty


def adjust_scores_with_recency(scores, history, gamma, h):
    """
    Adjust ALL template scores by subtracting their recency penalties.

    This is the function you call during template selection. It takes the
    smoothed scores from Bayesian smoothing and adjusts them based on
    what this specific user has seen recently.

    Args:
        scores: dict {template: smoothed_score} (from Bayesian smoothing)
        history: list of (template, days_ago) tuples for this specific user
        gamma: float, maximum penalty
        h: float, decay rate

    Returns:
        adjusted_scores: dict {template: adjusted_score}

    Example:
        scores = {"A": 0.03, "C": 0.05, "F": 0.01}
        history = [("A", 1.0), ("C", 0.3)]
        gamma = 0.1, h = 0.5

        penalty_A = 0.1 * exp(-0.5 * 1.0) = 0.061
        penalty_C = 0.1 * exp(-0.5 * 0.3) = 0.086
        penalty_F = 0.0  (not in history)

        adjusted_A = 0.03 - 0.061 = -0.031  [A gets demoted!]
        adjusted_C = 0.05 - 0.086 = -0.036  [C gets demoted even more — sent very recently]
        adjusted_F = 0.01 - 0.0   =  0.01   [F is untouched]

        So F would now be the top-scoring template, even though C had the best raw score.
        This is the "recovering" effect — C needs time to recover before being sent again.
    """
    adjusted_scores = {}

    for template, score in scores.items():
        penalty = compute_recency_penalty(template, history, gamma, h)
        adjusted_scores[template] = score - penalty

    return adjusted_scores


def explain_recency_adjustment(scores, history, gamma, h):
    """
    Same as adjust_scores_with_recency but prints a detailed explanation.
    Useful for debugging and understanding what's happening.

    Args:
        Same as adjust_scores_with_recency

    Returns:
        adjusted_scores: dict (same as above)
    """
    print(f"\n[recency_penalty] Adjusting scores with γ={gamma}, h={h}")
    print(f"[recency_penalty] User history: {history}")
    print(f"{'Template':<10} {'Raw Score':>10} {'Penalty':>10} {'Adjusted':>10} {'Note'}")
    print("-" * 55)

    adjusted_scores = {}
    for template in sorted(scores.keys()):
        score = scores[template]
        penalty = compute_recency_penalty(template, history, gamma, h)
        adjusted = score - penalty
        adjusted_scores[template] = adjusted

        # Find when this template was last sent
        note = ""
        for hist_t, days_ago in (history or []):
            if hist_t == template:
                note = f"sent {days_ago:.1f}d ago"
                break
        if not note:
            note = "never sent"

        print(f"{template:<10} {score:>+10.6f} {penalty:>10.6f} {adjusted:>+10.6f} {note}")

    return adjusted_scores
