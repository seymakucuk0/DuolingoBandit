"""
SOFTMAX (BOLTZMANN) SELECTION
=============================
This module selects a template using the softmax probability distribution.

THE EXPLORATION-EXPLOITATION DILEMMA:
  After scoring all templates, the simplest approach is: "always pick the
  highest-scoring template." This is called GREEDY selection.

  Problem: what if our scores are wrong? What if a template we've barely
  tried is actually the best, but we never discover it because we always
  pick the current leader?

  This is the exploration-exploitation tradeoff:
    - EXPLOITATION: pick the template we believe is best (high short-term reward)
    - EXPLORATION: try other templates to learn more (high long-term reward)

THE SOFTMAX SOLUTION:
  Instead of always picking the best, we assign probabilities to each template
  based on their scores, then SAMPLE from that distribution.

  P(a) = exp(τ × score(a)) / Σ exp(τ × score(t))   for all eligible t

  The temperature parameter τ controls the balance:
    - τ → 0:    uniform random (pure exploration, every template equally likely)
    - τ → ∞:    always pick the highest-scored (pure exploitation, greedy)
    - τ = 10:   moderate — best templates are more likely but others still get tried
    - τ = 50:   aggressive — strongly favors high scores but not fully greedy

  EXAMPLE (3 templates with scores 0.05, 0.02, -0.01):
    τ = 1:   P = [0.37, 0.34, 0.29]  — almost uniform
    τ = 10:  P = [0.51, 0.32, 0.17]  — slight preference for best
    τ = 50:  P = [0.82, 0.15, 0.03]  — strong preference for best
    τ = 100: P = [0.95, 0.05, 0.00]  — nearly greedy
"""

import numpy as np


def softmax_probabilities(eligible_templates, adjusted_scores, tau):
    """
    Compute the softmax probability distribution over eligible templates.

    Args:
        eligible_templates: list of template names eligible for this event
                           (e.g., ["A", "C", "F", "H"])
        adjusted_scores: dict {template: adjusted_score} from recency adjustment
        tau: float, temperature parameter (higher = more exploitative)

    Returns:
        dict {template: probability} summing to 1.0

    NUMERICAL STABILITY:
        Computing exp(τ × score) can overflow if τ × score is large (e.g., > 700).
        We subtract the maximum value before computing exp(). This doesn't change
        the final probabilities because:

        exp(x - max) / Σ exp(x_i - max) = exp(x) / Σ exp(x_i)

        The max cancels out in the ratio. This is a standard numerical trick.
    """
    if len(eligible_templates) == 0:
        return {}

    # If only one template is eligible, it gets probability 1.0
    if len(eligible_templates) == 1:
        return {eligible_templates[0]: 1.0}

    # Get scores for eligible templates only
    scores = np.array([adjusted_scores.get(t, 0.0) for t in eligible_templates])

    # Multiply by temperature
    scaled = tau * scores

    # Subtract max for numerical stability (prevents overflow in exp)
    scaled = scaled - np.max(scaled)

    # Compute softmax
    exp_scores = np.exp(scaled)
    total = np.sum(exp_scores)

    # Guard against division by zero (shouldn't happen, but be safe)
    if total == 0:
        # Fall back to uniform
        n = len(eligible_templates)
        return {t: 1.0 / n for t in eligible_templates}

    probs = exp_scores / total

    # Build the result dict
    result = {}
    for i, t in enumerate(eligible_templates):
        result[t] = float(probs[i])

    return result


def softmax_select(eligible_templates, adjusted_scores, tau, rng=None):
    """
    Select a template by sampling from the softmax distribution.

    Args:
        eligible_templates: list of template names eligible for this event
        adjusted_scores: dict {template: adjusted_score}
        tau: float, temperature parameter
        rng: numpy random generator (for reproducibility). If None, uses default.

    Returns:
        str: the selected template name

    Example:
        eligible = ["A", "C", "F"]
        scores = {"A": 0.01, "C": 0.05, "F": -0.02}
        tau = 50

        # Compute probabilities
        probs = softmax_probabilities(eligible, scores, tau)
        # probs might be {"A": 0.15, "C": 0.75, "F": 0.10}

        # Sample from distribution
        # Most likely returns "C" but sometimes returns "A" or "F"
        selected = "C"  # (randomly sampled)
    """
    if len(eligible_templates) == 0:
        raise ValueError("No eligible templates to select from!")

    if len(eligible_templates) == 1:
        return eligible_templates[0]

    # Get probabilities
    probs_dict = softmax_probabilities(eligible_templates, adjusted_scores, tau)

    # Convert to arrays for numpy sampling
    templates = list(probs_dict.keys())
    probs = np.array([probs_dict[t] for t in templates])

    # Ensure probabilities sum to 1 (fix floating point errors)
    probs = probs / probs.sum()

    # Sample one template
    if rng is None:
        rng = np.random.default_rng()

    selected_idx = rng.choice(len(templates), p=probs)
    return templates[selected_idx]


def explain_softmax_selection(eligible_templates, adjusted_scores, tau):
    """
    Show the full softmax calculation step by step.
    Useful for debugging and presentations.

    Args:
        eligible_templates: list of eligible template names
        adjusted_scores: dict {template: score}
        tau: float, temperature

    Returns:
        dict {template: probability}
    """
    print(f"\n[softmax] Computing selection probabilities (τ = {tau})")
    print(f"[softmax] Eligible templates: {eligible_templates}")

    scores = np.array([adjusted_scores.get(t, 0.0) for t in eligible_templates])
    scaled = tau * scores
    shifted = scaled - np.max(scaled)
    exp_vals = np.exp(shifted)
    total = np.sum(exp_vals)
    probs = exp_vals / total

    print(f"\n{'Template':<10} {'Score':>10} {'τ×Score':>10} {'Shifted':>10} {'exp()':>12} {'P(select)':>10}")
    print("-" * 72)

    result = {}
    for i, t in enumerate(eligible_templates):
        print(f"{t:<10} {scores[i]:>+10.6f} {scaled[i]:>+10.4f} {shifted[i]:>+10.4f} {exp_vals[i]:>12.4f} {probs[i]:>10.4f}")
        result[t] = float(probs[i])

    print(f"\n[softmax] Most likely: {eligible_templates[np.argmax(probs)]} ({probs.max():.1%})")

    return result
