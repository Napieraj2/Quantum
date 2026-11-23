"""Independent sixteen-secret posterior and stopping-rule controls.

The uniform four-bit prior, 99% declaration threshold, 100-attempt horizon,
and Bob-wins-a-same-attempt-tie rule come from the project proposal, §2.
Likelihood generation is intentionally outside this small control.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.special import logsumexp


SECRET_COUNT = 16


def posterior_from_log_likelihoods(log_likelihood: np.ndarray) -> np.ndarray:
    """Normalize cumulative evidence under the proposal's uniform prior."""
    scores = np.asarray(log_likelihood, dtype=float)
    if scores.shape != (SECRET_COUNT,):
        raise ValueError("Expected one log likelihood per four-bit secret (16 total)")
    if np.any(np.isnan(scores)) or np.all(np.isneginf(scores)):
        raise ValueError("No valid likelihoods to normalize")
    if np.any(np.isposinf(scores)):
        raise ValueError("Infinite positive likelihoods need explicit handling")
    return np.exp(scores - logsumexp(scores))


def first_declaration(log_likelihood_history, target: float = 0.99):
    """Find the first declaration from cumulative evidence at each attempt.

    Each history entry must contain all log-likelihood evidence available
    through that attempt; the entries are not per-attempt increments.
    """
    if not np.isfinite(target) or not 1 / SECRET_COUNT < target <= 1:
        raise ValueError("Target must be above the uniform prior and no greater than 1")
    for attempt, cumulative in enumerate(
        log_likelihood_history, start=1
    ):
        probabilities = posterior_from_log_likelihoods(cumulative)
        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])
        if confidence >= target:
            return attempt, best, confidence
    return None


def recovered_before_alarm(declaration, true_secret: int, first_alarm_attempt: int | None,
                           horizon: int = 100) -> bool:
    """An alarm on the same attempt counts before Eve can declare."""
    if (
        isinstance(true_secret, bool)
        or not isinstance(true_secret, Integral)
        or not 0 <= true_secret < SECRET_COUNT
        or isinstance(horizon, bool)
        or not isinstance(horizon, Integral)
        or horizon < 1
    ):
        raise ValueError("Bad secret or horizon")
    if (
        first_alarm_attempt is not None
        and (
            isinstance(first_alarm_attempt, bool)
            or not isinstance(first_alarm_attempt, Integral)
            or first_alarm_attempt < 1
        )
    ):
        raise ValueError("Alarm attempt must be a positive integer or None")
    if declaration is None:
        return False
    if not isinstance(declaration, tuple) or len(declaration) != 3:
        raise ValueError("Declaration must be an (attempt, guess, confidence) tuple")
    attempt, guess, _confidence = declaration
    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, Integral)
        or isinstance(guess, bool)
        or not isinstance(guess, Integral)
    ):
        raise ValueError("Declaration attempt and guess must be integers")
    return (1 <= attempt <= horizon and guess == true_secret and
            (first_alarm_attempt is None or attempt < first_alarm_attempt))
