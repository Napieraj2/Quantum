"""Sixteen-secret posterior probabilities and recovery-before-alarm logic."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


SECRET_COUNT = 16


def posterior_from_log_likelihoods(log_likelihood: np.ndarray) -> np.ndarray:
    scores = np.asarray(log_likelihood, dtype=float)
    if scores.shape != (SECRET_COUNT,):
        raise ValueError("Expected one log likelihood per four-bit secret (16 total)")
    if np.any(np.isnan(scores)) or np.all(np.isneginf(scores)):
        raise ValueError("No valid likelihoods to normalize")
    if np.any(np.isposinf(scores)):
        raise ValueError("Infinite positive likelihoods need explicit handling")
    return np.exp(scores - logsumexp(scores))


def first_declaration(log_likelihood_history, target: float = 0.99):
    """Find the first attempt when a candidate reaches the confidence target."""
    if not 1 / SECRET_COUNT < target <= 1:
        raise ValueError("Target must be above the uniform prior and no greater than 1")
    for attempt, log_likelihood in enumerate(log_likelihood_history, start=1):
        probabilities = posterior_from_log_likelihoods(log_likelihood)
        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])
        if confidence >= target:
            return attempt, best, confidence
    return None


def recovered_before_alarm(declaration, true_secret: int, first_alarm_attempt: int | None,
                           horizon: int = 100) -> bool:
    """An alarm on the same attempt counts before Eve can declare."""
    if not 0 <= true_secret < SECRET_COUNT or horizon < 1:
        raise ValueError("Bad secret or horizon")
    if declaration is None:
        return False
    attempt, guess, _confidence = declaration
    return (1 <= attempt <= horizon and guess == true_secret and
            (first_alarm_attempt is None or attempt < first_alarm_attempt))
