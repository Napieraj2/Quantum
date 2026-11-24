"""Paired QFT alarm and repeated-secret recovery study."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .qft_exchange import (
    CHECKS_PER_ATTEMPT,
    RECOVERY_CONFIDENCE,
    _eve_biases,
    _validate_path_settings,
    simulate_paths,
)


# Proposal §2 (Napieraj and Chuang, 2026) sets the 5% false-alarm target.
# The two QFT scores and the reproducible seed are project settings.
# See validation/MODEL_NOTES.md for the source map.
DETECTORS = ("Pooled check errors", "Joint-check surprisal")
TARGET_FALSE_ALARM = 0.05
DEFAULT_SEED = 20261122


# Project representation of the alarm rule fitted on noise-only trials in proposal §2.
@dataclass(frozen=True)
class _AlarmRule:
    looks: np.ndarray
    centers: np.ndarray
    scales: np.ndarray
    cutoff: float
    tie_probability: float


# Project implementation for comparing scores at different scheduled attempts.
def _standardize(values, centers, scales):
    delta = values - centers
    scores = np.zeros_like(delta, dtype=float)
    np.divide(delta, scales, out=scores, where=scales > 0)
    scores[(scales == 0) & (delta > 0)] = np.inf
    scores[(scales == 0) & (delta < 0)] = -np.inf
    return scores


# Proposal §2 targets any alarm in a noise-only trial. The simulation fits an
# empirical cutoff and randomizes decisions tied at that cutoff.
def _fit_cutoff(maxima, target=TARGET_FALSE_ALARM):
    rank = int(np.ceil((1 - target) * len(maxima))) - 1
    cutoff = float(np.partition(maxima, rank)[rank])
    above = np.mean(maxima > cutoff)
    tied = np.mean(maxima == cutoff)
    tie_probability = float(np.clip((target - above) / tied, 0.0, 1.0))
    return cutoff, tie_probability


# Project implementation of proposal §2's schedule-wide target. Fit the rule
# to the maximum standardized score across each noise-only trial's decisions.
def _fit_alarm_rule(values, looks, target=TARGET_FALSE_ALARM):
    selected = values[:, looks - 1]
    centers = selected.mean(axis=0)
    scales = selected.std(axis=0)
    maxima = _standardize(selected, centers, scales).max(axis=1)
    cutoff, tie_probability = _fit_cutoff(maxima, target)
    return _AlarmRule(looks, centers, scales, cutoff, tie_probability)


# Proposal §2 measures recovery before Bob's first alarm. Use the first
# qualifying decision and mark trials without an alarm as beyond the cap.
def _first_alarms(values, rule, tie_draws):
    scores = _standardize(
        values[:, rule.looks - 1], rule.centers, rule.scales
    )
    alarms = (scores > rule.cutoff) | (
        (scores == rule.cutoff)
        & (tie_draws[:, None] < rule.tie_probability)
    )
    has_alarm = alarms.any(axis=1)
    first = np.argmax(alarms, axis=1)
    return np.where(has_alarm, rule.looks[first], values.shape[1] + 1)


# Proposal §2's cumulative schedule uses the checks collected so far.
def _check_totals(paths):
    errors = np.cumsum(np.asarray(paths["check_errors"]), axis=1)
    joint = np.cumsum(np.asarray(paths["joint_score"]), axis=1)
    return errors, joint


# The QFT notebook uses proposal §2's pooled error rate and check-pattern
# likelihoods. Its joint score sums the negative log likelihoods.
def _detector_statistics(paths):
    errors, joint = _check_totals(paths)
    checks = CHECKS_PER_ATTEMPT * np.arange(1, errors.shape[1] + 1)
    pooled = errors / checks
    return dict(zip(DETECTORS, (pooled, joint)))


# Proposal §2 reports guesses at 99% confidence. The study records the first
# complete attempt at which Eve reaches that threshold.
def _declarations(paths, attempts):
    posterior = np.asarray(paths["posterior"])
    confidence = posterior.max(axis=2)
    guesses = posterior.argmax(axis=2)
    reached = confidence >= RECOVERY_CONFIDENCE
    has_declaration = reached.any(axis=1)
    first = np.argmax(reached, axis=1)
    declaration_attempt = np.where(has_declaration, first + 1, attempts + 1)
    declaration_guess = guesses[np.arange(len(guesses)), first]
    return declaration_attempt, declaration_guess


# Project plot metric from proposal §2's 16-key posterior. Select the marginal
# probability that Eve assigned to each bit Alice actually sent.
def _true_bit_probabilities(paths):
    secret = np.asarray(paths["secret"], dtype=int)
    secret_bits = (
        secret[:, None] >> np.arange(3, -1, -1)[None, :]
    ) & 1
    bit_one = np.asarray(paths["bit_posterior"], dtype=float)
    return np.where(secret_bits[:, None, :] == 1, bit_one, 1 - bit_one)


# Proposal §§2–3 ask for recovery before alarm, false alarms, and costs.
def _study_rows(
    paths, detector, alarm_attempt, false_alarm_attempt,
    declaration_attempt, declaration_guess, attempts, block_qubits,
):
    # Strictly before Bob's alarm is required by proposal §2, so Bob wins a tie.
    secret = np.asarray(paths["secret"], dtype=int)
    alarm_first = (
        (alarm_attempt <= attempts) & (alarm_attempt <= declaration_attempt)
    )
    declaration_first = (
        (declaration_attempt <= attempts) & (declaration_attempt < alarm_attempt)
    )
    correct = declaration_first & (declaration_guess == secret)
    incorrect = declaration_first & ~correct
    inconclusive = ~(alarm_first | declaration_first)
    # Count the costs requested in proposal §3 through Bob's alarm or the cap (§2).
    session_attempt = np.minimum(alarm_attempt, attempts)
    blocks_per_attempt = 4 // (block_qubits // 2)
    transmitted_per_attempt = 2 * block_qubits * blocks_per_attempt

    # Project acceptance rule in QFT_Repeated_Secret.ipynb. Use Bob's alarms
    # through the final decision and require his decoded key to match Alice's.
    accepted = (alarm_attempt > attempts) & paths["final_key_agrees"]
    accepted_sessions = int(accepted.sum())
    exposed_accepted_keys = int((accepted & correct).sum())

    summary = {
        "block_qubits": block_qubits,
        "detector": detector,
        "trials": len(secret),
        "false_alarm_rate": np.mean(false_alarm_attempt <= attempts),
        "alarm_rate": np.mean(alarm_first),
        "correct_recovery_rate": np.mean(correct),
        "incorrect_recovery_rate": np.mean(incorrect),
        "inconclusive_rate": np.mean(inconclusive),
        "mean_checks": np.mean(CHECKS_PER_ATTEMPT * session_attempt),
        "mean_transmitted_qubits": np.mean(
            session_attempt * transmitted_per_attempt
        ),
        "accepted_sessions": accepted_sessions,
        "exposed_accepted_keys": exposed_accepted_keys,
        "accepted_key_exposure_rate": (
            exposed_accepted_keys / accepted_sessions if accepted_sessions else np.nan
        ),
    }
    return summary, alarm_first, correct, declaration_first


# Project plots for proposal §3's alarm and recovery results over time.
def _curve_rows(
    paths, detector, alarm_attempt, declaration_attempt,
    alarm_first, correct, declaration_first, attempts, block_qubits,
):
    true_bits = _true_bit_probabilities(paths)
    # A same-attempt alarm suppresses that attempt's posterior update. A
    # declaration uses its triggering posterior. All later values are frozen.
    last_available = np.full(len(alarm_attempt), attempts - 1, dtype=int)
    last_available[alarm_first] = alarm_attempt[alarm_first] - 2
    last_available[declaration_first] = (
        declaration_attempt[declaration_first] - 1
    )

    rows = []
    for attempt in range(1, attempts + 1):
        indices = np.minimum(attempt - 1, last_available)
        probabilities = np.full((len(indices), 4), 0.5)
        available = indices >= 0
        probabilities[available] = true_bits[
            np.flatnonzero(available), indices[available]
        ]
        rows.append({
            "attempt": attempt,
            "block_qubits": block_qubits,
            "detector": detector,
            "cumulative_alarm_rate": np.mean(
                alarm_first & (alarm_attempt <= attempt)
            ),
            "cumulative_correct_recovery_rate": np.mean(
                correct & (declaration_attempt <= attempt)
            ),
            **{
                f"bit_{bit + 1}": probabilities[:, bit].mean()
                for bit in range(4)
            },
        })
    return rows


# Adapt proposal §2's check-budget fractions to fractions of the attempt cap.
def _percentage_looks(attempts):
    return np.ceil(attempts * np.asarray((0.25, 0.5, 1.0))).astype(int)


# Project validation for the QFT notebook's range of attempt caps and block settings.
def _validate_study(attempts, trials, block_qubits, intercept_probability):
    _validate_path_settings(trials, attempts, block_qubits, intercept_probability)
    if attempts < 4:
        raise ValueError("attempts must be an integer of at least four")


# Paired comparison for proposal §2. Default error rates come from Erven et al.
# (2009, §4). The notebook chooses the two Eve basis biases.
def run_qft_study(
    attempts=100,
    trials=256,
    block_qubits=2,
    intercept_probability=0.025,
    eve_z_probabilities=(0.1, 0.9),
    q_z=0.002,
    q_x=0.038,
    seed=DEFAULT_SEED,
):
    """Run paired pooled-error and joint-surprisal QFT detectors.

    ``trials`` independent paths are used for each of three domains: fitting
    stationary-null alarm rules, measuring held-out false alarms, and the
    configured multi-Eve attack. Detector comparisons share every physical
    attack path. Looks occur at 25%, 50%, and 100% of the attempt cap; this is
    separate from the proposal's 20/50/100 retained-check factorial.
    """
    _validate_study(attempts, trials, block_qubits, intercept_probability)
    eve_z_probabilities = _eve_biases(eve_z_probabilities)
    if not eve_z_probabilities:
        raise ValueError("eve_z_probabilities must contain at least one basis bias")

    # Proposal §2 calls for independent fitting and noise-only evaluation trials.
    looks = _percentage_looks(attempts)
    fit_root, null_root, attack_root, null_ties, attack_ties = (
        np.random.SeedSequence(seed).spawn(5)
    )
    common = dict(
        trials=trials,
        qubits=block_qubits,
        q_z=q_z,
        q_x=q_x,
        attempts=attempts,
    )
    fitted_null = simulate_paths(np.random.default_rng(fit_root), include_posterior=False, **common)
    heldout_null = simulate_paths(np.random.default_rng(null_root), include_posterior=False, **common)
    attack = simulate_paths(
        np.random.default_rng(attack_root),
        intercept_probability=intercept_probability,
        eve_z_probabilities=eve_z_probabilities,
        **common,
    )

    # Use the same checks for both detectors, as in proposal §2's paired comparison.
    fit_statistics = _detector_statistics(fitted_null)
    null_statistics = _detector_statistics(heldout_null)
    attack_statistics = _detector_statistics(attack)
    rules = {
        detector: _fit_alarm_rule(values, looks)
        for detector, values in fit_statistics.items()
    }
    null_draws = np.random.default_rng(null_ties).random(trials)
    attack_draws = np.random.default_rng(attack_ties).random(trials)
    false_alarms = {
        detector: _first_alarms(
            null_statistics[detector], rules[detector], null_draws
        )
        for detector in DETECTORS
    }
    attack_alarms = {
        detector: _first_alarms(
            attack_statistics[detector], rules[detector], attack_draws
        )
        for detector in DETECTORS
    }
    declaration_attempt, declaration_guess = _declarations(attack, attempts)

    # Package proposal §3 outcomes and project acceptance metrics for the notebook.
    summaries, curves = [], []
    for detector in DETECTORS:
        summary, alarm_first, correct, declaration_first = _study_rows(
            attack, detector, attack_alarms[detector], false_alarms[detector],
            declaration_attempt, declaration_guess, attempts, block_qubits,
        )
        summaries.append(summary)
        curves.extend(_curve_rows(
            attack, detector, attack_alarms[detector], declaration_attempt,
            alarm_first, correct, declaration_first, attempts, block_qubits,
        ))

    return pd.DataFrame(summaries), pd.DataFrame(curves)
