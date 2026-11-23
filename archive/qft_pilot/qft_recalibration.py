"""Follow-up demonstration of Bob's alarms under drifting channel noise.

The circuit uses Papadopoulos and Linvill, §4.1. The drift and adaptive
baseline updates are project choices outside the proposal recovery test.
https://arxiv.org/abs/2404.12507v2
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .qft_inference import qft_apply_pauli_channel_batch, qft_backend_cache
from .qft_protocol import (
    QFT_DEMO_Q_X,
    QFT_DEMO_Q_Z,
    SECRET_BITS,
    qft_matrix,
)
from .qft_trials import calibrate_bb84_cutoffs, standardize_look_scores


RECALIBRATION_ATTEMPT_CAP = 500
RECALIBRATION_LOOK_EVERY = 5
RECALIBRATION_SEED = 20261122


def _paths(values, name, attempts=None):
    paths = np.asarray(values, dtype=float)
    if paths.ndim != 2 or not paths.shape[0]:
        raise ValueError(f"{name} must be a nonempty two-dimensional array")
    if attempts is not None and paths.shape[1] != attempts:
        raise ValueError(f"{name} must have {attempts} attempt columns")
    if not np.isfinite(paths).all():
        raise ValueError(f"{name} must contain only finite values")
    return paths


def _window_scores(paths, look_every):
    return paths.reshape(len(paths), -1, look_every).sum(axis=2)


def _cumulative_residuals(window_scores, initial_center, update_weight):
    """Score each window, then update its expected score for the next look."""
    center = np.full(len(window_scores), initial_center, dtype=float)
    cumulative = np.zeros(len(window_scores), dtype=float)
    residuals = np.empty_like(window_scores, dtype=float)
    for look, observed in enumerate(window_scores.T):
        difference = observed - center
        cumulative += difference
        residuals[:, look] = cumulative
        center += update_weight * difference
    return residuals


def compare_alarm_policies(
    calibration,
    cases,
    *,
    look_every=RECALIBRATION_LOOK_EVERY,
    update_weight=0.5,
    target=0.05,
    seed=RECALIBRATION_SEED,
):
    """Compare fixed and post-decision EWMA centers on shared surprise paths.

    ``calibration`` and each value in ``cases`` have shape
    ``(trials, attempts)``. Decisions occur after every
    ``look_every``-attempt window, so the curve's ``attempt`` column is on the
    original absolute time axis. Both policies accumulate window residuals.
    The Frequent policy updates its expected window score only after recording
    the current decision statistic; an observation never affects its own look.

    Each policy's maximum standardized residual is calibrated separately on
    the same stationary null paths. Evaluation paths and one uniform tie draw
    per trial are shared across policies. ``fixed_only`` counts trials where
    Fixed alarms but Frequent never does and is reported on the Frequent row.
    """
    calibration = _paths(calibration, "calibration")
    if type(look_every) is not int or look_every < 1:
        raise ValueError("look_every must be a positive integer")
    attempts = calibration.shape[1]
    if not attempts or attempts % look_every:
        raise ValueError("attempts must be a positive multiple of look_every")
    if not 0.0 <= update_weight <= 1.0:
        raise ValueError("update_weight must be between 0 and 1")
    if not 0.0 < target < 1.0:
        raise ValueError("target must be between 0 and 1")
    if not isinstance(cases, Mapping) or not cases:
        raise ValueError("cases must be a nonempty mapping of names to arrays")

    look_attempts = np.arange(look_every, attempts + 1, look_every, dtype=int)
    null_windows = _window_scores(calibration, look_every)
    initial_center = float(null_windows.mean())
    policy_weights = {"Fixed": 0.0, "Frequent": update_weight}

    rules = {}
    for policy, weight in policy_weights.items():
        residuals = _cumulative_residuals(null_windows, initial_center, weight)
        centers, scales = residuals.mean(axis=0), residuals.std(axis=0)
        standardized = standardize_look_scores(residuals, centers, scales)
        cutoff, tie_probability = calibrate_bb84_cutoffs(
            standardized.max(axis=1, keepdims=True), target
        )
        rules[policy] = (centers, scales, cutoff[0], tie_probability[0])

    rng = np.random.default_rng(seed)
    summary_rows, curve_rows = [], []
    for condition, values in cases.items():
        if not isinstance(condition, str) or not condition:
            raise ValueError("case names must be nonempty strings")
        case_paths = _paths(values, f"cases[{condition!r}]", attempts)
        windows = _window_scores(case_paths, look_every)
        tie_draws = rng.random((len(windows), 1))
        condition_rows, final_alarms = [], {}

        for policy, weight in policy_weights.items():
            centers, scales, cutoff, tie_probability = rules[policy]
            residuals = _cumulative_residuals(windows, initial_center, weight)
            standardized = standardize_look_scores(residuals, centers, scales)
            alarms = (standardized > cutoff) | (
                (standardized == cutoff) & (tie_draws < tie_probability)
            )
            alarmed_by_look = np.logical_or.accumulate(alarms, axis=1)
            final_alarms[policy] = alarmed_by_look[:, -1]
            alarm_count = int(final_alarms[policy].sum())
            condition_rows.append({
                "condition": condition,
                "policy": policy,
                "alarm_count": alarm_count,
                "trials": len(windows),
                "rate": alarm_count / len(windows),
                "fixed_only": 0,
            })
            curve_rows.extend(
                {
                    "condition": condition,
                    "policy": policy,
                    "attempt": int(attempt),
                    "fraction_alarmed": float(rate),
                }
                for attempt, rate in zip(
                    look_attempts, alarmed_by_look.mean(axis=0)
                )
            )

        condition_rows[1]["fixed_only"] = int(np.sum(
            final_alarms["Fixed"] & ~final_alarms["Frequent"]
        ))
        summary_rows.extend(condition_rows)

    return pd.DataFrame(summary_rows), pd.DataFrame(curve_rows)


def _decoded_distribution(phase_diagonal, messages, leg_channels, cache, inverse_qft):
    """Bob's batched QFT outcome probabilities after both channel legs."""
    amplitudes = phase_diagonal / np.sqrt(phase_diagonal.shape[-1])
    density = amplitudes[..., :, None] * amplitudes.conj()[..., None, :]
    for q_z, q_x in leg_channels:
        density = qft_apply_pauli_channel_batch(density, q_z, q_x, cache, np)

    encoding = cache["encoding_diagonals"][messages]
    density = encoding[..., :, None] * density * encoding.conj()[..., None, :]
    for q_z, q_x in leg_channels:
        density = qft_apply_pauli_channel_batch(density, q_z, q_x, cache, np)

    decoder = inverse_qft * phase_diagonal.conj()[..., None, :]
    decoded = decoder @ density @ decoder.conj().swapaxes(-1, -2)
    probabilities = np.diagonal(decoded, axis1=-2, axis2=-1).real.clip(min=0)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def _sample_check_scores(rng, trials, noise_by_attempt, reference_noise, eve_channels=()):
    """Score four checks per attempt with shared or per-trial physical noise."""
    cache = qft_backend_cache(2, np)
    inverse_qft = qft_matrix(2).conj().T
    shape = (trials, SECRET_BITS)
    secret = rng.integers(0, 2, size=shape)
    scores = np.empty((trials, len(noise_by_attempt)))
    for attempt, noise in enumerate(noise_by_attempt):
        check_position = rng.integers(0, 2, size=shape)
        check_value = rng.integers(0, 2, size=shape)
        check_shift = 1 - check_position
        check_only = check_value << check_shift
        phases = rng.uniform(0.0, 2 * np.pi, size=(*shape, 2))
        scrambler = np.exp(1j * (phases @ cache["bits"].T))
        matching = (
            ((np.arange(4) >> check_shift[..., None]) & 1) == check_value[..., None]
        )

        actual_message = check_only + (secret << check_position)
        q_z, q_x = noise.T
        if noise.ndim == 2:
            # Each trial's rates apply to all four blocks in that attempt.
            q_z = q_z[:, None]
            q_x = q_x[:, None]
        physical = _decoded_distribution(
            scrambler, actual_message, ((q_z, q_x), *eve_channels), cache, inverse_qft,
        )
        chance_of_match = (physical * matching).sum(axis=-1)
        matched = rng.random(shape) < chance_of_match

        # Condition on Bob's phases and the disclosed check, and average the
        # attack-free distribution over the two unknown payload-bit values.
        candidates = np.stack((check_only, check_only + (1 << check_position)), axis=-1)
        nominal = _decoded_distribution(
            scrambler[..., None, :], candidates, (reference_noise,), cache, inverse_qft,
        ).mean(axis=-2)
        likelihood = (nominal * (matching == matched[..., None])).sum(axis=-1)
        scores[:, attempt] = -np.log(np.maximum(likelihood, np.finfo(float).tiny)).sum(axis=-1)
    return scores


def _sample_noise_multipliers(rng, attempts, trials):
    """Draw bounded mean-reverting noise from a separate stream."""
    persistence = 0.98
    innovation_sd = 0.30 * np.sqrt(1 - persistence**2)
    levels = np.ones((attempts, trials))
    for attempt in range(1, attempts):
        previous_offset = levels[attempt - 1] - 1
        jitter = rng.normal(0, innovation_sd, size=trials)
        levels[attempt] = np.clip(1 + persistence * previous_offset + jitter, 0.5, 2.0)
    return levels


def run_qft_recalibration_demo(
    trials=512, seed=RECALIBRATION_SEED, *, calibration_trials=2048,
    attempt_cap=RECALIBRATION_ATTEMPT_CAP,
):
    """Compare two alarm policies through the cap, with looks every five attempts."""
    if (
        type(attempt_cap) is not int
        or attempt_cap < RECALIBRATION_LOOK_EVERY
        or attempt_cap % RECALIBRATION_LOOK_EVERY
    ):
        raise ValueError(
            f"attempt_cap must be a positive multiple of {RECALIBRATION_LOOK_EVERY}"
        )
    reference_noise = (QFT_DEMO_Q_Z, QFT_DEMO_Q_X)
    stable = np.tile(reference_noise, (attempt_cap, 1))
    noise_seed, trials_seed = np.random.SeedSequence(seed).spawn(2)
    levels = _sample_noise_multipliers(
        np.random.default_rng(noise_seed), attempt_cap, trials,
    )
    drift = levels[..., None] * np.asarray(reference_noise)

    # For independent basis redraw, an unobserved measurement in basis B
    # maps rho to (rho + B rho B)/2. Average each Eve separately on each leg.
    # Bob's outcome is retained. Eve's individual records are not needed here.
    intercept_probability = 0.025
    eve_channels = tuple(
        (intercept_probability * (1 - p_z) / 2, intercept_probability * p_z / 2)
        for p_z in (0.1, 0.9)
    )
    settings = {
        "Stable noise": (stable, ()),
        "Drifting noise": (drift, ()),
        "Interception": (stable, eve_channels),
        "Drift and interception": (drift, eve_channels),
    }
    streams = trials_seed.spawn(len(settings) + 2)
    calibration = _sample_check_scores(
        np.random.default_rng(streams[0]), calibration_trials, stable, reference_noise,
    )
    cases = {
        condition: _sample_check_scores(
            np.random.default_rng(stream), trials, noise, reference_noise, eves,
        )
        for (condition, (noise, eves)), stream in zip(settings.items(), streams[1:-1])
    }
    return compare_alarm_policies(calibration, cases, seed=streams[-1])
