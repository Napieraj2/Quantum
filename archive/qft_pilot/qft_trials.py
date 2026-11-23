"""Archived alarm calibration and repeated-secret pilot trials.

The 20/50/100-check looks, 5% family-wise alarm target, and 99% recovery
declaration come from Napieraj and Chuang, "Detecting Patient Attacks on a
Repeated Secret," §2. This archive evaluates the joint check-pattern detector
with 100 null calibration trajectories, not every detector and calibration size
in the proposal. Each attempt retains four checks, so the 100-check terminal
look ends a trial after 25 attempts; ``MAX_ATTEMPTS`` is only a safety cap. The
QFT exchange is sourced in ``qft_protocol``; the sampled-phase posterior
limitation is documented in ``qft_inference``.
"""

import numpy as np
import pandas as pd

from .qft_inference import QFT_PHASE_SAMPLES, update_secret_posteriors_batched
from .qft_protocol import (
    QFT_DEMO_Q_X,
    QFT_DEMO_Q_Z,
    SECRET_BITS,
    qft_bits,
    simulate_qft_attempt,
)


TARGET_FALSE_ALARM = 0.05
MAX_ATTEMPTS = 100
RECOVERY_CONFIDENCE = 0.99
QFT_SEED = 20261102
QFT_ALARM_CALIBRATION_RUNS = 100
QFT_EVALUATION_RUNS = 8
QFT_TRIAL_BATCH_SIZE = 128
QFT_DEMO_BLOCK_QUBITS = 2
QFT_DEMO_CHECK_BUDGETS = (20, 50, 100)
QFT_DEMO_EVES = (
    {
        "intercept_probability": 0.025,
        "p_z": 0.5,
        "redraw_probability": 1.0,
    },
)


def calibrate_bb84_cutoffs(null_scores, target):
    """Return upper-tail cutoffs and randomized tie probabilities."""
    rank = int(np.ceil((1 - target) * len(null_scores))) - 1
    cutoffs = np.partition(null_scores, rank, axis=0)[rank]
    above = (null_scores > cutoffs).mean(axis=0)
    tied = (null_scores == cutoffs).mean(axis=0)
    return cutoffs, np.clip((target - above) / tied, 0.0, 1.0)


def standardize_look_scores(scores, centers, scales):
    """Standardize look scores while preserving zero-variance tails."""
    delta = scores - centers
    standardized = np.zeros_like(delta, dtype=float)
    np.divide(delta, scales, out=standardized, where=scales > 0)
    zero_scale = np.broadcast_to(scales == 0, delta.shape)
    standardized[zero_scale & (delta > 0)] = np.inf
    standardized[zero_scale & (delta < 0)] = -np.inf
    return standardized


def qft_score_trajectory(rng, qubits, q_z, q_x, budgets):
    secret = qft_bits(
        int(rng.integers(0, 2**SECRET_BITS)), SECRET_BITS
    )
    scores = []
    cumulative_score = 0.0
    retained_checks = 0
    maximum_budget = max(budgets)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        blocks = simulate_qft_attempt(
            rng, secret, qubits, q_z, q_x, (), [], attempt
        )
        for block in blocks:
            cumulative_score += block["score"]
            retained_checks += block["check_count"]
            if retained_checks in budgets:
                scores.append(cumulative_score)
            if retained_checks >= maximum_budget:
                return np.asarray(scores)
    raise RuntimeError("QFT check budget was not reached before the attempt limit")


def calibrate_qft_alarm(
    rng, qubits, q_z, q_x, budgets,
    runs=QFT_ALARM_CALIBRATION_RUNS,
):
    trajectories = np.stack([
        qft_score_trajectory(rng, qubits, q_z, q_x, budgets)
        for _ in range(runs)
    ])
    centers = trajectories.mean(axis=0)
    scales = trajectories.std(axis=0)
    standardized = standardize_look_scores(
        trajectories[:, :, None], centers[:, None], scales[:, None]
    )[:, :, 0]
    maxima = standardized.max(axis=1)
    cutoff, tie_rate = calibrate_bb84_cutoffs(
        maxima[:, None], TARGET_FALSE_ALARM
    )
    return {
        "budgets": tuple(budgets),
        "centers": centers,
        "scales": scales,
        "cutoff": cutoff[0],
        "tie_probability": tie_rate[0],
    }


def _trial_blocks(seed, secret, rng, qubits, q_z, q_x, eve_configs, alarm_rule, phase_samples):
    """Yield each block and its alarm decision in transmission order."""
    phase_rng = np.random.default_rng(seed)
    eve_bases = [
        "Z" if rng.random() < config["p_z"] else "X"
        for config in eve_configs
    ]
    tie_draw = rng.random()
    score = 0.0
    checks = transmitted = intercepts = look_index = 0
    budgets = alarm_rule["budgets"]
    maximum_budget = max(budgets)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        for block in simulate_qft_attempt(
            rng, secret, qubits, q_z, q_x, eve_configs, eve_bases, attempt,
        ):
            score += block["score"]
            checks += block["check_count"]
            transmitted += block["transmitted_qubits"]
            intercepts += len(block["records"])
            alarm = False
            if look_index < len(budgets) and checks == budgets[look_index]:
                delta = score - alarm_rule["centers"][look_index]
                scale = alarm_rule["scales"][look_index]
                if scale > 0:
                    standardized = delta / scale
                elif delta > 0:
                    standardized = np.inf
                elif delta < 0:
                    standardized = -np.inf
                else:
                    standardized = 0.0
                cutoff = alarm_rule["cutoff"]
                alarm = standardized > cutoff or (
                    standardized == cutoff and tie_draw < alarm_rule["tie_probability"]
                )
                look_index += 1

            # Bob wins a same-block timing tie. Only surviving blocks can update Eve.
            phases = (
                phase_rng.uniform(0.0, 2 * np.pi, size=(phase_samples, qubits))
                if not alarm and block["records"]
                else None
            )
            yield {
                "attempt": attempt,
                "block": block,
                "alarm": alarm,
                "retained_checks": checks,
                "transmitted_qubits": transmitted,
                "intercepts": intercepts,
                "posterior_phases": phases,
            }
            if alarm or checks >= maximum_budget:
                return
    raise RuntimeError("QFT check budget was not reached before the attempt limit")


def _trial_result(secret, event, posterior, recovered=False):
    guess = int(np.argmax(posterior))
    return {
        "secret": secret,
        "alarm": bool(event["alarm"]),
        "alarm_attempt": event["attempt"] if event["alarm"] else np.nan,
        "recovered": bool(recovered),
        "correct_recovery": bool(recovered and guess == secret),
        "recovery_attempt": event["attempt"] if recovered else np.nan,
        "posterior_confidence": float(np.exp(posterior[guess])),
        "retained_checks": event["retained_checks"],
        "transmitted_qubits": event["transmitted_qubits"],
        "intercepts": event["intercepts"],
    }


def _run_qft_batch(seeds, qubits, q_z, q_x, eve_configs, alarm_rule, phase_samples, backend):
    secrets, streams = [], []
    for seed in seeds:
        simulation_seed, posterior_seed = seed.spawn(2)
        rng = np.random.default_rng(simulation_seed)
        secret = int(rng.integers(0, 2**SECRET_BITS))
        secrets.append(secret)
        streams.append(_trial_blocks(
            posterior_seed, qft_bits(secret, SECRET_BITS), rng, qubits, q_z, q_x,
            eve_configs, alarm_rule, phase_samples,
        ))

    posteriors = np.full((len(seeds), 2**SECRET_BITS), -np.log(2**SECRET_BITS))
    results = [None] * len(seeds)
    active = list(range(len(seeds)))
    maximum_budget = max(alarm_rule["budgets"])
    while active:
        pending = []
        observations = []
        for index in active:
            event = next(streams[index])
            if event["alarm"]:
                results[index] = _trial_result(secrets[index], event, posteriors[index])
                continue
            pending.append((index, event))
            if event["posterior_phases"] is not None:
                observations.append((index, event))

        if observations:
            indices = [index for index, _ in observations]
            blocks = [event["block"] for _, event in observations]
            phases = np.stack([event["posterior_phases"] for _, event in observations])
            posteriors[indices] = update_secret_posteriors_batched(
                posteriors[indices], blocks, qubits, q_z, q_x, phases, backend=backend,
            )

        active = []
        for index, event in pending:
            posterior = posteriors[index]
            recovered = np.exp(posterior.max()) >= RECOVERY_CONFIDENCE
            if recovered or event["retained_checks"] >= maximum_budget:
                results[index] = _trial_result(secrets[index], event, posterior, recovered)
            else:
                active.append(index)
    return results


def run_qft_repeated_trials_batched(
    trial_seed_sequences, qubits, q_z, q_x, eve_configs, alarm_rule,
    *, phase_samples=QFT_PHASE_SAMPLES,
    batch_size=QFT_TRIAL_BATCH_SIZE,
    backend=None,
):
    """Advance independent trials a block at a time and batch their posterior updates."""
    seeds = list(trial_seed_sequences)
    if not seeds:
        raise ValueError("At least one trial seed is required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    rows = []
    for start in range(0, len(seeds), batch_size):
        rows.extend(_run_qft_batch(
            seeds[start:start + batch_size], qubits, q_z, q_x, eve_configs,
            alarm_rule, phase_samples, backend,
        ))
    return pd.DataFrame(rows)


def run_qft_demo(
    *,
    seed=QFT_SEED + 1,
    qubits=QFT_DEMO_BLOCK_QUBITS,
    q_z=QFT_DEMO_Q_Z,
    q_x=QFT_DEMO_Q_X,
    budgets=QFT_DEMO_CHECK_BUDGETS,
    eve_configs=QFT_DEMO_EVES,
    calibration_runs=QFT_ALARM_CALIBRATION_RUNS,
    evaluation_runs=QFT_EVALUATION_RUNS,
    phase_samples=QFT_PHASE_SAMPLES,
    batch_size=QFT_TRIAL_BATCH_SIZE,
    backend=None,
    return_diagnostics=False,
):
    """Run the canonical small repeated-secret demonstration on demand.

    By default only the per-trial DataFrame is returned.  With
    ``return_diagnostics=True`` the return value is
    ``(trials, diagnostics, alarm_rule)``.
    """
    calibration_seed, trial_root_seed = np.random.SeedSequence(seed).spawn(2)
    alarm_rule = calibrate_qft_alarm(
        np.random.default_rng(calibration_seed),
        qubits, q_z, q_x, budgets,
        runs=calibration_runs,
    )
    trials = run_qft_repeated_trials_batched(
        trial_root_seed.spawn(evaluation_runs),
        qubits, q_z, q_x, eve_configs, alarm_rule,
        phase_samples=phase_samples,
        batch_size=batch_size,
        backend=backend,
    )
    diagnostics = pd.Series({
        "runs": len(trials),
        "alarm_rate": trials["alarm"].mean(),
        "correct_recovery_rate": trials["correct_recovery"].mean(),
        "mean_transmitted_qubits": trials["transmitted_qubits"].mean(),
        "mean_intercepts": trials["intercepts"].mean(),
    })
    if return_diagnostics:
        return trials, diagnostics, alarm_rule
    return trials
