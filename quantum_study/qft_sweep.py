"""QFT factorial settings, deterministic seeds, and checkpointed runs."""

import hashlib
from itertools import product
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import qft_inference as _inference
from .qft_inference import QFT_PHASE_SAMPLES, qft_get_backend
from .qft_protocol import BLOCK_QUBITS, SECRET_BITS
from .qft_trials import (
    MAX_ATTEMPTS,
    QFT_ALARM_CALIBRATION_RUNS,
    QFT_DEMO_CHECK_BUDGETS,
    QFT_SEED,
    QFT_TRIAL_BATCH_SIZE,
    RECOVERY_CONFIDENCE,
    TARGET_FALSE_ALARM,
    calibrate_qft_alarm,
    run_qft_repeated_trials_batched,
)


EVE_COUNTS = (1, 2, 4)
EVE_INTERCEPT_RATES = (0.005, 0.01, 0.025)
EVE_Z_PROBABILITIES = (0.1, 0.5, 0.9)
EVE_BASIS_REDRAW_RATES = (0.1, 1.0)


SWEEP_NOISE_PROFILES = [
    ("No noise", 0.0, 0.0),
    ("Erven reference", 0.002, 0.038),
]
for _mean in (0.01, 0.02, 0.04):
    _low, _high = _mean * (1 - 0.9), _mean * (1 + 0.9)
    SWEEP_NOISE_PROFILES.extend([
        (f"{_mean:.0%} mean, balanced", _mean, _mean),
        (f"{_mean:.0%} mean, higher X", _low, _high),
        (f"{_mean:.0%} mean, higher Z", _high, _low),
    ])
del _mean, _low, _high


QFT_PROGRESS_EVERY = 10
QFT_CHECKPOINT_PATH = Path("results/qft_factorial_checkpoint.json")
QFT_RESULTS_PATH = Path("results/qft_factorial_results.csv")
QFT_RESUME_SWEEP = True
QFT_FULL_RUNS_PER_SETTING = 100
QFT_CHECKPOINT_SCHEMA = 1
QFT_SEED_VERSION = 1
QFT_POSTERIOR_VERSION = 1


def qft_json_ready(value):
    if isinstance(value, dict):
        return {str(key): qft_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [qft_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return qft_json_ready(value.tolist())
    if isinstance(value, np.generic):
        return qft_json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def qft_canonical_hash(value):
    encoded = json.dumps(
        qft_json_ready(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def qft_seed_sequence(domain, setting, trial_index=None):
    payload = {
        "domain": domain,
        "setting": qft_json_ready(setting),
        "trial_index": trial_index,
        "seed_version": QFT_SEED_VERSION,
    }
    digest = bytes.fromhex(qft_canonical_hash(payload))
    words = np.frombuffer(digest[:16], dtype="<u4").astype(int).tolist()
    return np.random.SeedSequence([QFT_SEED, *words])


def qft_trial_seed_sequences(setting, runs):
    return [
        qft_seed_sequence("qft-evaluation", setting, trial_index)
        for trial_index in range(runs)
    ]


def qft_factorial_config(
    runs_per_setting,
    *,
    backend=None,
    phase_samples=QFT_PHASE_SAMPLES,
    calibration_runs=QFT_ALARM_CALIBRATION_RUNS,
):
    _, backend_name = qft_get_backend(backend)
    return {
        "schema": QFT_CHECKPOINT_SCHEMA,
        "seed": QFT_SEED,
        "seed_version": QFT_SEED_VERSION,
        "backend": backend_name,
        # This is intentionally the configured GPU precision even when NumPy
        # is selected (NumPy itself remains complex128).  The canonical
        # notebook used this value in its checkpoint hash.
        "precision": _inference.QFT_GPU_PRECISION,
        "phase_samples": phase_samples,
        "posterior_version": QFT_POSTERIOR_VERSION,
        "calibration_runs": calibration_runs,
        "runs_per_setting": runs_per_setting,
        "secret_bits": SECRET_BITS,
        "recovery_confidence": RECOVERY_CONFIDENCE,
        "maximum_attempts": MAX_ATTEMPTS,
        "target_false_alarm": TARGET_FALSE_ALARM,
        "budgets": QFT_DEMO_CHECK_BUDGETS,
        "block_qubits": BLOCK_QUBITS,
        "noise_profiles": SWEEP_NOISE_PROFILES,
        "eve_counts": EVE_COUNTS,
        "intercept_rates": EVE_INTERCEPT_RATES,
        "eve_z_probabilities": EVE_Z_PROBABILITIES,
        "basis_redraw_rates": EVE_BASIS_REDRAW_RATES,
    }


def _write_qft_atomically(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_qft_checkpoint(path, payload):
    serialized = json.dumps(
        qft_json_ready(payload), indent=2, allow_nan=False
    )
    _write_qft_atomically(
        path,
        lambda temporary: temporary.write_text(serialized, encoding="utf-8"),
    )


def load_qft_checkpoint(path, config, resume):
    path = Path(path)
    config_hash = qft_canonical_hash(config)
    if not resume or not path.exists():
        return {
            "schema_version": QFT_CHECKPOINT_SCHEMA,
            "config_hash": config_hash,
            "config": qft_json_ready(config),
            "alarm_rules": {},
            "rows": [],
        }
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("schema_version") != QFT_CHECKPOINT_SCHEMA:
        raise ValueError("QFT checkpoint schema does not match this notebook")
    if checkpoint.get("config_hash") != config_hash:
        raise ValueError(
            "QFT checkpoint configuration differs; use a new path or resume=False"
        )
    keys = [row["setting_key"] for row in checkpoint.get("rows", [])]
    if len(keys) != len(set(keys)):
        raise ValueError("QFT checkpoint contains duplicate setting keys")
    return checkpoint


def restore_qft_alarm_rule(serialized):
    restored = dict(serialized)
    restored["budgets"] = tuple(restored["budgets"])
    restored["centers"] = np.asarray(restored["centers"], dtype=float)
    restored["scales"] = np.asarray(restored["scales"], dtype=float)
    return restored


def summarize_qft_trials(trials, setting):
    return {
        **setting,
        "runs": len(trials),
        "alarm_rate": trials["alarm"].mean(),
        "correct_recovery_rate": trials["correct_recovery"].mean(),
        "incorrect_recovery_rate": (
            trials["recovered"] & ~trials["correct_recovery"]
        ).mean(),
        "mean_posterior_confidence": trials[
            "posterior_confidence"
        ].mean(),
        "mean_retained_checks": trials["retained_checks"].mean(),
        "mean_transmitted_qubits": trials[
            "transmitted_qubits"
        ].mean(),
        "mean_intercepts": trials["intercepts"].mean(),
    }


def run_qft_factorial_sweep(
    runs_per_setting=QFT_FULL_RUNS_PER_SETTING,
    checkpoint_path=QFT_CHECKPOINT_PATH,
    results_path=QFT_RESULTS_PATH,
    resume=QFT_RESUME_SWEEP,
    *,
    backend=None,
    phase_samples=QFT_PHASE_SAMPLES,
    calibration_runs=QFT_ALARM_CALIBRATION_RUNS,
    batch_size=QFT_TRIAL_BATCH_SIZE,
    progress_every=QFT_PROGRESS_EVERY,
):
    config = qft_factorial_config(
        runs_per_setting,
        backend=backend,
        phase_samples=phase_samples,
        calibration_runs=calibration_runs,
    )
    checkpoint = load_qft_checkpoint(checkpoint_path, config, resume)
    write_qft_checkpoint(checkpoint_path, checkpoint)
    completed = {row["setting_key"] for row in checkpoint["rows"]}
    scenarios = [(0, 0.0, 0.5, 1.0, False)]
    scenarios.extend(
        (*parameters, True)
        for parameters in product(
            EVE_COUNTS, EVE_INTERCEPT_RATES,
            EVE_Z_PROBABILITIES, EVE_BASIS_REDRAW_RATES,
        )
    )
    total_settings = (
        len(BLOCK_QUBITS) * len(SWEEP_NOISE_PROFILES)
        * len(scenarios)
    )
    print(
        f"QFT sweep: {len(completed)}/{total_settings} settings "
        f"checkpointed; backend={config['backend']}"
    )

    for qubits in BLOCK_QUBITS:
        for noise_profile, q_z, q_x in SWEEP_NOISE_PROFILES:
            base_setting = {
                "block_qubits": qubits,
                "noise_profile": noise_profile,
                "q_z": q_z,
                "q_x": q_x,
            }
            group = {
                **base_setting,
                "budgets": QFT_DEMO_CHECK_BUDGETS,
                "calibration_runs": calibration_runs,
            }
            group_key = qft_canonical_hash(group)
            if group_key in checkpoint["alarm_rules"]:
                alarm_rule = restore_qft_alarm_rule(
                    checkpoint["alarm_rules"][group_key]
                )
            else:
                calibration_rng = np.random.default_rng(
                    qft_seed_sequence("qft-calibration", group)
                )
                alarm_rule = calibrate_qft_alarm(
                    calibration_rng, qubits, q_z, q_x,
                    QFT_DEMO_CHECK_BUDGETS,
                    runs=calibration_runs,
                )
                checkpoint["alarm_rules"][group_key] = qft_json_ready(alarm_rule)
                write_qft_checkpoint(checkpoint_path, checkpoint)

            for eve_count, intercept, eve_p_z, redraw, is_attack in scenarios:
                setting = {
                    **base_setting,
                    "eve_count": eve_count,
                    "intercept_probability": intercept,
                    "eve_p_z": eve_p_z,
                    "redraw_probability": redraw,
                    "is_attack": is_attack,
                }
                setting_key = qft_canonical_hash(setting)
                if setting_key in completed:
                    continue
                eve_configs = tuple({
                    "intercept_probability": intercept,
                    "p_z": eve_p_z,
                    "redraw_probability": redraw,
                } for _ in range(eve_count))
                started = time.perf_counter()
                trials = run_qft_repeated_trials_batched(
                    qft_trial_seed_sequences(setting, runs_per_setting),
                    qubits, q_z, q_x, eve_configs, alarm_rule,
                    phase_samples=phase_samples,
                    batch_size=batch_size,
                    backend=backend,
                )
                if not isinstance(trials, pd.DataFrame):
                    trials = pd.DataFrame(trials)
                row = summarize_qft_trials(trials, setting)
                row.update({
                    "setting_key": setting_key,
                    "compute_backend": config["backend"],
                    "phase_samples": phase_samples,
                    "calibration_runs": calibration_runs,
                    "seed_version": QFT_SEED_VERSION,
                    "posterior_version": QFT_POSTERIOR_VERSION,
                    "elapsed_seconds": time.perf_counter() - started,
                })
                checkpoint["rows"].append(qft_json_ready(row))
                completed.add(setting_key)
                write_qft_checkpoint(checkpoint_path, checkpoint)
                if (
                    len(completed) == total_settings
                    or len(completed) % max(progress_every, 1) == 0
                ):
                    print(
                        f"QFT sweep: {len(completed)}/{total_settings} "
                        f"settings checkpointed"
                    )

    results = pd.DataFrame(checkpoint["rows"])
    _write_qft_atomically(
        results_path,
        lambda temporary: results.to_csv(temporary, index=False),
    )
    return results
