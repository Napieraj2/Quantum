"""Compare out-of-proposal alarm schedules and responses for X-only Eves.

The random schedule varies when and how often Bob can alarm. The false-alarm
target is for at least one alarm in a noise-only trial. The reset policy can
dismiss alarms before the final attempt.
"""

import numpy as np
import pandas as pd

from .qft_exchange import simulate_paths
from .qft_study import (
    DEFAULT_SEED,
    DETECTORS,
    _check_totals,
    _declarations,
    _fit_cutoff,
    _percentage_looks,
    _standardize,
    _validate_study,
)


# Project follow-up in README.md#qft-alarm-response-results. Reuse the fixed
# schedule and randomly choose the number and timing of earlier decisions.
def _look_schedules(rng, trials, attempts, look_range):
    fewest, most = look_range
    fixed = _percentage_looks(attempts)
    # Pad shorter schedules with the final look. Replay only evaluates it once.
    random = np.full((trials, most), attempts)
    for row, count in zip(random, rng.integers(fewest, most + 1, trials)):
        row[:count - 1] = np.sort(rng.choice(attempts - 1, count - 1, replace=False) + 1)
    return {"Percentage": np.broadcast_to(fixed, (trials, len(fixed))), "Random": random}


# Proposal §2 (Napieraj and Chuang, 2026) sets the 5% chance of any noise-only
# alarm. This follow-up uses the same target for each schedule.
def _fit_schedule(totals, looks):
    # In stationary noise, the null distribution depends on the window length.
    # At a given length, pooled errors and the pooled error rate have the same z-score.
    centers, scales = totals.mean(axis=0), totals.std(axis=0)
    columns = looks - 1
    selected = np.take_along_axis(totals, columns, axis=1)
    maxima = _standardize(selected, centers[columns], scales[columns]).max(axis=1)
    cutoff, tie_probability = _fit_cutoff(maxima)
    return centers, scales, cutoff, tie_probability


# Project reset policy in validation/MODEL_NOTES.md#glossary. Clear Bob's
# accumulated score after an early alarm and keep testing the same key.
def _replay_alarms(totals, looks, rule, ties):
    """Return when Bob first alarms, whether he resets, and whether he alarms at the end."""
    trials, attempts = totals.shape
    centers, scales, cutoff, tie_probability = rule
    rows = np.arange(trials)
    first_crossing = np.full(trials, attempts + 1)
    last_reset = np.zeros(trials, dtype=int)
    discarded = np.zeros(trials, dtype=totals.dtype)
    tie_draw = ties[:, 0].copy()

    # Reuse the fitted noise rule on the checks collected since the last reset.
    def crosses(look):
        window = look - last_reset - 1
        segment = totals[rows, look - 1] - discarded
        scores = _standardize(segment, centers[window], scales[window])
        return (scores > cutoff) | ((scores == cutoff) & (tie_draw < tie_probability))

    for index, look in enumerate(looks.T[:-1], start=1):
        reset = (look < attempts) & crosses(look)
        first_crossing[reset] = np.minimum(first_crossing[reset], look[reset])
        last_reset[reset] = look[reset]
        discarded[reset] = totals[rows[reset], look[reset] - 1]
        tie_draw[reset] = ties[reset, index]

    # Under the follow-up policy, an alarm at the final attempt rejects the key.
    final_alarm = crosses(attempts)
    first_crossing[final_alarm] = np.minimum(first_crossing[final_alarm], attempts)
    return first_crossing, last_reset > 0, final_alarm


# Project follow-up comparing alarm schedules and responses for X-only Eves.
# See README.md#qft-alarm-response-results for the saved settings and results.
def run_qft_alarm_comparison(
    *, attempts=100, trials=1024, block_qubits=2, intercept_probability=0.025,
    coalition_sizes=(1, 2, 4, 8, 16), random_look_range=(2, 4),
    q_z=0.002, q_x=0.038, seed=DEFAULT_SEED,
):
    """Compare alarms and final key acceptance for X-only coalitions.

    ``exposed_after_reset`` counts exposed keys that Bob accepted in trials
    where he reset. Eve may have recovered the key before or after that reset.
    """
    _validate_study(attempts, trials, block_qubits, intercept_probability)
    coalition_sizes = tuple(coalition_sizes)
    if not coalition_sizes or any(type(size) is not int or size < 1 for size in coalition_sizes):
        raise ValueError("coalition_sizes must contain positive integers")
    fewest, most = random_look_range
    if type(fewest) is not int or type(most) is not int or not 2 <= fewest <= most <= attempts:
        raise ValueError("random_look_range must be within [2, attempts]")

    # Proposal §2 separates fitting, noise-only evaluation, and attack samples.
    fit_seed, noise_seed, attack_seed, schedule_seed, tie_seed = np.random.SeedSequence(seed).spawn(5)
    schedule_rng = np.random.default_rng(schedule_seed)
    fit_looks, noise_looks, attack_looks = (
        _look_schedules(schedule_rng, trials, attempts, random_look_range) for _ in range(3)
    )
    # Reuse a tie draw until the score resets, then draw independently for the fresh window.
    noise_ties, attack_ties = np.random.default_rng(tie_seed).random((2, trials, max(3, most)))
    common = dict(trials=trials, attempts=attempts, qubits=block_qubits, q_z=q_z, q_x=q_x)
    fit_totals = _check_totals(simulate_paths(
        np.random.default_rng(fit_seed), include_posterior=False, **common,
    ))
    rules = {
        (detector, schedule): _fit_schedule(totals, looks)
        for detector, totals in zip(DETECTORS, fit_totals)
        for schedule, looks in fit_looks.items()
    }
    del fit_totals
    noise_totals = _check_totals(simulate_paths(
        np.random.default_rng(noise_seed), include_posterior=False, **common,
    ))
    noise_rates = {}
    for detector, totals in zip(DETECTORS, noise_totals):
        for schedule, looks in noise_looks.items():
            key = (detector, schedule)
            first, _, final = _replay_alarms(totals, looks, rules[key], noise_ties)
            noise_rates[key] = np.mean(first <= attempts), final.mean()
    del noise_totals, totals

    # Reuse each attack path across both detectors, schedules, and alarm responses.
    results = []
    for eve_count, path_seed in zip(coalition_sizes, attack_seed.spawn(len(coalition_sizes))):
        paths = simulate_paths(
            np.random.default_rng(path_seed), intercept_probability=intercept_probability,
            eve_z_probabilities=(0.0,) * eve_count, **common,
        )
        declared_at, guess = _declarations(paths, attempts)
        recovered = (declared_at <= attempts) & (guess == paths["secret"])
        key_agrees = paths["final_key_agrees"]
        attack_totals = _check_totals(paths)
        del paths

        for detector, totals in zip(DETECTORS, attack_totals):
            for schedule, looks in attack_looks.items():
                key = (detector, schedule)
                first, reset, final = _replay_alarms(totals, looks, rules[key], attack_ties)
                crossed = first <= attempts
                noise_crossing, noise_final = noise_rates[key]
                settings = {
                    "eve_count": eve_count,
                    # Proposal §2 assumes independent interception. This is the
                    # chance that at least one Eve hits a qubit on a given leg.
                    "effective_intercept_probability": 1 - (1 - intercept_probability) ** eve_count,
                    "schedule": schedule,
                    "detector": detector,
                    "trials": trials,
                    "noise_crossing_rate": noise_crossing,
                    "attack_crossing_rate": crossed.mean(),
                    "recovered_before_first_crossing": int((recovered & (declared_at < first)).sum()),
                }
                # Project key-acceptance rules in validation/MODEL_NOTES.md#glossary.
                # Both responses require Bob's final key to match Alice's.
                for policy, rejected, noise_rejected, reset_sessions in (
                    ("Reject on alarm", crossed, noise_crossing, False),
                    ("Reset on interim alarm", final, noise_final, reset),
                ):
                    accepted = ~rejected & key_agrees
                    exposed = accepted & recovered
                    accepted_count = int(accepted.sum())
                    exposed_count = int(exposed.sum())
                    results.append({
                        **settings,
                        "alarm_policy": policy,
                        "noise_reject_rate": noise_rejected,
                        "sessions_reset": int(np.count_nonzero(reset_sessions)),
                        "accepted_sessions": accepted_count,
                        "exposed_accepted_keys": exposed_count,
                        "exposed_after_reset": int((exposed & reset_sessions).sum()),
                        "accepted_key_exposure_rate": (
                            exposed_count / accepted_count if accepted_count else np.nan
                        ),
                    })
    return pd.DataFrame(results)
