"""BB84 detector power study used by ``BB84.ipynb``.

The model samples already-sifted checks.  It is a controlled detector
comparison, not a finite-key security proof or a hardware channel model.
"""

from dataclasses import dataclass
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


DETECTORS = ("Pooled QBER", "Basis U2", "State U4")
CHECK_BUDGETS = (20, 50, 100)
LOOK_FRACTIONS = (0.25, 0.5, 1.0)


def _noise_profiles():
    profiles = [("No noise", 0.0, 0.0), ("Erven reference", 0.002, 0.038)]
    for mean in (0.01, 0.02, 0.04):
        low, high = mean * (1 - 0.9), mean * (1 + 0.9)
        profiles.extend([
            (f"{mean:.0%} mean, balanced", mean, mean),
            (f"{mean:.0%} mean, higher X", low, high),
            (f"{mean:.0%} mean, higher Z", high, low),
        ])
    return tuple(profiles)


@dataclass(frozen=True)
class BB84Config:
    noise_profiles: tuple
    intercept_fractions: tuple
    eve_z_probabilities: tuple
    calibration_sizes: tuple
    baseline_fits: int
    cutoff_trials: int
    evaluation_trials: int
    cumulative_cutoff_trials: int
    cumulative_evaluation_trials: int
    check_budgets: tuple = CHECK_BUDGETS
    look_fractions: tuple = LOOK_FRACTIONS
    target_false_alarm: float = 0.05
    terminal_seed: int = 20261031
    cumulative_seed: int = 20261101


def bb84_config(preset="quick"):
    """Return the fast notebook preset or the complete proposal grid."""
    if preset == "quick":
        return BB84Config(
            noise_profiles=(("Erven reference", 0.002, 0.038),),
            intercept_fractions=(0.025, 0.05, 0.10, 0.20, 0.50, 1.0),
            eve_z_probabilities=(0.5,),
            calibration_sizes=(1_000,),
            baseline_fits=2,
            cutoff_trials=3_000,
            evaluation_trials=3_000,
            cumulative_cutoff_trials=2_000,
            cumulative_evaluation_trials=2_000,
        )
    if preset == "broad":
        return BB84Config(
            noise_profiles=_noise_profiles(),
            intercept_fractions=(
                0.005, 0.01, 0.025, 0.05, 0.10,
                0.20, 0.35, 0.50, 0.75, 1.0,
            ),
            eve_z_probabilities=(0.1, 0.5, 0.9),
            calibration_sizes=(1_000, 10_000),
            baseline_fits=4,
            cutoff_trials=5_000,
            evaluation_trials=5_000,
            cumulative_cutoff_trials=3_000,
            cumulative_evaluation_trials=3_000,
        )
    raise ValueError("preset must be 'quick' or 'broad'")


def sample_checks(
    rng, intercept_fraction, eve_p_z, *, retained_checks, trials, baseline
):
    """Sample state totals and errors for fixed-size sifted BB84 trials."""
    counts = rng.multinomial(retained_checks, [0.25] * 4, size=trials)
    attack = 0.5 * intercept_fraction * np.repeat([1 - eve_p_z, eve_p_z], 2)
    rates = baseline + (1 - 2 * baseline) * attack
    return counts, rng.binomial(counts, rates)


def _conditioned_score(counts, errors, baseline):
    variable = (baseline > 0) & (baseline < 1)
    rates = baseline[variable]
    weights = (1 - 2 * rates) / (rates * (1 - rates))
    score = ((errors[:, variable] - counts[:, variable] * rates) * weights).sum(1)
    expected = counts[:, ~variable] * baseline[~variable]
    score[(errors[:, ~variable] != expected).any(1)] = np.inf
    return score


def detector_scores(counts, errors, fitted_baseline):
    """Return pooled QBER, basis-grouped U2, and state-conditioned U4."""
    basis_baseline = fitted_baseline.reshape(2, 2).mean(1)
    basis_counts = counts.reshape(-1, 2, 2).sum(2)
    basis_errors = errors.reshape(-1, 2, 2).sum(2)
    basis_score = _conditioned_score(basis_counts, basis_errors, basis_baseline)
    state_score = (
        basis_score
        if np.array_equal(fitted_baseline[::2], fitted_baseline[1::2])
        else _conditioned_score(counts, errors, fitted_baseline)
    )
    return np.column_stack((errors.sum(1) / counts.sum(1), basis_score, state_score))


def _cutoffs(null_scores, target):
    rank = int(np.ceil((1 - target) * len(null_scores))) - 1
    cutoffs = np.partition(null_scores, rank, axis=0)[rank]
    above = (null_scores > cutoffs).mean(0)
    tied = (null_scores == cutoffs).mean(0)
    return cutoffs, np.clip((target - above) / tied, 0.0, 1.0)


def _wilson(events, trials):
    rate, z = events / trials, 1.959963984540054
    denominator = 1 + z**2 / trials
    center = (rate + z**2 / (2 * trials)) / denominator
    radius = z * np.sqrt(
        rate * (1 - rate) / trials + z**2 / (4 * trials**2)
    ) / denominator
    return np.clip(center - radius, 0, 1), np.clip(center + radius, 0, 1)


def _normal_moments(checks, true_baseline, fraction, eve_p_z, fitted):
    attack = 0.5 * fraction * np.repeat([1 - eve_p_z, eve_p_z], 2)
    alternative = true_baseline + (1 - 2 * true_baseline) * attack
    pooled = alternative.mean()
    means, variances = [pooled], [pooled * (1 - pooled) / checks]
    for detector_fit in (fitted.reshape(2, 2).mean(1).repeat(2), fitted):
        if np.any((detector_fit <= 0) | (detector_fit >= 1)):
            means.append(np.nan)
            variances.append(np.nan)
            continue
        weights = (1 - 2 * detector_fit) / (detector_fit * (1 - detector_fit))
        state_means = weights * (alternative - detector_fit)
        per_bit_mean = state_means.mean()
        per_bit_variance = np.mean(
            weights**2 * alternative * (1 - alternative)
            + (state_means - per_bit_mean) ** 2
        )
        means.append(checks * per_bit_mean)
        variances.append(checks * per_bit_variance)
    return np.asarray(means), np.sqrt(variances)


def _normal_alarm(cutoffs, ties, means, deviations, checks):
    probabilities = np.empty(3)
    for index, (cutoff, tie, mean, deviation) in enumerate(
        zip(cutoffs, ties, means, deviations)
    ):
        if np.isnan(deviation):
            probabilities[index] = np.nan
        elif deviation == 0:
            probabilities[index] = (mean > cutoff) + tie * (mean == cutoff)
        elif index == 0:
            cutoff_count, mean_count = cutoff * checks, mean * checks
            deviation_count = deviation * checks
            upper = norm.sf((cutoff_count + 0.5 - mean_count) / deviation_count)
            tied = norm.cdf((cutoff_count + 0.5 - mean_count) / deviation_count)
            tied -= norm.cdf((cutoff_count - 0.5 - mean_count) / deviation_count)
            probabilities[index] = upper + tie * tied
        else:
            probabilities[index] = norm.sf((cutoff - mean) / deviation)
    return probabilities


def _fits(rng, true_rates, config):
    fits = [(0, 1, true_rates)]
    for size in config.calibration_sizes:
        errors = rng.binomial(size, true_rates, size=(config.baseline_fits, 4))
        fitted = (errors + 0.5) / (size + 1)
        fits.extend((size, index, rates) for index, rates in enumerate(fitted, 1))
    return fits


def _scenarios(config):
    return [(0.0, 0.5, False)] + [
        (fraction, eve_p_z, True)
        for fraction, eve_p_z in product(
            config.intercept_fractions, config.eve_z_probabilities
        )
    ]


def run_terminal(config):
    """Run the single-look proposal comparison and return tidy results."""
    fit_rng, cutoff_rng, null_rng, attack_rng = [
        np.random.default_rng(seed)
        for seed in np.random.SeedSequence(config.terminal_seed).spawn(4)
    ]
    rows = []
    for noise, q_z, q_x in config.noise_profiles:
        true_rates = np.repeat([q_z, q_x], 2)
        fits = _fits(fit_rng, true_rates, config)
        for budget in config.check_budgets:
            sampling = dict(retained_checks=budget, baseline=true_rates)
            cutoff_data = sample_checks(
                cutoff_rng, 0, 0.5, trials=config.cutoff_trials, **sampling
            )
            rules = [
                _cutoffs(detector_scores(*cutoff_data, fit), config.target_false_alarm)
                for _, _, fit in fits
            ]
            for fraction, eve_p_z, is_attack in _scenarios(config):
                rng = attack_rng if is_attack else null_rng
                data = sample_checks(
                    rng, fraction, eve_p_z,
                    trials=config.evaluation_trials, **sampling,
                )
                tie_draws = rng.random((config.evaluation_trials, 1))
                for (size, fit_id, fit), (cutoffs, ties) in zip(fits, rules):
                    scores = detector_scores(*data, fit)
                    alarms = (scores > cutoffs) | (
                        (scores == cutoffs) & (tie_draws < ties)
                    )
                    events = alarms.sum(0)
                    lower, upper = _wilson(events, config.evaluation_trials)
                    means, deviations = _normal_moments(
                        budget, true_rates, fraction, eve_p_z, fit
                    )
                    approximation = _normal_alarm(
                        cutoffs, ties, means, deviations, budget
                    )
                    for index, detector in enumerate(DETECTORS):
                        rows.append({
                            "noise_profile": noise, "q_z": q_z, "q_x": q_x,
                            "retained_checks": budget,
                            "calibration_trials_per_state": size,
                            "baseline_fit": fit_id,
                            "qhat_z0": fit[0], "qhat_z1": fit[1],
                            "qhat_x_plus": fit[2], "qhat_x_minus": fit[3],
                            "intercept_fraction": fraction, "eve_p_z": eve_p_z,
                            "is_attack": is_attack, "detector": detector,
                            "alarm_count": events[index],
                            "trials": config.evaluation_trials,
                            "alarm_rate": events[index] / config.evaluation_trials,
                            "normal_approximation": approximation[index],
                            "lower_95": lower[index], "upper_95": upper[index],
                            "cutoff": cutoffs[index],
                            "cutoff_tie_probability": ties[index],
                        })
    return pd.DataFrame(rows)


def summarize_calibration(results):
    """Summarize variation across independently fitted state baselines."""
    fitted = results[results["calibration_trials_per_state"].gt(0)]
    keys = [
        "noise_profile", "retained_checks", "calibration_trials_per_state",
        "intercept_fraction", "eve_p_z", "is_attack", "detector",
    ]
    return fitted.groupby(keys, as_index=False).agg(
        mean_alarm_rate=("alarm_rate", "mean"),
        between_fit_std=("alarm_rate", "std"),
        minimum_alarm_rate=("alarm_rate", "min"),
        maximum_alarm_rate=("alarm_rate", "max"),
        mean_normal_approximation=("normal_approximation", "mean"),
    )


def _look_counts(budget, fractions):
    return np.ceil(budget * np.asarray(fractions)).astype(int)


def _sample_looks(rng, looks, fraction, eve_p_z, *, trials, baseline):
    counts = np.zeros((trials, 4), dtype=int)
    errors = np.zeros_like(counts)
    count_looks, error_looks, previous = [], [], 0
    for look in looks:
        increment = sample_checks(
            rng, fraction, eve_p_z, retained_checks=int(look - previous),
            trials=trials, baseline=baseline,
        )
        counts += increment[0]
        errors += increment[1]
        count_looks.append(counts.copy())
        error_looks.append(errors.copy())
        previous = look
    return np.stack(count_looks, 1), np.stack(error_looks, 1)


def _scores_at_looks(counts, errors, fitted):
    return np.stack([
        detector_scores(counts[:, look], errors[:, look], fitted)
        for look in range(counts.shape[1])
    ], 1)


def _standardize(scores, centers, scales):
    delta = scores - centers
    standardized = np.zeros_like(delta, dtype=float)
    np.divide(delta, scales, out=standardized, where=scales > 0)
    zero = np.broadcast_to(scales == 0, delta.shape)
    standardized[zero & (delta > 0)] = np.inf
    standardized[zero & (delta < 0)] = -np.inf
    return standardized


def _null_standardization(looks, fitted):
    centers, scales = [], []
    for look in looks:
        means, deviations = _normal_moments(look, fitted, 0, 0.5, fitted)
        centers.append((means[0], 0.0, 0.0))
        scales.append(np.nan_to_num(deviations, nan=0.0))
    return np.asarray(centers), np.asarray(scales)


def run_cumulative(config):
    """Run the familywise-calibrated repeated-look comparison."""
    fit_rng, cutoff_rng, null_rng, attack_rng = [
        np.random.default_rng(seed)
        for seed in np.random.SeedSequence(config.cumulative_seed).spawn(4)
    ]
    rows = []
    for noise, q_z, q_x in config.noise_profiles:
        true_rates = np.repeat([q_z, q_x], 2)
        fits = _fits(fit_rng, true_rates, config)
        for budget in config.check_budgets:
            looks = _look_counts(budget, config.look_fractions)
            cutoff_data = _sample_looks(
                cutoff_rng, looks, 0, 0.5,
                trials=config.cumulative_cutoff_trials, baseline=true_rates,
            )
            rules = []
            for _, _, fit in fits:
                centers, scales = _null_standardization(looks, fit)
                maxima = _standardize(
                    _scores_at_looks(*cutoff_data, fit), centers, scales
                ).max(1)
                cutoffs, ties = _cutoffs(maxima, config.target_false_alarm)
                rules.append((centers, scales, cutoffs, ties))

            for fraction, eve_p_z, is_attack in _scenarios(config):
                rng = attack_rng if is_attack else null_rng
                data = _sample_looks(
                    rng, looks, fraction, eve_p_z,
                    trials=config.cumulative_evaluation_trials,
                    baseline=true_rates,
                )
                tie_draws = rng.random((config.cumulative_evaluation_trials, 1, 1))
                for (size, fit_id, fit), rule in zip(fits, rules):
                    centers, scales, cutoffs, ties = rule
                    standardized = _standardize(
                        _scores_at_looks(*data, fit), centers, scales
                    )
                    alarms = (standardized > cutoffs) | (
                        (standardized == cutoffs)
                        & (tie_draws < ties.reshape(1, 1, -1))
                    )
                    events = alarms.any(1).sum(0)
                    lower, upper = _wilson(
                        events, config.cumulative_evaluation_trials
                    )
                    for index, detector in enumerate(DETECTORS):
                        rows.append({
                            "noise_profile": noise, "q_z": q_z, "q_x": q_x,
                            "retained_checks": budget,
                            "look_counts": tuple(int(value) for value in looks),
                            "calibration_trials_per_state": size,
                            "baseline_fit": fit_id,
                            "intercept_fraction": fraction, "eve_p_z": eve_p_z,
                            "is_attack": is_attack, "detector": detector,
                            "alarm_count": events[index],
                            "trials": config.cumulative_evaluation_trials,
                            "any_alarm_rate": (
                                events[index] / config.cumulative_evaluation_trials
                            ),
                            "lower_95": lower[index], "upper_95": upper[index],
                            "maximum_cutoff": cutoffs[index],
                            "cutoff_tie_probability": ties[index],
                        })
    return pd.DataFrame(rows)


def _selected(results, profile, calibration_size, fit, eve_p_z):
    selected = results[
        results["noise_profile"].eq(profile)
        & results["calibration_trials_per_state"].eq(calibration_size)
        & results["baseline_fit"].eq(fit)
    ]
    if selected.empty or not selected["is_attack"].any():
        raise ValueError("The selected profile, calibration, or fit is unavailable")
    if eve_p_z not in selected.loc[selected["is_attack"], "eve_p_z"].unique():
        raise ValueError("The selected Eve basis probability is unavailable")
    return selected


def plot_alarm_curves(
    results, config, *, profile, calibration_size, fit, eve_p_z,
    cumulative=False,
):
    """Plot held-out null false alarms at f=0 and attack power thereafter."""
    selected = _selected(results, profile, calibration_size, fit, eve_p_z)
    rate = "any_alarm_rate" if cumulative else "alarm_rate"
    null = selected[~selected["is_attack"]]
    attack = selected[selected["is_attack"] & selected["eve_p_z"].eq(eve_p_z)]
    fig, axes = plt.subplots(
        1, len(config.check_budgets), figsize=(13.5, 4), sharey=True,
        squeeze=False, layout="constrained",
    )
    for ax, budget in zip(axes[0], config.check_budgets):
        for detector in DETECTORS:
            null_row = null[
                null["retained_checks"].eq(budget) & null["detector"].eq(detector)
            ].iloc[0]
            curve = attack[
                attack["retained_checks"].eq(budget)
                & attack["detector"].eq(detector)
            ].sort_values("intercept_fraction")
            x = np.r_[0.0, curve["intercept_fraction"].to_numpy()]
            y = np.r_[null_row[rate], curve[rate].to_numpy()]
            lower = np.r_[null_row["lower_95"], curve["lower_95"].to_numpy()]
            upper = np.r_[null_row["upper_95"], curve["upper_95"].to_numpy()]
            line, = ax.plot(x, y, marker="o", ms=3, label=detector)
            ax.fill_between(x, lower, upper, color=line.get_color(), alpha=0.12)
            if not cumulative:
                approximation = np.r_[
                    null_row["normal_approximation"],
                    curve["normal_approximation"].to_numpy(),
                ]
                ax.plot(x, approximation, color=line.get_color(), ls="--", lw=1)
        ax.axhline(
            config.target_false_alarm, color="black", ls=":", lw=1,
            label="5% null target",
        )
        ax.axhline(0.8, color="0.5", ls=":", lw=1, label="80% power")
        ax.set(title=f"{budget} retained checks", xlabel="Interception fraction")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Probability of any alarm" if cumulative else "Alarm probability")
    axes[0, -1].legend(loc="lower right", fontsize=8)
    kind = "Multiple scheduled alarm decisions" if cumulative else "Final alarm decision only"
    fig.suptitle(
        f"{kind} · {profile} · Eve P(Z)={eve_p_z:g}\n"
        "f=0 is held-out noise-only evaluation"
    )
    table = null.pivot(
        index="retained_checks", columns="detector", values=rate
    ).reindex(columns=DETECTORS)
    table.insert(0, "Target", config.target_false_alarm)
    table.columns.name = None
    return fig, table
