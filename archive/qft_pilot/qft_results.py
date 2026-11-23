"""Validation, summaries, and plots for the QFT factorial result table."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


RATE_COLUMNS = (
    "alarm_before_recovery_rate",
    "correct_recovery_rate",
    "incorrect_recovery_rate",
    "any_recovery_rate",
    "inconclusive_rate",
)


def wilson_95(events, trials):
    """Return two-sided 95% score intervals from Wilson (1927), pp. 209–212.

    https://doi.org/10.1080/01621459.1927.10502953
    """
    events = np.asarray(events, dtype=float)
    trials = np.asarray(trials, dtype=float)
    rate = events / trials
    z = 1.959963984540054
    denominator = 1 + z**2 / trials
    center = (rate + z**2 / (2 * trials)) / denominator
    radius = z * np.sqrt(
        rate * (1 - rate) / trials + z**2 / (4 * trials**2)
    ) / denominator
    return (
        np.clip(center - radius, 0.0, 1.0),
        np.clip(center + radius, 0.0, 1.0),
    )


def prepare_qft_results(results: pd.DataFrame) -> pd.DataFrame:
    """Validate a setting table and add competing outcomes and rate intervals."""
    required = {
        "setting_key",
        "is_attack",
        "runs",
        "alarm_rate",
        "correct_recovery_rate",
        "incorrect_recovery_rate",
        "elapsed_seconds",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Missing QFT result columns: {sorted(missing)}")
    if results["setting_key"].duplicated().any():
        raise ValueError("QFT result contains duplicate setting keys")

    enriched = results.copy()
    enriched["alarm_before_recovery_rate"] = enriched["alarm_rate"]
    enriched["any_recovery_rate"] = (
        enriched["correct_recovery_rate"]
        + enriched["incorrect_recovery_rate"]
    )
    enriched["inconclusive_rate"] = (
        1
        - enriched["alarm_before_recovery_rate"]
        - enriched["any_recovery_rate"]
    )
    if not enriched["inconclusive_rate"].between(
        -1e-12, 1 + 1e-12
    ).all():
        raise ValueError("QFT outcome rates do not form valid probabilities")
    enriched["inconclusive_rate"] = enriched[
        "inconclusive_rate"
    ].clip(0.0, 1.0)

    for column in RATE_COLUMNS:
        events = np.rint(
            enriched[column] * enriched["runs"]
        ).astype(int)
        lower, upper = wilson_95(events, enriched["runs"])
        enriched[f"{column}_events"] = events
        enriched[f"{column}_lower_95"] = lower
        enriched[f"{column}_upper_95"] = upper
    return enriched


def build_qft_summary(
    results: pd.DataFrame,
    *,
    expected_settings: int,
    target_false_alarm: float,
) -> dict[str, object]:
    """Build proposal-facing audit, factor, cost, and false-alarm summaries."""
    enriched = prepare_qft_results(results)
    required = {
        "block_qubits",
        "noise_profile",
        "eve_count",
        "intercept_probability",
        "eve_p_z",
        "redraw_probability",
        "compute_backend",
        "phase_samples",
        "mean_posterior_confidence",
        "mean_retained_checks",
        "mean_transmitted_qubits",
        "mean_intercepts",
    }
    missing = required.difference(enriched.columns)
    if missing:
        raise ValueError(f"Missing QFT summary columns: {sorted(missing)}")
    null = enriched[~enriched["is_attack"]].copy()
    attack = enriched[enriched["is_attack"]].copy()
    attack["effective_interception_intensity"] = (
        attack["eve_count"] * attack["intercept_probability"]
    )
    null_lookup = null[[
        "block_qubits",
        "noise_profile",
        "alarm_before_recovery_rate",
    ]].rename(columns={
        "alarm_before_recovery_rate": "matched_false_alarm_rate",
    })
    attack = attack.merge(
        null_lookup,
        on=["block_qubits", "noise_profile"],
        validate="many_to_one",
    )
    attack["alarm_lift"] = (
        attack["alarm_before_recovery_rate"]
        - attack["matched_false_alarm_rate"]
    )

    false_alarm_events = int(
        null["alarm_before_recovery_rate_events"].sum()
    )
    false_alarm_trials = int(null["runs"].sum())
    false_alarm_lower, false_alarm_upper = wilson_95(
        false_alarm_events, false_alarm_trials
    )
    recovery_events = int(attack["any_recovery_rate_events"].sum())
    attack_trials = int(attack["runs"].sum())

    audit = pd.Series({
        "observed / expected settings": (
            f"{len(enriched):,} / {expected_settings:,}"
        ),
        "unique setting keys": enriched["setting_key"].nunique(),
        "attack-free / attack settings": (
            f"{len(null)} / {len(attack):,}"
        ),
        "attack trials": attack_trials,
        "recovery declarations": recovery_events,
        "backend": ", ".join(sorted(
            enriched["compute_backend"].unique()
        )),
        "phase samples": ", ".join(map(
            str, sorted(enriched["phase_samples"].unique())
        )),
        "recorded setting runtime (minutes)": round(
            enriched["elapsed_seconds"].sum() / 60, 2
        ),
    }).to_frame("value")

    false_alarm = null[[
        "block_qubits",
        "noise_profile",
        "runs",
        "alarm_before_recovery_rate",
        "alarm_before_recovery_rate_lower_95",
        "alarm_before_recovery_rate_upper_95",
    ]].rename(columns={
        "alarm_before_recovery_rate": "false_alarm_rate",
        "alarm_before_recovery_rate_lower_95": "lower_95",
        "alarm_before_recovery_rate_upper_95": "upper_95",
    })
    false_alarm.insert(3, "target_false_alarm", target_false_alarm)

    primary = (
        attack.groupby(
            ["block_qubits", "noise_profile"], as_index=False
        )
        .agg(
            settings=("setting_key", "size"),
            alarm_before_recovery_rate=(
                "alarm_before_recovery_rate", "mean"
            ),
            alarm_lift=("alarm_lift", "mean"),
            correct_recovery_rate=("correct_recovery_rate", "mean"),
            inconclusive_rate=("inconclusive_rate", "mean"),
            mean_posterior_confidence=(
                "mean_posterior_confidence", "mean"
            ),
            mean_retained_checks=("mean_retained_checks", "mean"),
            mean_transmitted_qubits=(
                "mean_transmitted_qubits", "mean"
            ),
            mean_intercepts=("mean_intercepts", "mean"),
        )
    )

    factor_frames = []
    for factor in (
        "block_qubits",
        "eve_count",
        "intercept_probability",
        "effective_interception_intensity",
        "eve_p_z",
        "redraw_probability",
    ):
        frame = attack.groupby(factor, as_index=False).agg(
            settings=("setting_key", "size"),
            alarm_before_recovery_rate=(
                "alarm_before_recovery_rate", "mean"
            ),
            alarm_lift=("alarm_lift", "mean"),
            inconclusive_rate=("inconclusive_rate", "mean"),
            mean_posterior_confidence=(
                "mean_posterior_confidence", "mean"
            ),
            mean_retained_checks=("mean_retained_checks", "mean"),
            mean_intercepts=("mean_intercepts", "mean"),
        )
        frame.insert(0, "factor", factor)
        frame.insert(1, "level", frame[factor].astype(str))
        factor_frames.append(frame.drop(columns=factor))
    factors = pd.concat(factor_frames, ignore_index=True)

    leakage_watch = attack.nlargest(
        10, "mean_posterior_confidence"
    )[[
        "block_qubits",
        "noise_profile",
        "eve_count",
        "intercept_probability",
        "eve_p_z",
        "redraw_probability",
        "alarm_before_recovery_rate",
        "mean_posterior_confidence",
        "mean_retained_checks",
        "mean_intercepts",
    ]]

    return {
        "enriched": enriched,
        "null": null,
        "attack": attack,
        "audit": audit,
        "false_alarm": false_alarm,
        "primary": primary,
        "factors": factors,
        "leakage_watch": leakage_watch,
        "metrics": {
            "false_alarm_events": false_alarm_events,
            "false_alarm_trials": false_alarm_trials,
            "false_alarm_lower": float(false_alarm_lower),
            "false_alarm_upper": float(false_alarm_upper),
            "recovery_events": recovery_events,
            "attack_trials": attack_trials,
            "target_false_alarm": target_false_alarm,
        },
    }


def plot_qft_summary(
    summary: dict[str, object],
    *,
    block_sizes: Sequence[int],
    noise_order: Sequence[str],
):
    """Plot held-out false alarms and alarm response versus K times f."""
    import matplotlib.pyplot as plt

    false_alarm = summary["false_alarm"]
    attack = summary["attack"]
    metrics = summary["metrics"]
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 4.5), layout="constrained"
    )
    for block_qubits in block_sizes:
        curve = (
            false_alarm[false_alarm["block_qubits"].eq(block_qubits)]
            .set_index("noise_profile")
            .reindex(noise_order)
        )
        axes[0].plot(
            range(len(noise_order)),
            curve["false_alarm_rate"],
            marker="o",
            label=f"{block_qubits} qubits",
        )
    axes[0].axhline(
        metrics["target_false_alarm"],
        color="0.3",
        linestyle="--",
        label=f"{100 * metrics['target_false_alarm']:g}% target",
    )
    axes[0].set(
        xticks=range(len(noise_order)),
        xticklabels=noise_order,
        ylabel="Held-out false-alarm rate",
        title="Noise-only controls",
        ylim=(0, 0.2),
    )
    axes[0].tick_params(axis="x", rotation=70)
    axes[0].legend()

    intensity = (
        attack.groupby(
            ["block_qubits", "effective_interception_intensity"],
            as_index=False,
        )["alarm_before_recovery_rate"]
        .mean()
    )
    for block_qubits in block_sizes:
        curve = intensity[intensity["block_qubits"].eq(block_qubits)]
        axes[1].plot(
            curve["effective_interception_intensity"],
            curve["alarm_before_recovery_rate"],
            marker="o",
            label=f"{block_qubits} qubits",
        )
    axes[1].set(
        xlabel="Effective interception intensity, K × f",
        ylabel="Alarm before recovery",
        title="Equal-grid attack averages",
        ylim=(0, 1),
    )
    axes[1].legend()
    return fig
