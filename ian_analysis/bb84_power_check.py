"""Exact single-look BB84 detector power with known error rates.

The attack model and score definitions are the controls specified in the
project proposal, §2.  The reference basis-error rates come from Erven et al.,
§4 (https://arxiv.org/abs/0901.0960v2).  This file deliberately remains an
independent fixed-count calculation rather than importing the main study.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, norm


STATES = ("Z0", "Z1", "X+", "X-")
BASELINE = np.array([0.002, 0.002, 0.038, 0.038], dtype=float)
COUNTS = np.array([25, 25, 25, 25], dtype=int)
F_VALUES = (0.0, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20)
R_VALUES = (0.5, 0.9)
ALPHA = 0.05
TRIALS = 50000
SEED = 20261114


def bit_error_probs(
    f: float, r: float, q: np.ndarray = BASELINE
) -> np.ndarray:
    """Combine the background error rate with one intercept-resend attack."""
    if not 0 <= f <= 1 or not 0 <= r <= 1:
        raise ValueError("f and r must both be between 0 and 1")
    q = np.asarray(q, dtype=float)
    if (
        q.shape != (4,)
        or not np.isfinite(q).all()
        or np.any((q < 0) | (q > 1))
    ):
        raise ValueError("Expected four baseline probabilities between 0 and 1")
    attack = np.array([f * (1 - r) / 2] * 2 + [f * r / 2] * 2)
    return q + (1 - 2 * q) * attack


def count_grid(n: np.ndarray) -> np.ndarray:
    n = np.asarray(n)
    if (
        n.shape != (4,)
        or not np.issubdtype(n.dtype, np.integer)
        or np.any(n < 0)
    ):
        raise ValueError("Expected four nonnegative integer state counts")
    shape = tuple(int(k)+1 for k in n)
    return np.indices(shape, dtype=np.int16).reshape(4, -1).T


def joint_probability(grid: np.ndarray, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    n = np.asarray(n)
    p = np.asarray(p, dtype=float)
    if grid.ndim != 2 or grid.shape[1] != 4:
        raise ValueError("Expected one four-state error count per row")
    if n.shape != (4,) or p.shape != (4,):
        raise ValueError("Expected four state counts and four probabilities")
    if (
        not np.isfinite(p).all()
        or np.any((grid < 0) | (grid > n))
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError("Counts or probabilities are outside their support")
    pmf = np.ones(len(grid), dtype=float)
    for s in range(4):
        pmf *= binom.pmf(grid[:, s], int(n[s]), float(p[s]))
    return pmf


def statistics_from_counts(
    grid: np.ndarray, n: np.ndarray, q: np.ndarray
) -> dict[str, np.ndarray]:
    grid = np.asarray(grid)
    n = np.asarray(n)
    q = np.asarray(q, dtype=float)
    if (
        grid.ndim != 2
        or grid.shape[1] != 4
        or n.shape != (4,)
        or q.shape != (4,)
    ):
        raise ValueError("Expected four state counts and four baseline rates")
    if np.any((q <= 0) | (q >= 1)):
        raise ValueError("Score control requires baselines strictly between 0 and 1")
    if not (q[0] == q[1] and q[2] == q[3]):
        raise ValueError("This U2/U4 comparison assumes equal rates within each basis")
    total = grid.sum(axis=1).astype(float)
    conditioned = np.zeros(len(grid), dtype=float)
    for idx in ((0, 1), (2, 3)):
        q_b = np.mean(q[list(idx)])
        n_b = n[list(idx)].sum()
        e_b = grid[:, list(idx)].sum(axis=1)
        w_b = (1 - 2 * q_b) / (q_b * (1 - q_b))
        conditioned += w_b * (e_b - n_b * q_b)
    # Under this control's equal within-basis rates, the proposal's U4
    # expression reduces algebraically to U2.  Reuse the canonicalized score
    # under both proposal labels instead of evaluating the same statistic twice.
    conditioned = np.round(conditioned, 9)
    return {"pooled": total, "U2": conditioned, "U4": conditioned}


def cutoff_exact(
    statistic: np.ndarray, probability: np.ndarray, alpha: float = ALPHA
) -> float:
    """Cutoff for the strict rule statistic > cutoff (no random ties)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    statistic = np.asarray(statistic, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if statistic.ndim != 1 or probability.shape != statistic.shape:
        raise ValueError("Statistic and probability must be equal-length vectors")
    if not np.isfinite(statistic).all() or np.any(probability < 0):
        raise ValueError("Statistics must be finite and probabilities nonnegative")
    if not np.isclose(probability.sum(), 1.0, atol=1e-11, rtol=0):
        raise ValueError("Probabilities must sum to one")
    order = np.argsort(statistic, kind="stable")
    cdf = np.cumsum(probability[order])
    idx = min(int(np.searchsorted(cdf, 1 - alpha, side="left")), len(order)-1)
    return float(statistic[order[idx]])


def normal_power(
    name: str, cutoff: float, n: np.ndarray, q: np.ndarray,
    p: np.ndarray,
) -> float:
    if name == "pooled":
        mean = float(np.dot(n, p))
        variance = float(np.dot(n, p * (1 - p)))
        # T is an integer, so use a half-count correction for the strict cutoff.
        boundary = cutoff + 0.5
    elif name in ("U2", "U4"):
        b0 = np.mean(q[:2])
        b1 = np.mean(q[2:])
        detector_baseline = (
            np.array([b0, b0, b1, b1]) if name == "U2" else q
        )
        weights = (
            (1 - 2 * detector_baseline)
            / (detector_baseline * (1 - detector_baseline))
        )
        # Preserve the supplied per-state null means; detector_baseline only
        # selects the U2 basis weights or the genuine U4 state weights.
        mean = float(np.dot(weights * n, p - q))
        variance = float(np.dot(weights * weights * n, p * (1 - p)))
        boundary = cutoff
    else:
        raise ValueError("name must be 'pooled', 'U2', or 'U4'")
    if variance == 0:
        return float(mean > boundary)
    return float(norm.sf((boundary - mean) / np.sqrt(variance)))


def calculate(trials: int = TRIALS, seed: int = SEED):
    n = COUNTS.copy()
    q = BASELINE.copy()
    grid = count_grid(n)
    null_mass = joint_probability(grid, n, q)
    stats = statistics_from_counts(grid, n, q)
    names = ("pooled", "U2")
    cutoffs = {name: cutoff_exact(stats[name], null_mass) for name in names}
    masks = {name: stats[name] > cutoffs[name] for name in names}
    null_sizes = {
        name: float(null_mass[masks[name]].sum()) for name in names
    }
    if any(a > ALPHA + 1e-10 for a in null_sizes.values()):
        raise AssertionError("The exact single-look false-alarm target was exceeded")

    rng = np.random.default_rng(seed)
    rows = []
    for r in R_VALUES:
        for f in F_VALUES:
            p = bit_error_probs(f, r, q)
            attacked_mass = joint_probability(grid, n, p)
            observed_counts = rng.binomial(n, p, size=(trials, 4))
            observed_stats = statistics_from_counts(observed_counts, n, q)
            for name in names:
                cutoff = cutoffs[name]
                exact_power = float(attacked_mass[masks[name]].sum())
                empirical_power = float(np.mean(observed_stats[name] > cutoff))
                mc_se = np.sqrt(empirical_power * (1 - empirical_power) / trials)
                rows.append({
                    "n": int(n.sum()), "r_Z": r, "f": f, "test": name,
                    "null_alpha_exact": null_sizes[name],
                    "cutoff_strict_gt": cutoff,
                    "power_exact": exact_power,
                    "power_monte_carlo": empirical_power,
                    "mc_se": float(mc_se),
                    "power_normal_approx": normal_power(name, cutoff, n, q, p),
                    "qZ_attack": float(p[0]), "qX_attack": float(p[2]),
                    "trials": trials,
                })
            # U4 and U2 coincide under these known, within-basis equal rates.
            rows.append(dict(rows[-1], test="U4"))
    return rows


def save_results(rows: list[dict], out: Path):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "bb84_single_look_power.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for label, title, marker in (("pooled", "Pooled count", "o"),
                                 ("U2", "Basis-grouped score (U2 = U4)", "s")):
        selected = [row for row in rows if row["r_Z"] == 0.5 and row["test"] == label]
        ax.plot([row["f"] for row in selected], [row["power_exact"] for row in selected],
                marker=marker, label=title)
    ax.axhline(ALPHA, ls="--", color="gray", alpha=0.7, label="5% false-alarm target")
    ax.set(xlabel="Interception probability f", ylabel="Detection power",
           title="Single-look BB84 control: n=100, known baselines, uniform Eve")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "bb84_single_look_power.png", dpi=180)
    plt.close(fig)


def main():
    out = (Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()) / "results"
    rows = calculate()
    save_results(rows, out)
    print("Known baselines: qZ=0.002, qX=0.038, n=100 (25/state)")
    print("Exact null sizes: ", {x["test"]: round(x["null_alpha_exact"], 5)
                                  for x in rows if x["f"] == 0 and x["r_Z"] == 0.5})
    print("r=0.5: exact power by f")
    for f in F_VALUES:
        selected = [x for x in rows if x["r_Z"] == 0.5 and x["f"] == f]
        print(f"  f={f:6.3f}: pooled={selected[0]['power_exact']:.4f}; "
              f"U2={selected[1]['power_exact']:.4f}; U4={selected[2]['power_exact']:.4f}")
    print(f"Wrote files to {out}")


if __name__ == "__main__":
    main()
