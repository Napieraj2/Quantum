"""Check BB84 measurement probabilities and intercept-resend error rates.

Run from the repo root:
    python validation/independent_bb84_validation.py
"""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 20261026
ROUNDS_PER_TEST = 400_000
INTERCEPT_FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
STATES = ("Z0", "Z1", "X+", "X-")
# Some editors run the code without a file path. Use the repo folder in that case.
if "__file__" in globals():
    OUTPUT_DIR = Path(__file__).resolve().parent / "results"
else:
    current = Path.cwd()
    folder = current / "validation" if (current / "validation").is_dir() else current
    OUTPUT_DIR = folder / "results"


def measurement_probabilities():
    """Probabilities for the four BB84 states measured in Z or X."""
    zero = np.array([1.0, 0.0], dtype=complex)
    one = np.array([0.0, 1.0], dtype=complex)
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2)
    prepared = (zero, one, h @ zero, h @ one)

    # Each row is one prepared state; the two columns are Z and X measurements.
    return np.array([
        [np.abs(state) ** 2, np.abs(h @ state) ** 2]
        for state in prepared
    ])


def simulate_rounds(n, f, baseline_errors, seed):
    """Send n BB84 states through a partial intercept-resend attack."""
    q0 = np.asarray(baseline_errors, dtype=float)
    if n <= 0 or not 0 <= f <= 1:
        raise ValueError("n must be positive and f must be between 0 and 1")
    if q0.shape != (4,) or np.any((q0 < 0) | (q0 >= 0.5)):
        raise ValueError("Provide four baseline error rates between 0 and 0.5")

    rng = np.random.default_rng(seed)

    # Alice chooses a bit and basis. Bob independently chooses his basis.
    alice_basis = rng.integers(0, 2, n, dtype=np.int8)  # 0: Z, 1: X
    alice_bit = rng.integers(0, 2, n, dtype=np.int8)
    bob_basis = rng.integers(0, 2, n, dtype=np.int8)

    # Eve measures in a random basis on the rounds she intercepts.
    intercepted = rng.random(n) < f
    eve_basis = rng.integers(0, 2, n, dtype=np.int8)
    eve_random_bit = rng.integers(0, 2, n, dtype=np.int8)
    eve_bit = np.where(eve_basis == alice_basis, alice_bit, eve_random_bit)

    # Without Eve, Bob gets Alice's state. With Eve, he gets her resent state.
    received_basis = np.where(intercepted, eve_basis, alice_basis)
    received_bit = np.where(intercepted, eve_bit, alice_bit)
    bob_random_bit = rng.integers(0, 2, n, dtype=np.int8)
    bob_bit = np.where(bob_basis == received_basis, received_bit, bob_random_bit)

    # This extra error is a flip of Bob's measured bit, not a hardware noise model.
    state_id = 2 * alice_basis + alice_bit  # Z0, Z1, X+, X-
    baseline_flip = (rng.random(n) < q0[state_id]).astype(np.int8)
    bob_bit = np.bitwise_xor(bob_bit, baseline_flip)

    # Sifting: only compare rounds where Alice and Bob used the same basis.
    keep = alice_basis == bob_basis
    wrong_bit = bob_bit[keep] != alice_bit[keep]
    kept_states = state_id[keep]
    counts = np.bincount(kept_states, minlength=4)
    errors = np.bincount(kept_states[wrong_bit], minlength=4)

    total = int(counts.sum())
    if total == 0:
        raise ValueError("No rounds left after basis sifting")
    return {
        "transmitted": n,
        "intercept_fraction": f,
        "sifted": total,
        "errors": int(errors.sum()),
        "qber": int(errors.sum()) / total,
        "state_counts": counts,
        "state_errors": errors,
    }


def save_measurement_cases(folder):
    actual = measurement_probabilities()
    expected = np.array([
        [[1.0, 0.0], [0.5, 0.5]],  # Z0
        [[0.0, 1.0], [0.5, 0.5]],  # Z1
        [[0.5, 0.5], [1.0, 0.0]],  # X+
        [[0.5, 0.5], [0.0, 1.0]],  # X-
    ])
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)

    with (folder / "eight_case_probabilities.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["prepared", "measured_basis", "P(outcome0)", "P(outcome1)"])
        for i, state in enumerate(STATES):
            for j, basis in enumerate(("Z", "X")):
                writer.writerow([state, basis, f"{actual[i, j, 0]:.6f}",
                                 f"{actual[i, j, 1]:.6f}"])
    print("PASS: all eight state/basis measurement probabilities")


def save_ideal_qber(folder):
    rows = []
    for i, f in enumerate(INTERCEPT_FRACTIONS):
        result = simulate_rounds(ROUNDS_PER_TEST, f, (0, 0, 0, 0), SEED + i)
        predicted = f / 4
        # Standard error for the observed QBER over the sifted sample.
        se = np.sqrt(predicted * (1 - predicted) / result["sifted"])
        tolerance = 6 * se + 2 / result["sifted"]
        if abs(result["qber"] - predicted) > tolerance:
            raise AssertionError(f"f={f}: measured QBER is too far from f/4")

        rows.append((f, result["transmitted"], result["sifted"],
                     result["errors"], predicted, result["qber"], se))

    with (folder / "intercept_resend_ideal.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["intercept_fraction_f", "transmitted_rounds_N", "sifted_rounds_M",
                         "errors_E", "analytic_f_over_4", "empirical_QBER", "binomial_SE"])
        writer.writerows(rows)

    print("PASS: simulated QBER agrees with f/4")
    print("f      sifted bits   errors   simulated QBER   expected QBER")
    for f, _, m, e, predicted, measured, _ in rows:
        print(f"{f:0.1f}    {m:>8}       {e:>5}        {measured:.5f}          {predicted:.5f}")
    return rows


def save_state_errors(folder):
    f = 0.2
    q0 = np.array((0.01, 0.03, 0.05, 0.11))
    data = simulate_rounds(ROUNDS_PER_TEST, f, q0, SEED + 100)

    # A baseline flip and an attack error cancel if they both happen.
    # That gives q_s(f) = q0_s + (1 - 2*q0_s)*f/4.
    with (folder / "state_conditioned_xor_model.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["state", "m_s", "E_s", "q0_s", "analytic_q_s(f)",
                         "empirical_error_rate", "SE"])
        for i, state in enumerate(STATES):
            predicted = q0[i] + (1 - 2 * q0[i]) * f / 4
            measured = data["state_errors"][i] / data["state_counts"][i]
            se = np.sqrt(predicted * (1 - predicted) / data["state_counts"][i])
            if abs(measured - predicted) > 6 * se + 2 / data["state_counts"][i]:
                raise AssertionError(f"{state}: error rate is outside the expected range")
            writer.writerow([state, data["state_counts"][i], data["state_errors"][i],
                             q0[i], predicted, measured, se])
    print("PASS: four-state baseline error check")


def save_plot(rows, folder):
    fractions = np.array([row[0] for row in rows])
    measured = np.array([row[5] for row in rows])
    errors = np.array([1.96 * row[6] for row in rows])
    x_line = np.linspace(0, 1, 101)

    fig, ax = plt.subplots(figsize=(7.6, 4.15), dpi=170)
    ax.plot(100 * x_line, 25 * x_line, label="Expected QBER = f/4", lw=1.8)
    ax.errorbar(100 * fractions, 100 * measured, yerr=100 * errors,
                fmt="o", capsize=3, label="Simulated QBER (approx. 95% error bars)")
    ax.set(title="BB84 intercept-resend: theory vs simulation",
           xlabel="Intercepted signals (%)", ylabel="Sifted-bit error rate (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 28)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(folder / "bb84_ideal_validation.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_measurement_cases(OUTPUT_DIR)
    rows = save_ideal_qber(OUTPUT_DIR)
    save_state_errors(OUTPUT_DIR)
    save_plot(rows, OUTPUT_DIR)
    print("Results saved in validation/results/")


if __name__ == "__main__":
    main()
