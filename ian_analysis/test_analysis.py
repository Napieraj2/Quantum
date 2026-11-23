import unittest

import numpy as np

from bb84_power_check import (BASELINE, COUNTS, bit_error_probs, calculate,
                              count_grid, cutoff_exact, joint_probability,
                              normal_power, statistics_from_counts)
from qft_ideal_check import decode_no_attack, example_check, qft_matrix
from recovery_posterior import (first_declaration, posterior_from_log_likelihoods,
                                recovered_before_alarm)


class BB84ControlTests(unittest.TestCase):
    def test_no_attack_rates(self):
        np.testing.assert_allclose(bit_error_probs(0.0, 0.5), BASELINE)

    def test_uniform_eve_and_biased_eve(self):
        uniform = bit_error_probs(0.2, 0.5, np.zeros(4))
        np.testing.assert_allclose(uniform, np.full(4, 0.05))
        biased = bit_error_probs(0.2, 0.9, np.zeros(4))
        np.testing.assert_allclose(biased, [0.01, 0.01, 0.09, 0.09])

    def test_joint_mass_sums_to_one(self):
        grid = count_grid(COUNTS)
        self.assertAlmostEqual(float(joint_probability(grid, COUNTS, BASELINE).sum()), 1.0, 10)

    def test_exact_null_size_and_basis_equivalence(self):
        grid = count_grid(COUNTS)
        null_mass = joint_probability(grid, COUNTS, BASELINE)
        stats = statistics_from_counts(grid, COUNTS, BASELINE)
        for name in stats:
            cut = cutoff_exact(stats[name], null_mass)
            self.assertLessEqual(float(null_mass[stats[name] > cut].sum()), 0.05 + 1e-11)
        # The implementation reuses U2 under the U4 label because equal
        # within-basis baselines make the two proposal formulas identical.
        self.assertIs(stats["U2"], stats["U4"])
        weights4 = (1 - 2 * BASELINE) / (BASELINE * (1 - BASELINE))
        direct_u4 = (grid - COUNTS * BASELINE) @ weights4
        np.testing.assert_allclose(stats["U2"], direct_u4, atol=5.1e-10, rtol=0)

    def test_basis_score_rejects_unequal_within_basis_rates(self):
        q = np.array([0.002, 0.003, 0.038, 0.038])
        with self.assertRaises(ValueError):
            statistics_from_counts(np.zeros((1, 4), dtype=int), COUNTS, q)
        almost_equal = BASELINE.copy()
        almost_equal[1] = np.nextafter(almost_equal[0], 1.0)
        with self.assertRaises(ValueError):
            statistics_from_counts(
                np.zeros((1, 4), dtype=int), COUNTS, almost_equal
            )

    def test_normal_u4_retains_state_specific_weights(self):
        n = np.array([10, 20, 30, 40])
        q = np.array([0.01, 0.03, 0.07, 0.20])
        p = np.array([0.02, 0.08, 0.11, 0.27])
        u2 = normal_power("U2", 3.7, n, q, p)
        u4 = normal_power("U4", 3.7, n, q, p)
        self.assertNotAlmostEqual(u2, u4, places=6)

    def test_exact_power_and_monte_carlo_agree(self):
        rows = calculate(trials=15000, seed=123)
        for row in rows:
            self.assertLess(abs(row["power_exact"] - row["power_monte_carlo"]), 0.025)
            self.assertLessEqual(row["null_alpha_exact"], 0.05 + 1e-10)
        for row in rows:
            if row["test"] == "U4":
                companion = next(x for x in rows if x["test"] == "U2" and
                                 x["f"] == row["f"] and x["r_Z"] == row["r_Z"])
                for key in (
                    "null_alpha_exact", "cutoff_strict_gt", "power_exact",
                    "power_monte_carlo", "mc_se", "power_normal_approx",
                ):
                    self.assertEqual(row[key], companion[key])


class PosteriorTests(unittest.TestCase):
    def test_uniform_prior(self):
        np.testing.assert_allclose(
            posterior_from_log_likelihoods(log_likelihood=np.zeros(16)),
            np.full(16, 1 / 16),
        )

    def test_confident_declaration_and_alarm(self):
        first = np.zeros(16)
        second = np.zeros(16)
        second[3] = 10
        d = first_declaration(log_likelihood_history=[first, second])
        self.assertEqual(d[0], 2)
        self.assertEqual(d[1], 3)
        self.assertTrue(recovered_before_alarm(d, true_secret=3, first_alarm_attempt=3))
        self.assertFalse(recovered_before_alarm(d, true_secret=3, first_alarm_attempt=2))
        self.assertFalse(recovered_before_alarm(d, true_secret=4, first_alarm_attempt=3))

    def test_no_valid_likelihood_rejected(self):
        with self.assertRaises(ValueError):
            posterior_from_log_likelihoods(np.full(16, -np.inf))

    def test_impossible_candidates_normalize(self):
        evidence = np.full(16, -np.inf)
        evidence[7] = -3.0
        expected = np.zeros(16)
        expected[7] = 1.0
        np.testing.assert_array_equal(
            posterior_from_log_likelihoods(evidence), expected
        )

    def test_bad_likelihood_shape_rejected(self):
        with self.assertRaises(ValueError):
            posterior_from_log_likelihoods(np.zeros(4))


class QFTControlTests(unittest.TestCase):
    def test_ideal_blocks(self):
        self.assertLess(example_check(), 1e-11)

    def test_qft_is_unitary_and_inputs_are_checked(self):
        for qubits in (2, 4):
            transform = qft_matrix(qubits)
            np.testing.assert_allclose(
                transform.conj().T @ transform,
                np.eye(1 << qubits),
                atol=1e-14,
            )
        with self.assertRaises(ValueError):
            decode_no_attack(1.5, np.zeros(2))
        with self.assertRaises(ValueError):
            decode_no_attack(1, np.array([0.0, np.nan]))


if __name__ == "__main__":
    unittest.main()
