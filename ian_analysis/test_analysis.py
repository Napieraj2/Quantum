import unittest

import numpy as np

from bb84_power_check import (BASELINE, COUNTS, bit_error_probs, calculate,
                              count_grid, cutoff_exact, joint_probability, statistics_from_counts)
from qft_ideal_check import decode_no_attack, example_check
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
        np.testing.assert_allclose(stats["U2"], stats["U4"], atol=2e-9, rtol=0)

    def test_basis_score_rejects_unequal_within_basis_rates(self):
        q = np.array([0.002, 0.003, 0.038, 0.038])
        with self.assertRaises(ValueError):
            statistics_from_counts(np.zeros((1, 4), dtype=int), COUNTS, q)

    def test_exact_power_and_monte_carlo_agree(self):
        rows = calculate(trials=15000, seed=123)
        for row in rows:
            self.assertLess(abs(row["power_exact"] - row["power_monte_carlo"]), 0.025)
            self.assertLessEqual(row["null_alpha_exact"], 0.05 + 1e-10)
        for row in rows:
            if row["test"] == "U4":
                companion = next(x for x in rows if x["test"] == "U2" and
                                 x["f"] == row["f"] and x["r_Z"] == row["r_Z"])
                self.assertAlmostEqual(row["power_exact"], companion["power_exact"], 10)


class PosteriorTests(unittest.TestCase):
    def test_uniform_prior(self):
        np.testing.assert_allclose(posterior_from_log_likelihoods(np.zeros(16)), np.full(16, 1/16))

    def test_confident_declaration_and_alarm(self):
        first = np.zeros(16)
        second = np.zeros(16)
        second[3] = 10
        d = first_declaration([first, second])
        self.assertEqual(d[0], 2)
        self.assertEqual(d[1], 3)
        self.assertTrue(recovered_before_alarm(d, true_secret=3, first_alarm_attempt=3))
        self.assertFalse(recovered_before_alarm(d, true_secret=3, first_alarm_attempt=2))
        self.assertFalse(recovered_before_alarm(d, true_secret=4, first_alarm_attempt=3))

    def test_no_valid_likelihood_rejected(self):
        with self.assertRaises(ValueError):
            posterior_from_log_likelihoods(np.full(16, -np.inf))

    def test_bad_likelihood_shape_rejected(self):
        with self.assertRaises(ValueError):
            posterior_from_log_likelihoods(np.zeros(4))


class QFTControlTests(unittest.TestCase):
    def test_ideal_blocks(self):
        self.assertLess(example_check(), 1e-11)

    def test_three_secret_block_values(self):
        for bits, m in ((2, 0), (2, 3), (4, 11)):
            amplitude = decode_no_attack(m, np.arange(bits)/5)
            self.assertAlmostEqual(float(abs(amplitude[m])**2), 1.0, 10)


if __name__ == "__main__":
    unittest.main()
