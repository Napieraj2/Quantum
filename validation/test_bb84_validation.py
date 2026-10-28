"""Small checks for the BB84 validation code."""

import unittest
from pathlib import Path

import numpy as np

from independent_bb84_validation import SEED, measurement_probabilities, simulate_rounds


class BB84ValidationTests(unittest.TestCase):
    def test_measurement_probabilities(self):
        table = measurement_probabilities()
        self.assertEqual(table.shape, (4, 2, 2))
        np.testing.assert_allclose(table.sum(axis=2), 1.0)
        np.testing.assert_allclose(table[0, 0], (1.0, 0.0))
        np.testing.assert_allclose(table[3, 1], (0.0, 1.0))
        np.testing.assert_allclose(table[2, 0], (0.5, 0.5))

    def test_no_eve_and_no_baseline_errors(self):
        result = simulate_rounds(50_000, 0.0, (0, 0, 0, 0), seed=42)
        self.assertEqual(result["errors"], 0)
        self.assertEqual(result["qber"], 0.0)

    def test_full_intercept(self):
        result = simulate_rounds(200_000, 1.0, (0, 0, 0, 0), seed=42)
        self.assertAlmostEqual(result["qber"], 0.25, delta=0.006)

    def test_unequal_baselines(self):
        f = 0.2
        q0 = np.array((0.01, 0.03, 0.05, 0.11))
        result = simulate_rounds(400_000, f, q0, seed=SEED + 100)
        predicted = q0 + (1 - 2 * q0) * f / 4
        measured = result["state_errors"] / result["state_counts"]
        standard_error = np.sqrt(predicted * (1 - predicted) / result["state_counts"])
        self.assertTrue(np.all(np.abs(measured - predicted) < 6 * standard_error))

    def test_same_seed_same_counts(self):
        first = simulate_rounds(20_000, 0.4, (0, 0, 0, 0), seed=17)
        second = simulate_rounds(20_000, 0.4, (0, 0, 0, 0), seed=17)
        self.assertEqual(first["sifted"], second["sifted"])
        self.assertEqual(first["errors"], second["errors"])

    def test_runs_without_file_variable(self):
        # Check the editor-style case where the runner does not set __file__.
        script = Path(__file__).with_name("independent_bb84_validation.py")
        namespace = {"__name__": "bb84_check"}
        exec(compile(script.read_text(encoding="utf-8"), "<exec>", "exec"), namespace)
        self.assertEqual(namespace["OUTPUT_DIR"].name, "results")

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            simulate_rounds(10, 1.5, (0, 0, 0, 0), seed=1)
        with self.assertRaises(ValueError):
            simulate_rounds(10, 0.2, (0.01, 0.03), seed=1)


if __name__ == "__main__":
    unittest.main()
