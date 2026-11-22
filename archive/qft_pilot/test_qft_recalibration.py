import unittest
from unittest.mock import patch

import numpy as np

from archive.qft_pilot.qft_inference import qft_backend_cache
from archive.qft_pilot.qft_protocol import qft_matrix, qft_null_outcome_distribution
from archive.qft_pilot.qft_recalibration import (
    _cumulative_residuals,
    _decoded_distribution,
    _sample_check_scores,
    _sample_noise_multipliers,
    compare_alarm_policies,
    run_qft_recalibration_demo,
)


class QFTRecalibrationTests(unittest.TestCase):
    def test_per_trial_batch_probabilities_match_qft_model(self):
        phases = np.array([[0.1, 1.2], [2.1, 0.6]])
        messages = np.array([[0, 1], [2, 3]])
        noise = np.array([[0.002, 0.038], [0.004, 0.076]])
        leg_rates = (noise[:, :1], noise[:, 1:])
        cache = qft_backend_cache(2, np)
        scrambler = np.exp(1j * (phases @ cache["bits"].T))
        observed = _decoded_distribution(
            scrambler[:, None, :], messages, (leg_rates,), cache, qft_matrix(2).conj().T,
        )
        expected = [
            [qft_null_outcome_distribution(2, message, pair, *rates) for message in row]
            for pair, row, rates in zip(phases, messages, noise)
        ]
        np.testing.assert_allclose(observed, expected, atol=1e-13)

    def test_live_baseline_only_changes_the_next_look(self):
        observed = np.array([[12.0, 14.0, 8.0]])
        scores = _cumulative_residuals(observed, initial_center=10.0, update_weight=0.5)
        np.testing.assert_array_equal(scores, [[2.0, 5.0, 0.5]])

    def test_identical_policies_have_no_paired_misses(self):
        calibration = np.zeros((200, 10))
        cases = {"Stationary": np.zeros((200, 10))}
        summary, curves = compare_alarm_policies(calibration, cases, update_weight=0.0)
        self.assertEqual(summary.loc[0, "alarm_count"], summary.loc[1, "alarm_count"])
        self.assertEqual(summary.loc[1, "fixed_only"], 0)
        self.assertEqual(curves["attempt"].tolist(), [5, 10, 5, 10])

    def test_noise_is_seeded_and_correlated_between_attempts(self):
        levels = _sample_noise_multipliers(np.random.default_rng(3), 500, 32)
        replay = _sample_noise_multipliers(np.random.default_rng(3), 500, 32)
        np.testing.assert_array_equal(levels, replay)
        correlation = np.corrcoef(levels[:-1].ravel(), levels[1:].ravel())[0, 1]
        self.assertGreater(correlation, 0.9)

    def test_demo_uses_paired_noise_through_attempt_500(self):
        with patch(
            "archive.qft_pilot.qft_recalibration._sample_check_scores",
            wraps=_sample_check_scores,
        ) as sampler:
            _, curves = run_qft_recalibration_demo(trials=2, calibration_trials=3)

        self.assertEqual(curves["attempt"].unique().tolist(), list(range(5, 501, 5)))
        paths = [call.args[2] for call in sampler.call_args_list]
        stable = [path for path in paths if path.ndim == 2]
        drifting = [path for path in paths if path.ndim == 3]
        self.assertEqual(len(stable), 3)
        self.assertEqual(len(drifting), 2)
        reference = np.array([0.002, 0.038])
        for path in stable:
            np.testing.assert_array_equal(path, np.tile(reference, (500, 1)))
        drift = drifting[0]
        self.assertEqual(drift.shape, (500, 2, 2))
        np.testing.assert_array_equal(drift, drifting[1])
        np.testing.assert_array_equal(drift[0], np.tile(reference, (2, 1)))
        levels = drift / reference
        np.testing.assert_allclose(levels[..., 0], levels[..., 1])
        self.assertTrue(np.all((levels >= 0.5) & (levels <= 2.0)))
        self.assertTrue(np.any(np.diff(levels[100:], axis=0)))
        self.assertFalse(np.array_equal(levels[100:, 0], levels[100:, 1]))
