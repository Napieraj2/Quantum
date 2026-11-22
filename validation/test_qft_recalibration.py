import unittest

import numpy as np

from quantum_study.qft_experiment import (
    qft_backend_cache,
    qft_matrix,
    qft_null_outcome_distribution,
)
from quantum_study.qft_recalibration import (
    _cumulative_residuals,
    _decoded_distribution,
    compare_alarm_policies,
)


class QFTRecalibrationTests(unittest.TestCase):
    def test_batch_probabilities_match_qft_model(self):
        phases = np.array([[0.1, 1.2], [2.1, 0.6]])
        messages = np.array([[0, 1], [2, 3]])
        noise = (0.002, 0.038)
        cache = qft_backend_cache(2, np)
        scrambler = np.exp(1j * (phases @ cache["bits"].T))
        observed = _decoded_distribution(
            scrambler[:, None, :], messages, (noise,), cache, qft_matrix(2).conj().T,
        )
        expected = [
            [qft_null_outcome_distribution(2, message, pair, *noise) for message in row]
            for pair, row in zip(phases, messages)
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
