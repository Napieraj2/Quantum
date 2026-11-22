import unittest
from itertools import product

import numpy as np

from archive.qft_pilot.qft_inference import qft_record_likelihood_scalar
from archive.qft_pilot.qft_protocol import (
    qft_bits,
    qft_check_distribution,
    qft_integer,
    qft_message,
)
from quantum_study.qft_exchange import (
    _null_decoding_probabilities,
    _payload_likelihoods,
    _rotations,
)
from quantum_study.qft_study import (
    _curve_rows,
    _study_rows,
    run_qft_study,
)


class QFTStudyTests(unittest.TestCase):
    def test_return_only_records_do_not_reveal_the_message(self):
        for qubits in (2, 4):
            forward = np.full(qubits, -1)
            returned = np.full(qubits, -1)
            returned[0], returned[-1] = 0, 3
            likelihoods = _payload_likelihoods(_rotations(qubits), forward, returned, 0.038)
            np.testing.assert_array_equal(likelihoods, 1)

    def test_informative_record_pairs_match_exact_density_average(self):
        records = [
            dict(leg="forward", signal=0, basis="Z", outcome=0),
            dict(leg="forward", signal=0, basis="X", outcome=0),
            dict(leg="forward", signal=1, basis="X", outcome=1),
            dict(leg="return", signal=0, basis="X", outcome=1),
            dict(leg="return", signal=0, basis="Z", outcome=0),
            dict(leg="return", signal=1, basis="Z", outcome=0),
        ]
        for qubits in (2, 4):
            phase_grid = np.array(list(product((0, np.pi), repeat=qubits)))
            reference = np.array([
                qft_record_likelihood_scalar(qubits, message, records, 0.002, 0.038, phase_grid)
                for message in range(2**qubits)
            ])
            forward = np.full(qubits, -1)
            returned = np.full(qubits, -1)
            forward[:2], returned[:2] = (2, 3), (3, 0)
            likelihoods = _payload_likelihoods(_rotations(qubits), forward, returned, 0.038)
            np.testing.assert_allclose(likelihoods / likelihoods.sum(), reference / reference.sum())

    def test_joint_check_probability_matches_archived_scalar_model(self):
        q_z, q_x = 0.002, 0.038
        for qubits in (2, 4):
            phases = np.linspace(0.17, 2.41, qubits)
            checks = tuple(range(0, qubits, 2))
            values = tuple((index + 1) % 2 for index in range(len(checks)))
            reference = qft_check_distribution(
                qubits, checks, values, phases, q_z, q_x
            )
            width = qubits // 2
            candidates = np.array([
                qft_message(checks, values, qft_bits(payload, width), qubits)
                for payload in range(2**width)
            ])
            probabilities = _null_decoding_probabilities(
                np.exp(1j * phases), candidates, _rotations(qubits), q_z, q_x
            )
            for outcome in (0, 2 ** (qubits - 1), 2**qubits - 1):
                outcome_bits = qft_bits(outcome, qubits)
                pattern = qft_integer([
                    outcome_bits[position] for position in checks
                ])
                candidate = sum(
                    probability for word, probability in enumerate(probabilities)
                    if all(qft_bits(word, qubits)[position] == outcome_bits[position] for position in checks)
                )
                self.assertAlmostEqual(candidate, reference[pattern], places=14)

    def test_same_attempt_alarm_wins_and_freezes_prior_posterior(self):
        attempts = 4
        paths = {
            "secret": np.array([0]),
            "check_count": np.full((1, attempts), 4),
            "bit_posterior": np.array([[
                [0.1, 0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3, 0.3],
                [0.4, 0.4, 0.4, 0.4],
            ]]),
        }
        alarm_attempt = np.array([2])
        declaration_attempt = np.array([2])
        summary, alarm_first, correct, declaration_first, _ = _study_rows(
            paths,
            "Joint-check surprisal",
            alarm_attempt,
            np.array([attempts + 1]),
            declaration_attempt,
            np.array([0]),
            attempts,
            block_qubits=2,
        )
        self.assertEqual(summary["alarm_rate"], 1.0)
        self.assertEqual(summary["correct_recovery_rate"], 0.0)
        self.assertEqual(summary["mean_checks"], 8.0)
        self.assertTrue(alarm_first[0])
        self.assertFalse(correct[0])
        self.assertFalse(declaration_first[0])

        curves = _curve_rows(
            paths,
            "Joint-check surprisal",
            alarm_attempt,
            declaration_attempt,
            alarm_first,
            correct,
            declaration_first,
            attempts,
            block_qubits=2,
        )
        self.assertTrue(all(row["bit_1"] == 0.9 for row in curves))

    def test_no_interception_keeps_paired_recovery_uninformed(self):
        for qubits in (2, 4):
            summary, curves = run_qft_study(
                attempts=4,
                trials=8,
                block_qubits=qubits,
                intercept_probability=0.0,
                q_z=0.0,
                q_x=0.0,
                seed=20261122,
            )
            self.assertEqual(set(summary["detector"]), {
                "Pooled check errors",
                "Joint-check surprisal",
            })
            np.testing.assert_array_equal(summary["trials"], 8)
            np.testing.assert_allclose(summary["correct_recovery_rate"], 0.0)
            np.testing.assert_allclose(summary["incorrect_recovery_rate"], 0.0)
            for column in ("bit_1", "bit_2", "bit_3", "bit_4"):
                np.testing.assert_allclose(curves[column], 0.5)


if __name__ == "__main__":
    unittest.main()
