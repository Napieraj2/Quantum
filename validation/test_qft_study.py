import unittest

import numpy as np

from quantum_study.qft_alarm import _look_schedules, _replay_alarms
from quantum_study.qft_exchange import (
    _null_decoding_probabilities,
    _payload_likelihoods,
    _rotations,
    simulate_paths,
)
from quantum_study.qft_study import (
    _curve_rows,
    _declarations,
    _study_rows,
    run_qft_study,
)


# Independent full-density known answers for Papadopoulos–Linvill §4.1,
# Eqs. (1)–(3), with the proposal's local X(q_z)/Z(q_x) channel on both legs.
# The Eve fixture phase-twirls the input exactly to I/2**q before applying the
# ordered record projectors; the check fixture applies Scr, Enc, Scr†, and QFT†.
_RECORD_POSTERIORS = {
    2: np.tile([0.019, 0.481], 2),
    4: np.tile([0.00475, 0.12025], 8),
}
# Check values are 1 at position 0 (2q), and 1/0 at positions 0/2 (4q).
_CHECK_CANDIDATES = {
    2: np.array([2, 3]),
    4: np.array([8, 9, 12, 13]),
}
_CHECK_DISTRIBUTIONS = {
    2: np.array([0.10591964666739746, 0.8940803533326026]),
    4: np.array([
        0.09868142955120347,
        0.0362907458738645,
        0.7949530707668979,
        0.07007475380803409,
    ]),
}


class QFTStudyTests(unittest.TestCase):
    def test_random_alarm_looks_include_the_end_and_vary_by_session(self):
        schedules = _look_schedules(np.random.default_rng(17), 64, 100, (2, 4))
        self.assertTrue((schedules["Percentage"] == [25, 50, 100]).all())
        random = schedules["Random"]
        np.testing.assert_array_equal(random[:, -1], 100)
        for looks in random:
            interim = looks[looks < 100]
            self.assertTrue((interim > 0).all())
            self.assertTrue((np.diff(interim) > 0).all())
        self.assertEqual(set((random < 100).sum(axis=1) + 1), {2, 3, 4})

    def test_interim_alarm_reset_uses_only_checks_from_the_new_window(self):
        totals = np.array([[2, 5, 5, 5], [0, 0, 0, 6], [0, 5, 9, 14], [0, 0, 0, 0]])
        looks = np.array([[2, 4, 4, 4], [1, 3, 4, 4], [2, 3, 4, 4], [1, 2, 3, 4]])
        rule = (np.arange(1, 5), np.ones(4), 1.0, 0.0)
        first, reset, final = _replay_alarms(totals, looks, rule, np.ones((4, 4)))
        np.testing.assert_array_equal(first, [2, 4, 2, 5])
        np.testing.assert_array_equal(reset, [True, False, True, False])
        np.testing.assert_array_equal(final, [False, True, True, False])

    def test_cutoff_tie_draw_restarts_only_after_a_reset(self):
        looks = np.array([[1, 2, 4], [1, 2, 4]])
        ties = np.array([[0.1, 0.9, 0.1], [0.9, 0.1, 0.1]])
        rule = (np.zeros(4), np.zeros(4), 0.0, 0.5)
        first, reset, final = _replay_alarms(np.zeros((2, 4)), looks, rule, ties)
        np.testing.assert_array_equal(first, [1, 5])
        np.testing.assert_array_equal(reset, [True, False])
        np.testing.assert_array_equal(final, [False, False])

    def test_return_only_records_do_not_reveal_the_message(self):
        for qubits in (2, 4):
            forward = np.full(qubits, -1)
            returned = np.full(qubits, -1)
            returned[0], returned[-1] = 0, 3
            likelihoods = _payload_likelihoods(
                _rotations(qubits), forward, returned, 0.038
            )
            np.testing.assert_array_equal(likelihoods, 1)

    def test_informative_record_pairs_match_full_density_fixture(self):
        # Ordered full-density transcript: forward s0 Z0→X0, s1 X1;
        # return s0 X1→Z0, s1 Z0. The arrays retain the sufficient last
        # forward (X0, X1) and first return (X1, Z0) records.
        for qubits in (2, 4):
            forward = np.full(qubits, -1)
            returned = np.full(qubits, -1)
            forward[:2], returned[:2] = (2, 3), (3, 0)
            likelihoods = _payload_likelihoods(
                _rotations(qubits), forward, returned, 0.038
            )
            np.testing.assert_allclose(
                likelihoods / likelihoods.sum(),
                _RECORD_POSTERIORS[qubits],
                rtol=0,
                atol=1e-14,
            )

    def test_joint_check_probability_matches_full_density_fixture(self):
        q_z, q_x = 0.002, 0.038
        for qubits in (2, 4):
            phases = np.linspace(0.17, 2.41, qubits)
            checks = tuple(range(0, qubits, 2))
            probabilities = _null_decoding_probabilities(
                np.exp(1j * phases), _CHECK_CANDIDATES[qubits],
                _rotations(qubits), q_z, q_x,
            )
            words = np.arange(2**qubits)
            patterns = np.zeros_like(words)
            for position in checks:
                patterns = (
                    2 * patterns
                    + ((words >> (qubits - 1 - position)) & 1)
                )
            candidate = np.bincount(
                patterns, weights=probabilities, minlength=2 ** len(checks)
            )
            np.testing.assert_allclose(
                candidate, _CHECK_DISTRIBUTIONS[qubits], rtol=0, atol=1e-14
            )

    def test_same_attempt_alarm_wins_and_freezes_prior_posterior(self):
        attempts = 4
        paths = {
            "secret": np.array([0]),
            "final_key_agrees": np.array([True]),
            "bit_posterior": np.array([[
                [0.1, 0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3, 0.3],
                [0.4, 0.4, 0.4, 0.4],
            ]]),
        }
        alarm_attempt = np.array([2])
        declaration_attempt = np.array([2])
        summary, alarm_first, correct, declaration_first = _study_rows(
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

    def test_session_cost_continues_after_eves_declaration(self):
        paths = {"secret": np.zeros(2, dtype=int), "final_key_agrees": np.ones(2, dtype=bool)}
        for block_qubits in (2, 4):
            summary, *_ = _study_rows(
                paths, "Pooled check errors", np.array([5, 3]), np.full(2, 5),
                np.array([2, 2]), np.zeros(2, dtype=int), attempts=4, block_qubits=block_qubits,
            )
            self.assertEqual(summary["correct_recovery_rate"], 1.0)
            self.assertEqual(summary["mean_checks"], 14)
            self.assertEqual(summary["mean_transmitted_qubits"], 56)

    def test_accepted_key_exposure_uses_acceptance_denominator(self):
        attempts = 4
        paths = {
            "secret": np.zeros(4, dtype=int),
            # A later alarm, a final mismatch, an exposed acceptance, and an
            # accepted session without a declaration, respectively.
            "final_key_agrees": np.array([True, False, True, True]),
        }
        alarm_attempt = np.array([3, 5, 5, 5])
        declaration_attempt = np.array([2, 2, 2, 5])
        declaration_guess = np.zeros(4, dtype=int)
        summary, _, correct, _ = _study_rows(
            paths,
            "Joint-check surprisal",
            alarm_attempt,
            np.full(4, attempts + 1),
            declaration_attempt,
            declaration_guess,
            attempts,
            block_qubits=2,
        )
        np.testing.assert_array_equal(correct, [True, True, True, False])
        self.assertEqual(summary["accepted_sessions"], 2)
        self.assertEqual(summary["exposed_accepted_keys"], 1)
        self.assertEqual(summary["accepted_key_exposure_rate"], 0.5)

        no_acceptance, *_ = _study_rows(
            {
                "secret": paths["secret"][:2],
                "final_key_agrees": paths["final_key_agrees"][:2],
            },
            "Joint-check surprisal",
            alarm_attempt[:2],
            np.full(2, attempts + 1),
            declaration_attempt[:2],
            declaration_guess[:2],
            attempts,
            block_qubits=2,
        )
        self.assertEqual(no_acceptance["accepted_sessions"], 0)
        self.assertEqual(no_acceptance["exposed_accepted_keys"], 0)
        self.assertTrue(np.isnan(no_acceptance["accepted_key_exposure_rate"]))

    def test_noiseless_final_key_agrees_for_both_block_sizes(self):
        for qubits in (2, 4):
            paths = simulate_paths(
                np.random.default_rng(17),
                trials=8,
                qubits=qubits,
                q_z=0.0,
                q_x=0.0,
                attempts=2,
                include_posterior=False,
            )
            self.assertTrue(np.all(paths["final_key_agrees"]))

    def test_interception_stops_after_declaration_while_bob_finishes(self):
        attempts = 20
        for qubits in (2, 4):
            settings = dict(
                trials=32, qubits=qubits, q_z=0.0, q_x=0.0, attempts=attempts,
                intercept_probability=1.0, eve_z_probabilities=(0.0,),
            )
            paths = simulate_paths(np.random.default_rng(17), **settings)
            declarations, _ = _declarations(paths, attempts)
            early_trials = np.flatnonzero(declarations < attempts)
            self.assertGreater(len(early_trials), 0)
            for trial in early_trials:
                declaration = declarations[trial]
                np.testing.assert_array_equal(paths["check_errors"][trial, declaration:], 0)
                suffix = paths["posterior"][trial, declaration - 1:]
                np.testing.assert_array_equal(suffix, np.broadcast_to(suffix[0], suffix.shape))
            self.assertTrue(paths["final_key_agrees"][early_trials].all())

            without_posterior = simulate_paths(
                np.random.default_rng(17), include_posterior=False, **settings,
            )
            for name, values in without_posterior.items():
                np.testing.assert_array_equal(values, paths[name])

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
