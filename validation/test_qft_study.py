import unittest

import numpy as np
from pandas.testing import assert_frame_equal

from quantum_study.qft_alarm import _look_schedules, _replay_alarms
from quantum_study.qft_exchange import (
    _decode_wide_block,
    _fourier_bit_one_probabilities,
    _fourier_pattern_probability,
    _null_coherence,
    _null_decoding_probabilities,
    _payload_likelihoods,
    _prepared_state_score,
    _rotations,
    simulate_paths,
)
from quantum_study.qft_study import (
    PREPARED_STATE_DETECTOR,
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

    def test_check_scores_match_full_density_fixture(self):
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

            # Compare the weighted score of observing all zeros against the
            # mismatch marginals of the independent full-density fixture.
            values = np.array([1] if qubits == 2 else [1, 0])
            bits = (words[:, None] >> (qubits - 1 - np.array(checks))) & 1
            joint = _CHECK_DISTRIBUTIONS[qubits]
            rates = np.array(
                [joint[0]] if qubits == 2 else [joint[:2].sum(), joint[1::2].sum()]
            )
            expected = ((values - rates) * (1 - 2 * rates) / (rates * (1 - rates))).sum()
            score = _prepared_state_score(
                probabilities[None, None], bits[None, None], values[None, None],
                values[None, None].astype(bool),
            )
            np.testing.assert_allclose(score, [expected], atol=1e-12, rtol=0)

    def test_prepared_state_handles_checks_with_a_certain_noise_only_result(self):
        null = np.array([[[1., 0.]], [[1., 0.]]])
        candidate_checks = np.array([[[[0], [1]]], [[[0], [1]]]])
        expected = np.zeros((2, 1, 1), dtype=int)
        mismatches = np.array([[[False]], [[True]]])
        np.testing.assert_array_equal(
            _prepared_state_score(null, candidate_checks, expected, mismatches),
            [0., np.inf],
        )

    def test_prepared_state_comparison_preserves_shared_detector_results(self):
        settings = dict(attempts=12, trials=32, intercept_probability=0.2)
        summary, curves = run_qft_study(**settings)
        extended_summary, extended_curves = run_qft_study(**settings, compare_prepared_state=True)
        for baseline, extended in ((summary, extended_summary), (curves, extended_curves)):
            self.assertEqual(extended.detector.unique()[-1], PREPARED_STATE_DETECTOR)
            assert_frame_equal(baseline, extended.iloc[:len(baseline)], check_exact=True)

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
        for block_qubits in (1, 2, 4, 8, 16):
            summary, *_ = _study_rows(
                paths, "Pooled check errors", np.array([5, 3]), np.full(2, 5),
                np.array([2, 2]), np.zeros(2, dtype=int), attempts=4, block_qubits=block_qubits,
            )
            self.assertEqual(summary["correct_recovery_rate"], 1.0)
            expected = (42, 112) if block_qubits == 16 else (14, 56)
            self.assertEqual(summary["mean_checks"], expected[0])
            self.assertEqual(summary["mean_transmitted_qubits"], expected[1])

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

    def test_noiseless_checks_and_final_key_agree_for_all_block_sizes(self):
        for qubits in (1, 2, 4, 8, 16):
            paths = simulate_paths(
                np.random.default_rng(17),
                trials=8,
                qubits=qubits,
                q_z=0.0,
                q_x=0.0,
                attempts=2,
                include_posterior=False,
                include_prepared_state=True,
            )
            self.assertTrue(np.all(paths["final_key_agrees"]))
            for metric in ("check_errors", "joint_score", "prepared_state_score"):
                np.testing.assert_array_equal(paths[metric], 0)

    def test_sixteen_qubit_fourier_helpers_match_dense_distribution(self):
        qubits = 16
        q_z, q_x = 0.002, 0.038
        rotations = _rotations(qubits)
        phases = np.exp(1j * np.linspace(0.17, 2.41, qubits))[None, None]
        weights = 2 ** np.arange(qubits - 1, -1, -1)
        payload = np.array([1, 6, 11, 14])
        payload_bits = (
            np.arange(16)[:, None] >> np.arange(3, -1, -1)
        ) & 1
        payload_words = payload_bits @ weights[payload]
        payload_mask = weights[payload].sum()
        candidates = ((0xA53C & ~payload_mask) + payload_words)[None, None]
        outcomes = ((0x39A7 & ~payload_mask) + payload_words)[None, None]

        coherence = _null_coherence(
            phases, candidates, rotations, q_z, q_x,
        )
        scale = 1 - q_z - 2 * q_x
        direct_coherence = phases / 2
        direct_coherence = scale * direct_coherence + q_z * direct_coherence.conj()
        direct_coherence = direct_coherence[..., None, :] * rotations[candidates]
        direct_coherence = scale * direct_coherence + q_z * direct_coherence.conj()
        direct_coherence *= phases.conj()[..., None, :]
        np.testing.assert_allclose(coherence, direct_coherence, atol=2e-16, rtol=0)

        # Independently enumerate all 65,536 decoded words, but accumulate one
        # local Fourier factor at a time to avoid an outcome-by-signal tensor.
        probabilities = np.ones((16, 2**qubits))
        for signal in range(qubits):
            probabilities *= 0.5 + (
                direct_coherence[0, 0, :, signal, None]
                * rotations[:, signal].conj()
            ).real
        dense = probabilities.mean(axis=0)
        np.testing.assert_allclose(dense.sum(), 1.0, atol=2e-15, rtol=0)

        pattern = _fourier_pattern_probability(coherence, rotations, outcomes)
        np.testing.assert_allclose(
            pattern[0, 0], dense[outcomes[0, 0]].sum(), atol=2e-15, rtol=0,
        )

        decoded_bits = (
            np.arange(2**qubits)[:, None]
            >> np.arange(qubits - 1, -1, -1)
        ) & 1
        expected_marginals = dense @ decoded_bits
        np.testing.assert_allclose(
            _fourier_bit_one_probabilities(coherence)[0, 0],
            expected_marginals,
            atol=1e-13,
            rtol=0,
        )

        # Fourier-encoded basis words decode deterministically, including both
        # endpoint words and mixed high/low-bit patterns.
        words = np.array([0, 1, 0x00FF, 0x5A3C, 0x8000, 0xFFFF])
        states = np.stack(
            (np.ones_like(rotations[words]), rotations[words]), axis=-1,
        ) / np.sqrt(2)
        np.testing.assert_array_equal(
            _decode_wide_block(states, rotations, np.random.default_rng(23)),
            words,
        )

    def test_one_qubit_full_x_interception_recovers_each_payload_bit(self):
        paths = simulate_paths(
            np.random.default_rng(17), trials=32, qubits=1, q_z=0.0, q_x=0.0,
            attempts=2, intercept_probability=1.0, eve_z_probabilities=(0.0,),
        )
        first = paths["posterior"][:, 0]
        np.testing.assert_array_equal(first.argmax(axis=1), paths["secret"])
        np.testing.assert_allclose(first.max(axis=1), 1.0)
        np.testing.assert_array_equal(paths["posterior"][:, 1], first)
        np.testing.assert_array_equal(paths["check_errors"][:, 1], 0)
        self.assertTrue(paths["final_key_agrees"].all())

    def test_eight_qubit_phase_flip_probabilities_match_direct_formula(self):
        q_x = 0.038
        phases = np.exp(1j * np.linspace(0.17, 2.41, 8))[None, None]
        candidates = np.arange(16)[None, None]
        result = _null_decoding_probabilities(phases, candidates, _rotations(8), 0.0, q_x)
        differences = candidates[0, 0, :, None] - np.arange(256)
        angles = np.pi * differences[..., None] / 2 ** np.arange(8)
        # With only Pauli Z noise, undoing the scramble cancels all private phases.
        expected = (0.5 + 0.5 * (1 - 2 * q_x) ** 2 * np.cos(angles)).prod(axis=-1).mean(axis=0)
        np.testing.assert_allclose(result[0, 0], expected, atol=1e-14, rtol=0)

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
        for qubits in (1, 2, 4, 8, 16):
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
