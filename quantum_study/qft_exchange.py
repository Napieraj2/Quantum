"""Sample the proposal's two-pass exchange and exact Eve likelihood.

Phase scrambling, encoding, and decoding follow Papadopoulos and Linvill,
§4.1, Eqs. (1)–(3). Each block uses the hidden-check variant in §6.1.
https://arxiv.org/abs/2404.12507v2

Proposal §2 specifies the per-leg Pauli channel. Its reference rates come
from Erven et al., §4. The focused run gives every Eve the same interception
rate and draws a fresh independent basis at every hit, corresponding to the
proposal's λ=1 case.
https://arxiv.org/abs/0901.0960v2
"""

import numpy as np
from scipy.special import logsumexp


# Proposal §2 (Napieraj and Chuang, 2026) uses a four-bit secret, half-block
# checks, and 99% confidence for a guess. See validation/MODEL_NOTES.md.
CHECKS_PER_ATTEMPT = 4
RECOVERY_CONFIDENCE = 0.99
# Numerical safeguards for probabilities in the project implementation below.
_TINY = np.finfo(float).tiny
_NEAR_ONE = 1 - 16 * np.finfo(float).eps
# Proposal §2 specifies X/Z measurements. Here 0/1 are Z outcomes, 2/3 are X
# outcomes, and -1 means no record.
_EIGENSTATES = np.array([[1, 0], [0, 1], [1, 1], [1, -1]], dtype=complex)
_EIGENSTATES[2:] /= np.sqrt(2)


# Project bit ordering for the integer block word in Papadopoulos and Linvill, §4.1.
def _bits(width):
    return (np.arange(2**width)[:, None] >> np.arange(width - 1, -1, -1)) & 1


# Alice's local phase rotations follow Papadopoulos and Linvill, §4.1, Eq. (2).
def _rotations(qubits):
    dimension = 2**qubits
    weights = 2 ** np.arange(qubits - 1, -1, -1)
    return np.exp(2j * np.pi * np.outer(np.arange(dimension), weights) / dimension)


# Proposal §2 puts Pauli X, Z, or identity before Eve on each leg. Erven et al.
# (2009, §4) report the reference rates used by callers.
def _sample_paulis(states, rng, q_z, q_x):
    draws = rng.random(states.shape[:-1])
    flip = draws < q_z
    states[flip] = states[flip, ::-1]
    phase_flip = (draws >= q_z) & (draws < q_z + q_x)
    states[phase_flip, 1] *= -1


# Proposal §2 specifies independent hits by sequential X/Z intercept-resend
# devices. The record selection supports the likelihood reduction below.
def _intercept_leg(states, rng, rate, eve_z_probabilities, keep_first):
    """Sample every Eve, retaining the first or last record on each signal."""
    records = np.full(states.shape[:-1], -1, dtype=np.int8)
    if not np.any(rate):
        return records
    for p_z in eve_z_probabilities:
        hit = rng.random(records.shape) < rate
        x_basis = rng.random(hit.sum()) >= p_z
        measured = states[hit]
        p_zero = np.where(x_basis, abs(measured.sum(axis=1))**2 / 2, abs(measured[:, 0])**2)
        outcome = rng.random(len(p_zero)) >= p_zero.clip(0, 1)
        codes = 2 * x_basis + outcome
        states[hit] = _EIGENSTATES[codes]
        if keep_first:
            empty = records[hit] == -1
            records[hit] = np.where(empty, codes, records[hit])
        else:
            records[hit] = codes
    return records


# Project density-matrix calculation using the proposal §2 Pauli channel and
# the decoder from Papadopoulos and Linvill, §4.1, Eq. (3).
def _null_decoding_probabilities(phases, candidates, rotations, q_z, q_x):
    """Average Bob's null outcome probabilities over unknown payload values."""
    # A local Pauli channel leaves equatorial states with diagonals (1/2, 1/2).
    # Its only changing entry is the coherence rho[1, 0].
    coherence_scale = 1 - q_z - 2 * q_x
    coherence = phases / 2
    coherence = coherence_scale * coherence + q_z * coherence.conj()
    coherence = coherence[..., None, :] * rotations[candidates]
    coherence = coherence_scale * coherence + q_z * coherence.conj()
    coherence *= phases.conj()[..., None, :]
    # A Fourier basis vector factors into local phase states. Projection of the
    # product density onto that vector is the product of local probabilities.
    probabilities = (0.5 + (coherence[..., None, :] * rotations.conj()).real).prod(axis=-1)
    return probabilities.mean(axis=-2)


# Adapt the weighted error score in proposal §2 to QFT's prepared phases and
# announced checks. Use the marginal error probability for each retained bit.
def _prepared_state_score(null, candidate_checks, values, mismatches):
    possible_errors = candidate_checks != values[..., None, :]
    rates = (null[..., None] * possible_errors).sum(axis=-2).clip(0.0, 1.0)
    variable = (rates > 1 - _NEAR_ONE) & (rates < _NEAR_ONE)
    weights = np.divide(
        1 - 2 * rates, rates * (1 - rates),
        out=np.zeros_like(rates), where=variable,
    )
    score = ((mismatches - rates) * weights).sum(axis=(1, 2))
    impossible = ~variable & (mismatches != (rates >= 0.5))
    score[impossible.any(axis=(1, 2))] = np.inf
    return score


# Project derivation for proposal §2's pooled Eve records. The full-density
# reference values are checked in validation/test_qft_study.py.
def _payload_likelihoods(encodings, last_forward, first_return, q_x):
    """Exact likelihoods up to a factor common to all candidate messages."""
    # Uniform phases give I/2 per signal. Without a forward record the signal
    # stays mixed. A last-forward Z or first-return Z also gives no information
    # about phase encoding. Records after the first return collapse contribute
    # factors common to all messages. Only an X-to-X pair affects the posterior.
    informative = (last_forward >= 2) & (first_return >= 2)
    sign = np.where(last_forward == first_return, 1, -1)
    paired = (1 + sign[..., None, :] * (1 - 2 * q_x) * encodings.real) / 2
    return np.where(informative[..., None, :], paired, 1).prod(axis=-1)


# Project input checks for proposal §2's 2- and 4-qubit blocks and per-leg interception.
def _validate_path_settings(trials, attempts, qubits, intercept_probability):
    if type(trials) is not int or trials < 1:
        raise ValueError("trials must be a positive integer")
    if type(attempts) is not int or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    if type(qubits) is not int or qubits not in (2, 4):
        raise ValueError("qubits must be 2 or 4")
    if not 0 <= intercept_probability <= 1:
        raise ValueError("intercept_probability must be between zero and one")


# Project input checks for the Z-basis probabilities defined in proposal §2.
def _eve_biases(probabilities):
    biases = tuple(probabilities)
    if any(not 0 <= probability <= 1 for probability in biases):
        raise ValueError("eve_z_probabilities must lie between zero and one")
    return biases


# Combine Papadopoulos and Linvill's §4.1 circuit with proposal §2's repeated-secret model.
def simulate_paths(
    rng, *, trials, qubits, q_z, q_x, attempts,
    intercept_probability=0.0, eve_z_probabilities=(), include_posterior=True,
    include_prepared_state=False,
):
    """Sample exchanges through the cap, with interception ending at Eve's declaration.

    Stopping interception and confirming the final decoded key are policies
    introduced for this experiment.
    """
    _validate_path_settings(trials, attempts, qubits, intercept_probability)
    if not (q_z >= 0 and q_x >= 0 and q_z + q_x <= 1):
        raise ValueError("q_z and q_x must be nonnegative and sum to at most one")
    eve_z_probabilities = _eve_biases(eve_z_probabilities)

    # Proposal §2 uses the same secret in each attempt and fresh checks in each block.
    width = qubits // 2
    blocks = 4 // width
    rotations = _rotations(qubits)
    block_bits = _bits(qubits)
    payload_bits = _bits(width)
    secret_bits = _bits(4)
    weights = 2 ** np.arange(qubits - 1, -1, -1)
    payload_weights = 2 ** np.arange(width - 1, -1, -1)
    payload_index = secret_bits.reshape(16, blocks, width) @ payload_weights
    secret = rng.integers(16, size=trials)
    interception_rates = np.full(trials, intercept_probability, dtype=float)
    log_posterior = np.full((trials, 16), -np.log(16))
    if include_posterior:
        posterior = np.empty((trials, attempts, 16))
    errors = np.empty((trials, attempts), dtype=int)
    scores = np.empty((trials, attempts))
    if include_prepared_state:
        prepared_state_scores = np.empty_like(scores)
    signal_shape = (trials, blocks, qubits)

    for attempt in range(attempts):
        # Hide the check positions until after measurement as in Papadopoulos
        # and Linvill, §6.1. Proposal §2 assigns half the block to checks.
        order = rng.random(signal_shape).argsort(axis=-1)
        checks = np.sort(order[..., :width], axis=-1)
        payload = np.sort(order[..., width:], axis=-1)
        values = rng.integers(2, size=checks.shape)
        check_message = (values * weights[checks]).sum(axis=-1)
        candidates = check_message[..., None] + weights[payload] @ payload_bits.T
        message = np.take_along_axis(
            candidates, payload_index[secret, :, None], axis=-1,
        )[..., 0]

        # Follow the phase sequence in Papadopoulos and Linvill, §4.1, Eqs. (1)–(3).
        # Proposal §2 adds channel noise and the Eve chain on both legs.
        phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=signal_shape))
        states = np.stack((np.ones_like(phases), phases), axis=-1) / np.sqrt(2)
        _sample_paulis(states, rng, q_z, q_x)
        forward = _intercept_leg(
            states, rng, interception_rates[:, None, None], eve_z_probabilities,
            keep_first=False,
        )
        states[..., 1] *= rotations[message]
        _sample_paulis(states, rng, q_z, q_x)
        returned = _intercept_leg(
            states, rng, interception_rates[:, None, None], eve_z_probabilities,
            keep_first=True,
        )
        states[..., 1] *= phases.conj()

        # Sample the inverse-QFT measurement in Papadopoulos and Linvill, Eq. (3).
        amplitudes = states[..., None, :, 0] + states[..., None, :, 1] * rotations.conj()
        output_probabilities = (abs(amplitudes)**2 / 2).prod(axis=-1)
        cumulative = output_probabilities.cumsum(axis=-1)
        cumulative /= cumulative[..., -1:]
        outcomes = (rng.random((trials, blocks, 1)) > cumulative).sum(axis=-1)
        decoded_blocks = block_bits[outcomes]
        observed = np.take_along_axis(decoded_blocks, checks, axis=-1)
        mismatches = observed != values
        errors[:, attempt] = mismatches.sum(axis=(1, 2))

        # Proposal §2 uses joint check-pattern likelihoods. The notebook adds
        # their negative logs across blocks as Bob's score.
        null = _null_decoding_probabilities(phases, candidates, rotations, q_z, q_x)
        candidate_checks = block_bits[:, checks].transpose(1, 2, 0, 3)
        same_pattern = (candidate_checks == observed[..., None, :]).all(axis=-1)
        likelihood = (null * same_pattern).sum(axis=-1)
        likelihood[likelihood >= _NEAR_ONE] = 1.0
        scores[:, attempt] = -np.log(np.maximum(likelihood, _TINY)).sum(axis=-1)
        if include_prepared_state:
            prepared_state_scores[:, attempt] = _prepared_state_score(
                null, candidate_checks, values, mismatches,
            )

        # Proposal §2 updates Eve's probabilities over the 16 secrets. The
        # project stops Eve's interception after she makes a guess.
        attacking = interception_rates > 0
        if attacking.any():
            record_likelihood = _payload_likelihoods(
                rotations[candidates[attacking]], forward[attacking], returned[attacking], q_x,
            )
            active_log_posterior = log_posterior[attacking]
            for block in range(blocks):
                hypothesis_likelihood = record_likelihood[:, block, payload_index[:, block]]
                active_log_posterior += np.log(np.maximum(hypothesis_likelihood, _TINY))
            active_log_posterior -= logsumexp(active_log_posterior, axis=-1, keepdims=True)
            log_posterior[attacking] = active_log_posterior
            declared = np.exp(log_posterior.max(axis=1)) >= RECOVERY_CONFIDENCE
            interception_rates[declared] = 0.0
        if include_posterior:
            posterior[:, attempt] = np.exp(log_posterior)

    # Extract the payload as in Papadopoulos and Linvill, §4.1, "Extraction."
    # The notebook compares Bob's final decoded key with Alice's as a project policy.
    final_key = np.take_along_axis(decoded_blocks, payload, axis=-1).reshape(trials, 4)
    final_key_agrees = (final_key == secret_bits[secret]).all(axis=1)

    # Project outputs for the detector comparison and bit-confidence plots.
    paths = {
        "secret": secret,
        "check_errors": errors,
        "joint_score": scores,
        "final_key_agrees": final_key_agrees,
    }
    if include_prepared_state:
        paths["prepared_state_score"] = prepared_state_scores
    if include_posterior:
        paths["posterior"] = posterior
        paths["bit_posterior"] = posterior @ secret_bits
    return paths
