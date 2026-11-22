"""Sample the proposal's two-pass exchange and exact Eve likelihood.

Phase scrambling, encoding, and decoding follow Papadopoulos and Linvill,
§4.1, Eqs. (1)–(3). Each block uses the hidden-check variant in §6.1.
https://arxiv.org/abs/2404.12507v2

Proposal §2 specifies the per-leg Pauli channel. Its reference rates come
from Erven et al., §4. The focused run redraws each Eve's basis independently.
https://arxiv.org/abs/0901.0960v2
"""

import numpy as np
from scipy.special import logsumexp


CHECKS_PER_ATTEMPT = 4
_TINY = np.finfo(float).tiny
_NEAR_ONE = 1 - 16 * np.finfo(float).eps
# Codes 0/1 are Z outcomes, 2/3 are X outcomes, and -1 means no record.
_EIGENSTATES = np.array([[1, 0], [0, 1], [1, 1], [1, -1]], dtype=complex)
_EIGENSTATES[2:] /= np.sqrt(2)


def _bits(width):
    return (np.arange(2**width)[:, None] >> np.arange(width - 1, -1, -1)) & 1


def _rotations(qubits):
    dimension = 2**qubits
    weights = 2 ** np.arange(qubits - 1, -1, -1)
    return np.exp(2j * np.pi * np.outer(np.arange(dimension), weights) / dimension)


def _sample_paulis(states, rng, q_z, q_x):
    draws = rng.random(states.shape[:-1])
    flip = draws < q_z
    states[flip] = states[flip, ::-1]
    phase_flip = (draws >= q_z) & (draws < q_z + q_x)
    states[phase_flip, 1] *= -1


def _intercept_leg(states, rng, rate, eve_z_probabilities, keep_first):
    """Sample every Eve, retaining the first or last record on each signal."""
    records = np.full(states.shape[:-1], -1, dtype=np.int8)
    if rate == 0:
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


def simulate_paths(
    rng, *, trials, qubits, q_z, q_x, attempts,
    intercept_probability=0.0, eve_z_probabilities=(), include_posterior=True,
):
    """Sample shared physical paths, final key agreement, and optionally Eve's posterior."""
    if not (q_z >= 0 and q_x >= 0 and q_z + q_x <= 1):
        raise ValueError("q_z and q_x must be nonnegative and sum to at most one")
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
    if include_posterior:
        log_posterior = np.full((trials, 16), -np.log(16))
        posterior = np.empty((trials, attempts, 16))
    errors = np.empty((trials, attempts), dtype=int)
    scores = np.empty((trials, attempts))
    signal_shape = (trials, blocks, qubits)

    for attempt in range(attempts):
        # Select half of each block as fresh checks. Payload order is public.
        order = rng.random(signal_shape).argsort(axis=-1)
        checks = np.sort(order[..., :width], axis=-1)
        payload = np.sort(order[..., width:], axis=-1)
        values = rng.integers(2, size=checks.shape)
        check_message = (values * weights[checks]).sum(axis=-1)
        candidates = check_message[..., None] + weights[payload] @ payload_bits.T
        message = np.take_along_axis(
            candidates, payload_index[secret, :, None], axis=-1,
        )[..., 0]

        phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=signal_shape))
        states = np.stack((np.ones_like(phases), phases), axis=-1) / np.sqrt(2)
        _sample_paulis(states, rng, q_z, q_x)
        forward = _intercept_leg(
            states, rng, intercept_probability, eve_z_probabilities, keep_first=False,
        )
        states[..., 1] *= rotations[message]
        _sample_paulis(states, rng, q_z, q_x)
        returned = _intercept_leg(
            states, rng, intercept_probability, eve_z_probabilities, keep_first=True,
        )
        states[..., 1] *= phases.conj()

        # Undoing the private phases followed by QFT† gives one decoded word.
        amplitudes = states[..., None, :, 0] + states[..., None, :, 1] * rotations.conj()
        output_probabilities = (abs(amplitudes)**2 / 2).prod(axis=-1)
        cumulative = output_probabilities.cumsum(axis=-1)
        cumulative /= cumulative[..., -1:]
        outcomes = (rng.random((trials, blocks, 1)) > cumulative).sum(axis=-1)
        decoded_blocks = block_bits[outcomes]
        observed = np.take_along_axis(decoded_blocks, checks, axis=-1)
        errors[:, attempt] = (observed != values).sum(axis=(1, 2))

        null = _null_decoding_probabilities(phases, candidates, rotations, q_z, q_x)
        candidate_checks = block_bits[:, checks].transpose(1, 2, 0, 3)
        same_pattern = (candidate_checks == observed[..., None, :]).all(axis=-1)
        likelihood = (null * same_pattern).sum(axis=-1)
        likelihood[likelihood >= _NEAR_ONE] = 1.0
        scores[:, attempt] = -np.log(np.maximum(likelihood, _TINY)).sum(axis=-1)

        if include_posterior:
            record_likelihood = _payload_likelihoods(rotations[candidates], forward, returned, q_x)
            for block in range(blocks):
                hypothesis_likelihood = record_likelihood[:, block, payload_index[:, block]]
                log_posterior += np.log(np.maximum(hypothesis_likelihood, _TINY))
            log_posterior -= logsumexp(log_posterior, axis=-1, keepdims=True)
            posterior[:, attempt] = np.exp(log_posterior)

    final_key_agrees = np.zeros(trials, dtype=bool)
    if attempts:
        # Drop the check bits and concatenate the final exchange's payload blocks.
        final_key = np.take_along_axis(decoded_blocks, payload, axis=-1).reshape(trials, 4)
        final_key_agrees = (final_key == secret_bits[secret]).all(axis=1)

    paths = {
        "secret": secret,
        "check_errors": errors,
        "joint_score": scores,
        "final_key_agrees": final_key_agrees,
    }
    if include_posterior:
        paths["posterior"] = posterior
        paths["bit_posterior"] = posterior @ secret_bits
    return paths
