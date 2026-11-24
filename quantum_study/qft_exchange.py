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

from functools import cache

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
    if qubits == 16:
        # Reduce the angle before taking the exponential to keep precision.
        periods = 2 ** np.arange(1, qubits + 1)
        return np.exp(2j * np.pi * (np.arange(dimension)[:, None] % periods) / periods)
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


# Periodic expansion of the Fourier projection in Papadopoulos and Linvill, §4.1, Eq. (3).
def _null_decoding_probabilities_8(coherence, rotations):
    """Expand the periodic local factors without an outcome×qubit tensor."""
    payload_count = coherence.shape[-2]
    batch_count = int(np.prod(coherence.shape[:-2]))
    # Bound the largest recurrence array near 16 MiB of float64.
    payload_chunk = max(1, (1 << 21) // (batch_count * len(rotations)))
    payload_chunk = min(payload_count, payload_chunk)
    averaged = np.zeros(coherence.shape[:-2] + (len(rotations),))
    for start in range(0, payload_count, payload_chunk):
        selected = coherence[..., start:start + payload_chunk, :]
        probabilities = np.ones(selected.shape[:-1] + (1,))
        for signal in range(8):
            span = 1 << signal
            factor = (
                selected[..., signal, None]
                * rotations[:span, signal].conj()
            ).real
            factor += 0.5
            expanded = np.empty(selected.shape[:-1] + (2 * span,))
            np.multiply(probabilities, factor, out=expanded[..., :span])
            # The second half-period flips the local contribution's sign.
            np.subtract(1, factor, out=factor)
            np.multiply(probabilities, factor, out=expanded[..., span:])
            probabilities = expanded
        averaged += probabilities.sum(axis=-2) / payload_count
    return averaged


# Project density-matrix calculation using the proposal §2 Pauli channel and
# the decoder from Papadopoulos and Linvill, §4.1, Eq. (3).
def _null_coherence(phases, candidates, rotations, q_z, q_x):
    # A local Pauli channel leaves equatorial states with diagonals (1/2, 1/2).
    # Its only changing entry is the coherence rho[1, 0].
    coherence_scale = 1 - q_z - 2 * q_x
    coherence = phases / 2
    coherence = coherence_scale * coherence + q_z * coherence.conj()
    coherence = coherence[..., None, :] * rotations[candidates]
    coherence = coherence_scale * coherence + q_z * coherence.conj()
    coherence *= phases.conj()[..., None, :]
    return coherence


def _null_decoding_probabilities(phases, candidates, rotations, q_z, q_x):
    """Average Bob's null outcome probabilities over unknown payload values."""
    coherence = _null_coherence(phases, candidates, rotations, q_z, q_x)
    if phases.shape[-1] == 8:
        return _null_decoding_probabilities_8(coherence, rotations)
    # A Fourier basis vector factors into local phase states. Projection of the
    # product density onto that vector is the product of local probabilities.
    probabilities = (0.5 + (coherence[..., None, :] * rotations.conj()).real).prod(axis=-1)
    return probabilities.mean(axis=-2)


# Use the periodic Fourier factors in Papadopoulos and Linvill, §4.1, Eq. (3).
def _decode_wide_block(states, rotations, rng):
    """Sample the Fourier measurement from its least significant bit upward."""
    coherence = states[..., 1] * states[..., 0].conj()
    outcomes = np.zeros(coherence.shape[:-1], dtype=int)
    draws = rng.random(coherence.shape)
    for signal in range(coherence.shape[-1]):
        # Only the lowest signal + 1 bits affect this local factor. Summing
        # over later bits leaves a normalized conditional on the known suffix.
        rotation = rotations[outcomes, signal]
        p_zero = 0.5 + (coherence[..., signal] * rotation.conj()).real
        bit = draws[..., signal] >= p_zero.clip(0.0, 1.0)
        outcomes |= bit.astype(int) << signal
    return outcomes


# Proposal §2's joint score needs only the outcomes consistent with the
# observed checks. Each summand is a Fourier projection as in §4.1, Eq. (3).
def _fourier_pattern_probability(coherence, rotations, outcomes):
    """Sum over output completions and average over possible encoded payloads."""
    output_phases = rotations[outcomes].conj()
    probabilities = np.ones(coherence.shape[:-1] + (outcomes.shape[-1],))
    for signal in range(coherence.shape[-1]):
        local = coherence[..., signal, None]
        rotation = output_phases[..., None, :, signal]
        probabilities *= 0.5 + (local * rotation).real
    return probabilities.sum(axis=-1).mean(axis=-1)


# Characteristic functions of the centered suffix have even real parts and
# odd imaginary parts. Interpolate Re F(f) and Im F(f)/f in f² on [0, 1].
@cache
def _half_frequency_transfer():
    # For a centered suffix x in [-1/2, 1/2], the cosine Taylor remainder
    # after 14 terms is at most pi**28 / 28! before interpolation.
    angles = np.pi * (np.arange(14) + 0.5) / 14
    nodes = (1 - np.cos(angles)) / 2
    grid = np.sqrt(nodes)
    shifts = np.stack((grid / 2, (1 + grid) / 2, (1 - grid) / 2), axis=-1)
    frequencies = np.append(shifts.ravel(), 0.5)
    weights = np.sin(angles)
    weights[1::2] *= -1
    interpolation = weights / (frequencies[:, None]**2 - nodes)
    interpolation /= interpolation.sum(axis=1, keepdims=True)
    real_transfer = interpolation[:-1].T.copy()
    imag_transfer = (interpolation[:-1] * frequencies[:-1, None]).T.copy()
    return grid, real_transfer, imag_transfer, interpolation[-1]


# Check-bit marginals from the same Fourier factors, §4.1, Eq. (3).
def _fourier_bit_one_probabilities(coherence):
    """Approximate marginal probabilities in decoded word order (MSB first)."""
    grid, real_transfer, imag_transfer, at_half = _half_frequency_transfer()
    cosine = np.cos(np.pi * grid / 2)
    sine = np.sin(np.pi * grid / 2)
    marginal = np.empty(coherence.shape[:-2] + (coherence.shape[-1],))
    shape = coherence.shape[:-1] + (len(grid),)
    real = np.ones(shape)
    imag_scaled = np.zeros(shape)
    for signal in range(coherence.shape[-1]):
        # x = (suffix + 1/2)/2**signal - 1/2 centers the suffix. The next
        # bit has P(1 | x) = 1/2 - Re(aligned * exp(-i*pi*x)).
        rotation = np.exp(-1j * np.pi * (0.5 - 0.5**(signal + 1)))
        aligned = coherence[..., signal] * rotation
        a, b = aligned.real[..., None], aligned.imag[..., None]
        weighted = (a * real - 0.5 * b * imag_scaled).mean(axis=-2)
        marginal[..., signal] = 0.5 - weighted @ at_half
        if signal + 1 == coherence.shape[-1]:
            break

        # Appending bit d sends x to (x + d - 1/2)/2. Its two conditional
        # probabilities shift F(f/2) to F((f+1)/2) and F((f-1)/2); the
        # negative frequency is the conjugate of the positive one.
        r_half, r_plus, r_minus = np.moveaxis((real @ real_transfer).reshape(shape + (3,)), -1, 0)
        i_half, i_plus, i_minus = np.moveaxis((imag_scaled @ imag_transfer).reshape(shape + (3,)), -1, 0)
        real = cosine * r_half - sine * (a * (i_plus - i_minus) + b * (r_plus - r_minus))
        imag = cosine * i_half + sine * (a * (r_plus + r_minus) - b * (i_plus + i_minus))
        imag_scaled = imag / grid
    return marginal[..., ::-1]


# Adapt the weighted error score in proposal §2 to QFT's prepared phases and
# announced checks. Use the marginal error probability for each retained bit.
def _prepared_state_score(null, candidate_checks, values, mismatches):
    possible_errors = candidate_checks != values[..., None, :]
    rates = (null[..., None] * possible_errors).sum(axis=-2)
    return _weighted_check_errors(rates, mismatches)


def _weighted_check_errors(rates, mismatches):
    rates = rates.clip(0.0, 1.0)
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


# Proposal §2 uses 2- and 4-qubit blocks. The block-size comparison also
# uses dedicated one-qubit blocks and single eight- or sixteen-qubit blocks.
def _checks_per_attempt(qubits):
    return max(CHECKS_PER_ATTEMPT, qubits - 4)


def _validate_path_settings(trials, attempts, qubits, intercept_probability):
    if type(trials) is not int or trials < 1:
        raise ValueError("trials must be a positive integer")
    if type(attempts) is not int or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    if type(qubits) is not int or qubits not in (1, 2, 4, 8, 16):
        raise ValueError("qubits must be 1, 2, 4, 8, or 16")
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
    single_qubit = qubits == 1
    wide_block = qubits == 16
    checks_per_attempt = _checks_per_attempt(qubits)
    check_axis = 1 if single_qubit else -1
    width = min(4, max(1, qubits // 2))
    checks_per_block = qubits - width
    payload_blocks = 4 // width
    blocks = (4 + checks_per_attempt) // qubits
    rotations = _rotations(qubits)
    block_bits = _bits(qubits)
    payload_bits = _bits(width)
    secret_bits = _bits(4)
    weights = 2 ** np.arange(qubits - 1, -1, -1)
    payload_weights = 2 ** np.arange(width - 1, -1, -1)
    payload_index = secret_bits.reshape(16, payload_blocks, width) @ payload_weights
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
        # and Linvill, §6.1. The 16-qubit extension keeps four payload bits.
        order = rng.random(signal_shape).argsort(axis=check_axis)
        if single_qubit:
            # Randomly assign whole blocks to checks when a block is one bit.
            checks = np.sort(order[:, :CHECKS_PER_ATTEMPT], axis=1)
            payload = np.sort(order[:, CHECKS_PER_ATTEMPT:], axis=1)
            values = rng.integers(2, size=checks.shape)
            message = np.empty(signal_shape, dtype=int)
            np.put_along_axis(message, checks, values, axis=1)
            np.put_along_axis(message, payload, secret_bits[secret, :, None], axis=1)
            message = message[..., 0]
        else:
            checks = np.sort(order[..., :checks_per_block], axis=-1)
            payload = np.sort(order[..., checks_per_block:], axis=-1)
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
        if wide_block:
            outcomes = _decode_wide_block(states, rotations, rng)
        else:
            amplitudes = states[..., None, :, 0] + states[..., None, :, 1] * rotations.conj()
            output_probabilities = (abs(amplitudes)**2 / 2).prod(axis=-1)
            cumulative = output_probabilities.cumsum(axis=-1)
            cumulative /= cumulative[..., -1:]
            outcomes = (rng.random((trials, blocks, 1)) > cumulative).sum(axis=-1)
        decoded_blocks = block_bits[outcomes]
        observed = np.take_along_axis(decoded_blocks, checks, axis=check_axis)
        mismatches = observed != values
        errors[:, attempt] = mismatches.sum(axis=(1, 2))

        # Proposal §2 uses joint check-pattern likelihoods. The notebook adds
        # their negative logs across blocks as Bob's score.
        if wide_block:
            coherence = _null_coherence(phases, candidates, rotations, q_z, q_x)
            # The encoded candidates and the possible decoded words share the
            # same payload positions, but the observed check values can differ.
            check_offset = ((observed - values) * weights[checks]).sum(axis=-1)
            output_words = candidates + check_offset[..., None]
            likelihood = _fourier_pattern_probability(coherence, rotations, output_words)
            if include_prepared_state:
                p_one = _fourier_bit_one_probabilities(coherence)
                check_ones = np.take_along_axis(p_one, checks, axis=-1)
                rates = np.where(values != 0, 1 - check_ones, check_ones)
                prepared_state_scores[:, attempt] = _weighted_check_errors(rates, mismatches)
        else:
            if single_qubit:
                check_phases = np.take_along_axis(phases, checks, axis=1)
                null = _null_decoding_probabilities(check_phases, values, rotations, q_z, q_x)
                candidate_checks = block_bits
            else:
                null = _null_decoding_probabilities(phases, candidates, rotations, q_z, q_x)
                candidate_checks = block_bits[:, checks].transpose(1, 2, 0, 3)
            same_pattern = (candidate_checks == observed[..., None, :]).all(axis=-1)
            likelihood = (null * same_pattern).sum(axis=-1)
            if include_prepared_state:
                prepared_state_scores[:, attempt] = _prepared_state_score(
                    null, candidate_checks, values, mismatches,
                )
        likelihood[likelihood >= _NEAR_ONE] = 1.0
        scores[:, attempt] = -np.log(np.maximum(likelihood, _TINY)).sum(axis=-1)

        # Proposal §2 updates Eve's probabilities over the 16 secrets. The
        # project stops Eve's interception after she makes a guess.
        attacking = interception_rates > 0
        if attacking.any():
            forward_records, return_records = forward[attacking], returned[attacking]
            if single_qubit:
                # A dedicated check's disclosed value gives no evidence about the key.
                forward_records = np.take_along_axis(forward_records, payload[attacking], axis=1)
                return_records = np.take_along_axis(return_records, payload[attacking], axis=1)
                encodings = rotations
            else:
                encodings = rotations[candidates[attacking]]
            record_likelihood = _payload_likelihoods(
                encodings, forward_records, return_records, q_x,
            )
            active_log_posterior = log_posterior[attacking]
            for block in range(payload_blocks):
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
    final_key = np.take_along_axis(decoded_blocks, payload, axis=check_axis).reshape(trials, 4)
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
