"""Two-pass QFT states, channel dynamics, and Bob's check probabilities.

Scrambling, encoding, and decoding follow Papadopoulos and Linvill, §4.1,
Fig. 2 and Eqs. (1)–(3). Hidden checks use the layout in §6.1.
https://arxiv.org/abs/2404.12507v2

Napieraj and Chuang, "Detecting Patient Attacks on a Repeated Secret," §2,
defines this pilot's per-leg Pauli channel and repeated-secret exchange.
Erven et al., §4, supply the reference rates (0.002, 0.038).
https://arxiv.org/abs/0901.0960v2
"""

import numpy as np


BLOCK_QUBITS = (2, 4)
SECRET_BITS = 4
QFT_DEMO_Q_Z = 0.002
QFT_DEMO_Q_X = 0.038


PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY_2 = np.eye(2, dtype=complex)


def qft_matrix(qubits):
    dimension = 2**qubits
    values = np.arange(dimension)
    return np.exp(2j * np.pi * np.outer(values, values) / dimension) / np.sqrt(dimension)


def qft_encoding_matrix(qubits, message):
    dimension = 2**qubits
    return np.diag(np.exp(2j * np.pi * message * np.arange(dimension) / dimension))


def qft_scrambler(phases):
    qubits = len(phases)
    diagonal = []
    for value in range(2**qubits):
        phase = sum(
            phases[qubit] * ((value >> (qubits - 1 - qubit)) & 1)
            for qubit in range(qubits)
        )
        diagonal.append(np.exp(1j * phase))
    return np.diag(diagonal)


def qft_bits(value, qubits):
    return tuple((value >> (qubits - 1 - index)) & 1 for index in range(qubits))


def qft_integer(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def qft_message(check_positions, check_values, payload_bits, qubits):
    checks = dict(zip(check_positions, check_values))
    payload = iter(payload_bits)
    bits = [
        checks[index] if index in checks else next(payload)
        for index in range(qubits)
    ]
    return qft_integer(bits)


def local_operator(operator, qubit, qubits):
    factors = [
        operator if index == qubit else IDENTITY_2
        for index in range(qubits)
    ]
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def apply_pauli_channel(density, qubits, q_z, q_x):
    if min(q_z, q_x) < 0 or q_z + q_x > 1:
        raise ValueError("q_z and q_x must be nonnegative and sum to at most one")
    for qubit in range(qubits):
        x_operator = local_operator(PAULI_X, qubit, qubits)
        z_operator = local_operator(PAULI_Z, qubit, qubits)
        density = (
            (1 - q_z - q_x) * density
            + q_z * x_operator @ density @ x_operator
            + q_x * z_operator @ density @ z_operator
        )
    return density


def _prepare_block(qubits, phases):
    plus = np.ones(2**qubits, dtype=complex) / np.sqrt(2**qubits)
    scrambler = qft_scrambler(phases)
    state = scrambler @ plus
    return np.outer(state, state.conj()), scrambler


def _decoded_probabilities(qubits, density, scrambler):
    decoder = qft_matrix(qubits).conj().T @ scrambler.conj().T
    density = decoder @ density @ decoder.conj().T
    probabilities = np.clip(np.real(np.diag(density)), 0.0, None)
    return probabilities / probabilities.sum()


def qft_null_outcome_distribution(qubits, message, phases, q_z, q_x):
    density, scrambler = _prepare_block(qubits, phases)
    encoding = qft_encoding_matrix(qubits, message)
    density = apply_pauli_channel(density, qubits, q_z, q_x)
    density = encoding @ density @ encoding.conj().T
    density = apply_pauli_channel(density, qubits, q_z, q_x)
    return _decoded_probabilities(qubits, density, scrambler)


def qft_check_distribution(
    qubits, check_positions, check_values, phases, q_z, q_x
):
    payload_positions = [
        index for index in range(qubits) if index not in check_positions
    ]
    pattern_probabilities = np.zeros(2 ** len(check_positions))
    for payload_value in range(2 ** len(payload_positions)):
        payload_bits = qft_bits(payload_value, len(payload_positions))
        message = qft_message(
            check_positions, check_values, payload_bits, qubits
        )
        outcome_probabilities = qft_null_outcome_distribution(
            qubits, message, phases, q_z, q_x
        )
        for outcome, probability in enumerate(outcome_probabilities):
            outcome_bits = qft_bits(outcome, qubits)
            pattern = qft_integer([
                outcome_bits[position] for position in check_positions
            ])
            pattern_probabilities[pattern] += probability
    pattern_probabilities /= 2 ** len(payload_positions)
    return pattern_probabilities / pattern_probabilities.sum()


def qft_check_surprisal(
    outcome, qubits, check_positions, check_values, phases, q_z, q_x
):
    distribution = qft_check_distribution(
        qubits, check_positions, check_values, phases, q_z, q_x
    )
    outcome_bits = qft_bits(outcome, qubits)
    pattern = qft_integer([
        outcome_bits[position] for position in check_positions
    ])
    return -np.log(max(distribution[pattern], np.finfo(float).tiny))


def measurement_projector(basis, outcome, qubit, qubits):
    if basis == "Z":
        state = np.array([1, 0], dtype=complex) if outcome == 0 else np.array([0, 1], dtype=complex)
    elif basis == "X":
        sign = 1 if outcome == 0 else -1
        state = np.array([1, sign], dtype=complex) / np.sqrt(2)
    else:
        raise ValueError("basis must be Z or X")
    return local_operator(np.outer(state, state.conj()), qubit, qubits)


def sample_pauli_trajectory(density, qubits, q_z, q_x, rng):
    if min(q_z, q_x) < 0 or q_z + q_x > 1:
        raise ValueError("q_z and q_x must be nonnegative and sum to at most one")
    for qubit in range(qubits):
        choice = rng.choice(3, p=(1 - q_z - q_x, q_z, q_x))
        if choice:
            operator = local_operator(
                PAULI_X if choice == 1 else PAULI_Z, qubit, qubits
            )
            density = operator @ density @ operator
    return density


def sample_eve_chain(
    density, qubits, leg, attempt, block, eve_configs, eve_bases, rng, records
):
    for signal in range(qubits):
        for eve_index, config in enumerate(eve_configs):
            basis = eve_bases[eve_index]
            if rng.random() < config["intercept_probability"]:
                probabilities = []
                projectors = []
                for outcome in (0, 1):
                    projector = measurement_projector(
                        basis, outcome, signal, qubits
                    )
                    projectors.append(projector)
                    probabilities.append(np.real(np.trace(projector @ density)))
                probabilities = np.clip(probabilities, 0.0, None)
                probabilities = np.asarray(probabilities) / np.sum(probabilities)
                outcome = int(rng.choice(2, p=probabilities))
                projector = projectors[outcome]
                density = projector @ density @ projector / probabilities[outcome]
                records.append({
                    "attempt": attempt,
                    "block": block,
                    "leg": leg,
                    "signal": signal,
                    "eve": eve_index,
                    "basis": basis,
                    "outcome": outcome,
                })
            if rng.random() < config["redraw_probability"]:
                eve_bases[eve_index] = (
                    "Z" if rng.random() < config["p_z"] else "X"
                )
    return density


def simulate_qft_block(
    rng, qubits, message, q_z, q_x, eve_configs, eve_bases,
    attempt, block, phases=None
):
    dimension = 2**qubits
    if phases is None:
        phases = rng.uniform(0.0, 2 * np.pi, size=qubits)
    density, scrambler = _prepare_block(qubits, phases)
    records = []

    density = sample_pauli_trajectory(density, qubits, q_z, q_x, rng)
    density = sample_eve_chain(
        density, qubits, "forward", attempt, block,
        eve_configs, eve_bases, rng, records,
    )
    encoding = qft_encoding_matrix(qubits, message)
    density = encoding @ density @ encoding.conj().T
    density = sample_pauli_trajectory(density, qubits, q_z, q_x, rng)
    density = sample_eve_chain(
        density, qubits, "return", attempt, block,
        eve_configs, eve_bases, rng, records,
    )

    return {
        "outcome": int(rng.choice(dimension, p=_decoded_probabilities(
            qubits, density, scrambler,
        ))),
        "phases": np.asarray(phases),
        "records": records,
    }


def simulate_qft_attempt(
    rng, secret, qubits, q_z, q_x, eve_configs, eve_bases, attempt
):
    payload_per_block = qubits // 2
    blocks = []
    for block_index, start in enumerate(
        range(0, SECRET_BITS, payload_per_block), start=1
    ):
        secret_indices = tuple(range(
            start, min(start + payload_per_block, SECRET_BITS)
        ))
        check_positions = tuple(sorted(
            rng.choice(qubits, size=qubits // 2, replace=False)
        ))
        check_values = tuple(
            int(value) for value in rng.integers(0, 2, size=qubits // 2)
        )
        payload = [secret[index] for index in secret_indices]
        message = qft_message(
            check_positions, check_values, payload, qubits
        )
        simulated = simulate_qft_block(
            rng, qubits, message, q_z, q_x,
            eve_configs, eve_bases, attempt, block_index,
        )
        score = qft_check_surprisal(
            simulated["outcome"], qubits, check_positions,
            check_values, simulated["phases"], q_z, q_x,
        )
        blocks.append({
            "attempt": attempt,
            "block": block_index,
            "secret_indices": secret_indices,
            "check_positions": check_positions,
            "check_values": check_values,
            "message": message,
            "outcome": simulated["outcome"],
            "records": simulated["records"],
            "score": score,
            "check_count": len(check_positions),
            "transmitted_qubits": 2 * qubits,
        })
    return blocks
