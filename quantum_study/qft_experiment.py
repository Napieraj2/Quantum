"""Two-pass QFT repeated-secret simulation and factorial sweep.

The module is extracted from the canonical ``BB84.ipynb`` implementation.
Importing it performs no CUDA probe, validation run, simulation, or file write.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp


TARGET_FALSE_ALARM = 0.05
BLOCK_QUBITS = (2, 4)
SECRET_BITS = 4
MAX_ATTEMPTS = 100
EVE_COUNTS = (1, 2, 4)
EVE_INTERCEPT_RATES = (0.005, 0.01, 0.025)
EVE_Z_PROBABILITIES = (0.1, 0.5, 0.9)
EVE_BASIS_REDRAW_RATES = (0.1, 1.0)
RECOVERY_CONFIDENCE = 0.99

SWEEP_NOISE_PROFILES = [
    ("No noise", 0.0, 0.0),
    ("Erven reference", 0.002, 0.038),
]
for _mean in (0.01, 0.02, 0.04):
    _low, _high = _mean * (1 - 0.9), _mean * (1 + 0.9)
    SWEEP_NOISE_PROFILES.extend([
        (f"{_mean:.0%} mean, balanced", _mean, _mean),
        (f"{_mean:.0%} mean, higher X", _low, _high),
        (f"{_mean:.0%} mean, higher Z", _high, _low),
    ])
del _mean, _low, _high

QFT_SEED = 20261102
QFT_PHASE_SAMPLES = 16
QFT_ALARM_CALIBRATION_RUNS = 100
QFT_EVALUATION_RUNS = 8
QFT_COMPUTE_BACKEND = "auto"
QFT_GPU_PRECISION = "complex64"
QFT_GPU_VALIDATION_RTOL = 2e-5
QFT_TRIAL_BATCH_SIZE = 128
QFT_PROGRESS_EVERY = 10
QFT_CHECKPOINT_PATH = Path("results/qft_factorial_checkpoint.json")
QFT_RESULTS_PATH = Path("results/qft_factorial_results.csv")
QFT_RESUME_SWEEP = True
QFT_DEMO_BLOCK_QUBITS = 2
QFT_DEMO_Q_Z = 0.002
QFT_DEMO_Q_X = 0.038
QFT_DEMO_CHECK_BUDGETS = (20, 50, 100)
QFT_DEMO_EVES = (
    {
        "intercept_probability": 0.025,
        "p_z": 0.5,
        "redraw_probability": 1.0,
    },
)
QFT_FULL_RUNS_PER_SETTING = 100
QFT_CHECKPOINT_SCHEMA = 1
QFT_SEED_VERSION = 1
QFT_POSTERIOR_VERSION = 1


@dataclass(frozen=True)
class BackendStatus:
    """Selected numerical backend and its visible device metadata."""

    name: str
    array_module: Any
    precision: str
    device_name: str | None
    device_count: int


cp = None
QFT_CUPY_DEVICE_COUNT = 0
QFT_CUPY_AVAILABLE = False
QFT_CUPY_IMPORT_ERROR = None
QFT_ARRAY_MODULE = np
QFT_SELECTED_BACKEND = "numpy"


def configure_qft_backend(
    requested=QFT_COMPUTE_BACKEND, gpu_precision="complex64"
):
    """Select NumPy or probe and select CuPy without import-time side effects."""
    global cp
    global QFT_ARRAY_MODULE, QFT_SELECTED_BACKEND, QFT_GPU_PRECISION
    global QFT_CUPY_AVAILABLE, QFT_CUPY_DEVICE_COUNT, QFT_CUPY_IMPORT_ERROR

    if requested not in ("auto", "numpy", "cupy"):
        raise ValueError("QFT backend must be 'auto', 'numpy', or 'cupy'")
    if gpu_precision not in ("complex64", "complex128"):
        raise ValueError("gpu_precision must be 'complex64' or 'complex128'")
    QFT_GPU_PRECISION = gpu_precision

    if requested == "numpy":
        QFT_ARRAY_MODULE = np
        QFT_SELECTED_BACKEND = "numpy"
        return BackendStatus("numpy", np, "complex128", None, 0)

    try:
        import cupy as cupy_module

        device_count = cupy_module.cuda.runtime.getDeviceCount()
        if device_count < 1:
            raise RuntimeError("CuPy found no visible CUDA devices")
        cuda_probe = cupy_module.exp(
            1j * cupy_module.arange(8, dtype=cupy_module.float32)
        ).astype(cupy_module.complex64).reshape(2, 2, 2)
        cuda_probe_result = cupy_module.asnumpy(
            cupy_module.matmul(cuda_probe, cuda_probe)
        )
        cupy_module.cuda.Stream.null.synchronize()
        if not np.isfinite(cuda_probe_result).all():
            raise RuntimeError("CuPy CUDA matrix probe returned nonfinite values")
        properties = cupy_module.cuda.runtime.getDeviceProperties(0)
        device_name = properties["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode()
    except Exception as error:
        QFT_CUPY_AVAILABLE = False
        QFT_CUPY_DEVICE_COUNT = 0
        QFT_CUPY_IMPORT_ERROR = error
        if requested == "cupy":
            raise RuntimeError(
                "CuPy with a visible CUDA device is required for the requested backend"
            ) from error
        QFT_ARRAY_MODULE = np
        QFT_SELECTED_BACKEND = "numpy"
        return BackendStatus("numpy", np, "complex128", None, 0)

    cp = cupy_module
    QFT_CUPY_AVAILABLE = True
    QFT_CUPY_DEVICE_COUNT = device_count
    QFT_CUPY_IMPORT_ERROR = None
    QFT_ARRAY_MODULE = cp
    QFT_SELECTED_BACKEND = "cupy"
    return BackendStatus(
        "cupy", cp, gpu_precision, str(device_name), device_count
    )


def qft_get_backend(requested=None):
    """Return an array module and stable backend name."""
    if requested is None:
        return QFT_ARRAY_MODULE, QFT_SELECTED_BACKEND
    if isinstance(requested, BackendStatus):
        return requested.array_module, requested.name
    if requested is np:
        return np, "numpy"
    if cp is not None and requested is cp:
        return cp, "cupy"
    if requested == "numpy":
        return np, "numpy"
    if requested in ("auto", "cupy"):
        status = configure_qft_backend(requested, QFT_GPU_PRECISION)
        return status.array_module, status.name
    raise ValueError("QFT backend must be 'auto', 'numpy', 'cupy', or BackendStatus")


def qft_backend_dtype(array_module):
    if array_module is np:
        return np.complex128
    if QFT_GPU_PRECISION not in ("complex64", "complex128"):
        raise ValueError("QFT_GPU_PRECISION must be 'complex64' or 'complex128'")
    return getattr(array_module, QFT_GPU_PRECISION)


def calibrate_bb84_cutoffs(null_scores, target):
    """Return upper-tail cutoffs and randomized tie probabilities."""
    rank = int(np.ceil((1 - target) * len(null_scores))) - 1
    cutoffs = np.partition(null_scores, rank, axis=0)[rank]
    above = (null_scores > cutoffs).mean(axis=0)
    tied = (null_scores == cutoffs).mean(axis=0)
    return cutoffs, np.clip((target - above) / tied, 0.0, 1.0)


def standardize_look_scores(scores, centers, scales):
    """Standardize look scores while preserving zero-variance tails."""
    delta = scores - centers
    standardized = np.zeros_like(delta, dtype=float)
    np.divide(delta, scales, out=standardized, where=scales > 0)
    zero_scale = np.broadcast_to(scales == 0, delta.shape)
    standardized[zero_scale & (delta > 0)] = np.inf
    standardized[zero_scale & (delta < 0)] = -np.inf
    return standardized


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


def qft_null_outcome_distribution(qubits, message, phases, q_z, q_x):
    dimension = 2**qubits
    plus = np.ones(dimension, dtype=complex) / np.sqrt(dimension)
    scrambler = qft_scrambler(phases)
    encoding = qft_encoding_matrix(qubits, message)
    inverse_qft = qft_matrix(qubits).conj().T

    state = scrambler @ plus
    density = np.outer(state, state.conj())
    density = apply_pauli_channel(density, qubits, q_z, q_x)
    density = encoding @ density @ encoding.conj().T
    density = apply_pauli_channel(density, qubits, q_z, q_x)
    decoded = inverse_qft @ scrambler.conj().T
    density = decoded @ density @ decoded.conj().T
    probabilities = np.real(np.diag(density))
    probabilities = np.clip(probabilities, 0.0, None)
    return probabilities / probabilities.sum()


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
    plus = np.ones(dimension, dtype=complex) / np.sqrt(dimension)
    scrambler = qft_scrambler(phases)
    state = scrambler @ plus
    density = np.outer(state, state.conj())
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

    decoded = qft_matrix(qubits).conj().T @ scrambler.conj().T
    density = decoded @ density @ decoded.conj().T
    probabilities = np.clip(np.real(np.diag(density)), 0.0, None)
    probabilities /= probabilities.sum()
    outcome = int(rng.choice(dimension, p=probabilities))
    return {
        "outcome": outcome,
        "phases": np.asarray(phases),
        "records": records,
    }


QFT_BATCH_CACHE = {}


def qft_backend_cache(qubits, array_module):
    dtype = qft_backend_dtype(array_module)
    backend_name = "numpy" if array_module is np else "cupy"
    key = (backend_name, qubits, str(dtype))
    if key in QFT_BATCH_CACHE:
        return QFT_BATCH_CACHE[key]

    dimension = 2**qubits
    real_dtype = (
        array_module.float32
        if dtype == array_module.complex64
        else array_module.float64
    )
    bit_matrix = np.asarray([
        qft_bits(value, qubits) for value in range(dimension)
    ])
    encoding_angles = (
        2 * np.pi
        * np.arange(dimension)[:, None]
        * np.arange(dimension)[None, :]
        / dimension
    )
    cache = {
        "bits": array_module.asarray(bit_matrix, dtype=real_dtype),
        "encoding_diagonals": array_module.asarray(
            np.exp(1j * encoding_angles), dtype=dtype
        ),
        "x_operators": tuple(
            array_module.asarray(
                local_operator(PAULI_X, qubit, qubits), dtype=dtype
            )
            for qubit in range(qubits)
        ),
        "z_operators": tuple(
            array_module.asarray(
                local_operator(PAULI_Z, qubit, qubits), dtype=dtype
            )
            for qubit in range(qubits)
        ),
        "projectors": {
            (basis, outcome, qubit): array_module.asarray(
                measurement_projector(
                    basis, outcome, qubit, qubits
                ),
                dtype=dtype,
            )
            for basis in ("Z", "X")
            for outcome in (0, 1)
            for qubit in range(qubits)
        },
        "dtype": dtype,
        "real_dtype": real_dtype,
    }
    QFT_BATCH_CACHE[key] = cache
    return cache


def qft_apply_pauli_channel_batch(
    density, q_z, q_x, cache, array_module
):
    if min(q_z, q_x) < 0 or q_z + q_x > 1:
        raise ValueError(
            "q_z and q_x must be nonnegative and sum to at most one"
        )
    for x_operator, z_operator in zip(
        cache["x_operators"], cache["z_operators"]
    ):
        x_density = array_module.matmul(
            array_module.matmul(x_operator, density), x_operator
        )
        z_density = array_module.matmul(
            array_module.matmul(z_operator, density), z_operator
        )
        density = (
            (1 - q_z - q_x) * density
            + q_z * x_density
            + q_x * z_density
        )
    return density


def qft_apply_fixed_records_batch(
    density, records, leg, cache, array_module
):
    for record in records:
        if record["leg"] != leg:
            continue
        projector = cache["projectors"][(
            record["basis"], record["outcome"], record["signal"]
        )]
        density = array_module.matmul(
            array_module.matmul(projector, density), projector
        )
    return density


def qft_record_likelihood_scalar(
    qubits, message, records, q_z, q_x, phase_samples
):
    encoding = qft_encoding_matrix(qubits, message)
    likelihood = 0.0
    for phases in phase_samples:
        dimension = 2**qubits
        plus = np.ones(dimension, dtype=complex) / np.sqrt(dimension)
        state = qft_scrambler(phases) @ plus
        density = np.outer(state, state.conj())
        density = apply_pauli_channel(density, qubits, q_z, q_x)
        density = apply_fixed_eve_records(
            density, records, "forward", qubits
        )
        density = encoding @ density @ encoding.conj().T
        density = apply_pauli_channel(density, qubits, q_z, q_x)
        density = apply_fixed_eve_records(
            density, records, "return", qubits
        )
        likelihood += max(np.real(np.trace(density)), 0.0)
    return likelihood / len(phase_samples)


def qft_record_likelihoods_batched(
    qubits, messages, records, q_z, q_x, phase_samples,
    array_module=None,
):
    array_module = (
        QFT_ARRAY_MODULE if array_module is None else array_module
    )
    messages = np.asarray(messages, dtype=int)
    if messages.ndim != 1:
        raise ValueError("messages must be a one-dimensional sequence")
    if not len(messages):
        return np.empty(0)

    cache = qft_backend_cache(qubits, array_module)
    dimension = 2**qubits
    phases = array_module.asarray(
        phase_samples, dtype=cache["real_dtype"]
    )
    angles = array_module.matmul(phases, cache["bits"].T)
    states = array_module.exp(1j * angles).astype(
        cache["dtype"], copy=False
    ) / np.sqrt(dimension)
    density = (
        states[:, :, None]
        * array_module.conj(states[:, None, :])
    )
    density = qft_apply_pauli_channel_batch(
        density, q_z, q_x, cache, array_module
    )
    density = qft_apply_fixed_records_batch(
        density, records, "forward", cache, array_module
    )

    message_indices = array_module.asarray(messages, dtype=int)
    encodings = cache["encoding_diagonals"][message_indices]
    density = (
        density[None, :, :, :]
        * encodings[:, None, :, None]
        * array_module.conj(encodings[:, None, None, :])
    )
    density = qft_apply_pauli_channel_batch(
        density, q_z, q_x, cache, array_module
    )
    density = qft_apply_fixed_records_batch(
        density, records, "return", cache, array_module
    )

    traces = array_module.real(
        array_module.trace(density, axis1=-2, axis2=-1)
    )
    likelihoods = array_module.maximum(traces, 0).mean(axis=1)
    if array_module is np:
        return np.asarray(likelihoods, dtype=float)
    return cp.asnumpy(likelihoods).astype(float, copy=False)


def qft_record_likelihood(
    qubits, message, records, q_z, q_x, phase_samples
):
    return float(qft_record_likelihoods_batched(
        qubits, [message], records, q_z, q_x, phase_samples
    )[0])


def apply_fixed_eve_records(density, records, leg, qubits):
    for record in records:
        if record["leg"] != leg:
            continue
        projector = measurement_projector(
            record["basis"], record["outcome"],
            record["signal"], qubits,
        )
        density = projector @ density @ projector
    return density


def qft_apply_heterogeneous_records_batch(
    density, records_by_block, leg, cache, array_module
):
    sequences = [
        [record for record in records if record["leg"] == leg]
        for records in records_by_block
    ]
    maximum_records = max((len(records) for records in sequences), default=0)
    if not maximum_records:
        return density

    dimension = density.shape[-1]
    identity = array_module.eye(dimension, dtype=cache["dtype"])
    batch_size = len(sequences)
    projector_shape = (
        (batch_size,)
        + (1,) * (density.ndim - 3)
        + (dimension, dimension)
    )
    for record_index in range(maximum_records):
        projectors = array_module.stack([
            cache["projectors"][(
                records[record_index]["basis"],
                records[record_index]["outcome"],
                records[record_index]["signal"],
            )]
            if record_index < len(records)
            else identity
            for records in sequences
        ]).reshape(projector_shape)
        density = array_module.matmul(
            array_module.matmul(projectors, density), projectors
        )
    return density


def qft_record_likelihoods_wavefront(
    qubits, candidate_messages, records_by_block, q_z, q_x,
    phase_samples_by_block, array_module=None,
):
    array_module = (
        QFT_ARRAY_MODULE if array_module is None else array_module
    )
    candidate_messages = np.asarray(candidate_messages, dtype=int)
    phase_samples_by_block = np.asarray(phase_samples_by_block)
    if candidate_messages.ndim != 2:
        raise ValueError(
            "candidate_messages must have shape (blocks, candidates)"
        )
    if phase_samples_by_block.ndim != 3:
        raise ValueError(
            "phase_samples_by_block must have shape "
            "(blocks, phase_samples, qubits)"
        )
    batch_size = len(records_by_block)
    if (
        candidate_messages.shape[0] != batch_size
        or phase_samples_by_block.shape[0] != batch_size
        or phase_samples_by_block.shape[2] != qubits
    ):
        raise ValueError("wavefront batch dimensions do not match")

    cache = qft_backend_cache(qubits, array_module)
    dimension = 2**qubits
    phases = array_module.asarray(
        phase_samples_by_block, dtype=cache["real_dtype"]
    )
    angles = array_module.matmul(phases, cache["bits"].T)
    states = array_module.exp(1j * angles).astype(
        cache["dtype"], copy=False
    ) / np.sqrt(dimension)
    density = (
        states[:, :, :, None]
        * array_module.conj(states[:, :, None, :])
    )
    density = qft_apply_pauli_channel_batch(
        density, q_z, q_x, cache, array_module
    )
    density = qft_apply_heterogeneous_records_batch(
        density, records_by_block, "forward", cache, array_module
    )

    message_indices = array_module.asarray(
        candidate_messages, dtype=int
    )
    encodings = cache["encoding_diagonals"][message_indices]
    density = (
        density[:, None, :, :, :]
        * encodings[:, :, None, :, None]
        * array_module.conj(encodings[:, :, None, None, :])
    )
    density = qft_apply_pauli_channel_batch(
        density, q_z, q_x, cache, array_module
    )
    density = qft_apply_heterogeneous_records_batch(
        density, records_by_block, "return", cache, array_module
    )

    traces = array_module.real(
        array_module.trace(density, axis1=-2, axis2=-1)
    )
    likelihoods = array_module.maximum(traces, 0).mean(axis=-1)
    if array_module is np:
        return np.asarray(likelihoods, dtype=float)
    return cp.asnumpy(likelihoods).astype(float, copy=False)


def update_secret_posteriors_batched(
    log_posteriors, blocks, qubits, q_z, q_x,
    phase_samples_by_block, backend=None,
):
    log_posteriors = np.asarray(log_posteriors, dtype=float)
    if log_posteriors.ndim != 2 or log_posteriors.shape[1] != 2**SECRET_BITS:
        raise ValueError(
            "log_posteriors must have shape (blocks, 2**SECRET_BITS)"
        )
    if len(blocks) != len(log_posteriors):
        raise ValueError("blocks and log_posteriors must have equal length")
    phase_samples_by_block = np.asarray(phase_samples_by_block)
    if phase_samples_by_block.shape[0] != len(blocks):
        raise ValueError(
            "phase_samples_by_block must contain one batch per block"
        )

    array_module, _ = qft_get_backend(backend)

    updated = log_posteriors.copy()
    active = np.asarray([
        index for index, block in enumerate(blocks)
        if block["records"]
    ], dtype=int)
    if not len(active):
        updated -= logsumexp(updated, axis=1, keepdims=True)
        return updated

    candidate_messages = np.empty(
        (len(active), 2**SECRET_BITS), dtype=int
    )
    active_records = []
    for active_index, block_index in enumerate(active):
        block = blocks[block_index]
        active_records.append(block["records"])
        for candidate in range(2**SECRET_BITS):
            secret = qft_bits(candidate, SECRET_BITS)
            payload = [
                secret[index] for index in block["secret_indices"]
            ]
            candidate_messages[active_index, candidate] = qft_message(
                block["check_positions"], block["check_values"],
                payload, qubits,
            )

    unique_rows = []
    inverse_rows = []
    for messages in candidate_messages:
        unique, inverse = np.unique(messages, return_inverse=True)
        unique_rows.append(unique)
        inverse_rows.append(inverse)
    maximum_unique = max(len(unique) for unique in unique_rows)
    unique_messages = np.empty(
        (len(active), maximum_unique), dtype=int
    )
    for active_index, unique in enumerate(unique_rows):
        unique_messages[active_index, :len(unique)] = unique
        unique_messages[active_index, len(unique):] = unique[0]
    inverse_messages = np.stack(inverse_rows)

    unique_likelihoods = qft_record_likelihoods_wavefront(
        qubits, unique_messages, active_records, q_z, q_x,
        phase_samples_by_block[active], array_module=array_module,
    )
    likelihoods = unique_likelihoods[
        np.arange(len(active))[:, None], inverse_messages
    ]
    updated[active] += np.log(np.maximum(
        likelihoods, np.finfo(float).tiny
    ))
    updated -= logsumexp(updated, axis=1, keepdims=True)
    return updated



def update_secret_posterior(
    log_posterior, blocks, qubits, q_z, q_x, phase_rng,
    phase_samples=QFT_PHASE_SAMPLES,
):
    candidate_logs = log_posterior.copy()
    for block in blocks:
        if not block["records"]:
            continue
        common_phases = phase_rng.uniform(
            0.0, 2 * np.pi, size=(phase_samples, qubits)
        )
        candidate_messages = []
        for candidate in range(2**SECRET_BITS):
            secret = qft_bits(candidate, SECRET_BITS)
            payload = [
                secret[index] for index in block["secret_indices"]
            ]
            candidate_messages.append(qft_message(
                block["check_positions"], block["check_values"],
                payload, qubits,
            ))
        unique_messages, inverse = np.unique(
            candidate_messages, return_inverse=True
        )
        unique_likelihoods = qft_record_likelihoods_batched(
            qubits, unique_messages, block["records"], q_z, q_x,
            common_phases,
        )
        likelihoods = unique_likelihoods[inverse]
        candidate_logs += np.log(np.maximum(
            likelihoods, np.finfo(float).tiny
        ))
    candidate_logs -= logsumexp(candidate_logs)
    return candidate_logs


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


def qft_score_trajectory(rng, qubits, q_z, q_x, budgets):
    secret = qft_bits(
        int(rng.integers(0, 2**SECRET_BITS)), SECRET_BITS
    )
    scores = []
    cumulative_score = 0.0
    retained_checks = 0
    maximum_budget = max(budgets)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        blocks = simulate_qft_attempt(
            rng, secret, qubits, q_z, q_x, (), [], attempt
        )
        for block in blocks:
            cumulative_score += block["score"]
            retained_checks += block["check_count"]
            if retained_checks in budgets:
                scores.append(cumulative_score)
            if retained_checks >= maximum_budget:
                return np.asarray(scores)
    raise RuntimeError("QFT check budget was not reached before the attempt limit")


def calibrate_qft_alarm(
    rng, qubits, q_z, q_x, budgets,
    runs=QFT_ALARM_CALIBRATION_RUNS,
):
    trajectories = np.stack([
        qft_score_trajectory(rng, qubits, q_z, q_x, budgets)
        for _ in range(runs)
    ])
    centers = trajectories.mean(axis=0)
    scales = trajectories.std(axis=0)
    standardized = standardize_look_scores(
        trajectories[:, :, None], centers[:, None], scales[:, None]
    )[:, :, 0]
    maxima = standardized.max(axis=1)
    cutoff, tie_rate = calibrate_bb84_cutoffs(
        maxima[:, None], TARGET_FALSE_ALARM
    )
    return {
        "budgets": tuple(budgets),
        "centers": centers,
        "scales": scales,
        "cutoff": cutoff[0],
        "tie_probability": tie_rate[0],
    }


def run_qft_repeated_trial(
    rng, phase_rng, qubits, q_z, q_x, eve_configs, alarm_rule
):
    secret_value = int(rng.integers(0, 2**SECRET_BITS))
    secret = qft_bits(secret_value, SECRET_BITS)
    log_posterior = np.full(2**SECRET_BITS, -np.log(2**SECRET_BITS))
    eve_bases = [
        "Z" if rng.random() < config["p_z"] else "X"
        for config in eve_configs
    ]
    tie_draw = rng.random()
    cumulative_score = 0.0
    retained_checks = 0
    transmitted_qubits = 0
    total_intercepts = 0
    look_index = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        blocks = simulate_qft_attempt(
            rng, secret, qubits, q_z, q_x,
            eve_configs, eve_bases, attempt,
        )
        for block in blocks:
            cumulative_score += block["score"]
            retained_checks += block["check_count"]
            transmitted_qubits += block["transmitted_qubits"]
            total_intercepts += len(block["records"])

            alarm = False
            if (
                look_index < len(alarm_rule["budgets"])
                and retained_checks == alarm_rule["budgets"][look_index]
            ):
                center = alarm_rule["centers"][look_index]
                scale = alarm_rule["scales"][look_index]
                if scale > 0:
                    standardized = (cumulative_score - center) / scale
                elif cumulative_score > center:
                    standardized = np.inf
                elif cumulative_score < center:
                    standardized = -np.inf
                else:
                    standardized = 0.0
                alarm = (
                    standardized > alarm_rule["cutoff"]
                    or (
                        standardized == alarm_rule["cutoff"]
                        and tie_draw < alarm_rule["tie_probability"]
                    )
                )
                look_index += 1

            # An alarm at this block wins a timing tie with coalition recovery.
            if alarm:
                return {
                    "secret": secret_value,
                    "alarm": True,
                    "alarm_attempt": attempt,
                    "recovered": False,
                    "correct_recovery": False,
                    "recovery_attempt": np.nan,
                    "posterior_confidence": np.exp(log_posterior).max(),
                    "retained_checks": retained_checks,
                    "transmitted_qubits": transmitted_qubits,
                    "intercepts": total_intercepts,
                }

            log_posterior = update_secret_posterior(
                log_posterior, [block], qubits, q_z, q_x, phase_rng
            )
            declared = int(np.argmax(log_posterior))
            confidence = float(np.exp(log_posterior[declared]))
            if confidence >= RECOVERY_CONFIDENCE:
                return {
                    "secret": secret_value,
                    "alarm": False,
                    "alarm_attempt": np.nan,
                    "recovered": True,
                    "correct_recovery": declared == secret_value,
                    "recovery_attempt": attempt,
                    "posterior_confidence": confidence,
                    "retained_checks": retained_checks,
                    "transmitted_qubits": transmitted_qubits,
                    "intercepts": total_intercepts,
                }

            if retained_checks >= max(alarm_rule["budgets"]):
                return {
                    "secret": secret_value,
                    "alarm": False,
                    "alarm_attempt": np.nan,
                    "recovered": False,
                    "correct_recovery": False,
                    "recovery_attempt": np.nan,
                    "posterior_confidence": np.exp(log_posterior).max(),
                    "retained_checks": retained_checks,
                    "transmitted_qubits": transmitted_qubits,
                    "intercepts": total_intercepts,
                }
    raise RuntimeError("QFT repeated-secret trial exceeded the attempt limit")



def presimulate_qft_trial_path(
    seed_sequence, qubits, q_z, q_x, eve_configs, alarm_rule,
    phase_samples=QFT_PHASE_SAMPLES,
):
    """Generate one independently seeded path through alarm or final budget."""
    simulation_seed, posterior_seed = seed_sequence.spawn(2)
    rng = np.random.default_rng(simulation_seed)
    phase_rng = np.random.default_rng(posterior_seed)

    secret_value = int(rng.integers(0, 2**SECRET_BITS))
    secret = qft_bits(secret_value, SECRET_BITS)
    eve_bases = [
        "Z" if rng.random() < config["p_z"] else "X"
        for config in eve_configs
    ]
    tie_draw = rng.random()
    cumulative_score = 0.0
    retained_checks = 0
    transmitted_qubits = 0
    total_intercepts = 0
    look_index = 0
    maximum_budget = max(alarm_rule["budgets"])
    events = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        blocks = simulate_qft_attempt(
            rng, secret, qubits, q_z, q_x,
            eve_configs, eve_bases, attempt,
        )
        for block in blocks:
            cumulative_score += block["score"]
            retained_checks += block["check_count"]
            transmitted_qubits += block["transmitted_qubits"]
            total_intercepts += len(block["records"])

            alarm = False
            if (
                look_index < len(alarm_rule["budgets"])
                and retained_checks == alarm_rule["budgets"][look_index]
            ):
                center = alarm_rule["centers"][look_index]
                scale = alarm_rule["scales"][look_index]
                if scale > 0:
                    standardized = (cumulative_score - center) / scale
                elif cumulative_score > center:
                    standardized = np.inf
                elif cumulative_score < center:
                    standardized = -np.inf
                else:
                    standardized = 0.0
                alarm = (
                    standardized > alarm_rule["cutoff"]
                    or (
                        standardized == alarm_rule["cutoff"]
                        and tie_draw < alarm_rule["tie_probability"]
                    )
                )
                look_index += 1

            posterior_phases = None
            if not alarm and block["records"]:
                posterior_phases = phase_rng.uniform(
                    0.0, 2 * np.pi, size=(phase_samples, qubits)
                )
            events.append({
                "attempt": attempt,
                "block": block,
                "alarm": alarm,
                "retained_checks": retained_checks,
                "transmitted_qubits": transmitted_qubits,
                "intercepts": total_intercepts,
                "posterior_phases": posterior_phases,
            })
            if alarm or retained_checks >= maximum_budget:
                return {"secret": secret_value, "events": events}

    raise RuntimeError("QFT repeated-secret path exceeded the attempt limit")


def qft_trial_result(path, event, log_posterior, *, alarm=False, recovered=False):
    declared = int(np.argmax(log_posterior))
    confidence = float(np.exp(log_posterior[declared]))
    return {
        "secret": path["secret"],
        "alarm": bool(alarm),
        "alarm_attempt": event["attempt"] if alarm else np.nan,
        "recovered": bool(recovered),
        "correct_recovery": bool(
            recovered and declared == path["secret"]
        ),
        "recovery_attempt": event["attempt"] if recovered else np.nan,
        "posterior_confidence": confidence,
        "retained_checks": event["retained_checks"],
        "transmitted_qubits": event["transmitted_qubits"],
        "intercepts": event["intercepts"],
    }


def evaluate_qft_trial_paths_batched(
    paths, qubits, q_z, q_x, alarm_rule, *, backend=None
):
    trial_count = len(paths)
    log_posteriors = np.full(
        (trial_count, 2**SECRET_BITS), -np.log(2**SECRET_BITS)
    )
    results = [None] * trial_count
    active = list(range(trial_count))
    wavefront = 0
    maximum_budget = max(alarm_rule["budgets"])

    while active:
        current_events = {}
        posterior_indices = []
        next_active = []

        for trial_index in active:
            path = paths[trial_index]
            if wavefront >= len(path["events"]):
                raise RuntimeError(
                    "An active trial exhausted its presimulated path"
                )
            event = path["events"][wavefront]
            current_events[trial_index] = event
            if event["alarm"]:
                results[trial_index] = qft_trial_result(
                    path, event, log_posteriors[trial_index], alarm=True
                )
                continue
            if event["block"]["records"]:
                posterior_indices.append(trial_index)
            next_active.append(trial_index)

        if posterior_indices:
            phases = np.stack([
                current_events[index]["posterior_phases"]
                for index in posterior_indices
            ])
            updated = update_secret_posteriors_batched(
                log_posteriors[posterior_indices],
                [
                    current_events[index]["block"]
                    for index in posterior_indices
                ],
                qubits, q_z, q_x, phases, backend=backend,
            )
            log_posteriors[posterior_indices] = np.asarray(updated)

        still_active = []
        for trial_index in next_active:
            path = paths[trial_index]
            event = current_events[trial_index]
            confidence = float(np.exp(
                log_posteriors[trial_index].max()
            ))
            if confidence >= RECOVERY_CONFIDENCE:
                results[trial_index] = qft_trial_result(
                    path, event, log_posteriors[trial_index],
                    recovered=True,
                )
            elif event["retained_checks"] >= maximum_budget:
                results[trial_index] = qft_trial_result(
                    path, event, log_posteriors[trial_index]
                )
            else:
                still_active.append(trial_index)

        active = still_active
        wavefront += 1

    return results


def run_qft_repeated_trials_batched(
    trial_seed_sequences, qubits, q_z, q_x, eve_configs, alarm_rule,
    *, phase_samples=QFT_PHASE_SAMPLES,
    batch_size=QFT_TRIAL_BATCH_SIZE,
    backend=None,
):
    """Run independent trial paths in GPU-sized posterior wavefronts."""
    trial_seed_sequences = list(trial_seed_sequences)
    if not trial_seed_sequences:
        raise ValueError("At least one trial seed is required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    rows = []
    for start in range(0, len(trial_seed_sequences), batch_size):
        seed_chunk = trial_seed_sequences[start:start + batch_size]
        paths = [
            presimulate_qft_trial_path(
                seed_sequence, qubits, q_z, q_x,
                eve_configs, alarm_rule, phase_samples,
            )
            for seed_sequence in seed_chunk
        ]
        rows.extend(evaluate_qft_trial_paths_batched(
            paths, qubits, q_z, q_x, alarm_rule, backend=backend
        ))
    return pd.DataFrame(rows)


def validate_qft_model(backend=None, *, verbose=False):
    """Run the canonical core, trajectory, batch, and positive-control checks.

    Validation is explicit rather than an import side effect.  The returned
    tables retain the numerical evidence behind each assertion.
    """
    array_module, backend_name = qft_get_backend(backend)
    messages = []

    core_rows = []
    for qubits in BLOCK_QUBITS:
        phases = np.linspace(0.17, 2.41, qubits)
        for message in range(2**qubits):
            probabilities = qft_null_outcome_distribution(
                qubits, message, phases, 0.0, 0.0
            )
            core_rows.append({
                "qubits": qubits,
                "message": message,
                "decoded": int(np.argmax(probabilities)),
                "correct_probability": probabilities[message],
            })
    core = pd.DataFrame(core_rows)
    if not (
        core["message"].eq(core["decoded"]).all()
        and np.allclose(core["correct_probability"], 1.0)
    ):
        raise AssertionError(
            "Noiseless QFT round trip did not decode every message"
        )

    trajectory_rng = np.random.default_rng(QFT_SEED)
    for qubits in BLOCK_QUBITS:
        block = simulate_qft_block(
            trajectory_rng, qubits, 2**qubits - 1, 0.0, 0.0,
            (), [], attempt=1, block=1,
        )
        if block["outcome"] != 2**qubits - 1 or block["records"]:
            raise AssertionError(
                "Noiseless trajectory did not preserve the QFT message"
            )
    messages.append(
        "PASS: noiseless 2- and 4-qubit QFT trajectories decode exactly"
    )

    validation_phase_rng = np.random.default_rng(QFT_SEED + 17)
    likelihood_rows = []
    for qubits in BLOCK_QUBITS:
        dimension = 2**qubits
        records = [
            {
                "leg": "forward",
                "signal": 0,
                "basis": "Z",
                "outcome": 0,
            },
            {
                "leg": "return",
                "signal": qubits - 1,
                "basis": "X",
                "outcome": 1,
            },
        ]
        phase_samples = validation_phase_rng.uniform(
            0.0, 2 * np.pi, size=(8, qubits)
        )
        candidate_messages = np.arange(dimension)
        scalar_likelihoods = np.asarray([
            qft_record_likelihood_scalar(
                qubits, message, records, 0.002, 0.038,
                phase_samples,
            )
            for message in candidate_messages
        ])
        batched_cpu = qft_record_likelihoods_batched(
            qubits, candidate_messages, records, 0.002, 0.038,
            phase_samples, array_module=np,
        )
        if not np.allclose(
            scalar_likelihoods, batched_cpu, rtol=1e-12, atol=1e-12
        ):
            raise AssertionError(
                "Batched NumPy QFT likelihoods differ from the scalar "
                "reference"
            )

        duplicate_likelihoods = qft_record_likelihoods_batched(
            qubits, [0, 0, dimension - 1], records, 0.002, 0.038,
            phase_samples, array_module=np,
        )
        if duplicate_likelihoods[0] != duplicate_likelihoods[1]:
            raise AssertionError(
                "Duplicate QFT messages must have identical likelihoods"
            )

        gpu_max_abs_error = np.nan
        if backend_name == "cupy":
            batched_gpu = qft_record_likelihoods_batched(
                qubits, candidate_messages, records, 0.002, 0.038,
                phase_samples, array_module=array_module,
            )
            if not np.allclose(
                batched_cpu, batched_gpu,
                rtol=QFT_GPU_VALIDATION_RTOL,
                atol=QFT_GPU_VALIDATION_RTOL,
            ):
                raise AssertionError(
                    "CuPy QFT likelihoods differ from the NumPy batch"
                )
            gpu_max_abs_error = float(np.max(np.abs(
                batched_cpu - batched_gpu
            )))
        likelihood_rows.append({
            "qubits": qubits,
            "scalar_batch_max_abs_error": float(np.max(np.abs(
                scalar_likelihoods - batched_cpu
            ))),
            "gpu_batch_max_abs_error": gpu_max_abs_error,
        })
    messages.append(
        "PASS: batched posterior likelihoods match the scalar NumPy reference"
    )

    wavefront_blocks = [
        {
            "records": [
                {
                    "leg": "forward",
                    "signal": 0,
                    "basis": "Z",
                    "outcome": 0,
                },
                {
                    "leg": "return",
                    "signal": 3,
                    "basis": "X",
                    "outcome": 1,
                },
            ],
            "secret_indices": (0, 1),
            "check_positions": (1, 3),
            "check_values": (0, 1),
        },
        {
            "records": [{
                "leg": "return",
                "signal": 1,
                "basis": "Z",
                "outcome": 1,
            }],
            "secret_indices": (2, 3),
            "check_positions": (0, 2),
            "check_values": (1, 0),
        },
        {
            "records": [],
            "secret_indices": (0, 1),
            "check_positions": (2, 3),
            "check_values": (1, 1),
        },
    ]
    wavefront_phases = validation_phase_rng.uniform(
        0.0, 2 * np.pi, size=(len(wavefront_blocks), 8, 4)
    )
    wavefront_prior = np.full(
        (len(wavefront_blocks), 2**SECRET_BITS),
        -np.log(2**SECRET_BITS),
    )
    wavefront_cpu = update_secret_posteriors_batched(
        wavefront_prior, wavefront_blocks, 4, 0.002, 0.038,
        wavefront_phases, backend="numpy",
    )
    wavefront_scalar = wavefront_prior.copy()
    for block_index, block in enumerate(wavefront_blocks):
        if not block["records"]:
            continue
        for candidate in range(2**SECRET_BITS):
            secret = qft_bits(candidate, SECRET_BITS)
            payload = [
                secret[index] for index in block["secret_indices"]
            ]
            message = qft_message(
                block["check_positions"], block["check_values"],
                payload, 4,
            )
            likelihood = qft_record_likelihood_scalar(
                4, message, block["records"], 0.002, 0.038,
                wavefront_phases[block_index],
            )
            wavefront_scalar[block_index, candidate] += np.log(max(
                likelihood, np.finfo(float).tiny
            ))
    wavefront_scalar -= logsumexp(
        wavefront_scalar, axis=1, keepdims=True
    )
    if not np.allclose(
        wavefront_cpu, wavefront_scalar, rtol=1e-12, atol=1e-12
    ):
        raise AssertionError(
            "Cross-trial NumPy wavefront differs from scalar updates"
        )

    wavefront_gpu_max_abs_error = np.nan
    if backend_name == "cupy":
        wavefront_gpu = update_secret_posteriors_batched(
            wavefront_prior, wavefront_blocks, 4, 0.002, 0.038,
            wavefront_phases, backend=backend,
        )
        if not np.allclose(
            wavefront_cpu, wavefront_gpu,
            rtol=QFT_GPU_VALIDATION_RTOL,
            atol=QFT_GPU_VALIDATION_RTOL,
        ):
            raise AssertionError(
                "CuPy cross-trial wavefront differs from NumPy"
            )
        wavefront_gpu_max_abs_error = float(np.max(np.abs(
            wavefront_cpu - wavefront_gpu
        )))
    messages.append(
        "PASS: heterogeneous cross-trial posterior wavefront matches "
        "scalar updates"
    )

    positive_control_alarm_rule = {
        "budgets": QFT_DEMO_CHECK_BUDGETS,
        "centers": np.zeros(len(QFT_DEMO_CHECK_BUDGETS)),
        "scales": np.ones(len(QFT_DEMO_CHECK_BUDGETS)),
        "cutoff": np.inf,
        "tie_probability": 0.0,
    }
    positive_control_eves = ({
        "intercept_probability": 1.0,
        "p_z": 0.0,
        "redraw_probability": 1.0,
    },)
    positive_control = run_qft_repeated_trials_batched(
        np.random.SeedSequence(QFT_SEED + 101).spawn(4),
        2, 0.0, 0.0,
        positive_control_eves,
        positive_control_alarm_rule,
        backend=backend,
    )
    if not (
        positive_control["recovered"].all()
        and positive_control["correct_recovery"].all()
    ):
        raise AssertionError(
            "Extreme X-basis interception should recover the repeated secret"
        )
    messages.append(
        "PASS: extreme X-basis interception recovered 4/4 secrets "
        "(posterior positive control)"
    )

    if verbose:
        for message in messages:
            print(message)
    return {
        "backend": backend_name,
        "core": core,
        "likelihood_parity": pd.DataFrame(likelihood_rows),
        "wavefront_scalar_max_abs_error": float(np.max(np.abs(
            wavefront_cpu - wavefront_scalar
        ))),
        "wavefront_gpu_max_abs_error": wavefront_gpu_max_abs_error,
        "positive_control": positive_control,
        "messages": tuple(messages),
    }


def run_qft_demo(
    *,
    seed=QFT_SEED + 1,
    qubits=QFT_DEMO_BLOCK_QUBITS,
    q_z=QFT_DEMO_Q_Z,
    q_x=QFT_DEMO_Q_X,
    budgets=QFT_DEMO_CHECK_BUDGETS,
    eve_configs=QFT_DEMO_EVES,
    calibration_runs=QFT_ALARM_CALIBRATION_RUNS,
    evaluation_runs=QFT_EVALUATION_RUNS,
    phase_samples=QFT_PHASE_SAMPLES,
    batch_size=QFT_TRIAL_BATCH_SIZE,
    backend=None,
    return_diagnostics=False,
):
    """Run the canonical small repeated-secret demonstration on demand.

    By default only the per-trial DataFrame is returned.  With
    ``return_diagnostics=True`` the return value is
    ``(trials, diagnostics, alarm_rule)``.
    """
    calibration_seed, trial_root_seed = np.random.SeedSequence(seed).spawn(2)
    alarm_rule = calibrate_qft_alarm(
        np.random.default_rng(calibration_seed),
        qubits, q_z, q_x, budgets,
        runs=calibration_runs,
    )
    trials = run_qft_repeated_trials_batched(
        trial_root_seed.spawn(evaluation_runs),
        qubits, q_z, q_x, eve_configs, alarm_rule,
        phase_samples=phase_samples,
        batch_size=batch_size,
        backend=backend,
    )
    diagnostics = pd.Series({
        "runs": len(trials),
        "alarm_rate": trials["alarm"].mean(),
        "correct_recovery_rate": trials["correct_recovery"].mean(),
        "mean_transmitted_qubits": trials["transmitted_qubits"].mean(),
        "mean_intercepts": trials["intercepts"].mean(),
    })
    if return_diagnostics:
        return trials, diagnostics, alarm_rule
    return trials


def qft_json_ready(value):
    if isinstance(value, dict):
        return {str(key): qft_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [qft_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return qft_json_ready(value.tolist())
    if isinstance(value, np.generic):
        return qft_json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def qft_canonical_hash(value):
    encoded = json.dumps(
        qft_json_ready(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def qft_seed_sequence(domain, setting, trial_index=None):
    payload = {
        "domain": domain,
        "setting": qft_json_ready(setting),
        "trial_index": trial_index,
        "seed_version": QFT_SEED_VERSION,
    }
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).digest()
    words = np.frombuffer(digest[:16], dtype="<u4").astype(int).tolist()
    return np.random.SeedSequence([QFT_SEED, *words])


def qft_trial_seed_sequences(setting, runs):
    return [
        qft_seed_sequence("qft-evaluation", setting, trial_index)
        for trial_index in range(runs)
    ]


def qft_factorial_config(
    runs_per_setting,
    *,
    backend=None,
    phase_samples=QFT_PHASE_SAMPLES,
    calibration_runs=QFT_ALARM_CALIBRATION_RUNS,
):
    _, backend_name = qft_get_backend(backend)
    return {
        "schema": QFT_CHECKPOINT_SCHEMA,
        "seed": QFT_SEED,
        "seed_version": QFT_SEED_VERSION,
        "backend": backend_name,
        # This is intentionally the configured GPU precision even when NumPy
        # is selected (NumPy itself remains complex128).  The canonical
        # notebook used this value in its checkpoint hash.
        "precision": QFT_GPU_PRECISION,
        "phase_samples": phase_samples,
        "posterior_version": QFT_POSTERIOR_VERSION,
        "calibration_runs": calibration_runs,
        "runs_per_setting": runs_per_setting,
        "secret_bits": SECRET_BITS,
        "recovery_confidence": RECOVERY_CONFIDENCE,
        "maximum_attempts": MAX_ATTEMPTS,
        "target_false_alarm": TARGET_FALSE_ALARM,
        "budgets": QFT_DEMO_CHECK_BUDGETS,
        "block_qubits": BLOCK_QUBITS,
        "noise_profiles": SWEEP_NOISE_PROFILES,
        "eve_counts": EVE_COUNTS,
        "intercept_rates": EVE_INTERCEPT_RATES,
        "eve_z_probabilities": EVE_Z_PROBABILITIES,
        "basis_redraw_rates": EVE_BASIS_REDRAW_RATES,
    }


def write_qft_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(qft_json_ready(payload), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_qft_checkpoint(path, config, resume):
    path = Path(path)
    config_hash = qft_canonical_hash(config)
    if not resume or not path.exists():
        return {
            "schema_version": QFT_CHECKPOINT_SCHEMA,
            "config_hash": config_hash,
            "config": qft_json_ready(config),
            "alarm_rules": {},
            "rows": [],
        }
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("schema_version") != QFT_CHECKPOINT_SCHEMA:
        raise ValueError("QFT checkpoint schema does not match this notebook")
    if checkpoint.get("config_hash") != config_hash:
        raise ValueError(
            "QFT checkpoint configuration differs; use a new path or resume=False"
        )
    keys = [row["setting_key"] for row in checkpoint.get("rows", [])]
    if len(keys) != len(set(keys)):
        raise ValueError("QFT checkpoint contains duplicate setting keys")
    return checkpoint


def restore_qft_alarm_rule(serialized):
    restored = dict(serialized)
    restored["budgets"] = tuple(restored["budgets"])
    restored["centers"] = np.asarray(restored["centers"], dtype=float)
    restored["scales"] = np.asarray(restored["scales"], dtype=float)
    return restored


def summarize_qft_trials(trials, setting):
    return {
        **setting,
        "runs": len(trials),
        "alarm_rate": trials["alarm"].mean(),
        "correct_recovery_rate": trials["correct_recovery"].mean(),
        "incorrect_recovery_rate": (
            trials["recovered"] & ~trials["correct_recovery"]
        ).mean(),
        "mean_posterior_confidence": trials[
            "posterior_confidence"
        ].mean(),
        "mean_retained_checks": trials["retained_checks"].mean(),
        "mean_transmitted_qubits": trials[
            "transmitted_qubits"
        ].mean(),
        "mean_intercepts": trials["intercepts"].mean(),
    }


def run_qft_trials_for_setting(
    setting, runs, qubits, q_z, q_x, eve_configs, alarm_rule,
    *,
    backend=None,
    phase_samples=QFT_PHASE_SAMPLES,
    batch_size=QFT_TRIAL_BATCH_SIZE,
):
    trial_seeds = qft_trial_seed_sequences(setting, runs)
    trials = run_qft_repeated_trials_batched(
        trial_seeds, qubits, q_z, q_x, eve_configs, alarm_rule,
        phase_samples=phase_samples,
        batch_size=batch_size,
        backend=backend,
    )
    return trials if isinstance(trials, pd.DataFrame) else pd.DataFrame(trials)


def run_qft_factorial_sweep(
    runs_per_setting=QFT_FULL_RUNS_PER_SETTING,
    checkpoint_path=QFT_CHECKPOINT_PATH,
    results_path=QFT_RESULTS_PATH,
    resume=QFT_RESUME_SWEEP,
    *,
    backend=None,
    phase_samples=QFT_PHASE_SAMPLES,
    calibration_runs=QFT_ALARM_CALIBRATION_RUNS,
    batch_size=QFT_TRIAL_BATCH_SIZE,
    progress_every=QFT_PROGRESS_EVERY,
):
    config = qft_factorial_config(
        runs_per_setting,
        backend=backend,
        phase_samples=phase_samples,
        calibration_runs=calibration_runs,
    )
    checkpoint = load_qft_checkpoint(checkpoint_path, config, resume)
    write_qft_checkpoint(checkpoint_path, checkpoint)
    completed = {row["setting_key"] for row in checkpoint["rows"]}
    attack_settings_per_group = (
        len(EVE_COUNTS) * len(EVE_INTERCEPT_RATES)
        * len(EVE_Z_PROBABILITIES) * len(EVE_BASIS_REDRAW_RATES)
    )
    total_settings = (
        len(BLOCK_QUBITS) * len(SWEEP_NOISE_PROFILES)
        * (1 + attack_settings_per_group)
    )
    print(
        f"QFT sweep: {len(completed)}/{total_settings} settings "
        f"checkpointed; backend={config['backend']}"
    )

    for qubits in BLOCK_QUBITS:
        for noise_profile, q_z, q_x in SWEEP_NOISE_PROFILES:
            group = {
                "block_qubits": qubits,
                "noise_profile": noise_profile,
                "q_z": q_z,
                "q_x": q_x,
                "budgets": QFT_DEMO_CHECK_BUDGETS,
                "calibration_runs": calibration_runs,
            }
            group_key = qft_canonical_hash(group)
            if group_key in checkpoint["alarm_rules"]:
                alarm_rule = restore_qft_alarm_rule(
                    checkpoint["alarm_rules"][group_key]
                )
            else:
                calibration_rng = np.random.default_rng(
                    qft_seed_sequence("qft-calibration", group)
                )
                alarm_rule = calibrate_qft_alarm(
                    calibration_rng, qubits, q_z, q_x,
                    QFT_DEMO_CHECK_BUDGETS,
                    runs=calibration_runs,
                )
                checkpoint["alarm_rules"][group_key] = qft_json_ready(alarm_rule)
                write_qft_checkpoint(checkpoint_path, checkpoint)

            settings = [({
                "block_qubits": qubits,
                "noise_profile": noise_profile,
                "q_z": q_z,
                "q_x": q_x,
                "eve_count": 0,
                "intercept_probability": 0.0,
                "eve_p_z": 0.5,
                "redraw_probability": 1.0,
                "is_attack": False,
            }, ())]
            for eve_count, intercept, eve_p_z, redraw in product(
                EVE_COUNTS, EVE_INTERCEPT_RATES,
                EVE_Z_PROBABILITIES, EVE_BASIS_REDRAW_RATES,
            ):
                eve_config = {
                    "intercept_probability": intercept,
                    "p_z": eve_p_z,
                    "redraw_probability": redraw,
                }
                setting = {
                    "block_qubits": qubits,
                    "noise_profile": noise_profile,
                    "q_z": q_z,
                    "q_x": q_x,
                    "eve_count": eve_count,
                    "intercept_probability": intercept,
                    "eve_p_z": eve_p_z,
                    "redraw_probability": redraw,
                    "is_attack": True,
                }
                settings.append((
                    setting, tuple(eve_config.copy() for _ in range(eve_count))
                ))

            for setting, eve_configs in settings:
                setting_key = qft_canonical_hash(setting)
                if setting_key in completed:
                    continue
                started = time.perf_counter()
                trials = run_qft_trials_for_setting(
                    setting, runs_per_setting, qubits, q_z, q_x,
                    eve_configs, alarm_rule,
                    backend=backend,
                    phase_samples=phase_samples,
                    batch_size=batch_size,
                )
                row = summarize_qft_trials(trials, setting)
                row.update({
                    "setting_key": setting_key,
                    "compute_backend": config["backend"],
                    "phase_samples": phase_samples,
                    "calibration_runs": calibration_runs,
                    "seed_version": QFT_SEED_VERSION,
                    "posterior_version": QFT_POSTERIOR_VERSION,
                    "elapsed_seconds": time.perf_counter() - started,
                })
                checkpoint["rows"].append(qft_json_ready(row))
                completed.add(setting_key)
                write_qft_checkpoint(checkpoint_path, checkpoint)
                if (
                    len(completed) == total_settings
                    or len(completed) % max(progress_every, 1) == 0
                ):
                    print(
                        f"QFT sweep: {len(completed)}/{total_settings} "
                        f"settings checkpointed"
                    )

    results = pd.DataFrame(checkpoint["rows"])
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_results = results_path.with_suffix(
        results_path.suffix + ".tmp"
    )
    try:
        results.to_csv(temporary_results, index=False)
        os.replace(temporary_results, results_path)
    finally:
        if temporary_results.exists():
            temporary_results.unlink()
    return results
