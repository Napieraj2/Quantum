"""Historical Eve likelihoods for the archived factorial pilot.

The likelihood uses the two-pass encoding in Papadopoulos and Linvill,
§4.1, Eqs. (1)–(2). The pilot marginalizes Alice's private phases with
16 Monte Carlo samples per observed block. That sampled approximation is
preserved for artifact reproducibility; it is not an exact phase marginal and
can make return-leg-only records artificially message-informative.
CPU/GPU batching changes execution only, not the sampled likelihood model.
https://arxiv.org/abs/2404.12507v2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .qft_protocol import (
    PAULI_X,
    PAULI_Z,
    SECRET_BITS,
    _prepare_block,
    apply_pauli_channel,
    local_operator,
    measurement_projector,
    qft_bits,
    qft_encoding_matrix,
    qft_message,
)


QFT_PHASE_SAMPLES = 16
QFT_COMPUTE_BACKEND = "auto"
QFT_GPU_PRECISION = "complex64"


@dataclass(frozen=True)
class BackendStatus:
    """Selected numerical backend and its visible device metadata."""

    name: str
    array_module: Any
    precision: str
    device_name: str | None
    device_count: int


cp = None
QFT_CUPY_IMPORT_ERROR = None
QFT_ARRAY_MODULE = np
QFT_SELECTED_BACKEND = "numpy"


def configure_qft_backend(
    requested=QFT_COMPUTE_BACKEND, gpu_precision="complex64"
):
    """Select NumPy or probe and select CuPy without import-time side effects."""
    global cp
    global QFT_ARRAY_MODULE, QFT_SELECTED_BACKEND, QFT_GPU_PRECISION
    global QFT_CUPY_IMPORT_ERROR

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
        QFT_CUPY_IMPORT_ERROR = error
        if requested == "cupy":
            raise RuntimeError(
                "CuPy with a visible CUDA device is required for the requested backend"
            ) from error
        QFT_ARRAY_MODULE = np
        QFT_SELECTED_BACKEND = "numpy"
        return BackendStatus("numpy", np, "complex128", None, 0)

    cp = cupy_module
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
    if requested not in ("auto", "cupy"):
        raise ValueError(
            "QFT backend must be 'auto', 'numpy', 'cupy', or BackendStatus"
        )
    if cp is not None and QFT_SELECTED_BACKEND == "cupy":
        return cp, "cupy"
    if requested == "auto" and QFT_CUPY_IMPORT_ERROR is not None:
        return np, "numpy"
    status = configure_qft_backend(requested, QFT_GPU_PRECISION)
    return status.array_module, status.name


def qft_backend_dtype(array_module):
    if array_module is np:
        return np.complex128
    if QFT_GPU_PRECISION not in ("complex64", "complex128"):
        raise ValueError("QFT_GPU_PRECISION must be 'complex64' or 'complex128'")
    return getattr(array_module, QFT_GPU_PRECISION)


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
    """Apply scalar rates or rates broadcastable across the batch dimensions."""
    if np.isscalar(q_z) and np.isscalar(q_x):
        valid_rates = q_z >= 0 and q_x >= 0 and q_z + q_x <= 1
    else:
        q_z = array_module.asarray(q_z, dtype=cache["real_dtype"])
        q_x = array_module.asarray(q_x, dtype=cache["real_dtype"])
        valid_rates = array_module.all((q_z >= 0) & (q_x >= 0) & (q_z + q_x <= 1))
        q_z = q_z[..., None, None]
        q_x = q_x[..., None, None]
    if not valid_rates:
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


def qft_record_likelihood_scalar(
    qubits, message, records, q_z, q_x, phase_samples
):
    encoding = qft_encoding_matrix(qubits, message)
    likelihood = 0.0
    for phases in phase_samples:
        density, _ = _prepare_block(qubits, phases)
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
    phase_samples = array_module.asarray(phase_samples)
    return qft_record_likelihoods_wavefront(
        qubits,
        messages[None, :],
        [records],
        q_z,
        q_x,
        phase_samples[None, :, :],
        array_module=array_module,
    )[0]


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
    if candidate_messages.ndim != 2:
        raise ValueError(
            "candidate_messages must have shape (blocks, candidates)"
        )
    cache = qft_backend_cache(qubits, array_module)
    phases = array_module.asarray(
        phase_samples_by_block, dtype=cache["real_dtype"]
    )
    if phases.ndim != 3:
        raise ValueError(
            "phase_samples_by_block must have shape "
            "(blocks, phase_samples, qubits)"
        )
    batch_size = len(records_by_block)
    if (
        candidate_messages.shape[0] != batch_size
        or phases.shape[0] != batch_size
        or phases.shape[2] != qubits
    ):
        raise ValueError("wavefront batch dimensions do not match")

    dimension = 2**qubits
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

    # Each block exposes one or two payload bits. Evaluate those candidates
    # directly and map their likelihoods back to the sixteen full secrets.
    observed = [blocks[index] for index in active]
    hypotheses = np.arange(2**SECRET_BITS)
    payload_bits = qubits // 2
    messages, payload_indices = [], []
    for block in observed:
        indices = block["secret_indices"]
        if len(indices) != payload_bits:
            raise ValueError("QFT blocks must use half their bits for the payload")
        messages.append([
            qft_message(
                block["check_positions"], block["check_values"],
                qft_bits(payload, payload_bits), qubits,
            )
            for payload in range(2**payload_bits)
        ])
        payload = np.zeros_like(hypotheses)
        for index in indices:
            bit = (hypotheses >> (SECRET_BITS - 1 - index)) & 1
            payload = 2 * payload + bit
        payload_indices.append(payload)

    likelihoods_by_payload = qft_record_likelihoods_wavefront(
        qubits, messages, [block["records"] for block in observed], q_z, q_x,
        phase_samples_by_block[active], array_module=array_module,
    )
    likelihoods = likelihoods_by_payload[
        np.arange(len(active))[:, None], np.stack(payload_indices)
    ]
    updated[active] += np.log(np.maximum(
        likelihoods, np.finfo(float).tiny
    ))
    updated -= logsumexp(updated, axis=1, keepdims=True)
    return updated
