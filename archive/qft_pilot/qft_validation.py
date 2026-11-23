"""Noiseless, inference-parity, and recovery checks for the archived pilot.

The noiseless check exercises the two-pass round trip in Papadopoulos and
Linvill, §4.1, Eqs. (1)–(3). The parity checks establish agreement between
the scalar and batched implementations of the historical sampled-phase
likelihood; they do not establish exact phase marginalization.
https://arxiv.org/abs/2404.12507v2
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .qft_inference import (
    qft_get_backend,
    qft_record_likelihood_scalar,
    qft_record_likelihoods_batched,
    update_secret_posteriors_batched,
)
from .qft_protocol import (
    BLOCK_QUBITS,
    SECRET_BITS,
    qft_bits,
    qft_message,
    qft_null_outcome_distribution,
    simulate_qft_block,
)
from .qft_trials import QFT_DEMO_CHECK_BUDGETS, QFT_SEED, run_qft_repeated_trials_batched


QFT_GPU_VALIDATION_RTOL = 2e-5


def validate_qft_model(backend=None, *, verbose=False):
    """Run the canonical core, trajectory, batch, and positive-control checks.

    Validation is explicit rather than an import side effect.  The returned
    tables retain the numerical evidence behind each assertion.
    """
    array_module, backend_name = qft_get_backend(backend)
    q_z, q_x = 0.002, 0.038
    messages = []

    def record(leg, signal, basis, outcome):
        return dict(leg=leg, signal=signal, basis=basis, outcome=outcome)

    def validation_block(records, secret_indices, checks):
        positions, values = zip(*checks)
        return dict(
            records=records, secret_indices=secret_indices,
            check_positions=positions, check_values=values,
        )

    def scalar_likelihoods(qubits, candidate_messages, records, phases):
        """Independent, deliberately unbatched oracle for validation."""
        return np.asarray([
            qft_record_likelihood_scalar(
                qubits, message, records, q_z, q_x, phases
            )
            for message in candidate_messages
        ])

    def require_close(reference, candidate, failure, *, gpu=False):
        tolerance = QFT_GPU_VALIDATION_RTOL if gpu else 1e-12
        if not np.allclose(reference, candidate, rtol=tolerance, atol=tolerance):
            raise AssertionError(failure)
        return float(np.max(np.abs(reference - candidate)))

    def gpu_error(reference, calculation, failure):
        return (
            require_close(reference, calculation(), failure, gpu=True)
            if backend_name == "cupy" else np.nan
        )

    core_rows = []
    for qubits in BLOCK_QUBITS:
        phases = np.linspace(0.17, 2.41, qubits)
        for message in range(2**qubits):
            probabilities = qft_null_outcome_distribution(
                qubits, message, phases, 0.0, 0.0
            )
            core_rows.append((
                qubits, message, int(np.argmax(probabilities)),
                probabilities[message],
            ))
    core = pd.DataFrame(
        core_rows,
        columns=("qubits", "message", "decoded", "correct_probability"),
    )
    if not (
        core["message"].eq(core["decoded"]).all()
        and np.allclose(core["correct_probability"], 1.0)
    ):
        raise AssertionError("Noiseless QFT round trip did not decode every message")

    trajectory_rng = np.random.default_rng(QFT_SEED)
    for qubits in BLOCK_QUBITS:
        block = simulate_qft_block(
            trajectory_rng, qubits, 2**qubits - 1, 0.0, 0.0,
            (), [], attempt=1, block=1,
        )
        if block["outcome"] != 2**qubits - 1 or block["records"]:
            raise AssertionError("Noiseless trajectory did not preserve the QFT message")
    messages.append(
        "PASS: noiseless 2- and 4-qubit QFT trajectories decode exactly"
    )

    validation_phase_rng = np.random.default_rng(QFT_SEED + 17)
    likelihood_rows = []
    for qubits in BLOCK_QUBITS:
        dimension = 2**qubits
        records = [
            record("forward", 0, "Z", 0),
            record("return", qubits - 1, "X", 1),
        ]
        phase_samples = validation_phase_rng.uniform(0.0, 2 * np.pi, (8, qubits))
        candidate_messages = np.arange(dimension)
        scalar_values = scalar_likelihoods(qubits, candidate_messages, records, phase_samples)
        batched_cpu = qft_record_likelihoods_batched(
            qubits, candidate_messages, records, q_z, q_x,
            phase_samples, array_module=np,
        )
        scalar_error = require_close(
            scalar_values, batched_cpu,
            "Batched NumPy QFT likelihoods differ from the scalar reference",
        )

        duplicate_likelihoods = qft_record_likelihoods_batched(
            qubits, [0, 0, dimension - 1], records, q_z, q_x,
            phase_samples, array_module=np,
        )
        if duplicate_likelihoods[0] != duplicate_likelihoods[1]:
            raise AssertionError("Duplicate QFT messages must have identical likelihoods")

        gpu_max_abs_error = gpu_error(
            batched_cpu,
            lambda: qft_record_likelihoods_batched(
                qubits, candidate_messages, records, q_z, q_x,
                phase_samples, array_module=array_module,
            ),
            "CuPy QFT likelihoods differ from the NumPy batch",
        )
        likelihood_rows.append((qubits, scalar_error, gpu_max_abs_error))
    messages.append(
        "PASS: batched posterior likelihoods match the scalar NumPy reference"
    )

    wavefront_blocks = [
        validation_block(
            [record("forward", 0, "Z", 0), record("return", 3, "X", 1)],
            (0, 1), ((1, 0), (3, 1)),
        ),
        validation_block(
            [record("return", 1, "Z", 1)],
            (2, 3), ((0, 1), (2, 0)),
        ),
        validation_block([], (0, 1), ((2, 1), (3, 1))),
    ]
    wavefront_phases = validation_phase_rng.uniform(
        0.0, 2 * np.pi, (len(wavefront_blocks), 8, 4)
    )
    wavefront_prior = np.full(
        (len(wavefront_blocks), 2**SECRET_BITS),
        -np.log(2**SECRET_BITS),
    )
    wavefront_cpu = update_secret_posteriors_batched(
        wavefront_prior, wavefront_blocks, 4, q_z, q_x,
        wavefront_phases, backend="numpy",
    )
    wavefront_scalar = wavefront_prior.copy()
    for block_index, (block, phases) in enumerate(
        zip(wavefront_blocks, wavefront_phases)
    ):
        if not block["records"]:
            continue
        candidate_messages = []
        for candidate in range(2**SECRET_BITS):
            secret = qft_bits(candidate, SECRET_BITS)
            payload = [secret[index] for index in block["secret_indices"]]
            candidate_messages.append(qft_message(
                block["check_positions"], block["check_values"],
                payload, 4,
            ))
        likelihoods = scalar_likelihoods(4, candidate_messages, block["records"], phases)
        wavefront_scalar[block_index] += np.log(np.maximum(
            likelihoods, np.finfo(float).tiny
        ))
    wavefront_scalar -= logsumexp(wavefront_scalar, axis=1, keepdims=True)
    wavefront_scalar_error = require_close(
        wavefront_cpu, wavefront_scalar,
        "Cross-trial NumPy wavefront differs from scalar updates",
    )

    wavefront_gpu_max_abs_error = gpu_error(
        wavefront_cpu,
        lambda: update_secret_posteriors_batched(
            wavefront_prior, wavefront_blocks, 4, q_z, q_x,
            wavefront_phases, backend=backend,
        ),
        "CuPy cross-trial wavefront differs from NumPy",
    )
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
    positive_control_eves = (dict(
        intercept_probability=1.0, p_z=0.0, redraw_probability=1.0,
    ),)
    positive_control = run_qft_repeated_trials_batched(
        np.random.SeedSequence(QFT_SEED + 101).spawn(4),
        2, 0.0, 0.0,
        positive_control_eves,
        positive_control_alarm_rule,
        backend=backend,
    )
    recovered = positive_control[["recovered", "correct_recovery"]].all()
    if not recovered.all():
        raise AssertionError("Extreme X-basis interception should recover the repeated secret")
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
        "likelihood_parity": pd.DataFrame(likelihood_rows, columns=(
            "qubits", "scalar_batch_max_abs_error",
            "gpu_batch_max_abs_error",
        )),
        "wavefront_scalar_max_abs_error": wavefront_scalar_error,
        "wavefront_gpu_max_abs_error": wavefront_gpu_max_abs_error,
        "positive_control": positive_control,
        "messages": tuple(messages),
    }
