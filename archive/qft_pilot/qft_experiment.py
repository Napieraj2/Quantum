"""Compatibility imports for the QFT study. New code can import the owning modules."""

from . import qft_inference as _inference
from .qft_protocol import (
    BLOCK_QUBITS,
    IDENTITY_2,
    PAULI_X,
    PAULI_Z,
    QFT_DEMO_Q_X,
    QFT_DEMO_Q_Z,
    SECRET_BITS,
    apply_pauli_channel,
    local_operator,
    measurement_projector,
    qft_bits,
    qft_check_distribution,
    qft_check_surprisal,
    qft_encoding_matrix,
    qft_integer,
    qft_matrix,
    qft_message,
    qft_null_outcome_distribution,
    qft_scrambler,
    sample_eve_chain,
    sample_pauli_trajectory,
    simulate_qft_attempt,
    simulate_qft_block,
)
from .qft_inference import (
    BackendStatus,
    QFT_BATCH_CACHE,
    QFT_COMPUTE_BACKEND,
    QFT_PHASE_SAMPLES,
    apply_fixed_eve_records,
    configure_qft_backend,
    qft_apply_heterogeneous_records_batch,
    qft_apply_pauli_channel_batch,
    qft_backend_cache,
    qft_backend_dtype,
    qft_get_backend,
    qft_record_likelihood_scalar,
    qft_record_likelihoods_batched,
    qft_record_likelihoods_wavefront,
    update_secret_posteriors_batched,
)
from .qft_trials import (
    MAX_ATTEMPTS,
    QFT_ALARM_CALIBRATION_RUNS,
    QFT_DEMO_BLOCK_QUBITS,
    QFT_DEMO_CHECK_BUDGETS,
    QFT_DEMO_EVES,
    QFT_EVALUATION_RUNS,
    QFT_SEED,
    QFT_TRIAL_BATCH_SIZE,
    RECOVERY_CONFIDENCE,
    TARGET_FALSE_ALARM,
    calibrate_bb84_cutoffs,
    calibrate_qft_alarm,
    qft_score_trajectory,
    run_qft_demo,
    run_qft_repeated_trials_batched,
    standardize_look_scores,
)
from .qft_validation import (
    QFT_GPU_VALIDATION_RTOL,
    validate_qft_model,
)
from .qft_sweep import (
    EVE_BASIS_REDRAW_RATES,
    EVE_COUNTS,
    EVE_INTERCEPT_RATES,
    EVE_Z_PROBABILITIES,
    QFT_CHECKPOINT_PATH,
    QFT_CHECKPOINT_SCHEMA,
    QFT_FULL_RUNS_PER_SETTING,
    QFT_POSTERIOR_VERSION,
    QFT_PROGRESS_EVERY,
    QFT_RESULTS_PATH,
    QFT_RESUME_SWEEP,
    QFT_SEED_VERSION,
    SWEEP_NOISE_PROFILES,
    load_qft_checkpoint,
    qft_canonical_hash,
    qft_factorial_config,
    qft_json_ready,
    qft_seed_sequence,
    qft_trial_seed_sequences,
    restore_qft_alarm_rule,
    run_qft_factorial_sweep,
    summarize_qft_trials,
    write_qft_checkpoint,
)


_BACKEND_FIELDS = {
    "QFT_ARRAY_MODULE",
    "QFT_CUPY_IMPORT_ERROR",
    "QFT_GPU_PRECISION",
    "QFT_SELECTED_BACKEND",
    "cp",
}


def __getattr__(name):
    if name in _BACKEND_FIELDS:
        return getattr(_inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
