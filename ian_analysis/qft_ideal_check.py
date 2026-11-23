"""Ideal phase encoding and inverse-QFT decoding for small registers."""

from __future__ import annotations

import numpy as np


def qft_matrix(qubits: int) -> np.ndarray:
    if qubits not in (2, 4):
        raise ValueError("This project control uses 2 or 4 qubits")
    size = 1 << qubits
    x = np.arange(size)
    return np.exp(2j*np.pi*np.outer(x, x)/size) / np.sqrt(size)


def decode_no_attack(secret_block: int, theta: np.ndarray) -> np.ndarray:
    """Encode a block, remove the private phase, and apply the inverse QFT."""
    theta = np.asarray(theta, dtype=float)
    qubits = len(theta)
    transform = qft_matrix(qubits)
    size = 1 << qubits
    if not 0 <= secret_block < size:
        raise ValueError("Block value is outside register width")
    indices = np.arange(size)
    bit_matrix = ((indices[:, None] >> np.arange(qubits)) & 1)
    private_phase = bit_matrix @ theta
    start = np.exp(1j*private_phase) / np.sqrt(size)
    encoded = start * np.exp(2j*np.pi*secret_block*indices/size)
    returned = encoded * np.exp(-1j*private_phase)
    return np.conjugate(transform).T @ returned


def example_check(seed: int = 20261114) -> float:
    rng = np.random.default_rng(seed)
    worst_error = 0.0
    for qubits in (2, 4):
        for block in range(1 << qubits):
            amplitudes = decode_no_attack(block, rng.uniform(0, 2*np.pi, size=qubits))
            expected = np.zeros(1 << qubits, dtype=complex)
            expected[block] = 1
            worst_error = max(worst_error, float(np.max(np.abs(amplitudes-expected))))
    return worst_error


if __name__ == "__main__":
    worst = example_check()
    print(f"Maximum ideal no-attack decoding amplitude error: {worst:.3e}")
    if worst >= 1e-11:
        raise SystemExit("Ideal QFT check did not match the expected block")
