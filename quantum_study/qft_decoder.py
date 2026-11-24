"""Small Qiskit benchmark for the known-word inverse-QFT decoder."""

from math import tau


_CASE_NAMES = ("Control", "Inverse QFT")


def _validate_width(width):
    if type(width) is not int or width not in (2, 4):
        raise ValueError("width must be 2 or 4")


def decoder_circuits(width):
    """Return matched controls and inverse-QFT circuits for every input word."""
    _validate_width(width)
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QFTGate

    circuits, cases = [], []
    for case in _CASE_NAMES:
        for word in range(2**width):
            name = f"qft_decoder_{case.lower().replace(' ', '_')}_{word:0{width}b}"
            circuit = QuantumCircuit(width, width, name=name)
            if case == "Control":
                for qubit in range(width):
                    if (word >> qubit) & 1:
                        circuit.x(qubit)
                circuit.barrier()
            else:
                # Qiskit's full QFT gate includes the output swaps. In its
                # little-endian qubit order, qubit q has this phase period.
                for qubit in range(width):
                    period = 2 ** (width - qubit)
                    circuit.h(qubit)
                    circuit.p(tau * (word % period) / period, qubit)
                # Keep transpilation from absorbing known-word preparation
                # into the decoder being benchmarked.
                circuit.barrier()
                circuit.append(QFTGate(width).inverse(), range(width))
            circuit.measure(range(width), range(width))
            circuits.append(circuit)
            cases.append((case, word))
    return circuits, cases


def decoder_summary(width, cases, counts_by_circuit):
    """Aggregate word and bit errors from Qiskit single-register bitstrings."""
    _validate_width(width)
    totals = {
        case: {"shots": 0, "word_errors": 0, "bit_errors": [0] * width}
        for case in _CASE_NAMES
    }
    for (case, expected), counts in zip(cases, counts_by_circuit, strict=True):
        total = totals[case]
        shots = sum(counts.values())
        if shots < 1:
            raise ValueError("each circuit result must contain at least one shot")
        total["shots"] += int(shots)
        for outcome, count in counts.items():
            count = int(count)
            wrong_bits = int(outcome.replace(" ", ""), 2) ^ expected
            total["word_errors"] += count * (wrong_bits != 0)
            for bit in range(width):
                total["bit_errors"][bit] += count * ((wrong_bits >> bit) & 1)

    rows = []
    for case, total in totals.items():
        shots = total["shots"]
        if shots == 0:
            raise ValueError(f"no results for {case}")
        rows.append({
            "case": case,
            "shots": shots,
            "word_errors": total["word_errors"],
            "word_error_rate": total["word_errors"] / shots,
            **{
                f"bit_{bit}_error_rate": total["bit_errors"][bit] / shots
                for bit in range(width)
            },
        })
    return rows
