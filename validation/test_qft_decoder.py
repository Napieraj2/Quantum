import importlib.util
from collections import Counter
import unittest

import numpy as np

from quantum_study.qft_decoder import decoder_circuits, decoder_summary


_HAS_QISKIT = importlib.util.find_spec("qiskit") is not None


def _instruction_parts(instruction):
    """Support current and tuple-style Qiskit circuit instructions."""
    if hasattr(instruction, "operation"):
        return instruction.operation, instruction.qubits, instruction.clbits
    return instruction


@unittest.skipUnless(_HAS_QISKIT, "Qiskit is required for decoder circuit tests")
class QFTDecoderCircuitTests(unittest.TestCase):
    def test_every_two_and_four_qubit_word_decodes_to_its_integer(self):
        from qiskit.quantum_info import Statevector

        for width in (2, 4):
            circuits, cases = decoder_circuits(width)
            expected_cases = {
                (case, word)
                for case in ("Control", "Inverse QFT")
                for word in range(2**width)
            }
            self.assertEqual(Counter(cases), Counter(expected_cases))
            self.assertEqual(len(circuits), len(cases))

            for circuit, (case, word) in zip(circuits, cases):
                with self.subTest(width=width, case=case, word=word):
                    measurements = [
                        instruction
                        for instruction in circuit.data
                        if _instruction_parts(instruction)[0].name == "measure"
                    ]
                    self.assertEqual(len(measurements), width)
                    for instruction in measurements:
                        _, qubits, clbits = _instruction_parts(instruction)
                        self.assertEqual(
                            circuit.find_bit(qubits[0]).index,
                            circuit.find_bit(clbits[0]).index,
                        )

                    unmeasured = circuit.remove_final_measurements(inplace=False)
                    probabilities = Statevector.from_instruction(unmeasured).probabilities()
                    expected = np.zeros(2**width)
                    expected[word] = 1.0
                    np.testing.assert_allclose(
                        probabilities, expected, rtol=0, atol=1e-12
                    )

    def test_control_preparation_is_the_requested_computational_word(self):
        for width in (2, 4):
            circuits, cases = decoder_circuits(width)
            for circuit, (case, word) in zip(circuits, cases):
                if case != "Control":
                    continue
                with self.subTest(width=width, word=word):
                    prepared = []
                    for instruction in circuit.data:
                        operation, qubits, _ = _instruction_parts(instruction)
                        if operation.name in {"barrier", "measure"}:
                            continue
                        self.assertEqual(operation.name, "x")
                        self.assertEqual(len(qubits), 1)
                        prepared.append(circuit.find_bit(qubits[0]).index)
                    expected = [
                        qubit
                        for qubit in range(width)
                        if (word >> qubit) & 1
                    ]
                    self.assertEqual(sorted(prepared), expected)

    def test_inverse_qft_uses_analytical_product_preparation(self):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        for width in (2, 4):
            circuits, cases = decoder_circuits(width)
            for circuit, (case, word) in zip(circuits, cases):
                if case != "Inverse QFT":
                    continue
                with self.subTest(width=width, word=word):
                    barriers = [
                        index
                        for index, instruction in enumerate(circuit.data)
                        if _instruction_parts(instruction)[0].name == "barrier"
                    ]
                    self.assertTrue(barriers)
                    boundary = barriers[0]
                    preparation = QuantumCircuit(width)
                    for instruction in circuit.data[:boundary]:
                        operation, qubits, clbits = _instruction_parts(instruction)
                        self.assertNotIn("qft", operation.name.lower())
                        self.assertEqual(len(qubits), 1)
                        self.assertFalse(clbits)
                        indices = [circuit.find_bit(qubit).index for qubit in qubits]
                        preparation.append(operation, indices)

                    dimension = 2**width
                    target = np.exp(
                        2j * np.pi * word * np.arange(dimension) / dimension
                    ) / np.sqrt(dimension)
                    prepared = Statevector.from_instruction(preparation).data
                    fidelity = abs(np.vdot(target, prepared)) ** 2
                    self.assertAlmostEqual(fidelity, 1.0, places=12)

                    decoder = [
                        _instruction_parts(instruction)
                        for instruction in circuit.data[boundary + 1 :]
                        if _instruction_parts(instruction)[0].name != "measure"
                    ]
                    self.assertTrue(any(len(qubits) > 1 for _, qubits, _ in decoder))


class QFTDecoderSummaryTests(unittest.TestCase):
    def test_asymmetric_counts_preserve_little_endian_bits_and_word_errors(self):
        width = 4
        cases = [
            (case, word)
            for case in ("Control", "Inverse QFT")
            for word in range(2**width)
        ]
        counts = [{f"{word:04b}": 100} for _, word in cases]

        control = cases.index(("Control", 0b0110))
        counts[control] = {
            "0110": 50,
            "0111": 20,  # bit 0
            "0100": 10,  # bit 1
            "1110": 5,   # bit 3
            "0011": 15,  # bits 0 and 2, one word error
        }
        inverse = cases.index(("Inverse QFT", 0b1001))
        counts[inverse] = {
            "1001": 120,
            "1101": 30,  # bit 2
            "0011": 10,  # bits 1 and 3, one word error
        }

        rows = {row["case"]: row for row in decoder_summary(width, cases, counts)}
        self.assertEqual(rows["Control"], {
            "case": "Control",
            "shots": 1600,
            "word_errors": 50,
            "word_error_rate": 50 / 1600,
            "bit_0_error_rate": 35 / 1600,
            "bit_1_error_rate": 10 / 1600,
            "bit_2_error_rate": 15 / 1600,
            "bit_3_error_rate": 5 / 1600,
        })
        self.assertEqual(rows["Inverse QFT"], {
            "case": "Inverse QFT",
            "shots": 1660,
            "word_errors": 40,
            "word_error_rate": 40 / 1660,
            "bit_0_error_rate": 0.0,
            "bit_1_error_rate": 10 / 1660,
            "bit_2_error_rate": 30 / 1660,
            "bit_3_error_rate": 10 / 1660,
        })


if __name__ == "__main__":
    unittest.main()
