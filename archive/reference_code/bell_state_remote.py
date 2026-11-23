"""Transpile a Bell circuit for an IBM backend and print the circuit."""

from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService


def main():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    backend = QiskitRuntimeService().least_busy(simulator=False, operational=True)
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pass_manager.run(circuit)
    print(isa_circuit.draw(output="text", idle_wires=False))


if __name__ == "__main__":
    main()
