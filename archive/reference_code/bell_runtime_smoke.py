"""Sample and plot a three-qubit entangled circuit with Qiskit's local sampler."""

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram


def main():
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.x(0)
    circuit.measure_all()

    result = StatevectorSampler().run([circuit], shots=1024).result()
    counts = result[0].data.meas.get_counts()
    print(counts)
    plot_histogram(counts)
    plt.show()


if __name__ == "__main__":
    main()
