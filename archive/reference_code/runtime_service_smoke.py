"""List the IBM Quantum backends available to the saved account."""

from qiskit_ibm_runtime import QiskitRuntimeService


def main():
    backends = QiskitRuntimeService().backends()
    print("Connected to IBM Quantum. Available backends:")
    for backend in backends:
        print(f"- {backend.name}")


if __name__ == "__main__":
    main()
