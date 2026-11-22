from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
backends = service.backends()

print("Connected to IBM Quantum. Available backends:")
for backend in backends:
	print(f"- {backend.name}")