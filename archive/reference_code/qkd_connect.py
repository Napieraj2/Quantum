"""Save IBM Quantum account settings from the adjacent apikey.json file."""

import json
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService


def main():
    credentials_path = Path(__file__).with_name("apikey.json")
    with credentials_path.open(encoding="utf-8") as credentials_file:
        credentials = json.load(credentials_file)

    token = credentials.get("apikey")
    if not isinstance(token, str) or not token.strip():
        raise ValueError(f"{credentials_path.name} must contain a non-empty 'apikey' field")

    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=token.strip(),
        instance=credentials.get("crn") or None,
        overwrite=True,
    )
    print("IBM Quantum account saved.")


if __name__ == "__main__":
    main()
