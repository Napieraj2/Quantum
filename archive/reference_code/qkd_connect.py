import json
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService

credentials_path = Path(__file__).with_name("apikey.json")
with credentials_path.open(encoding="utf-8") as credentials_file:
    credentials = json.load(credentials_file)

QiskitRuntimeService.save_account(
    token=credentials["apikey"],
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/18607e1a0e9b4964b90790aed2b97495:d01630cb-6826-4404-ba0c-b2206bb552cd::",
    overwrite=True,
)

print("IBM Quantum account saved.")