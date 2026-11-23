# Archive

Historical files from earlier stages of the project.

- [QFT factorial pilot](qft_pilot/README.md) preserves the original GPU sweep, checkpoint workflow, diagnostic tests, and the separate 500-attempt Bob-only drift demo.
- [Earlier BB84 project plan](BB84_PROJECT_PLAN.md) preserves the initial scope, literature notes, and proposed responsibilities.

The links document the APIs and example workflows. The archived reference files
do not record their original sources.

| File | Project role | Source and proposal relation |
| --- | --- | --- |
| [BB84_Quick_Control.ipynb](BB84_Quick_Control.ipynb) | **Superseded study code.** One terminal detector check at 50 retained checks. | Earlier implementation of the proposal's BB84 comparison using the Erven profile and 5% target. [Erven et al., §4](https://arxiv.org/abs/0901.0960v2) · [Active terminal sweep](../BB84.ipynb#terminal) |
| [100_Qubit_Test.ipynb](reference_code/100_Qubit_Test.ipynb) | **Out-of-scope hardware prototype.** Builds and submits a 100-qubit GHZ correlation experiment. The saved plot measures separation along the logical chain, from qubit 99 toward qubit 0. | Uses a hardware workflow for a separate experiment. [Qiskit transpilation API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.generate_preset_pass_manager) · [IBM error-suppression guide](https://quantum.cloud.ibm.com/docs/en/guides/configure-error-suppression) |
| [First_Job.ipynb](reference_code/First_Job.ipynb) | **Setup-only prototype.** Estimates six Bell observables with IBM Runtime. Set `USE_FAKE_BACKEND = True` for a local Aer run using a device noise model. | The same workflow appears in the [IBM Hello world guide](https://quantum.cloud.ibm.com/docs/en/guides/hello-world). |
| [bell_runtime_smoke.py](reference_code/bell_runtime_smoke.py) | **Setup-only local smoke test.** Samples a three-qubit entangled circuit and plots counts. | Uses the Qiskit [`StatevectorSampler`](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.StatevectorSampler) and [`plot_histogram`](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.visualization.plot_histogram) APIs. |
| [bell_state_remote.py](reference_code/bell_state_remote.py) | **Out-of-scope hardware prototype.** Transpiles a Bell circuit for a Runtime backend and prints the result. | Uses the preparation and transpilation steps in the [IBM Hello world guide](https://quantum.cloud.ibm.com/docs/en/guides/hello-world). |
| [qkd_connect.py](reference_code/qkd_connect.py) | **Setup-only credential helper.** Saves an IBM Quantum account from the adjacent `apikey.json` file with an `apikey` field and an optional `crn`. | Uses the documented [QiskitRuntimeService.save_account](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/qiskit_ibm_runtime.QiskitRuntimeService#save_account) API. |
| [runtime_service_smoke.py](reference_code/runtime_service_smoke.py) | **Setup-only connection check.** Lists backends visible to an authenticated account. | Uses the documented [QiskitRuntimeService API](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/qiskit_ibm_runtime.QiskitRuntimeService). |

The two smoke scripts were renamed to keep them out of automatic test discovery.
[Gate Baseline](../Gate_Baseline.ipynb) is the current gate diagnostic.

- [Duplicate QFT export](results/qft_factorial_results_local.csv) preserves a conflicted merge of two copies of the same sweep. The archived pilot reads the [validated QFT CSV](results/qft_factorial_results.csv).

[Return to the current project guide](../README.md)
