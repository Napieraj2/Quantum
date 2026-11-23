# Script review

## Active study and controls

The proposal specifies the noise model, detector calibration, and recovery criterion. The code borrows the two-pass QFT circuit and hidden-check construction from Papadopoulos and Linvill. The Erven experiment reports the source-basis error rates, and the proposal uses those rates as simulation inputs. The BB84 material uses Bennett and Brassard for preparation and sifting, Lo et al. for basis-separated errors, and Lee et al. for the pooled-score normal approximation. [Proposal and source map](MODEL_NOTES.md#proposal-deliverables)

| Script | Dependencies and callers | Review result |
| --- | --- | --- |
| [quantum_study/__init__.py](../quantum_study/__init__.py) | Package marker for notebook and test imports. | Kept the minimal initializer. Callers load their own submodules. |
| [bb84.py](../quantum_study/bb84.py) | NumPy, SciPy, pandas, and Matplotlib. Drives `BB84.ipynb` using the proposal's sifted-check model. | Builds each scenario grid once and reuses decision counts per budget. The plot legend displays the configured false-alarm target. |
| [qft_exchange.py](../quantum_study/qft_exchange.py) | NumPy and SciPy. Supplies physical paths to `qft_study.py` and `qft_alarm.py`. | Shared input checks cover block size, attempts, trials, interception and basis probabilities. Removes the unused zero-attempt branch. The focused model draws independent bases on interception and uses a common interception probability. |
| [qft_study.py](../quantum_study/qft_study.py) | Active exchange, NumPy and pandas. Runs the focused notebook and supplies alarm helpers to the follow-up. | Transmission and check costs now run until Bob's alarm or the cap. Eve's declaration still determines the first-outcome result and stops her interception. The decision schedule uses percentages of the attempt cap. |
| [qft_alarm.py](../quantum_study/qft_alarm.py) | Active exchange and alarm helpers. Drives the alarm-response section of the QFT notebook. | Documents the follow-up's random count and spacing, first-crossing target, and the meaning of acceptance after reset. The policy outcomes are unchanged. |
| [independent_bb84_validation.py](independent_bb84_validation.py) | NumPy, Matplotlib and CSV. Standalone transmit–sift control for the ideal intercept–resend calculation. | Rejects non-finite baselines and calculates the total error count once. The control CSV values match the saved results. |
| [test_bb84_validation.py](test_bb84_validation.py) | Independent BB84 control and NumPy. | Checks states, sifting, reproducibility, and rejection of a non-finite baseline. |
| [test_qft_study.py](test_qft_study.py) | Active QFT modules and independent circuit and likelihood checks. | Checks the corrected session costs after Eve declares, as well as alarm order, recovery, and final acceptance. |

BB84 tests detector power on sifted bits with pooled, basis-grouped, and state-grouped scores. The QFT recovery experiment compares pooled check errors with a detector that scores each hidden-check pattern. The state-grouped U4 score is evaluated in the BB84 control. The [saved proposal sweep](../results/proposal_setting_results.zip) retains its first-outcome cost convention. The [alarm-policy CSV](../results/qft_alarm_policy_sweep.csv) reports crossing, recovery, and acceptance outcomes. [Model notes](MODEL_NOTES.md#reading-the-results)

## Ian's independent calculations

These scripts use the fixed-count known-rate BB84 control, the ideal QFT identity, and the proposal's uniform four-bit prior. Each can be checked without importing the primary simulator. [Derivations and references](../ian_analysis/RESULTS_AND_DERIVATION.md)

| Script | Dependencies and callers | Review result |
| --- | --- | --- |
| [bb84_power_check.py](../ian_analysis/bb84_power_check.py) | NumPy, SciPy binomial and normal distributions, and Matplotlib. Generates the standalone power CSV and figure. | Calculates the known-rate conditioned statistic once for U2 and U4, which agree under equal rates within each basis. Retains both result rows. |
| [qft_ideal_check.py](../ian_analysis/qft_ideal_check.py) | NumPy. Checks ideal encoding, phase removal, and inverse-QFT decoding. | Matches the active model's most-significant-bit-first block representation and validates the block and phase inputs. |
| [recovery_posterior.py](../ian_analysis/recovery_posterior.py) | NumPy and SciPy `logsumexp`. Consumes cumulative likelihoods for the 16 keys. | Clarifies the cumulative-evidence input and checks attempt and secret values for recovery decisions. |
| [test_analysis.py](../ian_analysis/test_analysis.py) | All three independent controls. | Checks the equivalent U2 and U4 formulas, ideal decoding, and posterior and alarm ordering. |

## Archived QFT pilot

This pilot retains its historical sampled-phase inference, shared Eve settings, and check budget. Its notebook, recalibration demo, GPU pipeline, and saved data are housed together under `archive/`. The active QFT inference uses exact phase averaging. [Pilot sources and settings](../archive/qft_pilot/README.md) · [Saved results and limitations](../archive/results/README.md)

| Script | Dependencies and callers | Review result |
| --- | --- | --- |
| [__init__.py](../archive/qft_pilot/__init__.py) | Package marker. | Kept for archive imports. |
| [qft_protocol.py](../archive/qft_pilot/qft_protocol.py) | NumPy. Supplies the pilot circuit, Pauli noise, and intercept-resend trajectory. | Checked the proposal's circuit and noise scope. The pilot tests use its scalar routines. |
| [qft_inference.py](../archive/qft_pilot/qft_inference.py) | Pilot circuit, NumPy, SciPy, optional CuPy. Supplies likelihoods and posterior updates. | Documents the sampled-phase approximation and potential spurious information from return-only records. |
| [qft_trials.py](../archive/qft_pilot/qft_trials.py) | Pilot circuit, inference and pandas. Runs per-trial stopping, calibration and recovery. | Clarifies the pilot's terminal budget of 100 retained checks, reached after 25 attempts. The calibration and stopping logic remain historical. |
| [qft_sweep.py](../archive/qft_pilot/qft_sweep.py) | Pilot trials, backend, pandas, and checkpoint files. | Removed a DataFrame conversion after a runner that already returns a DataFrame. Verified the published grid, configuration hash, and seed derivation. |
| [qft_results.py](../archive/qft_pilot/qft_results.py) | Saved CSV, pandas and Matplotlib. Builds dashboard summaries. | Reports missing summary columns explicitly and labels the cutoff target from the configured value. |
| [qft_validation.py](../archive/qft_pilot/qft_validation.py) | Pilot circuit, inference and trials. | Explains the scope of the noiseless circuit and backend checks. |
| [qft_recalibration.py](../archive/qft_pilot/qft_recalibration.py) | Pilot circuit, density-matrix channel, and cutoff helpers. | Reuses the window residual for the score and subsequent baseline update. The drift and baseline-update rules are project choices. |
| [qft_experiment.py](../archive/qft_pilot/qft_experiment.py) | Pilot submodules. Compatibility imports for the original study. | Retains referenced exports and uses the submodules as their implementations. |
| [test_qft_results.py](../archive/qft_pilot/test_qft_results.py) | Result and sweep helpers plus the saved CSV. | Tests the diagnostic for missing summary columns alongside the historical result checks. |
| [test_qft_recalibration.py](../archive/qft_pilot/test_qft_recalibration.py) | Drift demo and independent scalar circuit reference. | Confirms decoding, paired paths, post-decision updates, and time-correlated noise. |

## Proposal sweep runner

The bundled scripts reproduce the saved proposal sweep using its original inputs, seed order, and first-outcome cost convention. [Proposal results bundle](../results/proposal_setting_results.zip)

| Script in `qft_runner/` | Dependencies and callers | Review result |
| --- | --- | --- |
| `run.py` | NumPy, pandas, the bundled exchange, and the bundled alarm rules. | Defines the original proposal grid and retained-check budgets. Both smoke groups reproduced their previous CSVs. |
| `qft_proposal_alarm.py` | NumPy and the bundled recovery threshold. | Preserves the alarm and declaration rules from the original sweep revision. |
| `qft_proposal_exchange.py` | NumPy, SciPy, and the bundled basis implementation. Derived from the active exchange and exact phase marginal. | Retains per-check pooled and pattern scores so the 20, 50, and 100 check budgets can fall within an attempt. |
| `qft_proposal_basis.py` | NumPy. Supplies Eve measurement draws to the bundled exchange. | Retains the proposal's basis-redraw process and its independent-draw special case. |

## Notebooks and hardware references

The default study notebooks and local Bell control were checked on their local execution paths. IBM-specific account and transpilation calls were checked with mocks and a local fake backend. [Hardware source map](../archive/README.md)

| Script or notebook | Dependencies and callers | Review result |
| --- | --- | --- |
| [BB84.ipynb](../BB84.ipynb) | Active BB84 study and its plotting functions. | The final-only and scheduled decision figures and the calibration table serve separate parts of the proposal. |
| [QFT_Repeated_Secret.ipynb](../QFT_Repeated_Secret.ipynb) | Active exchange, focused study, and alarm-policy runner. | First outcomes, final acceptance, per-bit confidence, and the alarm-response comparison use their corresponding study output. |
| [Gate_Baseline.ipynb](../Gate_Baseline.ipynb) | Qiskit circuits, local Aer or IBM Runtime, and Matplotlib. | Skips simulator settings when they are inactive and labels a zero-noise run correctly. Control circuits and barriers preserve the comparison of the X/H round trips. |
| [BB84_Quick_Control.ipynb](../archive/BB84_Quick_Control.ipynb) | NumPy standalone earlier BB84 comparison using a strict cutoff. | Removed unused settings for larger sweeps. Kept the original single-budget demonstration. |
| [QFT_Factorial_Pilot.ipynb](../archive/qft_pilot/QFT_Factorial_Pilot.ipynb) | Archived result, sweep, validation, and recalibration modules. | The saved dashboard and optional reproduction runs use their original data and interfaces. |
| [First_Job.ipynb](../archive/reference_code/First_Job.ipynb) | IBM Runtime Estimator, Qiskit transpilation and local Aer for the fake backend. | Selects one backend and reads the result once. Both choices use the same six-observable circuit. |
| [100_Qubit_Test.ipynb](../archive/reference_code/100_Qubit_Test.ipynb) | IBM Runtime Estimator and a GHZ circuit on a logical chain. | Corrected the correlation label to match the anchored qubit in the Pauli strings. The original hardware results remain archived. |
| [bell_runtime_smoke.py](../archive/reference_code/bell_runtime_smoke.py) | Qiskit `StatevectorSampler` and Matplotlib. | Reads the local measurement counts once for printing and plotting. |
| [bell_state_remote.py](../archive/reference_code/bell_state_remote.py) | Runtime backend lookup and Qiskit transpiler. | Prints the transpiled Bell circuit and removes unused estimator and observable setup. |
| [qkd_connect.py](../archive/reference_code/qkd_connect.py) | JSON account configuration and Runtime `save_account`. | Gets the optional instance from the local configuration, validates the token field, and runs on explicit script execution. |
| [runtime_service_smoke.py](../archive/reference_code/runtime_service_smoke.py) | Runtime service and backend listing. | The account lookup runs on explicit script execution. |
