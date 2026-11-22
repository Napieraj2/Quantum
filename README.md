# Quantum eavesdropping detection studies

ECSE 6969 final project · Andrew Napieraj and Ian Chuang

## Run a notebook

Use the same `requirements.txt` for all three notebooks and the validation scripts. From the repository root, install the dependencies and launch Jupyter.

```sh
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open each notebook in a fresh Python kernel. Set the run options in the first code cell, then run the notebook top to bottom.

| Notebook | Purpose | Where the code lives |
| --- | --- | --- |
| [BB84](BB84.ipynb) | Compare pooled QBER, basis-grouped U2, and state-conditioned U4. Zero interception gives the noise-only alarm rate. | BB84 simulation and detectors are defined in the notebook. |
| [QFT repeated secret](QFT_Repeated_Secret.ipynb) | Read the saved 1,210-setting pilot. Optional cells validate the model, demonstrate channel drift, and reproduce the sweep. | The notebook calls [quantum_study/](quantum_study/). |
| [Gate baseline](Gate_Baseline.ipynb) | Run optional single-qubit gate and readout diagnostics in local Qiskit Aer. | The notebook defines its own Qiskit circuits and diagnostics. |

## QFT code and run controls

`quantum_study/` supports the QFT notebook and its validation tests.

- [Protocol physics](quantum_study/qft_protocol.py) and [posterior and backends](quantum_study/qft_inference.py)
- [Trials and alarms](quantum_study/qft_trials.py) and [channel drift and recalibration](quantum_study/qft_recalibration.py)
- [Sweeps and checkpoints](quantum_study/qft_sweep.py), [results and plots](quantum_study/qft_results.py), and [model checks](quantum_study/qft_validation.py)

Earlier imports still work through [qft_experiment.py](quantum_study/qft_experiment.py).

Set `RUN_RECALIBRATION_DEMO = True` for the 500-attempt Bob-side comparison of fixed and frequent recalibration. Drifting paths use a seeded, bounded, mean-reverting multiplier shared by `q_Z` and `q_X`, so nearby channel rates remain similar. Keep `RUN_FULL_QFT_SWEEP = False` when reviewing the saved results.

### Optional CUDA

The QFT posterior runs on CPU with NumPy. To enable GPU acceleration, install the CuPy package for your CUDA toolkit in the same Python environment as Jupyter. The tested RTX 5090 setup uses CUDA 13.x.

```sh
python -m pip install cupy-cuda13x
```

For CUDA 12.x, run `python -m pip install cupy-cuda12x`. Install only one CuPy distribution. Set `BACKEND = "auto"` to use CUDA when available with a CPU fallback. Choose `"numpy"` or `"cupy"` to require a specific backend.

## Study scope

The methods follow §2 of *Detecting Patient Attacks on a Repeated Secret* (Napieraj and Chuang, October 30, 2026). The BB84 notebook compares terminal and cumulative alarms with independent calibration and held-out false alarms. The QFT notebook uses two- and four-qubit exchanges, joint check-pattern likelihoods, and correct 99%-confidence recovery before Bob's first alarm.

The saved QFT pilot uses one joint-pattern alarm and a common interception rate across Eves. It caps trials at 100 retained checks (25 attempts). The proposal's 100-attempt recovery experiment and the comparison of detector choices within QFT remain open. [Pilot results and limits](results/README.md)

## Validate

```sh
python -m unittest discover -s validation -p "test_*.py" -v
```

The [validation guide](validation/README.md) covers the independent BB84 control. For assumptions and citations, see the [model notes](validation/MODEL_NOTES.md) and the methods sections in each notebook. Earlier plans, hardware prototypes, and the duplicate QFT export are in the [archive](archive/README.md).
