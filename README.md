# Quantum eavesdropping detection studies

ECSE 6969 final project · Andrew Napieraj and Ian Chuang

## Run a notebook

From the repository root, run `python -m pip install -r requirements.txt`. Open each notebook in a fresh Python kernel.

| Notebook | Purpose |
| --- | --- |
| [BB84](BB84.ipynb) | Compare pooled QBER, basis-grouped U2, and state-conditioned U4. Read the noise-only rates at zero interception and the detection rates above zero. |
| [QFT repeated secret](QFT_Repeated_Secret.ipynb) | Read the saved 1,210-setting pilot. Optional cells validate the model, demonstrate channel drift, and reproduce the sweep. |
| [Gate baseline](Gate_Baseline.ipynb) | Optional single-qubit gate and readout diagnostics in local Qiskit Aer. |

Set `RUN_RECALIBRATION_DEMO = True` for the 500-attempt Bob-side comparison of fixed and frequent recalibration. Drifting paths use a seeded, bounded, mean-reverting multiplier shared by `q_Z` and `q_X`, so nearby channel rates remain similar. Keep `RUN_FULL_QFT_SWEEP = False` when reviewing the saved results.

The QFT posterior uses NumPy on CPU and optionally CuPy on CUDA. Install `requirements-gpu.txt` for CUDA 13.x or substitute `cupy-cuda12x` for CUDA 12.x. Choose `BACKEND = "auto"` for selection with CPU fallback, or `"numpy"` or `"cupy"` explicitly.

## Study scope

The methods follow §2 of *Detecting Patient Attacks on a Repeated Secret* (Napieraj and Chuang, October 30, 2026). The BB84 notebook compares terminal and cumulative alarms with independent calibration and held-out false alarms. The QFT notebook uses two- and four-qubit exchanges, joint check-pattern likelihoods, and correct 99%-confidence recovery before Bob's first alarm.

The saved QFT pilot uses one joint-pattern alarm and a common interception rate across Eves. It caps trials at 100 retained checks (25 attempts). The proposal's 100-attempt recovery experiment and the comparison of detector choices within QFT remain open. [Pilot results and limits](results/README.md)

## Validate

```sh
python -m unittest discover -s validation -p "test_*.py" -v
```

The [validation guide](validation/README.md) covers the independent BB84 control. For assumptions and citations, see the [model notes](validation/MODEL_NOTES.md) and the methods sections in each notebook. Earlier plans, hardware prototypes, and the duplicate QFT export are in the [archive](archive/README.md).
