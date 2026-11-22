# Archived QFT factorial pilot

This directory preserves the superseded 1,210-setting repeated-secret pilot,
its Bob-only channel-drift demonstration, and their focused tests. The active
project uses a newer exchange model.

Open `QFT_Factorial_Pilot.ipynb` from either the repository root or this
directory. Its first code cell finds the repository root, loads the validated
CSV from `results/`, and exposes the optional run flags. Set
`RUN_RECALIBRATION_DEMO = True` for the 500-attempt Bob-alarm comparison or
`RUN_FULL_QFT_SWEEP = True` to reproduce the factorial pilot. Checkpoints and
reproduction CSVs remain under the repository-level `results/` directory.

Run the archived tests from the repository root:

```sh
python -m unittest discover -s archive/qft_pilot -p "test_*.py" -v
```

## Optional CUDA

The original RTX 5090 checkpoint uses CuPy and `complex64`. Install the
CuPy package for your CUDA toolkit in the notebook's Python environment.

```sh
python -m pip install cupy-cuda13x
```

For CUDA 12.x use `cupy-cuda12x`. Install one CuPy distribution. The pilot's
`BACKEND = "auto"` selects CUDA when available. Resume with the same backend
and precision as the checkpoint.

## Historical file map

| File | Pilot role |
| --- | --- |
| [QFT_Factorial_Pilot.ipynb](QFT_Factorial_Pilot.ipynb) | Pilot figures, validation controls, and optional run flags. |
| [qft_protocol.py](qft_protocol.py) | Circuit probabilities from [Papadopoulos and Linvill, §§4.1 and 6.1](https://arxiv.org/abs/2404.12507v2). |
| [qft_inference.py](qft_inference.py) | Historical phase-sampled posterior and NumPy/CuPy backend. |
| [qft_trials.py](qft_trials.py) | Pilot alarm calibration, stopping, and recovery outcomes. |
| [qft_sweep.py](qft_sweep.py) | Pilot settings grid and checkpoint reproduction. |
| [qft_results.py](qft_results.py) | Pilot figures and setting-level summaries. |
| [qft_validation.py](qft_validation.py) | Noiseless circuit and backend agreement checks. |
| [qft_recalibration.py](qft_recalibration.py) | Follow-up only. Bob's alarms during drift and baseline updates. |
| [test_qft_recalibration.py](test_qft_recalibration.py) | Follow-up tests for the drift demo. |
| [test_qft_results.py](test_qft_results.py) | Regression tests for the historical CSV and configuration. |
| [qft_experiment.py](qft_experiment.py) | Setup only. Compatibility imports. |
| [__init__.py](__init__.py) | Setup only. Python package marker. |

See the [pilot results and limitations](../../results/README.md).
