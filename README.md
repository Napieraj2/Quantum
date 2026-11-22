# Quantum eavesdropping detection studies

ECSE 6969 final project · Andrew Napieraj and Ian Chuang

## Run All

Use the same `requirements.txt` for all three notebooks and the validation scripts. From the repository root, install the dependencies and launch Jupyter.

```sh
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open a notebook, edit the first code cell if needed, and choose **Run All**. The figures appear directly under the cells that produce them.

| Notebook | Figures you get |
| --- | --- |
| [BB84](BB84.ipynb) | Terminal detection power and cumulative alarm probability for QBER, U2, and U4. Zero interception gives the held-out false-alarm rate. |
| [QFT repeated secret](QFT_Repeated_Secret.ipynb) | Held-out false alarms, recovery before Bob's alarm, and confidence in each secret bit over 100 attempts. Both detectors see the same transmissions. |
| [Gate baseline](Gate_Baseline.ipynb) | Wrong-outcome rates for each X/H round trip and its same-state control. The default runs locally in Aer. |

## Study scope

BB84 and QFT start with the Erven reference noise rates. The BB84 notebook offers a broader sweep in its first cell. The QFT notebook starts with two Eves using different basis biases and compares pooled check errors with the block check-pattern score. It uses the published two-pass circuit and the proposal's repeated four-bit secret. [Methods and deliverable map](validation/MODEL_NOTES.md#proposal-deliverables)

The original 1,210-setting GPU pilot, its checkpoint workflow, and the 500-attempt Bob-only drift demo are available in the archive. [QFT pilot and run instructions](archive/qft_pilot/README.md)

## Validation

```sh
python -m unittest discover -s validation -p "test_*.py" -v
```

See the [validation guide](validation/README.md) for the independent BB84 control and model tests. The notebook runs display the study figures.
