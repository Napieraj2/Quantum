# BB84 intercept-resend check

This folder checks the BB84 error model before we compare the two detectors. It covers the four states, Z/X measurements, and a partial intercept-resend attack.

## Run it

From the root of the Quantum repo:

```bash
python -m pip install -r validation/requirements.txt
python validation/independent_bb84_validation.py
python -m unittest discover -s validation -p "test_*.py" -v
```

On Windows, `py` works in place of `python`. Running the `.py` file from PowerShell is the simplest option. The code also handles editors that do not provide `__file__`, although their working folder controls where results are saved.

The script uses 400,000 transmitted rounds per interception fraction and a fixed seed (`20261026`). It saves the figure and CSV results in `validation/results/`. No IBM account or Qiskit installation is needed.

## What it checks

1. All four prepared states (`Z0`, `Z1`, `X+`, `X-`) against both Z and X measurements. Matching bases should give the correct bit; opposite bases give a 50/50 result.
2. Eve intercepts each signal with probability `f`, picks a random Z/X basis, measures, and resends her result. Alice and Bob keep only their matching-basis rounds. The expected error rate is `f/4`.
3. A separate test adds different baseline error rates to Bob's **decoded bits**: `(0.01, 0.03, 0.05, 0.11)`. That checks `q_s(f) = q0_s + (1 - 2*q0_s)*f/4` at `f = 0.2`.

For the ideal intercept-resend run:

| Intercepted fraction | Expected QBER | Simulated QBER |
| --- | --- | --- |
| 0% | 0% | 0% |
| 20% | 5.000% | 5.024% |
| 40% | 10.000% | 9.982% |
| 60% | 15.000% | 15.093% |
| 80% | 20.000% | 19.809% |
| 100% | 25.000% | 25.017% |

## Files

- `independent_bb84_validation.py` - runs the checks and writes the outputs.
- `test_bb84_validation.py` - tests the measurement table and the simulation logic.
- `results/bb84_ideal_validation.png` - theory vs simulation; this is the Figure 1 check for the progress report.
- `results/intercept_resend_ideal.csv` - simulated and expected QBER for each `f`.
- `results/eight_case_probabilities.csv` - Z/X probabilities for each prepared state.
- `results/state_conditioned_xor_model.csv` - error counts grouped by Alice's prepared state.
- `MODEL_NOTES.md` - assumptions and what these checks do not cover.

The ideal figure checks the attack model, **not** whether a state-conditioned detector beats a pooled-QBER detector. Detector thresholds, false-alarm rates, and power curves are still part of the next phase.
