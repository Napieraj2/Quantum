# Validation

From the repository root:

```sh
python -m pip install -r validation/requirements.txt
python -m unittest discover -s validation -p "test_*.py" -v
```

- **BB84** checks the four prepared states, sifting, ideal intercept-resend QBER of `f/4`, and state-conditioned baseline errors.
- **QFT** audits the saved sweep and checks that the recalibration demo uses the existing QFT probabilities and updates its live baseline for the next look.

Run `python validation/independent_bb84_validation.py` to regenerate the [independent BB84 figure and CSVs](results/). It uses 400,000 transmitted rounds per interception fraction and a fixed seed. The [model notes](MODEL_NOTES.md) explain the formulas and limitations.
