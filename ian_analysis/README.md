# Independent analysis controls

This folder contains a separate numerical check of pooled and conditioned BB84 error detectors. The calculations use one disclosed sample and known no-attack error rates. The QFT and posterior modules check smaller mathematical pieces of the repeated-secret model.

The controls are derived from the project proposal, §2. The ideal QFT identity follows [Papadopoulos and Linvill, §4.1, Eqs. (1)--(3)](https://arxiv.org/abs/2404.12507v2), and the reference basis-error rates come from [Erven et al., §4](https://arxiv.org/abs/0901.0960v2). The standard partial intercept-resend model is consistent with [Lee, Lee, and Lee, §III](https://dl.ifip.org/db/conf/ondm/ondm2023/1570879964.pdf). These scripts do not import the main study implementation.

## Model

The BB84 example uses 100 retained, matching-basis bits, split evenly across Z0, Z1, X+ and X-. The assumed benign wrong-bit rates are 0.002 in the Z basis and 0.038 in the X basis. These are model parameters, not measured device error rates.

Eve intercepts each signal with probability `f`, measures in Z with probability `r` (otherwise X), and resends. An independent symmetric baseline bit flip gives the state-dependent error probabilities. The detector cutoffs are nonrandomized and use a nominal false-alarm limit of 5%.

## Results

The script calculates exact binomial probabilities over all possible four-state error counts and checks them with Monte Carlo samples. For uniform Eve at `f = 0.025`, pooled QBER has 11.87% detection power and the basis/state scores each have 16.69%. Their realized false-alarm rates are 4.93% and 3.14%, respectively. The result does not establish an equal-size advantage.

U2 and U4 agree in this example because both prepared states within each basis have the same known baseline error rate. The implementation therefore calculates the conditioned score once and reports that result under both proposal labels. A direct four-state formula in the tests verifies the algebraic reduction. The basis weights need not remain equal when rates are estimated separately or when the benign noise model changes.

## Files

- `bb84_power_check.py` — exact single-look cutoffs and power, Monte Carlo estimates, and a normal approximation.
- `qft_ideal_check.py` — ideal phase encoding and inverse-QFT decoding for 2- and 4-qubit blocks.
- `recovery_posterior.py` — posterior normalization over 16 supplied cumulative secret likelihoods and a recovery-before-alarm decision rule.
- `test_analysis.py` — tests for model probabilities, score behavior, QFT decoding, and posterior logic.
- `RESULTS_AND_DERIVATION.md` — equations, numerical settings, results, and limitations.
- `results/` — the exact BB84 power CSV and figure.

The BB84 calculation fixes the number of bits per state and treats their errors as independent. It does not model estimated baseline rates, sequential alarm calibration, or temporal correlations. The QFT script checks ideal decoding algebra; the posterior module accepts candidate likelihoods rather than generating Eve observations. Neither module establishes repeated-secret recovery probabilities or key security.
