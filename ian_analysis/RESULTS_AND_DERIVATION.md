# BB84 detector calculation

This control implements the fixed-count, known-rate special case in the project proposal, §2. The reference rates are from [Erven et al., §4](https://arxiv.org/abs/0901.0960v2), and the partial intercept-resend model follows the standard BB84 calculation used by [Lee, Lee, and Lee, §III](https://dl.ifip.org/db/conf/ondm/ondm2023/1570879964.pdf).

## Noise and sample model

The single-look example fixes 25 retained check bits for each of four prepared states. The model uses independent symmetric baseline flips with rates 0.002 for Z0/Z1 and 0.038 for X+/X-.

| State | Baseline error rate | Retained bits |
| --- | ---: | ---: |
| Z0 | 0.002 | 25 |
| Z1 | 0.002 | 25 |
| X+ | 0.038 | 25 |
| X- | 0.038 | 25 |

Each signal is intercepted with probability `f`. On an intercepted signal Eve measures in the Z basis with probability `r`, otherwise X, then resends her result. For a sifted bit, Eve introduces errors when her measurement basis differs from Alice's and Bob's matching basis. The attack-induced bit error is

```
aZ = f * (1-r) / 2
aX = f * r / 2
```

Combining attack-induced errors and independent baseline flips:

```
pZ = qZ + (1 - 2*qZ) * aZ
pX = qX + (1 - 2*qX) * aX
```

For `f=0.025`, `r=0.5`, both attack terms equal 0.00625. Substituting:

```
pZ = 0.002 + 0.996 * 0.00625 = 0.008225
pX = 0.038 + 0.924 * 0.00625 = 0.043775
```

The expected number of errors rises from `50(0.002) + 50(0.038) = 2.0` without interception to `50(0.008225) + 50(0.043775) = 2.6` under this attack. At 2.5% interception, the model predicts 0.6 additional errors per 100 retained bits.

## Detection statistics and exact probabilities

Let `E_s` be the wrong-bit count for state `s`, `n_s = 25`, and `q_s` its no-attack error rate. Conditional on fixed state counts, the error counts have independent binomial distributions:

```
E_s ~ Binomial(n_s, p_s)
P(E_Z0=e0, E_Z1=e1, E_X+=e2, E_X-=e3)
    = product_s BinomialPMF(e_s; n_s, p_s)
```

With four state counts from 0 to 25, the calculation enumerates `26^4 = 456,976` combinations. The pooled statistic is the total wrong-bit count. For the known-rate conditioned score, the state weights are

```
w_s = (1 - 2*q_s) / (q_s * (1 - q_s))
U4  = sum_s w_s * (E_s - n_s*q_s)
```

The two states in each basis have the same rate and therefore the same weight. This gives

```
U4 = wZ * (E_Z0 + E_Z1 - 50*qZ)
   + wX * (E_X+ + E_X- - 50*qX)
   = U2
```

The code evaluates this conditioned statistic once and publishes the same value under the `U2` and `U4` proposal labels. The tests independently substitute the four state weights into the displayed `U4` expression and verify equality for every enumerated count vector.

For each statistic, the code selects a nonrandomized cutoff whose upper-tail probability under no attack does not exceed 5%. An alarm occurs only when the statistic is strictly greater than its cutoff. All powers and false-alarm probabilities below use that rule.

## Exact results for uniform Eve (`r = 0.5`)

| Interception `f` | Pooled | U2 | U4 |
| ---: | ---: | ---: | ---: |
| 0 | 4.93% | 3.14% | 3.14% |
| 0.005 | 6.06% | 5.48% | 5.48% |
| 0.01 | 7.32% | 8.04% | 8.04% |
| 0.025 | 11.87% | 16.69% | 16.69% |
| 0.05 | 21.63% | 32.35% | 32.35% |
| 0.10 | 45.08% | 60.15% | 60.15% |
| 0.20 | 81.82% | 89.25% | 89.25% |

The `f=0` row gives the *realized false-alarm probability*, not power against an eavesdropper. The two cutoff rules produce different actual null rejection rates: 4.93% for pooled and 3.14% for U2/U4. The discrete counts and nonrandomized ties prevent an exact 5% calibration for both rules. The observed power difference is therefore **not** a matched-size detector comparison.

For `f=0.025` and strongly Z-biased Eve (`r=0.9`), pooled power is approximately 11.53% and U2/U4 power approximately 7.23%. Thus, changing Eve's basis distribution can reverse the ranking of the detectors.

For each pair `(f, r)`, the code also generates 50,000 independent Monte Carlo count samples. The Monte Carlo estimates agree with the enumerated probabilities within sampling uncertainty. A normal approximation is recorded for comparison; it is less reliable when the expected counts are small.

## Ideal QFT calculation

This identity is derived from [Papadopoulos and Linvill, §4.1, Eqs. (1)--(3) and Fig. 2](https://arxiv.org/abs/2404.12507v2). For `b` qubits, set `D = 2^b`. The ideal encoded amplitudes are proportional to `exp(2*pi*i*m*x/D)`. After removing an independent private phase, applying the inverse QFT returns the computational-basis value `m`. The matrix check covers every possible `m` for `b=2` and `b=4`. The maximum calculated amplitude error is approximately `4.10e-15`, consistent with floating-point rounding.

This is a no-noise algebra check. It does not simulate Eve's measurements or evaluate the complete two-pass protocol.

## Recovery decision rule

For a four-bit repeated secret, the posterior is defined over 16 possible values. Given cumulative log likelihoods for the 16 candidates through each attempt, the recovery module applies the proposal's uniform prior, computes a normalized posterior, and finds the first attempt where the largest posterior is at least 0.99. History rows are cumulative evidence, not per-attempt likelihood increments.

A correct declaration counts as recovery only when it occurs no later than attempt 100 and strictly before Bob's first alarm. A declaration at the same attempt as the alarm does not count. The supplied log likelihoods must represent Eve's available measurements and public transcript, with Bob's unreleased private randomness marginalized. Likelihood generation is not part of this module.

These single-look BB84 probabilities and ideal QFT identities do not establish a repeated-secret recovery rate or a cryptographic security bound.
