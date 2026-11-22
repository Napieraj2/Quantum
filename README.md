# Quantum eavesdropping detection studies

**ECSE 6969 final project**
**Team:** Ian Chuang and Andrew Napieraj

## Choose a notebook

Run each notebook from the repository root in a fresh kernel.

| Notebook | Start here for |
| --- | --- |
| [BB84.ipynb](./BB84.ipynb) | Pooled QBER, U2, and U4 detector performance, including terminal and cumulative alarms |
| [QFT_Repeated_Secret.ipynb](./QFT_Repeated_Secret.ipynb) | The saved QFT sweep, its interpretation, and optional simulation and reproduction |
| [Gate_Baseline.ipynb](./Gate_Baseline.ipynb) | Optional single-qubit gate and readout checks in Qiskit Aer |

### Proposal and current QFT baseline

The scope comes from §2 of *Detecting Patient Attacks on a Repeated Secret* (Andrew Napieraj and Ian Chuang, ECSE 6969, October 30, 2026). The BB84 study compares pooled QBER, U2, and U4 on the same checks with independent calibration and held-out false-alarm estimates. The QFT study uses two- and four-qubit blocks, the stated channel and Eve settings, joint check-pattern likelihoods, and correct 99%-confidence recovery before Bob's first alarm. CUDA accelerates the same posterior calculation used on CPU.

The saved QFT run uses a common interception rate across Eves and a single joint-pattern alarm. Its cap of 100 retained checks amounts to 25 attempts. The proposal calls for up to 100 attempts. A proposal-complete run needs to specify the check schedule over that horizon and how to compare detector choices in the QFT exchange. [Saved run notes](./results/README.md)

## Project question

The initial question was whether a detector that accounts for errors separately across the four BB84 states detects partial intercept-resend attacks with greater statistical power than a pooled-QBER detector when no-attack error rates differ by state. Recent closely related papers mean the scope and novelty of this comparison are **provisional** until we review their methods and results.

The project is a controlled simulation and statistical comparison. It is not a physical QKD implementation, a complete security proof, or a claim that an elevated error rate identifies Eve.

## Background and contribution

State- or basis-resolved error estimation is not itself new. Lo, Chau, and Ardehali's efficient BB84 protocol estimates error rates separately for accepted data subsets organized by basis. Scarani et al. review practical QKD assumptions and implementation imperfections. Fung et al. analyze detector-efficiency mismatch as a practical security issue, and Lee et al. study detection of partial intercept-resend attacks.

Two newer works are especially relevant: Fiorini et al. (2025) compare threshold-based eavesdropper detection methods for partial intercept-resend attacks in the six-state protocol; Fiorini et al. (2026) address threshold-based detection of partial intercept-resend attacks in noisy BB84 directly. The 2026 paper is close enough to the proposed topic that we must compare its detector definitions, noise assumptions, calibration, sample sizes, false-alarm control, and power results before deciding what question remains.

Accordingly, the project’s contribution is not yet fixed. After that comparison, we will either narrow the work to a demonstrably unaddressed extension (for example, four-state conditioning with independently estimated state baselines and matched realized false-alarm rates, if these are not already treated), or reframe it as a reproduction/extension under explicitly selected conditions. We will not claim novelty based only on using a four-state score test or simulating partial intercept-resend.

## Protocol and simulation model

- Alice chooses one of `Z0`, `Z1`, `X+`, or `X-`; Bob independently chooses Z or X. Standard BB84 uses equal basis probabilities. Only rounds where their bases match are sifted and included in QBER tests.
- Eve intercepts each signal independently with probability `f`, measures it, and resends the measured state. The reference attack chooses Z/X uniformly. The comparison will also vary Eve's Z-basis probability `p_Z`; `p_Z = 0.5` is random-basis intercept-resend.
- With uniform Alice/Bob bases and random-basis Eve, the ideal attack-only sifted QBER is `f/4`. Under biased Eve basis choice, the pooled ideal QBER remains `f/4`, while the attack contribution depends on Alice's state basis: it is `f(1-p_Z)/2` for Z states and `f p_Z/2` for X states.
- Baseline errors are initially modeled as independent flips of Bob's decoded bit conditioned on Alice's prepared state. If the no-attack baseline for state `s` is `q0_s` and the attack-induced error probability is `a_s`, then the combined rate is `q_s = q0_s + (1 - 2 q0_s) a_s`; a baseline flip and an attack flip cancel when both occur.
- This classical conditional-flip model is intentionally simple. The current example rates `(0.01, 0.03, 0.05, 0.11)` are synthetic test values, not measurements from a cited experiment and not inferred hardware characteristics.

### Why state-dependent baselines are plausible

Real QKD systems can have implementation asymmetries, including detector-efficiency mismatch, and polarization/measurement alignment imperfections. Fung et al. discuss physical sources and security implications of detector-efficiency mismatch; Scarani et al. survey practical implementation issues. These papers motivate studying nonuniform errors, but do **not** justify the example four-state flip rates above. We will label any rates as synthetic unless we extract and justify them from a specific reported experiment. The initial comparison will vary the mean baseline level and its spread across states, then compare which basis has the higher baseline error.

## Detector comparison

1. **Pooled-QBER detector:** tests the total number of sifted errors against a no-attack pooled baseline.
2. **Four-state-conditioned detector:** scores the four error counts using their respective baseline rates and state counts. Its implementation and calibration procedure must be specified before reporting results.

We will compare a known-baseline case with an estimated-baseline case. When baselines are estimated, calibration data will be separate from attack-detection data so the effect of calibration uncertainty can be measured. Thresholds will be selected on null/calibration simulations to target a 5% false-alarm rate, then held fixed during evaluation; the realized false-alarm rate will be reported rather than assumed to equal 5%.

For each fixed number `m` of sifted bits, report empirical false-alarm rate and detection power as functions of interception fraction `f`, including uncertainty intervals. Sweep `m` so the result shows how sample size affects the false-alarm calibration and the interception level needed to reach 80% power. Compare simulated power with binomial/large-sample approximations and identify where those approximations are adequate.

## Planned experiments

| Factor | Comparison |
| --- | --- |
| Sifted sample size | Several fixed values of `m`, not just one operating point |
| Eve interception | `f` from no attack through partial interception |
| Eve basis choice | Uniform Z/X choice versus varying `p_Z` |
| Baseline mean | Low, medium, and higher average state error |
| Baseline spread | Equal versus unequal four-state baselines |
| Baseline orientation | Higher Z-basis versus higher X-basis errors |
| Baseline knowledge | Known rates versus rates estimated from independent calibration samples |
| Statistical calibration | Realized false-alarm rate and power at each sample size, with matched thresholds |

The detector study uses a state-conditioned flip model with independent validation. The optional gate baseline uses a separate Aer gate/readout noise model.

## Proposed team responsibilities

This is a proposed split for the progress check and should be confirmed by both teammates.

| Teammate | Proposed lead |
| --- | --- |
| Ian Chuang | Literature review and statistical detector design, including baseline estimation, false-alarm calibration, and power approximations |
| Andrew Napieraj | Qiskit/BB84 simulation implementation, experiment execution, and reproducible plots/tables |
| Both | Agree on assumptions and test matrix, review results, interpret limitations, and write the final report |

## Current repository status

- [validation/](./validation/) contains independent checks of BB84 state probabilities, ideal partial intercept-resend QBER, and the saved QFT result grid.
- [validation/MODEL_NOTES.md](./validation/MODEL_NOTES.md) records current model assumptions and known omissions.
- The BB84 study compares pooled QBER, basis-grouped U2, and state-conditioned U4 across the proposed sample sizes, calibration sizes, noise profiles, interception rates, and Eve basis biases. It reports held-out false alarms, power, confidence intervals, first tested 80% power points, and three-look familywise alarms.
- The QFT notebook loads the [completed 1,210-setting CUDA sweep](./results/qft_factorial_results.csv) by default. Its optional validation and demo exercise the two-pass model, joint check scores, sequential Eves, and inference over 16 candidate secrets. The [QFT experiment module](./quantum_study/qft_experiment.py) handles CPU/CUDA selection and checkpointed reproduction.
- The final report still needs QFT horizon/threshold/phase-convergence sensitivity checks, more stable false-alarm calibration, trial-level cost uncertainty, and the literature comparison with the 2025 six-state and 2026 noisy-BB84 studies. The completed baseline observed no 99%-confidence recovery declaration under the current 100-check horizon.
- The literature review must first establish how the 2025 six-state and 2026 noisy-BB84 studies overlap with the proposed detectors and experimental design; the project question may change based on that comparison.

Run the existing validation checks from the repository root:

```powershell
python -m pip install -r validation/requirements.txt
python validation/independent_bb84_validation.py
python -m unittest discover -s validation -p "test_*.py" -v
```

For optional NVIDIA GPU acceleration of the QFT posterior, install the CUDA-specific environment before opening the notebook:

```powershell
python -m pip install -r requirements-gpu.txt
```

The default GPU requirements target CUDA Toolkit 13.x on the project RTX 5090. CUDA 12.x systems use `cupy-cuda12x`. Install one CuPy CUDA distribution in the environment. The QFT notebook's optional simulation selects CuPy after a representative matrix-operation probe succeeds. Set `BACKEND = "numpy"` for CPU execution or `BACKEND = "cupy"` to require CUDA.

## References

1. C. H. Bennett and G. Brassard, “Quantum cryptography: Public key distribution and coin tossing” (1984), [arXiv:2003.06557](https://arxiv.org/abs/2003.06557).
2. V. Scarani et al., “The security of practical quantum key distribution,” *Reviews of Modern Physics*, 81, 1301 (2009), [arXiv:0802.4155v3](https://arxiv.org/abs/0802.4155v3).
3. H.-K. Lo, H. F. Chau, and M. Ardehali, “Efficient quantum key distribution scheme and proof of its unconditional security,” *Journal of Cryptology*, 18, 133–165 (2005), [arXiv:quant-ph/0011056](https://arxiv.org/abs/quant-ph/0011056).
4. C.-H. F. Fung, K. Tamaki, B. Qi, H.-K. Lo, and X. Ma, “Security proof of quantum key distribution with detection efficiency mismatch,” *Quantum Information and Computation*, 9, 0131–0165 (2009), [arXiv:0802.3788](https://arxiv.org/abs/0802.3788).
5. C. Lee, E. Lee, and W. Lee, “Analysis on eavesdropper detection in BB84 quantum key distribution protocol against partial intercept-and-resend attack” (2023), [paper PDF](https://dl.ifip.org/db/conf/ondm/ondm2023/1570879964.pdf).
6. F. Fiorini, R. G. Garroppo, M. Pagano, and R. Schiavini Yadzhak, “Eavesdropper Detection in Six-State Protocol Against Partial Intercept–Resend Attack,” *Future Internet*, 17(11), 500 (2025), [doi:10.3390/fi17110500](https://doi.org/10.3390/fi17110500). This is a related six-state study; its abstract describes threshold-based methods and false-positive/false-negative analysis.
7. F. Fiorini, R. G. Garroppo, and M. Pagano, “Threshold-based eavesdropper detection for partial intercept-resend attack in noisy BB84 quantum key distribution,” *Computer Networks*, 277, 112058 (2026), [doi:10.1016/j.comnet.2026.112058](https://doi.org/10.1016/j.comnet.2026.112058). This directly overlaps the proposal’s broad setting; its precise relationship to the planned pooled-versus-four-state comparison remains to be established by reading the paper.
