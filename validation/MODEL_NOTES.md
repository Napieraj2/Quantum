# Model notes

## Proposal deliverables

*Detecting Patient Attacks on a Repeated Secret* (Napieraj and Chuang, October 30, 2026, `QFT_QKD_With_Multiple_Eves.pdf`) specifies the methods in §2 and the project outputs in §3.

| Week | Output | Where to see it |
| --- | --- | --- |
| 1 | Circuit probabilities | The QFT exchange calculates Bob's check-pattern probabilities. The validation tests check circuit and inference invariants. |
| 2 | BB84 power and false alarms | [BB84](../BB84.ipynb) plots terminal and cumulative results. The independent control checks sifted `f/4` using 400,000 transmitted rounds per setting. |
| 3 | QFT recovery | [QFT Repeated Secret](../QFT_Repeated_Secret.ipynb) compares two detectors on shared transmissions over 100 attempts. It plots recovery, false alarms, and confidence in each bit. |
| 4 | Noise, schedules and hardware | The QFT configuration supports different fixed noise rates and the default uses independently redrawn bases. [Gate Baseline](../Gate_Baseline.ipynb) is hardware setup with a local Aer default. The archived pilot preserves its basis-redraw sweep and separate Bob-only drift demo. |
| 5 | Figures and report | The notebooks produce figures for selected settings. The QFT study also reports mean transmissions and retained checks. Broader sensitivity runs and final report figures are separate work. |
| 6 | Presentation and rehearsal | Project work outside the notebook runs. |

## Active files

| File | Proposal output or role | Source |
| --- | --- | --- |
| [BB84.ipynb](../BB84.ipynb) and [bb84.py](../quantum_study/bb84.py) | Week 2. Paired terminal and cumulative QBER, U2, and U4 power with held-out false alarms. | Bennett and Brassard provide the preparations, Lo et al. motivate basis grouping, Lee et al. inform the normal approximation, and proposal §2 defines the scores and calibration. [Notebook references](../BB84.ipynb#references) |
| [QFT_Repeated_Secret.ipynb](../QFT_Repeated_Secret.ipynb) and [qft_study.py](../quantum_study/qft_study.py) | Weeks 3 and 5. The paired recovery experiment, outcomes, and figures. | Proposal §2 supplies the repeated secret, recovery threshold, and independent calibration. The attempt-level alarm schedule and block surprisal are project choices. |
| [qft_exchange.py](../quantum_study/qft_exchange.py) | Weeks 1 and 3. Circuit, check likelihoods, and Eve's pooled record likelihood. | The two-pass circuit follows [Papadopoulos and Linvill, §4.1, Fig. 2, Eqs. (1)–(3), and §6.1](https://arxiv.org/abs/2404.12507v2). [Erven et al., §4](https://arxiv.org/abs/0901.0960v2) supplies the reference rates. Proposal §2 defines the Pauli model and pooled records. |
| [Gate_Baseline.ipynb](../Gate_Baseline.ipynb) | Week 4 setup. X/H round trips and same-state controls. | The four preparations and measurements follow Bennett and Brassard, pp. 175–177. The control design and synthetic noise rates are project choices. [Gate references](../Gate_Baseline.ipynb#interpretation-and-sources) |
| [quantum_study/__init__.py](../quantum_study/__init__.py) | Python package setup for the notebook imports. | Python package infrastructure. |
| [independent_bb84_validation.py](independent_bb84_validation.py) and [test_bb84_validation.py](test_bb84_validation.py) | §3 completed control and Week 2 validation. State probabilities, sifting, and ideal intercept-resend. | [Bennett and Brassard, pp. 175–177](https://arxiv.org/abs/2003.06557) and [Scarani et al., §I.B.5](https://arxiv.org/abs/0802.4155). |
| [test_qft_study.py](test_qft_study.py) | Weeks 1 and 3 validation. Scientific regression checks for the current QFT experiment. | The circuit reference and proposal outcome rules used by the QFT study. |

The [archive inventory](../archive/README.md) and the [archived QFT pilot](../archive/qft_pilot/README.md) cover superseded implementations and follow-up work.

## Reading the results

- **BB84.** With uniform Eve bases, sifted attack-only QBER is `f/4`. The project combines background bit flips and attack errors as `q + (1 - 2q) × attack_error`. Equal true rates within each basis make true-rate U2 and U4 coincide. The curves show realized false alarms at zero interception. Wilson intervals describe Monte Carlo uncertainty. The sharp Bernoulli bounds in [Mannalath et al.](https://arxiv.org/abs/2410.04095) remain outside these runs.
- **QFT.** Each attempt yields four fresh checks. Bob looks after 25%, 50%, and 100% of the configured attempt cap. Each detector targets a 5% chance of any alarm across those looks, and the figure shows its independent held-out rate. The joint detector scores the disclosed check pattern within each block. A correct declaration requires at least 99% posterior confidence before Bob's alarm. The exact Eve likelihood starts from the uniform phase mixture and uses the measurement records from both legs. The default is a focused comparison at one noise and attack setting.
- **Accepted-key exposure.** Under the final-check use rule, Bob accepts if every scheduled check passes and his final decoded four-bit key matches Alice's under ideal confirmation. A key is exposed if Eve's first 99%-confidence declaration was correct by then. Eve stops intercepting after her first declaration, and Bob continues checking through the final attempt. The table gives the fraction exposed among accepted attacked sessions. The fraction is undefined when no session accepts a key.
- **Historical pilot.** The old run ended at 100 retained checks, after 25 attempts. Its posterior used sampled phase averages, and the separate 500-attempt drift demo tracked Bob's alarms alone. [Pilot limitations](../results/README.md#scope)

## QFT research direction

**Alice's encoder as a phase oracle.** In the ideal circuit, applying Alice's encoding for an $n$-qubit block word $m$ to a $\lvert+\rangle^{\otimes n}$ probe produces $\mathrm{QFT}\lvert m\rangle$. Eve could store Bob's original block, send Alice the probe, and apply the inverse QFT to the returned probe to read the block word. Applying Alice's encoding to the stored block before forwarding it would give Bob the expected state, with the hidden checks intact. [Encoding circuit, §4.1 and §6.1](https://arxiv.org/abs/2404.12507v2)

This scenario assumes access to the whole block on both legs, that Alice accepts the substituted input, and that Eve can store and forward Bob's block within the exchange's timing. A follow-up study could test how much information Eve can recover with partial access and how memory noise or timing changes affect Bob's alarm. The current notebook models X/Z intercept-resend.
