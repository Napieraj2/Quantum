# Model notes

## Proposal deliverables

*Detecting Patient Attacks on a Repeated Secret* (Napieraj and Chuang, October 30, 2026, `QFT_QKD_With_Multiple_Eves.pdf`) specifies the methods in §2 and the project outputs in §3.

| Week | Output | Where to see it |
| --- | --- | --- |
| 1 | Circuit probabilities | The QFT exchange calculates Bob's check-pattern probabilities. The validation tests check circuit and inference invariants. |
| 2 | BB84 power and false alarms | [BB84](../BB84.ipynb) plots results with one final alarm decision and with multiple scheduled decisions. The independent control checks sifted `f/4` using 400,000 transmitted rounds per setting. |
| 3 | QFT recovery | [QFT Repeated Secret](../QFT_Repeated_Secret.ipynb) compares two detectors on shared transmissions over 100 attempts. It plots recovery, false alarms, and confidence in each bit. |
| 4 | Noise, schedules and hardware | The QFT configuration supports different fixed noise rates and the default uses independently redrawn bases. [Gate Baseline](../Gate_Baseline.ipynb) is hardware setup with a local Aer default. The archived pilot preserves its basis-redraw sweep and separate Bob-only drift demo. |
| 5 | Figures and report | The notebooks produce figures for selected settings. The QFT study also reports mean transmissions and retained checks. Broader sensitivity runs and final report figures are separate work. |
| 6 | Presentation and rehearsal | Project work outside the notebook runs. |

## Active files

| File | Proposal output or role | Source |
| --- | --- | --- |
| [BB84.ipynb](../BB84.ipynb) and [bb84.py](../quantum_study/bb84.py) | Week 2. QBER, U2, and U4 compare detection and held-out false alarms on shared trials within each alarm schedule. | Bennett and Brassard provide the preparations, Lo et al. motivate basis grouping, Lee et al. inform the normal approximation, and proposal §2 defines the scores and calibration. [Notebook references](../BB84.ipynb#references) |
| [QFT_Repeated_Secret.ipynb](../QFT_Repeated_Secret.ipynb) and [qft_study.py](../quantum_study/qft_study.py) | Weeks 3 and 5. The paired recovery experiment, outcomes, and figures. | Proposal §2 supplies the repeated secret, recovery threshold, and independent calibration. The attempt-level alarm schedule and block surprisal are project choices. |
| [qft_exchange.py](../quantum_study/qft_exchange.py) | Weeks 1 and 3. Circuit, check likelihoods, and Eve's pooled record likelihood. | The two-pass circuit follows [Papadopoulos and Linvill, §4.1, Fig. 2, Eqs. (1)–(3), and §6.1](https://arxiv.org/abs/2404.12507v2). [Erven et al., §4](https://arxiv.org/abs/0901.0960v2) supplies the reference rates. Proposal §2 defines the Pauli model and pooled records. |
| [Gate_Baseline.ipynb](../Gate_Baseline.ipynb) | Week 4 setup. X/H round trips and same-state controls. | The four preparations and measurements follow Bennett and Brassard, pp. 175–177. The control design and synthetic noise rates are project choices. [Gate references](../Gate_Baseline.ipynb#interpretation-and-sources) |
| [quantum_study/__init__.py](../quantum_study/__init__.py) | Python package setup for the notebook imports. | Python package infrastructure. |
| [independent_bb84_validation.py](independent_bb84_validation.py) and [test_bb84_validation.py](test_bb84_validation.py) | §3 completed control and Week 2 validation. State probabilities, sifting, and ideal intercept-resend. | [Bennett and Brassard, pp. 175–177](https://arxiv.org/abs/2003.06557) and [Scarani et al., §I.B.5](https://arxiv.org/abs/0802.4155). |
| [test_qft_study.py](test_qft_study.py) | Weeks 1 and 3 validation. Scientific regression checks for the current QFT experiment. | The circuit reference and proposal outcome rules used by the QFT study. |

The [archive inventory](../archive/README.md) and the [archived QFT pilot](../archive/qft_pilot/README.md) cover superseded implementations and follow-up work.

## Glossary

### Runs and channel settings

| Term | Definition |
| --- | --- |
| **QFT** | Quantum Fourier transform. Bob undoes his private phase rotations and applies the inverse QFT to decode Alice's block. |
| **BB84** | Bennett and Brassard's 1984 quantum key distribution protocol. Alice prepares bits in X or Z and Bob measures them. They keep the results where their bases match. |
| **Attempt** | One QFT round trip carrying Alice's complete four-bit candidate key and four fresh check bits. Alice repeats the same key across attempts within a trial. `ATTEMPTS` is the maximum number of round trips. |
| **Trial or session** | One simulated run of a setting. A QFT trial follows one candidate key. In the QFT notebook, `TRIALS = 1024` gives 1,024 independent runs in each batch used to fit the rule, test it under noise alone, or test it under interception. |
| **Block** | A group of two or four qubits encoded and decoded together. Half of each block carries payload bits and half carries hidden checks. One attempt uses four two-qubit blocks or two four-qubit blocks. |
| **Leg** | One direction of the quantum exchange. The forward leg goes from Bob to Alice. The return leg goes from Alice to Bob. |
| **Eve and coalition** | Eve is an interceptor on the link. A coalition is a group of Eves pooling their classical measurement records. |
| **Interception probability** | The chance one Eve intercepts a given qubit on a given leg. For $K$ Eves acting independently with the same probability $f$, **effective interception** is the chance at least one Eve intercepts that qubit on that leg, $1-(1-f)^K$. |
| **Measurement basis and redraw** | A Z-basis measurement gives `0` or `1`. An X-basis measurement gives `+` or `−`. `P(Z)` is Eve's probability of choosing Z on a basis draw, so `P(Z) = 0` means X-only. The redraw probability is the chance Eve chooses a basis again after a passing signal, including when she skips it. |
| **Channel noise (`q_z`, `q_x`)** | The subscripts name the bases in which the errors appear. On each QFT leg, Pauli X is applied with probability `q_z` (a Z-basis bit error) and Pauli Z with probability `q_x` (an X-basis bit error). A profile labeled **higher X** has larger `q_x`. |

### Checks, alarms, and calibration

| Term | Definition |
| --- | --- |
| **Hidden check and retained checks** | In QFT, Alice inserts test bits with fresh positions and values, then discloses them so Bob can compare the decoded values. A retained check is a bit included in the score. One complete QFT attempt yields four retained checks. BB84 retains sifted bits whose preparation and measurement bases match. |
| **Pooled check errors / QBER** | The fraction of retained checks with the wrong value. QBER stands for quantum bit error rate in BB84. |
| **Joint-check surprisal** | The negative log of the noise-only probability of Bob's observed check pattern within a QFT block. Rarer patterns give larger scores. Bob adds the scores across the blocks in his current evidence window. |
| **Basis U2 / State U4** | BB84 scores that weight error counts relative to the noise baseline. U2 groups counts by the two bases. U4 uses the four prepared states. |
| **Alarm look** | A scheduled point when Bob compares his score with the alarm cutoff. In the QFT notebook he collects check results on every attempt, and makes alarm decisions at the scheduled looks. |
| **Final alarm decision only** | Bob decides whether to alarm once, at the full check budget. The proposal results CSVs label this schedule `terminal`. |
| **Multiple scheduled alarm decisions** | Bob can alarm at several points. In the proposal runs, each decision uses all checks collected so far. With a 100-check budget, he can alarm at 25, 50, or 100 checks. The proposal results CSVs label this schedule `cumulative`. |
| **Percentage and random schedules (QFT)** | The percentage schedule makes decisions at 25%, 50%, and 100% of the attempt cap. The default random schedule chooses 2–4 decision points per session and includes the final attempt. |
| **Calibration and held-out noise** | Calibration uses noise-only trials to set the cutoff. A separate held-out noise-only batch measures the realized crossing rate. The 5% target refers to at least one crossing across the whole schedule. |
| **Threshold crossing** | Bob's score meets the calibrated alarm rule at a scheduled look. The policy comparison counts a trial as having crossed if this happens at least once, including a crossing Bob later dismisses. |
| **Attack crossing rate / Bob alarmed first** | The attack crossing rate counts any crossing in an attacked trial. The first-outcome chart counts Bob first when his alarm comes before or on the same attempt as Eve's declaration. |
| **Reject on alarm** | Bob rejects the candidate key at the first threshold crossing. |
| **Reset on interim alarm** | Bob treats an interim crossing as noise and clears his accumulated evidence. Later looks score checks collected after the reset using the same noise calibration. Alice continues with the same key and Eve keeps her evidence. A crossing at the final look rejects the key. |
| **Sessions reset** | The number of trials in which Bob dismissed at least one interim crossing. A trial with several resets counts once. |
| **Noise session rejection** | The fraction of held-out noise-only trials rejected by Bob's alarm response. A dismissed interim crossing can contribute to the noise crossing rate and still leave the session eligible for acceptance. |
| **Detection power** | In the BB84 results, the fraction of attacked trials that raise an alarm under the stated detector and check schedule. |

### Recovery and key acceptance

| Term | Definition |
| --- | --- |
| **Eve's posterior and declaration** | The posterior gives Eve's probability for each of the 16 possible four-bit keys. She declares the first candidate to reach at least 99% confidence and stops intercepting on subsequent attempts. Her declaration can be correct or incorrect. |
| **Mean probability of true bit** | The average probability Eve assigned to the value Alice actually sent at one bit position. Each line tracks one bit across trials. The plot carries forward each stopped trial's last available confidence. |
| **Full recovery before alarm** | Eve's first declaration correctly identifies all four bits before Bob's first crossing. Bob wins a same-attempt tie. The recovery curve shows the fraction of attacked trials recovered by each attempt, and the policy table reports the count before the first crossing. |
| **Reached cap** | In the first-outcome chart, the trial ended at the attempt cap with neither an alarm nor a declaration. Key acceptance is evaluated separately at the final check. |
| **Accepted session** | Bob reaches the final attempt, passes the alarm policy, and his final decoded key matches Alice's under ideal confirmation. The reset policy can accept a key after an interim crossing has been dismissed. |
| **Exposed accepted key** | An accepted key that Eve correctly declared by the cap. **Exposed acceptance after reset** is an exposed accepted key from a session where Bob dismissed at least one interim crossing. |
| **Exposure among accepted keys** | Exposed accepted keys divided by accepted sessions. This fraction is undefined when no key is accepted. Dividing by all attacked trials gives the overall frequency of exposed acceptance. |

For example, 8 exposed keys among 8 acceptances out of 1,024 attacked trials gives **100% exposure among accepted keys** and **0.78% exposed acceptance across all attacked trials**.

## Reading the results

- **BB84.** With uniform Eve bases, sifted attack-only QBER is `f/4`. The project combines background bit flips and attack errors as `q + (1 - 2q) × attack_error`. Equal true rates within each basis make true-rate U2 and U4 coincide. The curves show realized false alarms at zero interception. Wilson intervals describe Monte Carlo uncertainty. The sharp Bernoulli bounds in [Mannalath et al.](https://arxiv.org/abs/2410.04095) remain outside these runs.
- **QFT.** Each attempt yields four fresh checks. Bob looks after 25%, 50%, and 100% of the configured attempt cap. Each detector targets a 5% chance of any alarm across those looks, and the figure shows its independent held-out rate. The joint detector scores the disclosed check pattern within each block. A correct declaration requires at least 99% posterior confidence before Bob's alarm. The exact Eve likelihood starts from the uniform phase mixture and uses the measurement records from both legs. The default is a focused comparison at one noise and attack setting.
- **Accepted-key exposure.** In the focused reject-on-alarm comparison, Bob accepts if all alarm looks pass and his final decoded four-bit key matches Alice's under ideal confirmation. A key is exposed if Eve's first 99%-confidence declaration was correct by then. Eve stops intercepting after her first declaration, and Bob continues checking through the final attempt. The table gives the fraction exposed among accepted attacked sessions. The fraction is undefined when no session accepts a key.
- **Historical pilot.** The old run ended at 100 retained checks, after 25 attempts. Its posterior used sampled phase averages, and the separate 500-attempt drift demo tracked Bob's alarms alone. [Pilot limitations](../results/README.md#scope)

## QFT research direction

**Alice's encoder as a phase oracle.** In the ideal circuit, applying Alice's encoding for an $n$-qubit block word $m$ to a $\lvert+\rangle^{\otimes n}$ probe produces $\mathrm{QFT}\lvert m\rangle$. Eve could store Bob's original block, send Alice the probe, and apply the inverse QFT to the returned probe to read the block word. Applying Alice's encoding to the stored block before forwarding it would give Bob the expected state, with the hidden checks intact. [Encoding circuit, §4.1 and §6.1](https://arxiv.org/abs/2404.12507v2)

This scenario assumes access to the whole block on both legs, that Alice accepts the substituted input, and that Eve can store and forward Bob's block within the exchange's timing. A follow-up study could test how much information Eve can recover with partial access and how memory noise or timing changes affect Bob's alarm. The current notebook models X/Z intercept-resend.
