# Model notes

## Proposal deliverables

*Detecting Patient Attacks on a Repeated Secret* (Napieraj and Chuang, October 30, 2026, `QFT_QKD_With_Multiple_Eves.pdf`) specifies the methods in §2 and the project outputs in §3.

| Week | Output | Status and evidence |
| --- | --- | --- |
| 1 | Circuit probabilities | **Complete for simulation.** The exchange computes the noise-only check-pattern probabilities for two- and four-qubit blocks. Ian's ideal circuit check and the full-density validation fixtures provide independent checks. [Exchange](../quantum_study/qft_exchange.py) · [Ian's check](../ian_analysis/qft_ideal_check.py) · [Validation](test_qft_study.py) |
| 2 | BB84 power and false alarms | **Complete for the detector comparison.** The saved grid covers QBER, U2, and U4 on shared trials, both alarm schedules, true rates, repeated fits, and the normal approximation for a single final decision. The 400,000-round control verifies `f/4`. Ian's exact calculation checks one fixed-count, known-rate case. [Proposal sweep](../README.md#proposal-setting-results) · [BB84 notebook](../BB84.ipynb) · [Ian's calculation](../ian_analysis/RESULTS_AND_DERIVATION.md) |
| 3 | QFT recovery | **Baseline complete, prepared-state sweep pending.** The saved comparison reports pooled errors and the joint check score over the proposal grid. The active notebook can now add the prepared-state score on the same transmissions. Its calibration uses known channel rates and independent noise-only trials. Ian's posterior check verifies the decision given likelihoods. [QFT notebook](../QFT_Repeated_Secret.ipynb) · [Saved grid](../results/proposal_setting_results.zip) · [Ian's posterior check](../ian_analysis/recovery_posterior.py) |
| 4 | Noise, schedules, and hardware | **Simulation complete, hardware result open.** The saved grid exercises the ten distinct noise profiles, both redraw probabilities, and the retained-check schedules. The follow-up compares fixed and random decisions and resets. The gate notebook has a local Aer run and an IBM submission path. A study hardware result is still needed. [Proposal sweep](../README.md#proposal-setting-results) · [Follow-up sweep](../README.md#qft-alarm-response-results) · [Gate notebook](../Gate_Baseline.ipynb) |
| 5 | Figures, costs, and report | **Partial.** The root README and notebooks contain figures and tables. The focused QFT code reports mean transmissions and checks through Bob's alarm or the cap. The saved QFT grid uses the first event for its cost columns, and the BB84 detector CSV reports check budgets. A BB84 stopping-cost calculation and the final report are open. [Results inventory](SCRIPT_REVIEW.md) |
| 6 | Presentation and rehearsal | **Outside the checked-in work.** Completion of the presentation and rehearsal is unverified. |

Proposal §2 also calls for sharp Bernoulli bounds on calibration uncertainty. That calculation is open. The repeated BB84 fits measure variation, and the Wilson intervals describe sampling uncertainty in the reported alarm rates. [Mannalath, Zapatero, and Curty](https://arxiv.org/abs/2410.04095)

## Active files

| File | Proposal output or role | Source |
| --- | --- | --- |
| [BB84.ipynb](../BB84.ipynb) and [bb84.py](../quantum_study/bb84.py) | Week 2. QBER, U2, and U4 compare detection and held-out false alarms on shared trials within each alarm schedule. | Bennett and Brassard provide the preparations, Lo et al. motivate basis grouping, Lee et al. inform the normal approximation, and proposal §2 defines the scores and calibration. [Notebook references](../BB84.ipynb#references) |
| [QFT_Repeated_Secret.ipynb](../QFT_Repeated_Secret.ipynb) and [qft_study.py](../quantum_study/qft_study.py) | Weeks 3 and 5. The paired recovery experiment, including the prepared-state comparison. | Proposal §2 supplies the repeated secret, recovery threshold, and independent calibration. The attempt-level schedule, the joint score, and the adaptation to individual QFT checks are project choices. |
| [qft_exchange.py](../quantum_study/qft_exchange.py) | Weeks 1 and 3. Circuit, check likelihoods, and Eve's pooled record likelihood. | The two-pass circuit follows [Papadopoulos and Linvill, §4.1, Fig. 2, Eqs. (1)–(3), and §6.1](https://arxiv.org/abs/2404.12507v2). [Erven et al., §4](https://arxiv.org/abs/0901.0960v2) supplies the reference rates. Proposal §2 defines the Pauli model and pooled records. |
| [qft_alarm.py](../quantum_study/qft_alarm.py) | Follow-up. Compares fixed and random schedules and what Bob does when he alarms. | The 5% false-alarm target comes from the proposal. Random schedules, resets, and final key acceptance are project extensions. |
| [Gate_Baseline.ipynb](../Gate_Baseline.ipynb) | Week 4 setup. X/H round trips and same-state controls. | The four preparations and measurements follow Bennett and Brassard, pp. 175–177. The control design and synthetic noise rates are project choices. [Gate references](../Gate_Baseline.ipynb#interpretation-and-sources) |
| [quantum_study/__init__.py](../quantum_study/__init__.py) | Python package setup for the notebook imports. | Python package infrastructure. |
| [independent_bb84_validation.py](independent_bb84_validation.py) and [test_bb84_validation.py](test_bb84_validation.py) | §3 completed control and Week 2 validation. State probabilities, sifting, and ideal intercept-resend. | [Bennett and Brassard, pp. 175–177](https://arxiv.org/abs/2003.06557) and [Scarani et al., §I.B.5](https://arxiv.org/abs/0802.4155). |
| [test_qft_study.py](test_qft_study.py) | Weeks 1 and 3 validation. Scientific regression checks for the current QFT experiment. | The circuit reference and proposal outcome rules used by the QFT study. |

The [script review](SCRIPT_REVIEW.md) records the dependencies, sources, and audit result for each script and notebook.

## Glossary

### Runs and channel settings

| Term | Definition |
| --- | --- |
| **QFT** | Quantum Fourier transform. Bob undoes his private phase rotations and applies the inverse QFT to decode Alice's block. |
| **BB84** | Bennett and Brassard's 1984 quantum key distribution protocol. Alice prepares bits in X or Z and Bob measures them. They keep the results where their bases match. |
| **Attempt** | One QFT round trip carrying Alice's complete four-bit candidate key and fresh check bits. Block sizes 1, 2, 4, and 8 use four checks. The 16-qubit block uses twelve. Alice repeats the same key across attempts within a trial. `ATTEMPTS` is the maximum number of round trips. |
| **Trial or session** | One simulated run of a setting. A QFT trial follows one candidate key. In the QFT notebook, `TRIALS = 1024` gives 1,024 independent runs in each batch used to fit the rule, test it under noise alone, or test it under interception. |
| **Block** | A group of qubits encoded and decoded together. Block sizes 2, 4, and 8 mix equal numbers of payload and check bits within each block. With one-qubit blocks, Alice randomly assigns four whole blocks to checks and four to the payload. These layouts send eight qubits on each leg. The 16-qubit layout uses one block containing four payload bits and twelve checks. |
| **Leg** | One direction of the quantum exchange. The forward leg goes from Bob to Alice. The return leg goes from Alice to Bob. |
| **Eve and coalition** | Eve is an interceptor on the link. A coalition is a group of Eves pooling their classical measurement records. |
| **Interception probability** | The chance one Eve intercepts a given qubit on a given leg. For $K$ Eves acting independently with the same probability $f$, **effective interception** is the chance at least one Eve intercepts that qubit on that leg, $1-(1-f)^K$. |
| **Measurement basis and redraw** | A Z-basis measurement gives `0` or `1`. An X-basis measurement gives `+` or `−`. `P(Z)` is Eve's probability of choosing Z on a basis draw, so `P(Z) = 0` means X-only. The redraw probability is the chance Eve chooses a basis again after a passing signal, including when she skips it. |
| **Channel noise (`q_z`, `q_x`)** | The subscripts name the bases in which the errors appear. On each QFT leg, Pauli X is applied with probability `q_z` (a Z-basis bit error) and Pauli Z with probability `q_x` (an X-basis bit error). A profile labeled **higher X** has larger `q_x`. |

### Checks, alarms, and calibration

| Term | Definition |
| --- | --- |
| **Hidden check and retained checks** | In QFT, Alice inserts test bits with fresh positions and values, then discloses them so Bob can compare the decoded values. A retained check is a bit included in the score. One complete QFT attempt yields four retained checks for block sizes 1–8 and twelve for the 16-qubit block. BB84 retains sifted bits whose preparation and measurement bases match. |
| **Pooled check errors / QBER** | The fraction of retained checks with the wrong value. QBER stands for quantum bit error rate in BB84. |
| **Joint-check surprisal** | The negative log of the noise-only probability of Bob's observed check pattern within a QFT block. Rarer patterns give larger scores. Bob adds the scores across the blocks in his current evidence window. |
| **Prepared-state check errors** | A QFT score that weights each check using its noise-only chance of an error, given Bob's phases and Alice's disclosed check positions and values. |
| **Basis U2 / State U4** | BB84 scores that weight error counts relative to the noise baseline. U2 groups counts by the two bases. U4 uses the four prepared states. |
| **Final alarm decision only** | Bob decides whether to alarm once, at the full check budget. The proposal results CSVs label this schedule `terminal`. |
| **Multiple scheduled alarm decisions** | Bob can alarm at several points. In the proposal runs, each decision uses all checks collected so far. With a 100-check budget, he can alarm at 25, 50, or 100 checks. The proposal results CSVs label this schedule `cumulative`. |
| **Fixed and random schedules (QFT)** | Bob collects check bits on every attempt. On the fixed schedule he can alarm at 25%, 50%, and 100% of the attempt cap. By default the random schedule chooses 2–4 attempts and always includes the final attempt. |
| **Calibration and held-out noise** | One set of noise-only trials sets the cutoff. A separate set measures how often Bob alarms. The 5% target is for at least one alarm in a noise-only trial. |
| **Trials with an alarm** | The share of trials with at least one alarm, including alarms Bob dismisses. **Trials rejected by alarm** counts the share where an alarm caused Bob to reject the key. |
| **Bob alarmed first** | Bob alarmed before or on the same attempt as Eve's first guess, or he alarmed and Eve never guessed. |
| **Reject on alarm** | Bob rejects the key as soon as he alarms. |
| **Reset on early alarm** | Bob dismisses alarms before the final attempt and clears the evidence he has collected so far. Alice keeps sending the same key, and Eve keeps what she learned. An alarm at the final attempt still rejects the key. |
| **Sessions reset** | The number of trials where Bob dismissed at least one alarm. A trial with several resets counts once. |
| **Detection power** | In the BB84 results, the fraction of attacked trials that raise an alarm under the stated detector and check schedule. |

### Recovery and key acceptance

| Term | Definition |
| --- | --- |
| **Eve's confidence and guess** | Eve tracks a probability for each of the 16 keys. She guesses when one reaches at least 99% confidence and then stops intercepting. |
| **Mean probability of true bit** | The average probability Eve assigned to the value Alice actually sent at one bit position. Each line tracks one bit across trials. The plot carries forward each stopped trial's last available confidence. |
| **Full recovery before alarm** | Eve guessed all four bits correctly before Bob's first alarm, or she guessed correctly and Bob never alarmed. An alarm on the same attempt as her guess counts for Bob. |
| **Reached attempt cap** | The trial reached the last attempt without an alarm or a guess from Eve. Bob can still accept the key if it matches Alice's. |
| **Accepted session** | Bob reaches the final attempt, has no alarm that requires him to reject the key, and his final decoded key matches Alice's. |
| **Exposed accepted key** | Bob accepted the key and Eve guessed it correctly by the final attempt. The reset column counts these keys in trials where Bob dismissed at least one alarm. |
| **Exposure among accepted keys** | Exposed accepted keys divided by accepted sessions. This fraction is undefined when no key is accepted. Dividing by all attacked trials gives the overall frequency of exposed acceptance. |

For example, 8 exposed keys among 8 acceptances out of 1,024 attacked trials gives **100% exposure among accepted keys** and **0.78% exposed acceptance across all attacked trials**.

The saved sweep uses `noise_crossing_rate` and `attack_crossing_rate` for the shares of trials with an alarm. `recovered_before_first_crossing` counts trials where Eve recovered before Bob's alarm. [Alarm sweep CSV](../results/qft_alarm_policy_sweep.csv)

## Reading the results

- **BB84.** With uniform Eve bases, sifted attack-only QBER is `f/4`. The project combines background bit flips and attack errors as `q + (1 - 2q) × attack_error`. Equal true rates within each basis make true-rate U2 and U4 coincide. The curves show realized false alarms at zero interception. Wilson intervals describe Monte Carlo uncertainty. The sharp Bernoulli bounds in [Mannalath et al.](https://arxiv.org/abs/2410.04095) remain outside these runs.
- **QFT.** The first three notebook sections use one noise and attack setting with a fixed schedule. The left panel measures false alarms on separate noise-only trials. The joint detector scores the disclosed check pattern within each block. Eve's probabilities account for Bob's unknown phases and use measurement records from both legs.
- **QFT cost counts.** The notebook measures `mean_checks` and `mean_transmitted_qubits` until Bob alarms or reaches the cap. In the saved proposal sweep, `mean_retained_checks` and `mean_transmitted_qubits` count up to Bob's alarm, Eve's guess, or the check budget, whichever comes first. Each qubit sent on either leg counts as one transmission. [Proposal sweep](../results/proposal_setting_results.zip)

## Prepared-state comparison in QFT

QFT prepares a block with continuous private phases. For each check bit, Bob can use those phases and the disclosed check positions and values to compute the noise-only probability $q_i$ of a wrong decoded bit. The simulator averages over all payloads consistent with the disclosed checks. The prepared-state detector adapts the weighted error score from proposal §2. If $e_i$ is 1 for a mismatch and 0 for a match, it adds $(e_i-q_i)(1-2q_i)/(q_i(1-q_i))$ for each check. The detector uses marginal probabilities for individual bits. The existing joint-check score uses the probability of their whole pattern, including correlations within a block. Both derive their probabilities from the same density calculation. [Exchange and score](../quantum_study/qft_exchange.py)

The notebook uses each score on the same trials. It fits separate alarm cutoffs on noise-only trials to target a 5% chance of any alarm, then measures false alarms on fresh noise-only trials. This first implementation uses known channel rates for $q_i$ and the attempt schedule at 25%, 50%, and 100% of the cap. A prepared-state sweep over the full retained-check and basis-redraw grid, and fitted channel rates for the score, are still open. [QFT notebook](../QFT_Repeated_Secret.ipynb)

An initial proposal-range check used 4,096 trials in each batch, 100 attempts, four Eves with 2.5% interception each and `P(Z) = 0.1`, fresh basis draws, noise `(q_z, q_x) = (0.038, 0.002)`, and seed `20261122`. Each cell shows Eve's correct recoveries before Bob's alarm, out of 4,096 attacked trials.

| Block size | Pooled check errors | Joint-check surprisal | Prepared-state check errors |
| --- | ---: | ---: | ---: |
| Two qubits | 34 | 39 | 97 |
| Four qubits | 4 | 4 | 7 |

For the two-qubit run, the holdout false-alarm counts were 215, 219, and 243 out of 4,096, respectively. A second seed (`20261123`) gave 38, 39, and 100 recoveries, with 197, 194, and 219 false alarms. The new score allowed more recoveries in this setting under both seeds. The saved proposal results cover the wider two-detector comparison. [Saved proposal sweep](../results/proposal_setting_results.zip)

## Hardware-informed measurement option

The gate baseline exports wrong outcomes and shot counts for the prepared `|0⟩` and `|1⟩` same-state controls measured in Z. The QFT switch uses their fractions as effective assignment probabilities `P(1|0)` and `P(0|1)`. The measurements include state preparation. The model applies rates from one physical qubit independently to each decoded bit at Bob. His check scores and final key agreement both use the reported bits. Alarm fitting and held-out false-alarm trials use the same assignment model. The existing per-leg Pauli rates still describe the link.

The X–X and H–H round-trip bars benchmark single-qubit circuits. Characterizing the multi-qubit QFT gates would require data from representative transpiled QFT circuits at the chosen block size. [Gate baseline](../Gate_Baseline.ipynb) · [QFT settings](../QFT_Repeated_Secret.ipynb)

## QFT block-size comparison

The proposal used two- and four-qubit blocks. The notebook also runs one-, eight-, and sixteen-qubit blocks. Sizes 1–8 send four payload bits and four fresh check bits per attempt, for 16 qubit transmissions counting both legs. The one-qubit layout randomly selects four dedicated check blocks each attempt. A disturbance to a payload block cannot propagate to a separate check block. The eight-qubit layout puts the entire payload and all four checks in one block. The 16-qubit layout puts four payload bits and twelve fresh checks in one block, for 32 qubit transmissions counting both legs. Its results reflect the larger block and the larger check budget together. All layouts use the same link noise and ideal circuit operations. [Notebook settings](../QFT_Repeated_Secret.ipynb)

At 16 qubits the simulator samples the decoded word one bit at a time using the conditional Fourier probabilities. The joint score sums over the sixteen output words consistent with Bob's observed checks. The prepared-state score uses interpolation of the Fourier factors to compute marginal check probabilities. Validation compares those probabilities with enumeration of all 65,536 outputs. [Exchange implementation](../quantum_study/qft_exchange.py) · [Validation](test_qft_study.py)

The comparison uses 4,096 trials per batch for each size, a cap of 100 attempts, and alarm decisions at attempts 25, 50, and 100. Bob discards the key on any alarm. Four Eves each intercept with probability 2.5% per qubit per leg and choose Z with probability 0.1 on fresh basis draws. The noise rates are `(q_z, q_x) = (0.038, 0.002)` and the seed is `20261122`. Five block sizes and three detectors produce 15 rows. The CSV records `payload_bits_per_attempt` and `check_bits_per_attempt` so the expanded 16-qubit setting remains visible. The plot shows the observed shares of all noise-only or attacked trials with 95% Wilson intervals. The table also includes exposure among the keys Bob accepted. [Comparison plot](../results/qft_block_sizes/comparison.png) · [Counts and settings](../results/qft_block_sizes/comparison.csv)

## First bounds on Eve's information

Let $K$ be the uniform four-bit key and $T_t$ Eve's transcript through attempt $t$. The transcript contains her modeled measurements and the disclosed checks. Bob's private phases and alarm decisions are outside this transcript. For Eve's posterior $p_t$, the information gained in bits is $I(K;T_t)=4-\mathbb{E}[H(p_t)]$. The average over simulated trials estimates this quantity. We can also bound it directly from the modeled records.

On a given qubit, the secret can affect Eve's likelihood only if the last forward measurement and first return measurement are both in X. Write $p_{XX}$ for the probability of that pair while Eve is intercepting. Under independent hits and fresh bases it is the product of the two endpoint X probabilities. With ordered Eves of hit rates $f_i$ and Z probabilities $r_i$, the probability the first recorded measurement is in X is $\sum_i f_i(1-r_i)\prod_{j<i}(1-f_j)$. Use $j>i$ for the probability the last recorded measurement is in X. For an all-X coalition of size $k$ with common interception $f$, $p_{XX}=[1-(1-f)^k]^2$.

A paired X outcome carries at most $C=1-h_2(q_x)$ bits, where $h_2$ is binary entropy. The relative outcome is binary and its conditional error probability lies between $q_x$ and $1-q_x$ for the proposal noise rates. Let $N$ be the number of qubit positions per attempt: $N=8$ for block sizes 1–8 and $N=16$ for the 16-qubit block. Thus $I(K;T_t)\le\min(4,Nt\,p_{XX}C)$. Eve's early stop can only reduce the number of these outcomes. [Record likelihood](../quantum_study/qft_exchange.py)

A lower bound follows from the first attempt alone. The rotation on the first qubit encodes the parity of the block word. When its least significant bit is payload, a paired X outcome gives one secret bit through a binary flip with probability $q_x$. If $r$ is the probability that this position is payload and $B$ is the number of blocks, the contributions from different payload blocks give $I(K;T_t)\ge Brp_{XX}C$ for $t\ge1$. Sizes 2, 4, and 8 have $r=1/2$. The 16-qubit block has $r=4/16=1/4$. With one-qubit blocks, four of the eight dedicated blocks carry payload, so $Br=4$. A separate lower bound from Fano's inequality uses Eve's best-guess error probability $P_e$ and gives $\max(0,4-h_2(P_e)-P_e\log_2 15)$ bits.

The upper bound tightens by enumerating the check layouts, disclosed values, and qubit positions. In each case find the smallest and largest probabilities $a,b$ of equal X outcomes among the block words consistent with the disclosed checks. Maximize $h_2((1-u)a+ub)-(1-u)h_2(a)-uh_2(b)$ over $0\le u\le1$ to bound what the next pair can reveal under any prior. Averaging these capacities over the layouts, values, and positions replaces $C$ in the upper bound. At $q_x=0.038$ the averages are 0.34427 bits for two-qubit blocks and 0.39203 bits for four-qubit blocks.

For the notebook's default two-qubit, two-Eve settings, the chance of an X pair on a signal is 0.00060941 while Eve is intercepting. At 100 attempts the record bounds are about **0.00093 to 0.168 bits** with the tighter upper bound. As a check, averaging the posterior entropy over 4,096 trials with seed `7120` gave **0.166 bits** with a Monte Carlo standard error of **0.0052 bits**. This is an estimate of average information in Eve's modeled records. Recovery before alarm and exposure among accepted keys use their own denominators. An analysis conditioned on acceptance would start from $H(K\mid\text{accept})$ and include whatever Bob reveals about his decision.

## QFT research direction

**Alice's encoder as a phase oracle.** In the ideal circuit, applying Alice's encoding for an $n$-qubit block word $m$ to a $\lvert+\rangle^{\otimes n}$ probe produces $\mathrm{QFT}\lvert m\rangle$. Eve could store Bob's original block, send Alice the probe, and apply the inverse QFT to the returned probe to read the block word. Applying Alice's encoding to the stored block before forwarding it would give Bob the expected state, with the hidden checks intact. [Encoding circuit, §4.1 and §6.1](https://arxiv.org/abs/2404.12507v2)

This scenario assumes access to the whole block on both legs, that Alice accepts the substituted input, and that Eve can store and forward Bob's block within the exchange's timing. A follow-up study could test how much information Eve can recover with partial access and how memory noise or timing changes affect Bob's alarm. The current notebook models X/Z intercept-resend.
