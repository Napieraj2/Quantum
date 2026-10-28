# Model notes

## What the code represents

Alice sends one of four BB84 states: `Z0`, `Z1`, `X+`, or `X-`. Bob randomly measures in Z or X. They only compare rounds where their bases match.

Eve uses a partial intercept-resend attack. She intercepts a signal with probability `f`, picks Z or X with equal probability, measures, and sends Bob the corresponding state. If she picks the wrong basis, Bob has a 50% chance of a bit error even when he picked Alice's correct basis. So the ideal attack-only sifted QBER is:

`P(error) = f * (1/2) * (1/2) = f/4`

The first 1/2 is Eve picking the wrong basis. The second 1/2 is Bob getting the wrong bit after Eve resends from that basis. There is no extra 1/2 for sifting because QBER is measured **after** sifting.

## Adding unequal baseline errors

The second run uses different no-attack error rates for the four prepared states. After Bob's ideal measurement, his decoded bit is flipped independently with probability `q0_s`. If the attack error and the baseline flip both occur, they cancel. That gives:

`q_s(f) = q0_s + (1 - 2*q0_s)*f/4`

The test uses `q0 = (0.01, 0.03, 0.05, 0.11)` and `f = 0.2`. These values are chosen to exercise an unequal-baseline case; they are not measurements from a device. This is a simple classical flip model, not a general model of quantum hardware noise.

## Where this stops

- The ideal QBER figure uses zero baseline errors. It doesn't compare the two detectors.
- The simulation randomly generates the four state counts. The main detector study needs a fixed disclosed-bit budget `m` and specified counts `m_s` for a fair comparison.
- It does not model photon-number attacks, detector-control attacks, source leakage, drifting hardware, or a real optical channel.
- Detecting an abnormal error rate does not identify Eve or prove the key is secure.

For the main comparison, we still need a pooled-error detector, a four-state score test, separate calibration/evaluation data, and power curves at matched false-alarm rates. Thresholds should stay fixed when we test baseline mismatch.

## Background

- Bennett and Brassard, "Quantum cryptography: Public key distribution and coin tossing" (1984), https://arxiv.org/pdf/2003.06557 - BB84 and basis sifting.
- Scarani et al., "The security of practical quantum key distribution" (2009), https://arxiv.org/pdf/0802.4155 - assumptions and practical-security limits.
- Lee et al., "Analysis on eavesdropper detection in BB84 quantum key distribution protocol against partial intercept-and-resend attack" (2023), https://dl.ifip.org/db/conf/ondm/ondm2023/1570879964.pdf - partial-interception detection context.
