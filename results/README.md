# QFT factorial result

`qft_factorial_results.csv` is the completed setting-level output from the two-pass QFT repeated-secret experiment. It contains 1,210 unique settings: two block sizes, eleven noise profiles, twenty-two attack-free controls, and 1,188 attack settings. Every setting contains 100 Monte Carlo trials.

## Run audit

- Source commit: `8f49492b30fba181e3e1652f3b9d85f35edaf239`
- Backend: CuPy on an NVIDIA GeForce RTX 5090
- Numeric precision: complex64
- Phase samples per posterior update: 16
- Alarm-calibration trajectories per block/noise group: 100
- Evaluation trials: 121,000 total; 118,800 attacked
- Recorded setting runtime: 128.46 minutes; observed wall time: approximately 131 minutes
- Result SHA-256: `179f5a2a560cf7dd21b6dfcf3a0a2cb76bb3100206fcc25cc59097c3ed687974`

The matching checkpoint was validated against every CSV value. It remains a local resumable artifact and is excluded from Git.

## Headline interpretation

- No correct or incorrect recovery declaration occurred in any attacked trial at the configured 99% posterior threshold.
- This means recovery was not observed within the implemented 100-check horizon. It does not establish zero recovery probability under longer horizons, different thresholds, or a refined posterior calculation.
- Held-out attack-free alarms totaled 160 of 2,200 trials: 7.27% overall, with an approximate 95% Wilson interval of 6.26%–8.43%, above the nominal 5% target.
- Alarm-before-recovery increased strongly with total interception intensity. Noise generally masked attacks, and basis redraw had little marginal effect.
- `eve_p_z = 0.1` produced lower alarm rates and higher posterior confidence than larger Z-basis probabilities; this asymmetry requires a dedicated convention and sensitivity check.

## Reading the rate columns

For attack settings, `alarm_rate` is the probability that Bob alarmed before recovery or the final budget. It is a competing-risk outcome, not ordinary fixed-horizon power. `correct_recovery_rate` and `incorrect_recovery_rate` are declarations before Bob's alarm. The remaining probability is inconclusive:

```text
inconclusive_rate = 1 - alarm_rate
                    - correct_recovery_rate
                    - incorrect_recovery_rate
```

For attack-free rows, `alarm_rate` is the realized familywise false-alarm rate. `eve_p_z` and `redraw_probability` are inert placeholders whenever `eve_count` is zero.

## Limitations of this result version

- The run stops at 100 retained checks, which corresponds to roughly 25 attempts, although the proposal also mentions a 100-attempt horizon.
- Only one QFT alarm statistic, joint-pattern surprisal, is evaluated. The pooled/U2/U4 detector comparison belongs to the separate BB84 experiment.
- One hundred calibration and evaluation runs per setting are too small for stable 5% tail calibration and smooth cell-level comparisons.
- The CSV retains setting means rather than trial-level distributions, so uncertainty for costs and posterior confidence cannot be reconstructed.
- Phase-sample, precision, recovery-threshold, and horizon sensitivity analyses remain necessary before making a security claim.

The [QFT repeated-secret notebook](../QFT_Repeated_Secret.ipynb) validates completeness, derives outcome rates and confidence intervals, and plots the saved sweep by default.
