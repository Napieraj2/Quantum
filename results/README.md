# QFT factorial pilot

The [saved CSV](qft_factorial_results.csv) has one row per setting. It contains 1,210 distinct settings, with 100 trials per setting. The [run manifest](qft_factorial_run_manifest.json) records the source commit, RTX 5090 configuration, runtime, and checksums. The resumable checkpoint is local and excluded from Git.

## What was observed

- The 118,800 attacked trials produced zero full-secret declarations at the 99% posterior threshold within this run's check budget.
- Noise-only controls alarmed in 160 of 2,200 trials (7.27%, pooled Wilson 95% interval 6.26%–8.43%) against a nominal 5% target.
- Alarm-before-recovery rose with total interception intensity. Noise generally masked interception. The basis-bias asymmetry merits a sensitivity check.

## Read and reproduce

In an attack setting, `alarm_rate` measures alarms before recovery or the budget. The two recovery columns distinguish correct and incorrect declarations. The remaining fraction reached the budget with neither outcome. In a noise-only setting, `alarm_rate` is the probability of any false alarm across the looks, and `eve_p_z` and `redraw_probability` are placeholders.

The [QFT notebook](../QFT_Repeated_Secret.ipynb) loads the CSV, checks grid completeness, and plots it by default. Full reproduction is optional and supports checkpoint resume.

## Scope

This pilot used one joint-pattern alarm, fixed noise within a setting, and a cap of 100 retained checks (25 attempts). The proposal's 100-attempt recovery horizon and comparison of detector choices within QFT still need evaluation. The optional drift demo measures Bob's alarms over 100 attempts. The saved sweep contains setting means, so trial-level cost, confidence, and partial-recovery trajectories cannot be recovered from it.
