# Historical QFT factorial pilot

The [saved CSV](qft_factorial_results.csv) has one row per setting. It contains 1,210 distinct settings, with 100 trials per setting. The [run manifest](qft_factorial_run_manifest.json) records the source commit, RTX 5090 configuration, runtime, and checksums. The resumable checkpoint is local and excluded from Git.

## What was observed

- The 118,800 attacked trials produced zero full-secret declarations at the 99% posterior threshold within this run's check budget.
- Noise-only controls alarmed in 160 of 2,200 trials (7.27%, pooled Wilson 95% interval 6.26%–8.43%) against a nominal 5% target.
- Alarm-before-recovery rose with total interception intensity. Noise generally masked interception. The basis-bias asymmetry merits a sensitivity check.

## Read and reproduce

In an attack setting, `alarm_rate` measures alarms before recovery or the budget. The two recovery columns distinguish correct and incorrect declarations. The remaining fraction reached the budget with neither outcome. In a noise-only setting, `alarm_rate` is the probability of any false alarm across the looks, and `eve_p_z` and `redraw_probability` are placeholders.

The [archived pilot notebook](../archive/qft_pilot/QFT_Factorial_Pilot.ipynb) loads and plots this CSV. It supports checkpoint resume for reproduction. The [active QFT notebook](../QFT_Repeated_Secret.ipynb) runs the focused paired experiment with exact phase averaging.

## Scope

This pilot used one joint-pattern alarm, fixed noise within a setting, and a cap of 100 retained checks (25 attempts). The separate drift demo measures Bob's alarms over 500 attempts using seeded, bounded, time-correlated channel rates. The saved sweep contains setting means.

The pilot's 16-sample phase approximation can make an Eve observation from the return leg alone appear informative about the secret. The active comparison uses the exact phase average and a separate 100-attempt schedule. Its figures describe the selected configuration. [Archived posterior calculation](../archive/qft_pilot/qft_inference.py) · [Current exchange](../quantum_study/qft_exchange.py)
