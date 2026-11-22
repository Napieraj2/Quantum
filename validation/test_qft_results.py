import unittest
from pathlib import Path

import pandas as pd

from quantum_study.qft_results import build_qft_summary, prepare_qft_results


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "qft_factorial_results.csv"


class QFTResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = pd.read_csv(RESULT_PATH)

    def test_completed_grid_is_unique(self):
        prepared = prepare_qft_results(self.results)
        self.assertEqual(len(prepared), 1_210)
        self.assertEqual(prepared["setting_key"].nunique(), 1_210)
        self.assertEqual((~prepared["is_attack"]).sum(), 22)
        self.assertEqual(prepared["is_attack"].sum(), 1_188)

    def test_outcomes_and_audit_match_saved_run(self):
        summary = build_qft_summary(
            self.results,
            expected_settings=1_210,
            target_false_alarm=0.05,
        )
        metrics = summary["metrics"]
        self.assertEqual(metrics["attack_trials"], 118_800)
        self.assertEqual(metrics["recovery_events"], 0)
        self.assertEqual(metrics["false_alarm_events"], 160)
        self.assertEqual(metrics["false_alarm_trials"], 2_200)

        prepared = summary["enriched"]
        total = (
            prepared["alarm_before_recovery_rate"]
            + prepared["any_recovery_rate"]
            + prepared["inconclusive_rate"]
        )
        self.assertTrue(total.between(1 - 1e-12, 1 + 1e-12).all())


if __name__ == "__main__":
    unittest.main()
