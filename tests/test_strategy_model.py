"""Verify the timing rules that keep the strategy replay forward-only."""

import json
import unittest
from pathlib import Path

import numpy as np

from prepare_strategy_model import (HORIZON, REBALANCE_EVERY, PastOnlyScaler,
                                    known_training_indices, load_omo_policy, make_labels)


ROOT = Path(__file__).resolve().parents[1]


class StrategyTimingTests(unittest.TestCase):
    def test_label_starts_after_signal_and_enters_training_only_after_exit(self):
        levels = np.arange(20, dtype=float) ** 2
        labels = make_labels(levels, horizon=HORIZON)
        self.assertEqual(labels[0], levels[6] - levels[0])
        self.assertEqual(labels[3], levels[9] - levels[3])
        self.assertTrue(np.isnan(labels[-6:]).all())

        valid = np.isfinite(labels)
        # At close t=6, only the trade signalled at t=0 has completed.
        self.assertEqual(known_training_indices(6, valid).tolist(), [0])
        self.assertEqual(known_training_indices(7, valid).tolist(), [0, 1])

    def test_scaler_uses_fit_history_only(self):
        scaler = PastOnlyScaler().fit([[1.0, 2.0], [2.0, np.nan], [3.0, 4.0]])
        before = (scaler.median.copy(), scaler.lo.copy(), scaler.hi.copy(), scaler.mean.copy())
        scaler.transform([[100000.0, np.nan]])
        after = (scaler.median, scaler.lo, scaler.hi, scaler.mean)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)

    def test_published_paths_have_nonoverlapping_holding_periods(self):
        data = json.loads((ROOT / "strategy_model.json").read_text(encoding="utf-8"))
        self.assertEqual(data["meta"]["factor_count"], 14)
        for target in data["targets"].values():
            self.assertEqual(len(target["selected"]), 7)
            daily = {row["asof"]: row for row in target["daily_predictions"]}
            previous_exit = None
            for trade in target["trades"]:
                source = daily[trade["asof"]]
                self.assertEqual((trade["entry"], trade["exit"]), (source["entry"], source["exit"]))
                if previous_exit:
                    self.assertLess(previous_exit, trade["entry"])
                previous_exit = trade["exit"]
            self.assertEqual(len(target["phase_robustness"]), REBALANCE_EVERY)
            self.assertEqual(len(target["market_factors"]), 4)
            self.assertEqual(len(target["controls"]), 4)

    def test_omo_changes_are_effective_on_announced_dates(self):
        dates = ["2024-09-26", "2024-09-27", "2024-09-30", "2025-05-07", "2025-05-08"]
        rates, _ = load_omo_policy(dates)
        np.testing.assert_array_equal(rates, [1.70, 1.50, 1.50, 1.50, 1.40])


if __name__ == "__main__":
    unittest.main()
