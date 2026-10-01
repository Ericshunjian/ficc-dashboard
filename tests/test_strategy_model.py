"""Verify the timing rules that keep the strategy replay forward-only."""

import json
import unittest
from pathlib import Path

import numpy as np

from prepare_strategy_model import HORIZON, PastOnlyScaler, known_training_indices, make_labels


ROOT = Path(__file__).resolve().parents[1]


class StrategyTimingTests(unittest.TestCase):
    def test_label_starts_after_signal_and_enters_training_only_after_exit(self):
        levels = np.arange(20, dtype=float) ** 2
        labels = make_labels(levels, horizon=HORIZON)
        self.assertEqual(labels[0], levels[6] - levels[1])
        self.assertEqual(labels[3], levels[9] - levels[4])
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
                    self.assertLessEqual(previous_exit, trade["entry"])
                previous_exit = trade["exit"]
            self.assertEqual(len(target["phase_robustness"]), HORIZON)


if __name__ == "__main__":
    unittest.main()
