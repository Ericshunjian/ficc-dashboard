"""Regression checks for paired rank correlation and fractional-position accounting."""

import json
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from prepare_strategy_model_v2 import (_colwise_spearman, simulate_exit_rule,
                                       exit_daily_path)

ROOT = Path(__file__).resolve().parents[1]


class V2CalculationTests(unittest.TestCase):
    def test_spearman_is_one_for_nonlinear_monotone_target(self):
        x = np.arange(100, dtype=float)
        y = np.exp(x / 10)
        self.assertAlmostEqual(_colwise_spearman(x[:, None], y)[0], 1.0)

    def test_spearman_uses_each_columns_paired_observations_and_ties(self):
        rng = np.random.default_rng(123)
        x = rng.integers(0, 12, (120, 3)).astype(float)
        y = rng.integers(0, 8, 120).astype(float) ** 3
        x[::4, 0] = np.nan
        x[::5, 1] = np.nan
        y[::9] = np.nan
        actual = _colwise_spearman(x, y)
        for j in range(x.shape[1]):
            mask = np.isfinite(x[:, j]) & np.isfinite(y)
            self.assertAlmostEqual(actual[j], spearmanr(x[mask, j], y[mask]).statistic)

    def test_exit_curve_keeps_fractional_short_position(self):
        dates = [f"d{i}" for i in range(7)]
        level = np.array([100, 104, 106, 102, 103, 101, 108], dtype=float)
        base = [{"asof": "d0", "entry": "d1", "position": -0.25}]
        trades = simulate_exit_rule(base, [], level, dates, "fixed")
        path = exit_daily_path(trades, level, dates, "d6")
        self.assertEqual(trades[0]["position"], -0.25)
        self.assertEqual(trades[0]["pnl"], -2)
        self.assertEqual(path[-1]["equity"], -2)

    def test_early_exit_curve_uses_weight_and_stops_exposure(self):
        dates = [f"d{i}" for i in range(7)]
        level = np.array([100, 105, 200, 0, 0, 0, 0], dtype=float)
        base = [{"asof": "d0", "entry": "d1", "position": 0.4}]
        signals = [{"asof": "d1", "pred": -1, "train_target_std": 2,
                    "short1_pred": -0.1, "short2_pred": -0.1}]
        trades = simulate_exit_rule(base, signals, level, dates, "long_reverse")
        path = exit_daily_path(trades, level, dates, "d6")
        self.assertEqual(trades[0]["exit"], "d2")
        self.assertEqual([d["equity"] for d in path], [0, 2, 2, 2, 2, 2, 2])

    def test_published_exit_equity_matches_trade_ledger(self):
        data = json.loads((ROOT / "strategy_model_v2.json").read_text(encoding="utf-8"))
        self.assertIn("calculation_revision", data["meta"])
        for target in data["targets"].values():
            main = target["variants"]["rolling|k10|quantile"]
            for rule in main["exit_comparison"]["rules"]:
                self.assertAlmostEqual(rule["daily_path"][-1]["equity"],
                                       rule["metrics"]["total"], delta=0.02)
                self.assertAlmostEqual(sum(t["pnl"] for t in rule["trades"]),
                                       rule["daily_path"][-1]["equity"], delta=0.02)
            fixed = main["exit_comparison"]["rules"][0]
            self.assertAlmostEqual(main["metrics"]["model"]["total"],
                                   fixed["metrics"]["total"], delta=0.002)


if __name__ == "__main__":
    unittest.main()
