import json
import unittest
from pathlib import Path

import numpy as np

from prepare_curve_timing_percentile import signed_percentile_gate
from prepare_curve_timing_walkforward import simulate

ROOT = Path(__file__).resolve().parents[1]


class SignedPercentileTests(unittest.TestCase):
    def test_relative_direction_overrides_positive_negative_and_zero_forecast(self):
        for past, today, direction in (([1, 2, 3, 4, 5], .1, -1),
                                       ([-5, -4, -3, -2, -1], -.1, 1),
                                       ([1, 2, 3, 4, 5], 0, -1)):
            gate = signed_percentile_gate(past + [today], window=5, min_history=5)
            self.assertEqual(gate["signal"][-1], direction)

    def test_boundaries_and_ties_are_flat(self):
        for today, rank in ((2.5, 40), (3.5, 60)):
            gate = signed_percentile_gate([1, 2, 3, 4, 5, today], window=5, min_history=5)
            self.assertEqual(gate["percentile"][-1], rank)
            self.assertEqual(gate["signal"][-1], 0)
        gate = signed_percentile_gate(np.ones(65))
        self.assertEqual(gate["percentile"][60], 50)
        self.assertEqual(gate["signal"][60], 0)

    def test_history_excludes_today_future_and_missing_calendar_sessions(self):
        rng = np.random.default_rng(42)
        pred = rng.normal(size=400)
        pred[270:300] = np.nan
        before = signed_percentile_gate(pred)
        self.assertTrue(np.isnan(before["percentile"][:60]).all())
        self.assertEqual(before["history_n"][300], 220)
        pred[301:] *= -1000
        after = signed_percentile_gate(pred)
        for key in before:
            if key != "direction_basis":
                np.testing.assert_array_equal(before[key][:301], after[key][:301])

    def test_dynamic_exit_follows_percentile_direction_and_pnl_before_reversal(self):
        dates = [f"d{i}" for i in range(5)]
        gate = {"direction_basis": "percentile", "percentile": np.array([20, 20, 80, 50, 20]),
                "signal": np.array([-1, -1, 1, 0, -1]), "history_n": np.full(5, 60)}
        pred = np.ones(5)
        sim = simulate(dates, np.array([0, 1, 11, 12, 13.]), pred, np.ones(5), rule="dynamic", gate=gate)
        self.assertEqual(sim["trades"][0]["exit_signal_date"], "d2")
        self.assertEqual(sim["trades"][0]["reason"], "reverse")
        self.assertEqual(sim["position"][2], -1)
        self.assertEqual(sim["daily_pnl"][2], -10)
        self.assertEqual(sim["position"][3], 1)
        self.assertEqual(sim["trades"][1]["reason"], "weak")
        self.assertEqual(sim["decisions"][-1]["next_position"], -1)
        self.assertEqual(sum(t["pnl"] for t in sim["trades"]), sim["equity"][-1])

    def test_latest_plan_does_not_realize_exit_or_next_trade(self):
        gate = {"direction_basis": "percentile", "percentile": np.array([20, 20, 80]),
                "signal": np.array([-1, -1, 1]), "history_n": np.full(3, 60)}
        sim = simulate(["d0", "d1", "d2"], np.arange(3.), np.ones(3), np.ones(3), rule="dynamic", gate=gate)
        self.assertEqual(len(sim["trades"]), 1)
        self.assertFalse(sim["trades"][0]["closed"])
        self.assertEqual(sim["decisions"][-1]["next_position"], 1)
        self.assertEqual(sim["equity"][-1], -2)

    def test_published_comparison_reuses_models_and_reconciles_accounting(self):
        path = ROOT / "curve_timing_percentile.json"
        if not path.exists():
            self.skipTest("not generated yet")
        d = json.loads(path.read_text(encoding="utf-8"))
        source = json.loads((ROOT / "curve_timing_walkforward.json").read_text(encoding="utf-8"))
        self.assertEqual(d["daily"], {k: source["daily"][k] for k in d["daily"]})
        self.assertEqual(d["source_models"], source["refits"][0]["models"])
        for method, rules in d["strategies"].items():
            for rule, arms in rules.items():
                for arm, sim in arms.items():
                    self.assertAlmostEqual(sum(t["pnl"] for t in sim["trades"]), sim["equity"][-1])
                    np.testing.assert_allclose(np.cumsum(sim["daily_pnl"]), sim["equity"], atol=1e-12)
                    self.assertTrue(all(t["holding_sessions"] <= 6 for t in sim["trades"]))
                    for t in sim["trades"]:
                        if method == "signed":
                            self.assertTrue(t["entry_percentile"] < 40 if t["direction"] == -1 else t["entry_percentile"] > 60)
                    if method == "absolute":
                        self.assertEqual(sim, source["strategies"][rule][arm])


if __name__ == "__main__":
    unittest.main()
