"""Causal training, fair model comparison and executable close-proxy accounting."""
import json
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from prepare_curve_timing_walkforward import (completed_indices, forecast_metrics,
                                            fit_once_and_test, forecast_percentile_gate, position_for, simulate, walk_forward)
from prepare_direction_screen import RULES, rolling_spearman, screen_snapshot
from prepare_strategy_model import make_offset_labels, ridge_fit, ridge_predict

ROOT = Path(__file__).resolve().parents[1]


def feature(key, values):
    return {"id": key, "key": key, "group": "test", "label": key, "unit": "", "formula": "test", "values": values}


def empty_screen(dates, level, candidates, t):
    return {"selected_ids": [], "rows": [], "summary": {"selected": 0},
            "window": {"last_label_exit": dates[t]}}, None


class CurveTimingTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(48)
        dates = pd.bdate_range("2023-04-21", periods=420).strftime("%Y-%m-%d").tolist()
        level = 200 + np.cumsum(rng.normal(size=420))
        base = [feature("baseline|x", rng.normal(size=420)), feature("baseline|lag", np.r_[np.nan, level[:-1]])]
        return dates, level, base

    def test_six_day_label_includes_boundary_but_not_incomplete_label(self):
        level = np.arange(30, dtype=float)
        y = make_offset_labels(level, 6)
        idx = completed_indices(level, y, 20)
        self.assertEqual(idx[-1], 14)
        self.assertEqual(idx[-1] + 6, 20)
        level[15] = np.nan
        idx = completed_indices(level, make_offset_labels(level, 6), 22)
        self.assertNotIn(15, idx)
        self.assertEqual(idx[-1], 16)

    def test_future_changes_cannot_alter_historical_screen_or_fitted_forecast(self):
        dates, level, base = self.fixture()
        # This synthetic feature makes selection nonempty, exercising the full-model path.
        c = [feature("candidate", make_offset_labels(level, 6) + np.random.default_rng(39).normal(scale=.5, size=len(level)))]
        before = walk_forward(dates, level, base, c)
        level[351:] += 100000
        for f in base + c:
            f["values"][351:] -= 100000
        after = walk_forward(dates, level, base, c)
        for arm in ("baseline", "full"):
            np.testing.assert_array_equal(before["predictions"][arm][:351], after["predictions"][arm][:351])
        past_before = [r for r in before["refits"] if r["index"] <= 350]
        past_after = [r for r in after["refits"] if r["index"] <= 350]
        self.assertTrue(any(r["selected_ids"] for r in past_before))
        self.assertEqual(past_before, past_after)
        for r in before["refits"]:
            self.assertLessEqual(r["last_label_exit"], r["date"])

    def test_cached_trailing_ic_gives_same_selection_as_uncached_snapshot(self):
        dates, level, base = self.fixture()
        c = [feature("x", base[0]["values"])]
        y = make_offset_labels(level, 6)
        cached = rolling_spearman(c[0]["values"][:, None], y, RULES["icir_window"], RULES["icir_min_pairs"])
        regular, _ = screen_snapshot(dates, level, c, 350)
        fast, _ = screen_snapshot(dates, level, c, 350, precomputed_rolling=cached)
        self.assertEqual(regular, fast)

    def test_empty_selection_produces_identical_models_on_identical_training_dates(self):
        dates, level, base = self.fixture()
        result = walk_forward(dates, level, base, [], screen_fn=empty_screen)
        np.testing.assert_array_equal(result["predictions"]["baseline"], result["predictions"]["full"])
        self.assertEqual(result["refits"][0]["train_n"], 300)
        self.assertEqual(result["refits"][1]["train_n"], 320)
        self.assertEqual(result["refits"][0]["models"]["baseline"], result["refits"][0]["models"]["full"])

    def test_ridge_matches_reference_on_training_scaled_matrix(self):
        rng = np.random.default_rng(52)
        x = rng.normal(size=(320, 5))
        x[0, 0] = np.nan
        y = x[:, 1] * 0.3 + rng.normal(size=320)
        fitted = ridge_fit(x, y, 15)
        scaler, _, coef = fitted
        reference = Ridge(alpha=15).fit(scaler.transform(x), y)
        np.testing.assert_allclose(coef, reference.coef_, atol=1e-12)
        np.testing.assert_allclose(ridge_predict(fitted, x), reference.predict(scaler.transform(x)), atol=1e-12)

    def test_frozen_list_is_never_rescreened_and_coefficients_still_refit(self):
        dates, level, base = self.fixture()
        c = [feature("fixed", base[0]["values"].copy())]
        frozen = {"selected_ids": ["fixed"], "rows": [c[0] | {"values": None}],
                  "summary": {}, "window": {"start": "selected using full history"}}
        def forbidden_screen(*args, **kwargs):
            self.fail("固定名单时不应调用筛选函数")
        result = walk_forward(dates, level, base, c, frozen_snapshot=frozen, screen_fn=forbidden_screen)
        self.assertGreater(len(result["refits"]), 1)
        self.assertTrue(all(r["selected_ids"] == ["fixed"] for r in result["refits"]))
        self.assertNotEqual(result["refits"][0]["models"]["full"]["intercept"], result["refits"][1]["models"]["full"]["intercept"])

    def test_percentile_excludes_today_and_requires_strictly_more_than_sixty(self):
        pred = np.array([1, 2, 3, 4, 5, 3.5, -4.5], float)
        gate = forecast_percentile_gate(pred, window=5, min_history=5)
        self.assertTrue(np.isnan(gate["percentile"][:5]).all())
        self.assertEqual(gate["history_n"][5], 5)
        self.assertEqual(gate["percentile"][5], 60)
        self.assertEqual(gate["signal"][5], 0)
        self.assertEqual(gate["percentile"][6], 80)
        self.assertEqual(gate["signal"][6], -1)

    def test_static_model_fits_once_per_arm_and_test_does_not_change_fit_or_scaler(self):
        dates, level, base = self.fixture()
        c = [feature("fixed", base[0]["values"].copy())]
        frozen = {"selected_ids": ["fixed"], "rows": [c[0] | {"values": None}],
                  "summary": {}, "window": {}}
        with patch("prepare_curve_timing_walkforward.ridge_fit", wraps=ridge_fit) as fitter:
            before = fit_once_and_test(dates, level, base, c, frozen, train_through=dates[340])
            self.assertEqual(fitter.call_count, 2)
        self.assertEqual(len(before["refits"]), 1)
        self.assertEqual(before["refits"][0]["train_end"], dates[334])
        self.assertTrue(np.isnan(before["predictions"]["full"][:341]).all())
        self.assertTrue(np.all(before["refit_index"][341:] == 0))
        level[361:] += 10000
        for f in base + c:
            f["values"][361:] -= 10000
        after = fit_once_and_test(dates, level, base, c, frozen, train_through=dates[340])
        self.assertEqual(before["refits"], after["refits"])
        for arm in ("baseline", "full"):
            np.testing.assert_array_equal(before["predictions"][arm][:361], after["predictions"][arm][:361])
        gate = forecast_percentile_gate(before["predictions"]["full"])
        self.assertTrue(np.isnan(gate["percentile"][:401]).all())
        self.assertEqual(gate["history_n"][401], 60)

    def test_percentile_future_changes_cannot_alter_past_and_abs_sign_is_respected(self):
        rng = np.random.default_rng(21)
        pred = rng.normal(size=420)
        before = forecast_percentile_gate(pred)
        pred[301:] *= 100000
        after = forecast_percentile_gate(pred)
        for key in before:
            np.testing.assert_array_equal(before[key][:301], after[key][:301])
        flat = forecast_percentile_gate(np.ones(65))
        self.assertEqual(flat["percentile"][60], 50)
        self.assertEqual(flat["signal"][60], 0)
        all_zero = forecast_percentile_gate(np.zeros(65))
        self.assertFalse(all_zero["signal"].any())

    def test_percentile_window_uses_calendar_sessions_and_does_not_compress_missing(self):
        pred = np.ones(300)
        pred[260:290] = np.nan
        pred[299] = -2
        gate = forecast_percentile_gate(pred)
        self.assertEqual(gate["history_n"][299], 220)
        self.assertEqual(gate["percentile"][299], 100)
        self.assertEqual(gate["signal"][299], -1)

    def test_percentile_gate_drives_entries_and_dynamic_weak_exit(self):
        dates = [f"d{i}" for i in range(9)]
        pred = np.array([1, 2, 3, 4, 5, 3.5, 10, 1, 1], float)
        gate = forecast_percentile_gate(pred, window=5, min_history=5)
        sim = simulate(dates, np.arange(9, dtype=float), pred, gate["quantile"], rule="dynamic", gate=gate)
        self.assertEqual(sim["trades"][0]["signal_date"], "d6")
        self.assertEqual(sim["trades"][0]["entry_percentile"], 100)
        self.assertEqual(sim["trades"][0]["reason"], "weak")
        self.assertEqual(sim["trades"][0]["exit_signal_date"], "d7")
        self.assertEqual(sim["position"][7], 1)
        self.assertEqual(sim["position"][8], 0)

    def test_fixed_hold_is_exactly_six_intervals_and_daily_pnl_reconciles(self):
        dates = [f"d{i}" for i in range(15)]
        level = np.arange(15, dtype=float)
        pred, threshold = np.ones(15), np.full(15, 0.2)
        sim = simulate(dates, level, pred, threshold)
        self.assertEqual(sim["trades"][0]["holding_sessions"], 6)
        self.assertEqual(sim["trades"][0]["entry_session"], "d1")
        self.assertEqual(sim["trades"][0]["exit_session"], "d6")
        self.assertEqual(sim["trades"][0]["pnl"], 6)
        self.assertEqual(sim["position"][0], 0)
        self.assertEqual(sim["position"][1], 1)
        self.assertEqual(sum(t["pnl"] for t in sim["trades"]), sim["equity"][-1])
        self.assertLessEqual(sim["metrics"]["max_holding"], 6)

    def test_dynamic_reversal_occurs_after_signal_close_pnl_and_can_reverse(self):
        dates = [f"d{i}" for i in range(8)]
        level = np.array([0, 1, 11, 12, 13, 14, 15, 16], float)
        pred = np.array([1, 1, -1, -1, -1, -1, -1, -1], float)
        threshold = np.full(8, 0.2)
        sim = simulate(dates, level, pred, threshold, rule="dynamic")
        self.assertEqual(sim["position"][2], 1)
        self.assertEqual(sim["daily_pnl"][2], 10)
        self.assertEqual(sim["position"][3], -1)
        self.assertEqual(sim["trades"][0]["exit_signal_date"], "d2")
        self.assertEqual(sim["trades"][0]["exit_session"], "d3")
        self.assertEqual(sim["trades"][0]["pnl"], 11)
        self.assertEqual(sum(t["pnl"] for t in sim["trades"]), sim["equity"][-1])

    def test_flat_threshold_weak_exit_and_last_close_plan_not_realized_trade(self):
        self.assertEqual(position_for(0.2, 0.2), 0)
        self.assertEqual(position_for(np.nan, 0.2), 0)
        dates = [f"d{i}" for i in range(4)]
        pred = np.array([1, 1, 1, -1], float)
        sim = simulate(dates, np.arange(4, dtype=float), pred, np.full(4, .2), rule="dynamic")
        self.assertEqual(sim["trades"][0]["reason"], "open")
        self.assertEqual(sim["decisions"][-1]["next_position"], -1)
        self.assertEqual(sim["position"][-1], 1)
        weak = simulate(dates, np.arange(4, dtype=float), np.array([1, .1, 0, 0]), np.full(4, .2), rule="dynamic")
        self.assertEqual(weak["trades"][0]["reason"], "weak")
        self.assertEqual(weak["position"][2], 0)

    def test_drawdown_includes_zero_origin_and_missing_held_prices_fail(self):
        dates = [f"d{i}" for i in range(4)]
        sim = simulate(dates, np.array([10, 7, 9, 6]), np.ones(4), np.full(4, .2))
        self.assertEqual(sim["metrics"]["max_drawdown"], 4)
        with self.assertRaises(ValueError):
            simulate(dates, np.array([10, np.nan, 9, 6]), np.ones(4), np.full(4, .2))

    def test_published_artifact_labels_selections_and_pnl(self):
        path = ROOT / "curve_timing_walkforward.json"
        if not path.exists():
            self.skipTest("artifact not generated yet")
        d = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(all(v is None for v in d["daily"]["target"][-6:]))
        self.assertEqual(d["meta"]["baseline_count"], 8)
        fixed = json.loads((ROOT / "curve_timing_factor_set.json").read_text(encoding="utf-8"))
        self.assertEqual(d["meta"]["selection_mode"], "fixed-full-history")
        self.assertEqual(d["meta"]["training_mode"], "fixed-train-test")
        self.assertEqual(len(d["refits"]), 1)
        self.assertEqual(d["refits"][0]["date"], "2024-12-31")
        self.assertEqual(d["refits"][0]["train_n"], 406)
        self.assertEqual(d["refits"][0]["train_end"], "2024-12-23")
        self.assertEqual(d["daily"]["dates"][0], "2025-01-02")
        self.assertEqual(d["forecast_metrics"]["full"]["n"], 420)
        self.assertTrue(all(i == 0 for i in d["daily"]["refit_index"]))
        self.assertTrue(all(v is None for v in d["daily"]["percentile"]["full"][:60]))
        for r in d["refits"]:
            self.assertEqual(r["selected_ids"], fixed["selected_ids"])
            self.assertLessEqual(r["last_label_exit"], r["date"])
            self.assertLessEqual(len(r["selected_ids"]), 20)
            self.assertEqual(len(r["models"]["full"]["features"]), 8 + len(r["selected_ids"]))
            for row in r["selected"]:
                self.assertTrue(all(row["checks"].values()))
        for strategies in d["strategies"].values():
            for s in strategies.values():
                self.assertAlmostEqual(sum(t["pnl"] for t in s["trades"]), s["equity"][-1])
                self.assertTrue(set(s["position"]) <= {-1, 0, 1})
                self.assertTrue(all(t["holding_sessions"] <= 6 for t in s["trades"]))
                self.assertTrue(all(t["entry_percentile"] > 60 and t["entry_history_n"] >= 60 for t in s["trades"]))
                np.testing.assert_allclose(np.cumsum(s["daily_pnl"]), s["equity"], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
