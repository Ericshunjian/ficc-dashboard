"""Frozen-model reproduction, causal gates, data guards and daily archive checks."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import prepare_ridge_daily as daily
from prepare_curve_timing_percentile import signed_percentile_gate
from prepare_direction_screen import baseline_features, load_candidates
from research_inputs import research_input


class DailyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = daily.read(daily.MODELS)
        cls.report = daily.read(daily.OUTPUT)

    def test_all_archived_predictions_reproduce_without_fitting(self):
        old = daily.read("curve_timing_capacity.json")
        snapshot = research_input("factor_library.json", old["meta"]["input_hashes"]["factor_library.json"])
        with patch("prepare_factor_checkup.LIB_PATH", str(snapshot)), patch("prepare_strategy_model.BASE", str(snapshot.parent)):
            _, days, curve, candidates = load_candidates()
            baselines, _ = baseline_features(3 * curve["T主力"] - curve["TL主力"], curve, days)
        features = {f["id"]: f["values"] for f in candidates + baselines}
        axis = {d: i for i, d in enumerate(days)}
        for arm in daily.ARMS:
            model = self.frozen["models"][arm]
            for i, day in enumerate(old["daily"]["dates"]):
                expected = old["daily"]["prediction"][arm][i]
                if expected is None: continue
                values = [features[f["id"]][axis[day]] for f in model["features"]]
                result, _ = daily.infer(model, values)
                self.assertAlmostEqual(result, expected, places=11)

    def test_baselines_are_never_clipped_even_when_extrapolating(self):
        model = copy.deepcopy(self.frozen["models"]["f20"])
        values = [f["mean"] for f in model["features"]]
        values[4] = 1000
        _, audit = daily.infer(model, values)
        self.assertEqual(audit[4]["processed"], 1000)
        self.assertFalse(audit[4]["clipped"])
        self.assertTrue(all(not f["clip_enabled"] for f in model["features"][:8]))

    def test_missing_is_filled_only_with_frozen_train_median(self):
        model = self.frozen["models"]["f20"]
        values = [np.nan] * len(model["features"])
        _, audit = daily.infer(model, values)
        for f, a in zip(model["features"], audit):
            self.assertEqual(a["filled"], f["median"])
            self.assertTrue(a["missing"])

    def test_gate_excludes_current_and_obeys_strict_boundaries(self):
        values = np.r_[np.arange(250), 99.5, 1e9]
        gate = signed_percentile_gate(values)
        self.assertEqual(gate["percentile"][250], 40.)
        self.assertEqual(gate["signal"][250], 0)
        self.assertEqual(gate["signal"][251], 1)
        values[251] = -1e9
        again = signed_percentile_gate(values)
        self.assertEqual(again["percentile"][250], 40.)
        self.assertEqual(again["signal"][251], -1)

    def test_guards_do_not_use_stale_policy_or_absent_daily_report(self):
        info = {"omo_verified": "2026-10-09"}
        coverage, problems = daily.quality("2026-10-12", 259, np.ones(8), np.ones(20), info, set(), set())
        self.assertEqual(coverage, 1.)
        self.assertEqual(len(problems), 3)
        coverage, problems = daily.quality("2026-10-12", np.nan, np.r_[np.nan, np.ones(7)],
                                           np.r_[np.ones(17), np.full(3, np.nan)], info, set(), set())
        self.assertEqual(len(problems), 6)
        self.assertLess(coverage, .9)

    def test_frozen_membership_and_coefficients_match_authority(self):
        source = daily.read("curve_timing_capacity.json")
        daily.validate_models(self.frozen)
        for arm in daily.ARMS:
            self.assertEqual(self.frozen["models"][arm], source["models"][arm])
        self.assertEqual(self.report["meta"]["model_sha256"], daily.sha(daily.MODELS))

    def test_live_inputs_explain_full_prediction_and_baseline_quality(self):
        for arm in daily.ARMS:
            rec = self.report["latest"]["models"][arm]
            self.assertEqual(rec["status"], "issued")
            self.assertAlmostEqual(rec["prediction"], self.frozen["models"][arm]["intercept"] +
                                   sum(f["contribution"] for f in rec["inputs"]), places=12)
            self.assertTrue(all(not f["missing"] for f in rec["inputs"][:8]))

    def test_issued_records_and_signals_are_not_overwritten(self):
        rows = copy.deepcopy(self.report["records"])
        issued = copy.deepcopy([r for r in rows if r["origin"] == "live-issued"])
        daily.enrich(rows)
        self.assertEqual(issued, [r for r in rows if r["origin"] == "live-issued"])

    def test_current_neutral_exits_are_plans_not_next_day_realized_trades(self):
        rec = self.report["latest"]["models"]["f20"]
        if rec["signal"] == 0:
            self.assertEqual(self.report["positions"]["f20"]["next_position"], 0)
        trade = self.report["positions"]["f20"]["open_trade"]
        if trade:
            self.assertFalse(trade["closed"])
            self.assertIsNone(trade["exit_session"])

    def test_daily_charts_and_ledger_reconcile_with_saved_position_summary(self):
        for arm in daily.ARMS:
            sim = self.report["strategies"][arm]
            self.assertEqual(sim["metrics"], self.report["positions"][arm]["metrics"])
            self.assertEqual(len(sim["equity"]), len(self.report["records"]))
            np.testing.assert_allclose(np.cumsum(sim["daily_pnl"]), sim["equity"], atol=1e-9)
            self.assertAlmostEqual(sum(t["pnl"] for t in sim["trades"]), sim["equity"][-1], places=8)
            self.assertAlmostEqual(-min(sim["drawdown"]), sim["metrics"]["max_drawdown"], places=5)
            self.assertEqual(sim["decisions"][-1]["next_position"], self.report["positions"][arm]["next_position"])

    def test_rerun_is_idempotent_and_never_calls_training(self):
        names = (daily.MODELS, daily.OUTPUT, "factor_library.json", "data_version.json",
                 "curve_timing_logistic_models.json", "curve_timing_logistic.json")
        before = {n: daily.sha(n) for n in names}
        with patch("prepare_curve_timing_capacity.fit_model", side_effect=AssertionError("Unexpected fit")), \
             patch("prepare_direction_screen.screen_snapshot", side_effect=AssertionError("Unexpected screening")), \
             patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("Unexpected Logistic fit")):
            _, changes = daily.build()
        self.assertEqual(changes["new_signal_dates"], [])
        self.assertEqual(changes["new_logistic_dates"], [])
        self.assertFalse(changes["signal_changed"])
        self.assertFalse(changes["library_changed"])
        self.assertEqual(before, {n: daily.sha(n) for n in names})

    def test_all_comparison_charts_update_with_daily_signal(self):
        import prepare_curve_comparison_daily as comparison
        current = self.report['comparison']
        dates = [r['date'] for r in self.report['records']]
        self.assertEqual(current['daily']['dates'], dates)
        self.assertEqual(current['meta']['asof'], self.report['meta']['signal_date'])
        self.assertTrue(current['meta']['healthy'])
        for name, digest in current['meta']['frozen_hashes'].items():
            self.assertEqual(digest, daily.sha(name))
        for rule in ('dynamic', 'fixed'):
            self.assertEqual(set(current['strategies'][rule]), set(comparison.KEYS))
            for sim in current['strategies'][rule].values():
                self.assertEqual(len(sim['equity']), len(dates))
                self.assertEqual(len(sim['drawdown']), len(dates))
                self.assertEqual(len(sim['position']), len(dates))
                self.assertAlmostEqual(sum(sim['daily_pnl']), sim['equity'][-1], places=8)

    def test_parallel_logistic_signal_uses_frozen_probability_gate(self):
        import prepare_logistic_daily as logistic
        frozen = daily.read(logistic.MODELS)
        report = self.report["logistic"]
        logistic.validate_models(frozen)
        self.assertEqual(report["meta"]["model_sha256"], daily.sha(logistic.MODELS))
        for arm in logistic.ARMS:
            rec = report["latest"]["models"][arm]
            score = frozen["models"][arm]["intercept"] + sum(f["contribution"] for f in rec["inputs"])
            self.assertAlmostEqual(rec["prediction"], logistic.probability(score), places=12)
            expected = 1 if rec["prediction"] > .6 else -1 if rec["prediction"] < .4 else 0
            self.assertEqual(rec["signal"], expected)
            self.assertEqual(report["strategies"][arm]["metrics"], report["positions"][arm]["metrics"])


if __name__ == "__main__":
    unittest.main()
