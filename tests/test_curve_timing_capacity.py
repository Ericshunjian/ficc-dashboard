import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from prepare_curve_timing_capacity import (ALL_VARIANTS, TypeAwareScaler, clip_policy,
                                          fit_model, fit_variants, model_record, predict)

ROOT = Path(__file__).resolve().parents[1]


def feature(key, values):
    return {"id": key, "label": key, "unit": "", "values": np.array(values, float)}


class CapacityTests(unittest.TestCase):
    def test_prices_spreads_technical_and_bounded_features_never_clip(self):
        protected = ["baseline|spread_ma5", "baseline|spread_ma20", "baseline|spread_mom5",
                     "baseline|spread_z60", "baseline|y30_y10", "curve|T主力", "curve|10年国债",
                     "carry|价差|s3010", "tech|T|pct250", "tech|TF|rsi14",
                     "derived|days_ratio|券商|3-5年|国债", "derived|tenor_zdiff|保险|7-10|国债"]
        fs = [feature(key, []) for key in protected]
        scaler = TypeAwareScaler(fs).fit(np.tile(np.arange(100.)[:, None], (1, len(fs))))
        test = np.tile(np.array([-1000., 260., 1000.])[:, None], (1, len(fs)))
        np.testing.assert_array_equal(scaler.apply_limits(test), test)
        np.testing.assert_allclose(scaler.transform(test) * scaler.std + scaler.mean, test, atol=1e-12)
        self.assertFalse(scaler.clip_mask.any())
        self.assertTrue(np.isnan(scaler.lo).all())

    def test_only_amount_features_clip_and_missing_values_use_train_median(self):
        fs = [feature("baseline|spread_ma5", []), feature("cash|国债|券商|1-3年", [])]
        x = np.column_stack([np.arange(100.), np.arange(100.)])
        scaler = TypeAwareScaler(fs).fit(x)
        test = np.array([[260., 10000.], [np.nan, np.nan]])
        restored = scaler.transform(test) * scaler.std + scaler.mean
        self.assertEqual(restored[0, 0], 260.)
        self.assertAlmostEqual(restored[0, 1], 98.01)
        np.testing.assert_allclose(restored[1], [49.5, 49.5])
        self.assertEqual(clip_policy({"id": "repo|中小银行|rate_spread"})[0], False)
        with self.assertRaises(ValueError):
            clip_policy({"id": "unknown|price"})

    def test_fit_matches_sklearn_on_type_aware_matrix_and_records_reproduce(self):
        rng = np.random.default_rng(18)
        x = rng.normal(size=(320, 2));x[0, 1] = 1000
        fs = [feature("baseline|spread_ma5", []), feature("cash|国债|券商|1-3年", [])]
        y = 2 * x[:, 0] + rng.normal(size=320)
        model = fit_model(x, y, fs)
        scaler = model[0]
        reference = Ridge(alpha=15).fit(scaler.transform(x), y)
        np.testing.assert_allclose(predict(model, x), reference.predict(scaler.transform(x)), atol=1e-12)
        record = model_record(model, fs)
        self.assertIsNone(record["features"][0]["clip_high"])
        test = np.array([[500., 1000.]])
        z = []
        for j, f in enumerate(record["features"]):
            value = test[0, j]
            if f["clip_enabled"]:
                value = np.clip(value, f["clip_low"], f["clip_high"])
            z.append((value - f["mean"]) / f["std"])
        expected = record["intercept"] + np.array(z) @ np.array([f["coef"] for f in record["features"]])
        self.assertAlmostEqual(float(predict(model, test)[0]), expected)

    def test_four_fits_once_and_future_data_cannot_change_models_or_past_predictions(self):
        rng = np.random.default_rng(71)
        dates = pd.bdate_range("2023-04-21", periods=440).strftime("%Y-%m-%d").tolist()
        level = 250 + np.cumsum(rng.normal(size=440))
        bases = [feature("baseline|spread_ma5", level.copy())]
        cs = [feature(f"tech|T|fixture{i}", rng.normal(size=440)) for i in range(40)]
        sets = {f"f{n}": [f["id"] for f in cs[:n]] for n in (20, 30, 40)}
        with patch("prepare_curve_timing_capacity.fit_model", wraps=fit_model) as fitter:
            before = fit_variants(dates, level, bases, cs, sets, train_through=dates[340])
            self.assertEqual(fitter.call_count, 4)
        level[401:] += 100000
        for f in bases + cs:
            f["values"][401:] += 100000
        after = fit_variants(dates, level, bases, cs, sets, train_through=dates[340])
        for key in before["models"]:
            self.assertEqual(model_record(before["models"][key], before["features"][key]),
                             model_record(after["models"][key], after["features"][key]))
            np.testing.assert_array_equal(before["predictions"][key][:401], after["predictions"][key][:401])
            self.assertTrue(np.isnan(before["predictions"][key][:341]).all())
        self.assertEqual(before["train"][-1], 334)

    def test_artifact_nested_factors_target_integrity_protection_and_accounting(self):
        path = ROOT / "curve_timing_capacity.json"
        if not path.exists():
            self.skipTest("not generated")
        d = json.loads(path.read_text(encoding="utf-8"))
        old = json.loads((ROOT / "curve_timing_walkforward.json").read_text(encoding="utf-8"))
        signed = json.loads((ROOT / "curve_timing_percentile.json").read_text(encoding="utf-8"))
        frozen = json.loads((ROOT / "curve_timing_factor_set.json").read_text(encoding="utf-8"))
        self.assertEqual(d["daily"]["level"], old["daily"]["level"])
        self.assertEqual(d["daily"]["target"], old["daily"]["target"])
        self.assertEqual(d["daily"]["prediction"]["legacy20"], old["daily"]["prediction"]["full"])
        self.assertEqual(d["meta"]["train_n"], 406)
        self.assertEqual(d["meta"]["fit_count"], 4)
        self.assertEqual(d["forecast"]["f40"]["test"]["n"], 420)
        self.assertEqual(d["factor_sets"]["sets"]["f20"], frozen["selected_ids"])
        for n in (20, 30, 40):
            self.assertEqual(len(d["models"][f"f{n}"]["features"]), n + 8)
            self.assertEqual(d["factor_sets"]["sets"][f"f{n}"], d["factor_sets"]["sets"]["f40"][:n])
        for model in ("baseline", "f20", "f30", "f40"):
            for f in d["models"][model]["features"]:
                if f["id"].startswith(("baseline|", "tech|", "curve|")):
                    self.assertFalse(f["clip_enabled"])
                    self.assertIsNone(f["clip_high"])
        for rule, arms in d["strategies"].items():
            self.assertEqual(arms["legacy20"], signed["strategies"]["signed"][rule]["full"])
            for key in ALL_VARIANTS:
                sim = arms[key]
                self.assertAlmostEqual(sum(t["pnl"] for t in sim["trades"]), sim["equity"][-1])
                np.testing.assert_allclose(np.cumsum(sim["daily_pnl"]), sim["equity"], atol=1e-12)
                self.assertTrue(all(t["holding_sessions"] <= 6 for t in sim["trades"]))
                self.assertTrue(all(t["entry_percentile"] < 40 if t["direction"] == -1 else t["entry_percentile"] > 60 for t in sim["trades"]))
                self.assertAlmostEqual(sum(d["periods"][rule][key][y]["total"] for y in ("2025", "2026")), sim["metrics"]["total"], places=5)
                p = d["periods"][rule][key]["all"]
                self.assertAlmostEqual(p["long_pnl"] + p["short_pnl"], p["total"], places=5)
        for name, h in d["meta"]["input_hashes"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), h)


if __name__ == "__main__":
    unittest.main()
