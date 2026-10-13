import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from prepare_curve_timing_nonlinear import (chronological_split, fit_nonlinear,
                                           preprocessing_record, serialize_tree, xgboost)
from prepare_direction_screen import baseline_features, load_candidates

ROOT = Path(__file__).resolve().parents[1]


class NonlinearTests(unittest.TestCase):
    def test_holdout_purges_six_crossing_labels(self):
        signals = np.arange(406)
        fitting, validation, purged = chronological_split(signals)
        self.assertEqual((len(fitting), len(validation), len(purged)), (318, 82, 6))
        self.assertTrue(np.all(signals[fitting] + 6 < signals[validation[0]]))
        self.assertFalse(set(fitting) & set(validation))
        self.assertEqual(validation[-1], 405)

    def test_future_inputs_and_labels_cannot_change_selected_parameters_or_final_models(self):
        rng = np.random.default_rng(45)
        x = rng.normal(size=(440, 2))
        y = x[:, 0] * .2 + rng.normal(scale=.4, size=440)
        features = [{"id": "baseline|spread_ma5", "label": "spread", "unit": "点"},
                    {"id": "cash|国债|券商|1-3年", "label": "flow", "unit": "亿元"}]
        train = np.arange(406)
        before, scaler, selection = fit_nonlinear(x, y, train, features)
        x[406:] *= 100000;y[406:] += 100000
        after, scaler_after, selection_after = fit_nonlinear(x, y, train, features)
        self.assertEqual(selection, selection_after)
        self.assertEqual(preprocessing_record(scaler, features), preprocessing_record(scaler_after, features))
        self.assertEqual(serialize_tree(before["tree"], scaler, features), serialize_tree(after["tree"], scaler_after, features))
        self.assertEqual(before["xgb"].get_booster().save_raw(raw_format="json"),
                         after["xgb"].get_booster().save_raw(raw_format="json"))
        self.assertEqual(before["xgb"].n_estimators, selection["xgb_rounds"])
        self.assertIsNone(before["xgb"].early_stopping_rounds)

    def test_archived_tree_and_xgboost_reproduce_all_test_predictions(self):
        d = json.loads((ROOT / "curve_timing_nonlinear.json").read_text(encoding="utf-8"))
        archive = json.loads((ROOT / "curve_timing_nonlinear_models.json").read_text(encoding="utf-8"))
        _, dates, cmap, candidates = load_candidates()
        level = 3 * cmap["T主力"] - cmap["TL主力"]
        bases, _ = baseline_features(level, cmap, dates)
        values = {f["id"]: f["values"] for f in candidates + bases}
        test = np.array([dates.index(day) for day in d["daily"]["dates"]])
        for key, record in archive["models"].items():
            if record["family"] == "ridge":
                continue
            features = record["preprocessing"]
            raw = np.column_stack([values[f["id"]][test] for f in features])
            z = np.empty_like(raw)
            for j, f in enumerate(features):
                filled = np.where(np.isfinite(raw[:, j]), raw[:, j], f["median"])
                if f["clip_enabled"]:
                    filled = np.clip(filled, f["clip_low"], f["clip_high"])
                z[:, j] = (filled - f["mean"]) / f["std"]
                if f["id"].startswith("baseline|"):
                    self.assertFalse(f["clip_enabled"])
                    self.assertIsNone(f["clip_high"])
            if record["family"] == "tree":
                nodes = record["tree"]["nodes"]
                pred = []
                for row in z.astype(np.float32):
                    node = nodes[0]
                    while not node["leaf"]:
                        node = nodes[node["left"] if row[node["feature_index"]] <= node["threshold_scaled"] else node["right"]]
                    pred.append(node["value"])
                self.assertLessEqual(record["tree"]["depth"], 3)
                self.assertTrue(all(n["n_train"] >= 40 for n in nodes if n["leaf"]))
            else:
                booster = xgboost.Booster()
                booster.load_model(bytearray(json.dumps(record["booster"]).encode("utf-8")))
                pred = booster.predict(xgboost.DMatrix(z))
                self.assertEqual(booster.num_boosted_rounds(), record["params"]["n_estimators"])
            np.testing.assert_allclose(pred, np.array(d["daily"]["prediction"][key], float), atol=1e-12, rtol=0)

    def test_artifact_ridge_parity_validation_dates_and_accounting(self):
        d = json.loads((ROOT / "curve_timing_nonlinear.json").read_text(encoding="utf-8"))
        source = json.loads((ROOT / "curve_timing_capacity.json").read_text(encoding="utf-8"))
        self.assertEqual(d["daily"]["level"], source["daily"]["level"])
        self.assertEqual(d["daily"]["target"], source["daily"]["target"])
        self.assertEqual(len(d["forecast"]), 9)
        for n in (20, 30, 40):
            split = d["tuning"][f"f{n}"]["dates"]
            self.assertLess(split["inner_last_label_exit"], split["validation_start"])
            self.assertLessEqual(split["last_validation_label_exit"], "2024-12-31")
            self.assertEqual(split["validation_n"], 82)
            for family in ("ridge", "tree", "xgb"):
                key = f"{family}{n}"
                self.assertEqual(d["forecast"][key]["test"]["n"], 420)
                self.assertEqual(len(d["model_summary"][key]["preprocessing"]), n + 8)
                self.assertTrue(all(v is None for v in d["gates"][key]["percentile"][:60]))
                for rule in ("fixed", "dynamic"):
                    sim = d["strategies"][rule][key]
                    self.assertAlmostEqual(sum(t["pnl"] for t in sim["trades"]), sim["equity"][-1])
                    self.assertTrue(all(t["holding_sessions"] <= 6 for t in sim["trades"]))
                    self.assertTrue(all(t["entry_percentile"] < 40 if t["direction"] == -1 else t["entry_percentile"] > 60 for t in sim["trades"]))
                    np.testing.assert_allclose(np.cumsum(sim["daily_pnl"]), sim["equity"], atol=1e-12)
                    if family == "ridge":
                        self.assertEqual(sim, source["strategies"][rule][f"f{n}"])
        for name, h in d["meta"]["input_hashes"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), h)

    def test_requested_pages_removed_and_no_live_html_links_remain(self):
        for name in ("curve_timing_walkforward.html", "strategy_model.html"):
            self.assertFalse((ROOT / name).exists())
            for page in ROOT.glob("*.html"):
                self.assertNotIn('href="' + name + '"', page.read_text(encoding="utf-8"), page.name)


if __name__ == "__main__":
    unittest.main()
