"""Checks for the agreed global AND rule, as-of timing, grouping and no deduplication."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from prepare_direction_screen import (RULES, global_percentiles, rank_gate, quintiles,
                                      quarter_blocks, screen_snapshot, spearman_columns,
                                      technical_candidates)

ROOT = Path(__file__).resolve().parents[1]


def candidate(key, values):
    return {"id": key, "group": "test", "key": key, "label": key,
            "unit": "", "formula": "test fixture", "values": values}


class DirectionScreenTests(unittest.TestCase):
    def test_global_ranks_require_both_not_either_and_ignore_group(self):
        ic = np.array([0.06, -0.08, 0.12, -0.20])
        ir = np.array([1.2, -0.9, 0.6, -0.1])
        pic, pir = global_percentiles(ic, ir, np.ones(4, bool))
        self.assertFalse(any(rank_gate(a, b) for a, b in zip(pic, pir)))
        self.assertTrue(all(a >= 50 or b >= 50 for a, b in zip(pic, pir)))

    def test_paired_spearman_handles_ties_and_column_specific_missingness(self):
        rng = np.random.default_rng(44)
        a = rng.integers(0, 8, (110, 3)).astype(float)
        y = rng.integers(0, 10, 110).astype(float) ** 3
        a[::4, 0] = np.nan; a[::7, 1] = np.nan; y[::9] = np.nan
        actual = spearman_columns(a, y)
        for j in range(3):
            mask = np.isfinite(a[:, j]) & np.isfinite(y)
            self.assertAlmostEqual(actual[j], spearmanr(a[mask, j], y[mask]).statistic)

    def test_quintiles_respect_negative_direction_and_do_not_split_ties(self):
        x = np.repeat(np.arange(5), 40).astype(float)
        result = quintiles(x, -2 * x, -1)
        self.assertEqual(result["counts"], [40] * 5)
        self.assertEqual(result["same_steps"], 4)
        self.assertEqual(result["difference"], -8)
        self.assertTrue(result["trend_pass"])
        tied = quintiles(np.zeros(200), np.arange(200), 1)
        self.assertEqual(tied["counts"], [200, 0, 0, 0, 0])
        self.assertFalse(tied["count_pass"])

    def test_stability_uses_last_four_actual_blocks(self):
        x = np.r_[np.arange(100), np.tile(np.arange(63), 4)].astype(float)
        y = x.copy(); y[226:289] *= -1
        dates = [f"d{i}" for i in range(len(x))]
        blocks = quarter_blocks(x, y, dates, 1)
        self.assertEqual(blocks[0]["start"], "d100")
        self.assertEqual(blocks[-1]["end"], "d351")
        self.assertEqual([b["n"] for b in blocks], [63] * 4)
        self.assertEqual(sum(b["same"] for b in blocks), 3)

    def test_future_data_cannot_change_historical_screen(self):
        rng = np.random.default_rng(91)
        dates = pd.bdate_range("2023-04-21", periods=620).strftime("%Y-%m-%d").tolist()
        level = 100 + np.cumsum(rng.normal(size=620))
        x = rng.normal(size=620)
        before, _ = screen_snapshot(dates, level, [candidate("x", x)], 450)
        level[451:] += 10000; x[451:] += 10000
        after, _ = screen_snapshot(dates, level, [candidate("x", x)], 450)
        self.assertEqual(before, after)
        self.assertEqual(before["window"]["last_label_exit"], dates[450])
        self.assertEqual(before["window"]["end"], dates[444])

    def test_twenty_cap_keeps_correlated_candidates_without_deduplication(self):
        rng = np.random.default_rng(3)
        dates = pd.bdate_range("2023-04-21", periods=430).strftime("%Y-%m-%d").tolist()
        level = 100 + np.cumsum(rng.normal(size=430))
        y = np.r_[level[6:] - level[:-6], np.full(6, np.nan)]
        x = y + rng.normal(scale=0.3, size=len(y))
        result, _ = screen_snapshot(dates, level, [candidate(f"x{i:02}", x.copy()) for i in range(25)], 429)
        self.assertEqual(result["summary"]["qualified"], 25)
        self.assertEqual(len(result["selected_ids"]), 20)
        self.assertEqual(sum(r["status"] == "capped" for r in result["rows"]), 5)

    def test_technical_features_are_causal(self):
        rng = np.random.default_rng(12)
        p = 100 + np.cumsum(rng.normal(size=350))
        cmap = {k + "主力": p.copy() for k in ("T", "TF", "TS", "TL")}
        before = technical_candidates(cmap)
        for a in cmap.values(): a[280:] += 10000
        after = technical_candidates(cmap)
        self.assertEqual(len(before), 32)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a["values"][:280], b["values"][:280])

    def test_published_results_follow_agreed_rules(self):
        d = json.loads((ROOT / "direction_screen.json").read_text(encoding="utf-8"))
        self.assertEqual(d["summary"]["total"], 844)
        self.assertEqual(d["meta"]["history"], "all")
        self.assertFalse(d["meta"]["deduplication"])
        self.assertFalse(d["meta"]["model_increment"])
        rows = {r["id"]: r for r in d["rows"]}
        self.assertLessEqual(len(d["selected_ids"]), 20)
        for key in d["selected_ids"]:
            r = rows[key]
            self.assertTrue(all(r["checks"].values()))
            self.assertGreaterEqual(r["ic_percentile"], 50)
            self.assertGreaterEqual(r["icir_percentile"], 50)
        matrix = json.loads((ROOT / "direction_screen_inputs.json").read_text(encoding="utf-8"))
        self.assertEqual(len(matrix["features"]), len(d["selected_ids"]) + 8)
        self.assertTrue(all(y is None for y in matrix["target_h6"][-6:]))
        self.assertTrue(all(len(f["values"]) == len(matrix["dates"]) for f in matrix["features"]))


if __name__ == "__main__":
    unittest.main()
