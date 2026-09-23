import unittest

import numpy as np

import prepare_factor_checkup as checkup


class CheckupTradingCalendarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.dates, _, cls.curves = checkup.load_library()

    def test_calendar_contains_only_futures_sessions(self):
        self.assertTrue(np.isfinite(self.curves["T主力"]).all())
        self.assertEqual(len(self.dates), len(set(self.dates)))

    def test_five_session_labels_are_raw_spread_changes(self):
        for target, weights in [("tl3t", {"T主力": 3, "TL主力": -1}),
                                ("ts4t", {"TS主力": 4, "T主力": -1})]:
            spec = next(x for x in checkup.TARGETS if x["id"] == target)
            level, denom = checkup.build_target(spec, self.curves, len(self.dates))
            y = checkup.target_change(spec, level, denom, 5, len(self.dates))
            price_ok = np.isfinite(y)
            i = int(np.flatnonzero(price_ok)[10])
            expected = sum(w * (self.curves[name][i + 5] - self.curves[name][i])
                           for name, w in weights.items())
            self.assertAlmostEqual(y[i], expected, places=10)


if __name__ == "__main__":
    unittest.main()
