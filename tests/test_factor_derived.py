import unittest

import numpy as np

import factor_derived

INSTITUTIONS = ["大型银行", "中小型银行", "证券公司", "保险公司",
                "基金公司及产品", "货币市场基金",
                "理财子公司及理财类产品", "其他"]


class DerivedFactorTenorTests(unittest.TestCase):
    def test_each_source_maturity_is_its_own_bucket(self):
        self.assertEqual(len(factor_derived.TENORS), 9)
        self.assertTrue(all(name == mats[0] and len(mats) == 1
                            for name, mats in factor_derived.TENORS))

        dates = [f"2026-01-{i + 1:02d}" for i in range(30)]
        records = []
        for date in dates:
            records.extend([
                {"date": date, "bond_type": "国债", "institution": "大型银行",
                 "maturity": "7-10年", "value": 1.0},
                {"date": date, "bond_type": "国债", "institution": "大型银行",
                 "maturity": "5-7年", "value": -100.0},
            ])
        merged = {"meta": {"institutions": INSTITUTIONS}, "detail": records}
        factors, _ = factor_derived.build(merged, dict(zip(dates, range(30))), 30)
        ratio_7 = next(x for x in factors if x["op"] == "days_ratio"
                       and x["tenor"] == "7-10年")
        ratio_5 = next(x for x in factors if x["op"] == "days_ratio"
                       and x["tenor"] == "5-7年")
        self.assertEqual(ratio_7["v"][-1], 100.0)
        self.assertEqual(ratio_5["v"][-1], 0.0)

    def test_missing_is_not_zero_in_path_or_cross_institution_sum(self):
        path = np.array([1.0] * 9 + [np.nan])
        self.assertTrue(np.isnan(factor_derived._cum_slope(path, 10)[-1]))

        dates = [f"d{i}" for i in range(30)]
        insts = ["大型银行", "理财子公司及理财类产品", "保险公司",
                 "基金公司及产品", "中小型银行"]
        records = [
            {"date": date, "bond_type": "国债", "institution": inst,
             "maturity": "7-10年", "value": 1.0}
            for date in dates[10:] for inst in insts
        ]
        merged = {"meta": {"institutions": INSTITUTIONS}, "detail": records}
        factors, _ = factor_derived.build(merged, dict(zip(dates, range(30))), 30)
        spread = next(x for x in factors if x["op"] == "cfg_trd"
                      and x["tenor"] == "7-10年")
        self.assertGreaterEqual(spread["i0"], 19)
        self.assertNotEqual(spread["v"][-1], 0.0)

    def test_off_futures_date_does_not_enter_rolling_window(self):
        dates = [f"d{i}" for i in range(31)]
        records = [
            {"date": date, "bond_type": "国债", "institution": "大型银行",
             "maturity": "7-10年", "value": -100.0 if i == 25 else 1.0}
            for i, date in enumerate(dates)
        ]
        merged = {"meta": {"institutions": INSTITUTIONS}, "detail": records}
        factors, _ = factor_derived.build(
            merged, dict(zip(dates, range(31))), 31,
            session_dates=[d for i, d in enumerate(dates) if i != 25])
        ratio = next(x for x in factors if x["op"] == "days_ratio"
                     and x["tenor"] == "7-10年")
        self.assertIsNone(ratio["v"][25 - ratio["i0"]])
        self.assertEqual(ratio["v"][-1], 100.0)


if __name__ == "__main__":
    unittest.main()
