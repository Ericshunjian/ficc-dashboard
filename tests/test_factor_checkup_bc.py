# -*- coding: utf-8 -*-
"""B/C 类检验列与回测口径的回归测试。

两条红线：
1. COLS 的新列只能 append 在末尾。主循环里有 `rows[i][28] = sig` 这类硬编码索引，
   一旦有人在中间插队，写错列不会报错、只会悄悄把统计写进别人的字段。
2. 回测页（JS）的「日度净值」定义必须与「每笔下注收益」自洽：
   日度合计 == (1/N)·Σ w_t·fwd_t。改任一侧都要改另一侧。
"""
import unittest

import numpy as np

import prepare_factor_checkup as checkup


class ColsLayoutTests(unittest.TestCase):
    def test_new_bc_cols_are_appended_at_tail(self):
        cols = checkup.COLS
        self.assertEqual(cols[28], "sig", "sig 的索引被挪动了，主循环写 sig 会写错列")
        self.assertEqual(cols[39], "rglv")
        self.assertEqual(cols[40:], ["crange", "cprange", "cmono", "cvr", "tr"])


class BcStatsTests(unittest.TestCase):
    @staticmethod
    def _cond(p, ys):
        return checkup.cond_dist(p, ys)

    def test_cond_dist_has_p95(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0, 100, 600)
        ys = rng.normal(0, 1, 600)
        cond = self._cond(p, ys)
        self.assertEqual(len(cond), len(checkup.COND_EDGES) - 1)
        for row in cond:
            self.assertEqual(len(row), 7, "每区间应为 [n, 均值, 中位, 标准差, 上行%, p05, p95]")
            if row[0] >= 10:
                self.assertLessEqual(row[5], row[6], "5% 分位不应大于 95% 分位")

    def test_monotonic_factor_looks_like_A(self):
        """A 类：区间均值单调 ⇒ |cmono| 接近 1"""
        rng = np.random.default_rng(2)
        p = rng.uniform(0, 100, 1200)
        ys = (p / 100.0) * 2.0 + rng.normal(0, 0.6, 1200)     # 强单调
        cr, cpr, cmono, cvr, tr = checkup.bc_stats(self._cond(p, ys))
        self.assertGreater(abs(cmono), 0.85)
        self.assertGreater(cr, 1.0)

    def test_u_shaped_factor_looks_like_B(self):
        """B 类：U 形 ⇒ 区间极差大但 cmono 接近 0（线性 IC 会把它当噪声）"""
        rng = np.random.default_rng(3)
        p = rng.uniform(0, 100, 1200)
        ys = np.where(p < 15, 2.0, np.where(p > 85, 1.5, -0.5)) + rng.normal(0, 0.8, 1200)
        cr, cpr, cmono, cvr, tr = checkup.bc_stats(self._cond(p, ys))
        self.assertGreater(cr, 1.5)
        self.assertLess(abs(cmono), 0.6)
        self.assertGreater(cpr, 20, "两端与中间的上行概率应明显拉开")

    def test_vol_factor_looks_like_C(self):
        """C 类：均值几乎不变，但高区分辨率/波动明显更大"""
        rng = np.random.default_rng(4)
        p = rng.uniform(0, 100, 1500)
        sd = np.where(p > 70, 3.0, 1.0)
        ys = rng.normal(0, sd)
        cr, cpr, cmono, cvr, tr = checkup.bc_stats(self._cond(p, ys))
        self.assertGreater(cvr, 1.8)
        self.assertLess(cr, 1.5, "C 类因子的区间均值差应该很小")

    def test_too_few_bins_returns_none(self):
        none_row = [0, None, None, None, None, None, None]
        self.assertEqual(checkup.bc_stats([none_row] * 6), [None] * 5)


class BacktestParityTests(unittest.TestCase):
    """回测页 JS 的日度净值口径 = 总名义敞口固定 1 单位、每期用 1/N 资金新开一笔。
    Python 侧复刻同一定义，断言它与「每笔下注收益」的恒等关系成立。"""

    def test_daily_equity_equals_average_of_bet_pnl(self):
        rng = np.random.default_rng(5)
        D, N = 400, 5
        lvl = np.cumsum(rng.normal(0, 0.02, D))          # 目标水平
        delta = np.zeros(D)
        delta[1:] = np.diff(lvl) * 100                    # 日度变动（bp）
        w = rng.choice([-1.0, 0.0, 1.0], size=D)          # 每日信号
        fwd = np.full(D, np.nan)
        fwd[:D - N] = (lvl[N:] - lvl[:-N]) * 100          # 未来 N 日变动
        # 最后 N-1 天开不出完整的 N 日期：回测页里 w 会被置 0（fwd 为 null 就不下注），
        # 否则恒等式会被尾部截断破坏
        w[D - N:] = 0.0

        pos = np.zeros(D)
        for j in range(1, D):
            pos[j] = w[max(0, j - N):j].sum() / N         # 与 backtest_dashboard.html 同式
        daily = pos * delta

        lhs = daily.sum()
        rhs = np.nansum(w * np.nan_to_num(fwd)) / N
        self.assertAlmostEqual(lhs, rhs, places=8,
                               msg="日度净值合计必须等于 (1/N)·Σ w·fwd，否则 JS 与体检口径脱节")


if __name__ == "__main__":
    unittest.main()
