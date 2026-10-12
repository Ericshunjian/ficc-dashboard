# -*- coding: utf-8 -*-
"""v2 多因子岭回归策略：按「|IC| 组内前50% + |ICIR| 组内前50%」筛因子，滚动重筛 + walk-forward 岭回归。

相对 v1 的三处关键改动：
1) 因子集不再人工硬编码，而是按统一规则筛；分两种模式
   - rolling：每次重训只用**当时可得**的历史数据重跑整套筛选流程（无前视，主口径）
   - static ：用全样本筛一次后固定（含前视，仅作对照）
2) 下注口径三选一
   - sign     ：pred 符号（v1 口径；训练集标准化后样本外打分整体偏移时会系统性站错边）
   - quantile ：pred 放到它自己过去 250 次预测的分布里取分位 q，仓位 w = 2q−1（免疫分布漂移，主口径）
   - qband    ：q≥0.7 做多 / q≤0.3 做空 / 中间空仓
3) 自动拉黑机械恒等因子：|Spearman(因子, 目标水平)| ≥ 0.90（水平→变动的伪信号）

其余沿用 v1：预测 t→t+6 变动、t+1 按 t 收盘代理入场、每 6 日交易、每 20 日重训、
岭回归 α=15、零成本、六条离场规则、价差自身 / 价差＋市场 / 等权打分三个基线。
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import prepare_factor_checkup as checkup

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "strategy_model_v2.json")

HORIZON = 5
REBALANCE_EVERY = HORIZON + 1
REFIT_EVERY = 20
RIDGE_ALPHA = 15.0
DR_WINDOW = 10
WEAK_FRAC = 0.20

# 筛选规则（事先固定，不搜索）
MECH_CUT = 0.70      # |ρ(因子, 目标水平)| 超此值 = 与水平共线（收益率水平 / 期货主力 / 隐含利率 / 共同趋势），拉黑
MIN_COV = 0.80       # 窗口内覆盖率
MIN_SAME = 0.80      # 分年 IC 同号率
MIN_YEARS = 3        # 年块少于此数则不要求同号率
RHO_CUT = 0.70       # 相关性去重阈值
BLOCK = 20           # ICIR 用的分块长度（交易日）
QS = 50              # 「前 50%」分位

MODES = ("rolling", "static")
KS = (5, 10, 15)
STAKES = ("quantile", "qband", "sign")
MAIN = ("rolling", 10, "quantile")   # 主口径：完整离场规则对照只在这一组上算

MIN_TRAIN = {"tl3t": 300, "s3010": 500}
TARGETS = {
    "tl3t": {"name": "3T−TL", "unit": "点", "formula": "3×T主力 − TL主力", "kind": "期货报价价差"},
    "s3010": {"name": "30Y−10Y 国债收益率利差", "unit": "bp", "formula": "100×(30年国债收益率 − 10年国债收益率)", "kind": "收益率利差方向分数"},
}
EXIT_RULES = [
    ("fixed", "固定 t+6 收盘", "原策略基准"),
    ("long_reverse", "长周期预测反向退出", "持仓方向与每日长周期预测相反时，次日代理开盘退出"),
    ("long_weak", "长周期预测转弱退出", "|预测| < 0.20×仅用历史训练标签计算的标准差时退出"),
    ("long_both", "长周期反向或转弱退出", "满足反向或转弱任一条件即退出"),
    ("short1_reverse", "1日预测反向退出", "另训未来1日模型，仅用于离场判断"),
    ("short2_reverse", "2日预测反向退出", "另训未来2日模型，仅用于离场判断"),
]


def _round(x, n=4):
    return round(float(x), n) if x is not None and np.isfinite(x) else None


def _rank_corr(x, y):
    if len(x) < 3:
        return None
    from scipy.stats import spearmanr
    v = spearmanr(x, y).statistic
    return _round(v)


def _trailing_z(a, window=60):
    out = np.full(len(a), np.nan)
    for t in range(window - 1, len(a)):
        hist = a[t - window + 1:t + 1]
        if not np.isfinite(hist).all():
            continue
        sd = hist.std(ddof=1)
        if sd > 1e-10:
            out[t] = (a[t] - hist.mean()) / sd
    return out


def _trailing_mean(a, window):
    out = np.full(len(a), np.nan)
    for t in range(window - 1, len(a)):
        hist = a[t - window + 1:t + 1]
        if np.isfinite(hist).all():
            out[t] = hist.mean()
    return out


def load_omo_policy(dates):
    with open(os.path.join(BASE, "omo_policy_7d.json"), encoding="utf-8") as f:
        policy = json.load(f)
    changes = policy["changes"]
    if [c["date"] for c in changes] != sorted(c["date"] for c in changes):
        raise ValueError("OMO 利率调整日期必须按升序排列")
    out = np.full(len(dates), np.nan)
    i, current = 0, np.nan
    for t, day in enumerate(dates):
        while i < len(changes) and changes[i]["date"] <= day:
            current = float(changes[i]["rate"])
            i += 1
        out[t] = current
    return out, policy


def make_labels(level, horizon=HORIZON):
    out = np.full(len(level), np.nan)
    for t in range(len(level) - horizon - 1):
        if np.isfinite(level[t]) and np.isfinite(level[t + 1 + horizon]):
            out[t] = level[t + 1 + horizon] - level[t]
    return out


def make_offset_labels(level, offset):
    out = np.full(len(level), np.nan)
    if len(level) > offset:
        start, end = level[:-offset], level[offset:]
        valid = np.isfinite(start) & np.isfinite(end)
        out[np.flatnonzero(valid)] = end[valid] - start[valid]
    return out


def known_training_indices(t, valid, horizon=HORIZON):
    return np.flatnonzero(valid[:max(0, t - horizon)])


def known_offset_indices(t, valid, offset):
    return np.flatnonzero(valid[:max(0, t - offset + 1)])


class PastOnlyScaler:
    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.median = np.nanmedian(x, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        filled = np.where(np.isfinite(x), x, self.median)
        self.lo = np.quantile(filled, 0.01, axis=0)
        self.hi = np.quantile(filled, 0.99, axis=0)
        clipped = np.clip(filled, self.lo, self.hi)
        self.mean = clipped.mean(axis=0)
        self.std = clipped.std(axis=0)
        self.std = np.where(self.std > 1e-10, self.std, 1.0)
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        filled = np.where(np.isfinite(x), x, self.median)
        return (np.clip(filled, self.lo, self.hi) - self.mean) / self.std


def ridge_fit(x, y, alpha=RIDGE_ALPHA):
    scaler = PastOnlyScaler().fit(x)
    z = scaler.transform(x)
    ym = float(np.mean(y))
    gram = z.T @ z + alpha * np.eye(z.shape[1])
    coef = np.linalg.solve(gram, z.T @ (y - ym))
    return scaler, ym, coef


def ridge_predict(model, x):
    scaler, ym, coef = model
    return ym + scaler.transform(x) @ coef


# ---------------------------------------------------------------- 筛选器


def _colwise_spearman(A, y):
    """A (n,m) 含 NaN，y (n,) 无 NaN（调用前已按 y 过滤行）。返回 (m,) 精确 Spearman。"""
    # 每列都在因子与目标的成对有效观测上分别取秩；目标也必须取秩。
    # 仅对 A 取秩再与原值 y 相关，并不是 Spearman。
    M = np.isfinite(A) & np.isfinite(y)[:, None]
    R = pd.DataFrame(np.where(M, A, np.nan)).rank().to_numpy()
    YR = pd.DataFrame(np.where(M, y[:, None], np.nan)).rank().to_numpy()
    R0 = np.where(M, R, 0.0)
    ycol = np.where(M, YR, 0.0)
    cnt = M.sum(0)
    ok = cnt >= 60
    c = np.where(cnt > 0, cnt, 1)
    sx, sy = R0.sum(0), ycol.sum(0)
    sxy = (R0 * ycol).sum(0)
    sxx = (R0 * R0).sum(0)
    syy = (ycol * ycol).sum(0)
    cov = sxy / c - (sx / c) * (sy / c)
    vx = sxx / c - (sx / c) ** 2
    vy = syy / c - (sy / c) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(vx * vy)
    return np.where(ok, r, np.nan)


def _blocked_corr(A, y, size):
    """按连续块切分，块内 Pearson（NaN 逐元素剔除）。返回 (nb, m)。"""
    n = (len(A) // size) * size
    if n < size:
        return np.zeros((0, A.shape[1]))
    Xb = A[:n].reshape(-1, size, A.shape[1])
    Yb = y[:n].reshape(-1, size)
    M = np.isfinite(Xb) & np.isfinite(Yb)[:, :, None]
    X0 = np.where(M, Xb, 0.0)
    Y0 = np.where(M, Yb[:, :, None], 0.0)
    cnt = M.sum(1)
    sx, sy = X0.sum(1), Y0.sum(1)
    sxy = (X0 * Y0).sum(1)
    sxx = (X0 * X0).sum(1)
    syy = (Y0 * Y0).sum(1)
    c = np.where(cnt > 0, cnt, 1)
    cov = sxy / c - (sx / c) * (sy / c)
    vx = sxx / c - (sx / c) ** 2
    vy = syy / c - (sy / c) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(vx * vy)
    return np.where(cnt >= max(8, size // 2), r, np.nan)


def _year_blocks(idx, years):
    out = []
    for y in sorted(set(years)):
        sel = idx[years[idx] == y]
        if len(sel) >= 30:
            out.append(sel)
    return out


class Screener:
    """在给定历史窗口上执行整套筛选规则，输出按 |ICIR| 降序的候选池。"""

    def __init__(self, arrays, groups, labels):
        self.A = np.column_stack(arrays)           # (N, M)
        self.groups = np.asarray(groups)
        self.labels = list(labels)
        self.M = self.A.shape[1]
        self._cache = {}

    def run(self, win, level, y, years):
        """win: 可用行下标（已保证 y 有效）。返回 pool: list of dict（按 |ICIR| 降序）。"""
        key = (int(win[0]) if len(win) else -1, int(win[-1]) if len(win) else -1)
        if key in self._cache:
            return self._cache[key]
        A = self.A[win]
        lvl = level[win]
        yy = y[win]
        yrs = years[win]
        ic = _colwise_spearman(A, yy)
        mech = _colwise_spearman(A, lvl)
        blk = _blocked_corr(A, yy, BLOCK)
        nb = blk.shape[0]
        if nb >= 6:
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = np.nanmean(blk, 0)
                sd = np.nanstd(blk, 0, ddof=1)
                icir = np.where(sd > 1e-12, mu / np.where(sd > 1e-12, sd, 1.0), np.nan)
        else:
            icir = np.full(self.M, np.nan)
        yb = _year_blocks(np.arange(len(win)), yrs)
        same = np.full(self.M, np.nan)
        if len(yb) >= MIN_YEARS:
            sy = []
            for sel in yb:
                sy.append(_colwise_spearman(A[sel], yy[sel]))
            S = np.array(sy)                                    # (nyr, M)
            sgn = np.sign(np.where(np.isfinite(ic), ic, 0.0))[None, :]
            good = np.isfinite(S) & (np.abs(S) > 1e-12)
            with np.errstate(invalid="ignore"):
                same = np.where(good.sum(0) >= MIN_YEARS,
                                (np.sign(S) == sgn).sum(0) / np.maximum(good.sum(0), 1), np.nan)
        cov = np.isfinite(A).sum(0) / max(len(win), 1)

        alive = np.isfinite(ic) & np.isfinite(icir)
        alive &= ~(np.isfinite(mech) & (np.abs(mech) >= MECH_CUT))
        alive &= cov >= MIN_COV
        alive &= ~(np.isfinite(same) & (same < MIN_SAME))

        # 组内 50% 分位
        passed = np.zeros(self.M, bool)
        for g in set(self.groups[alive].tolist()):
            gi = np.flatnonzero(alive & (self.groups == g))
            if len(gi) < 4:
                continue
            t1 = np.percentile(np.abs(ic[gi]), QS)
            t2 = np.percentile(np.abs(icir[gi]), QS)
            passed[gi[(np.abs(ic[gi]) >= t1) & (np.abs(icir[gi]) >= t2)]] = True
        cand = np.flatnonzero(passed)
        order = cand[np.argsort(-np.abs(icir[cand]))]

        # 贪心去重（逐对精确 Spearman）
        kept = []
        for j in order:
            dup = False
            for k2 in kept:
                a, b = A[:, j], A[:, k2]
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 60:
                    continue
                ra = pd.Series(a[m]).rank().to_numpy()
                rb = pd.Series(b[m]).rank().to_numpy()
                if abs(np.corrcoef(ra, rb)[0, 1]) >= RHO_CUT:
                    dup = True
                    break
            if not dup:
                kept.append(int(j))
        pool = [{"j": j, "label": self.labels[j], "group": self.groups[j],
                 "ic": float(ic[j]), "icir": float(icir[j]), "cov": float(cov[j]),
                 "same": (float(same[j]) if np.isfinite(same[j]) else None),
                 "mech": (float(mech[j]) if np.isfinite(mech[j]) else None)} for j in kept]
        self._cache[key] = pool
        return pool


# ---------------------------------------------------------------- 策略构建


def _metrics(trades, field, unit):
    if not trades:
        return {"trades": 0}
    pnl = np.asarray([x[field] for x in trades], dtype=float)
    equity = np.cumsum(pnl)
    drawdown = equity - np.maximum.accumulate(np.r_[0.0, equity])[1:]
    n = len(pnl)
    return {
        "trades": n, "total": _round(equity[-1], 3),
        "mean_trade": _round(pnl.mean(), 4),
        "win_rate": _round(np.mean(pnl > 0), 4),
        "max_drawdown": _round(drawdown.min(), 3), "unit": unit,
        "t": _round(pnl.mean() / (pnl.std(ddof=1) / np.sqrt(n)), 3) if n > 2 and pnl.std(ddof=1) > 0 else None,
    }


def simulate_exit_rule(fixed_trades, daily, level, dates, rule_id):
    by_day = {d["asof"]: d for d in daily}
    day_index = {day: i for i, day in enumerate(dates)}
    out, equity = [], 0.0
    for base in fixed_trades:
        t0 = day_index[base["asof"]]
        weight = base["position"]
        if weight == 0:
            continue
        position = 1 if weight > 0 else -1
        price_t = t0 + REBALANCE_EVERY
        reason, execution = "最长持有期", "收盘"
        if rule_id != "fixed":
            for t in range(t0 + 1, t0 + REBALANCE_EVERY):
                signal = by_day.get(dates[t])
                if signal is None:
                    continue
                long_pred = signal["pred"]
                reverse = position * long_pred < 0
                weak = abs(long_pred) < WEAK_FRAC * signal["train_target_std"]
                short1_reverse = position * signal["short1_pred"] < 0
                short2_reverse = position * signal["short2_pred"] < 0
                triggered = {
                    "long_reverse": reverse, "long_weak": weak, "long_both": reverse or weak,
                    "short1_reverse": short1_reverse, "short2_reverse": short2_reverse,
                }[rule_id]
                if triggered:
                    price_t = t
                    reason = ("长周期反向" if reverse and rule_id == "long_both" else
                              "长周期转弱" if rule_id in ("long_weak", "long_both") else
                              "长周期反向" if rule_id == "long_reverse" else
                              "1日预测反向" if rule_id == "short1_reverse" else "2日预测反向")
                    execution = "次日代理开盘"
                    break
        exit_day = dates[price_t] if price_t == t0 + REBALANCE_EVERY else dates[price_t + 1]
        pnl = _round(weight * (level[price_t] - level[t0]), 4)
        equity += pnl
        out.append({
            "asof": base["asof"], "entry": base["entry"], "exit": exit_day,
            "exit_price_asof": dates[price_t], "exit_timing": execution,
            "position": weight, "direction": position, "exit_reason": reason,
            "holding_sessions": price_t - t0, "early_exit": price_t < t0 + REBALANCE_EVERY,
            "pnl": pnl, "equity": _round(equity, 4),
        })
    return out


def exit_metrics(trades, unit):
    result = _metrics(trades, "pnl", unit)
    if trades:
        result.update({
            "early_exits": sum(d["early_exit"] for d in trades),
            "avg_hold": _round(np.mean([d["holding_sessions"] for d in trades]), 2),
            "exposure_sessions": sum(d["holding_sessions"] for d in trades),
        })
    return result


def exit_daily_path(trades, level, dates, end_date):
    if not trades:
        return []
    ix = {day: i for i, day in enumerate(dates)}
    start, end = ix[trades[0]["asof"]], ix[end_date]
    changes = np.zeros(end - start + 1)
    for trade in trades:
        a, b = ix[trade["asof"]], ix[trade["exit_price_asof"]]
        segment = np.asarray(level[a:b + 1], dtype=float)
        if not np.isfinite(segment).all():
            raise ValueError("持仓期间缺少价格，无法计算逐日盯市结果")
        changes[a + 1 - start:b + 1 - start] += trade["position"] * np.diff(segment)
    return [{"date": dates[start + i], "equity": _round(value, 4)}
            for i, value in enumerate(np.cumsum(changes))]


def build_predictions(target, dates, level, y, y1, y2, screener, x_controls, x_market,
                      mode, k, min_train):
    """逐日 walk-forward，产出每日预测与当时选中的因子。"""
    n = len(dates)
    years = np.array([int(str(d)[:4]) for d in dates])
    x_market_base = np.column_stack([x_controls, x_market])
    x_ok = np.isfinite(x_controls).all(axis=1) & np.isfinite(x_market).all(axis=1)
    train_ok = x_ok & np.isfinite(y)
    short1_ok, short2_ok = x_ok & np.isfinite(y1), x_ok & np.isfinite(y2)

    fixed_pool = None
    if mode == "static":
        win = np.flatnonzero(np.isfinite(y))
        fixed_pool = screener.run(win, level, y, years)

    daily, refit_at, state = [], -REFIT_EVERY, {}
    prev_pool, turnover = None, []
    for t in range(n):
        if not x_ok[t] or not np.isfinite(level[t]):
            continue
        train_idx = known_training_indices(t, train_ok)
        if len(train_idx) < min_train:
            continue
        if not state or t - refit_at >= REFIT_EVERY:
            win = np.flatnonzero(np.isfinite(y)[:max(0, t - HORIZON)])
            pool = fixed_pool if fixed_pool is not None else screener.run(win, level, y, years)
            cur = pool[:k]
            if not cur and fixed_pool is None:
                cur = screener.run(win, level, y, years)[:k]
            if not cur:
                continue
            if prev_pool is not None:
                prev_lab = {p["label"] for p in prev_pool}
                now_lab = {p["label"] for p in cur}
                turnover.append(len(prev_lab & now_lab) / max(len(now_lab), 1))
            prev_pool = cur
            cols = [p["j"] for p in cur]
            xf = screener.A[:, cols]
            x_full = np.column_stack([xf, x_market_base])
            state = {
                "pool": cur, "cols": cols,
                "model": ridge_fit(x_full[train_idx], y[train_idx]),
                "baseline": ridge_fit(x_controls[train_idx], y[train_idx]),
                "market": ridge_fit(x_market_base[train_idx], y[train_idx]),
                "short1": ridge_fit(x_full[known_offset_indices(t, short1_ok, 1)],
                                    y1[known_offset_indices(t, short1_ok, 1)]),
                "short2": ridge_fit(x_full[known_offset_indices(t, short2_ok, 2)],
                                    y2[known_offset_indices(t, short2_ok, 2)]),
                "tstd": float(np.std(y[train_idx], ddof=1)),
            }
            scaler = PastOnlyScaler().fit(xf[train_idx])
            z = scaler.transform(xf[train_idx])
            cov = z.T @ (y[train_idx] - np.mean(y[train_idx]))
            state["scaler"] = scaler
            state["signs"] = np.where(cov >= 0, 1, -1)
            refit_at = t
        xf_t = screener.A[t:t + 1, state["cols"]]
        x_full_t = np.column_stack([xf_t, x_market_base[t:t + 1]])
        pred = float(ridge_predict(state["model"], x_full_t)[0])
        daily.append({
            "t": t, "asof": dates[t],
            "entry": dates[t + 1] if t + 1 < n else None,
            "exit": dates[t + 1 + HORIZON] if t + 1 + HORIZON < n else None,
            "pred": _round(pred),
            "price_pred": _round(float(ridge_predict(state["baseline"], x_controls[t:t + 1])[0])),
            "market_pred": _round(float(ridge_predict(state["market"], x_market_base[t:t + 1])[0])),
            "short1_pred": _round(float(ridge_predict(state["short1"], x_full_t)[0])),
            "short2_pred": _round(float(ridge_predict(state["short2"], x_full_t)[0])),
            "train_target_std": _round(state["tstd"]),
            "equal_score": _round(float(np.mean(state["scaler"].transform(xf_t)[0] * state["signs"]))),
            "actual": _round(y[t]), "train_n": len(train_idx),
            "pool": [{"label": p["label"], "group": p["group"], "icir": _round(p["icir"], 3),
                      "ic": _round(p["ic"], 3)} for p in state["pool"]],
        })
    return daily, (float(np.mean(turnover)) if turnover else None)


def stake_weights(daily, stake, window=250):
    """把预测序列变成仓位；只用截至当时的历史预测，无前视。"""
    preds = np.array([d["pred"] for d in daily], dtype=float)
    out = np.full(len(daily), np.nan)
    hist = []
    for i, p in enumerate(preds):
        if stake == "sign":
            out[i] = 1.0 if p >= 0 else -1.0
            continue
        if len(hist) >= 60:
            arr = np.array(hist[-window:])
            q = float(np.mean(arr <= p))
            out[i] = (2 * q - 1) if stake == "quantile" else (1.0 if q >= 0.7 else (-1.0 if q <= 0.3 else 0.0))
        else:
            out[i] = 0.0 if stake == "qband" else np.nan
        hist.append(p)
    return out


def make_trades(daily, weights, phase, field_actual="actual"):
    trades, eq = [], {"pnl": 0.0, "price_pnl": 0.0, "market_pnl": 0.0, "equal_pnl": 0.0}
    first = daily[0]["t"] if daily else None
    for i, d in enumerate(daily):
        if d["actual"] is None or (d["t"] - first) % REBALANCE_EVERY != phase:
            continue
        w = weights[i]
        if not np.isfinite(w) or w == 0:
            continue
        actual = d["actual"]
        item = {k: d[k] for k in ("asof", "entry", "exit", "pred", "actual", "train_n")}
        # 保留计算精度，舍入只交给前端显示，避免退出路径另用舍入仓位。
        item["position"] = float(w)
        item["pnl"] = _round(w * actual, 4)
        bq = 1 if d["price_pred"] >= 0 else -1
        mq = 1 if d["market_pred"] >= 0 else -1
        eq_q = 1 if d["equal_score"] >= 0 else -1
        item["price_pnl"] = _round(bq * actual, 4)
        item["market_pnl"] = _round(mq * actual, 4)
        item["equal_pnl"] = _round(eq_q * actual, 4)
        for key in ("pnl", "price_pnl", "market_pnl", "equal_pnl"):
            eq[key] += item[key]
            item["equity" if key == "pnl" else key.replace("pnl", "equity")] = _round(eq[key], 4)
        trades.append(item)
    return trades


def perm_pvalue(trades, n_perm=500, seed=7):
    """循环移位保持重叠结构：p = P(随机累计 ≤ 真实累计)，越大越好。"""
    if len(trades) < 10:
        return None
    w = np.array([t["position"] for t in trades], dtype=float)
    a = np.array([t["actual"] for t in trades], dtype=float)
    real = float((w * a).sum())
    rng = np.random.default_rng(seed)
    hits, tot = 0, 0
    for _ in range(n_perm):
        s = rng.integers(len(w))
        v = float((np.roll(w, s) * a).sum())
        tot += 1
        hits += v <= real
    return _round(hits / tot, 4)


_DAILY_CACHE = {}
_SCREENER_CACHE = {}


def build_target(target, dates, series, cmap, omo, mode, k, stake):
    ck2 = (target, mode, k)
    if ck2 not in _DAILY_CACHE:
        level = (3 * cmap["T主力"] - cmap["TL主力"]) if target == "tl3t" else 100 * (cmap["30年国债"] - cmap["10年国债"])
        y = make_labels(level)
        y1, y2 = make_offset_labels(level, 1), make_offset_labels(level, 2)
        arrays = [np.asarray(a, float) for _, _, _, a in series]
        # k 与下注口径不改变历史筛选统计，共用同目标的历史窗口缓存。
        if target not in _SCREENER_CACHE:
            _SCREENER_CACHE[target] = Screener(arrays, [g for g, _, _, _ in series],
                                             [l for _, _, l, _ in series])
        screener = _SCREENER_CACHE[target]
        level_z = _trailing_z(level)
        momentum5 = np.full(len(level), np.nan)
        momentum5[5:] = level[5:] - level[:-5]
        x_controls = np.column_stack([level_z, momentum5, _trailing_mean(level, 5), _trailing_mean(level, 20)])
        y10, y2_, y30 = (cmap[kk] for kk in ("10年国债", "2年国债", "30年国债"))
        x_market = np.column_stack([100 * (cmap["DR001-MA10"] - omo), 100 * (y10 - cmap["DR001-MA10"]),
                                    100 * (y10 - y2_), 100 * (y30 - y10)])
        _DAILY_CACHE[ck2] = build_predictions(target, dates, level, y, y1, y2, screener,
                                              x_controls, x_market, mode, k, MIN_TRAIN[target])
    daily, turnover = _DAILY_CACHE[ck2]
    unit = TARGETS[target]["unit"]
    if not daily:
        return None
    weights = stake_weights(daily, stake)

    def pack(phase):
        return make_trades(daily, weights, phase)

    trades = pack(0)
    phases = [pack(p) for p in range(REBALANCE_EVERY)]
    known = [d for d in daily if d["actual"] is not None]
    preds = [d["pred"] for d in known]
    acts = [d["actual"] for d in known]
    res = {
        "mode": mode, "k": k, "stake": stake,
        "turnover": _round(turnover, 3),
        "trades": trades,
        "metrics": {
            "model": _metrics(trades, "pnl", unit),
            "price_only": _metrics(trades, "price_pnl", unit),
            "market_only": _metrics(trades, "market_pnl", unit),
            "equal_weight": _metrics(trades, "equal_pnl", unit),
        },
        "perm_p": perm_pvalue(trades),
        "phase_robustness": [{"phase": p, "model": _metrics(tp, "pnl", unit),
                              "price_only": _metrics(tp, "price_pnl", unit),
                              "market_only": _metrics(tp, "market_pnl", unit),
                              "equal_weight": _metrics(tp, "equal_pnl", unit)}
                             for p, tp in enumerate(phases)],
        "forecast_ic": _rank_corr(preds, acts),
        "period": [trades[0]["entry"], trades[-1]["exit"]] if trades else None,
        "last_pool": daily[-1]["pool"] if daily else None,
    }
    flat = [{"date": d["asof"], "pred": d["pred"], "weight": _round(weights[i], 3),
             "actual": d["actual"]} for i, d in enumerate(daily) if d["actual"] is not None]
    res["signal_path"] = flat
    return res


def main():
    _, dates, series, cmap = checkup.load_library()
    omo, omo_policy = load_omo_policy(dates)
    targets = {}
    for target in ("tl3t", "s3010"):
        variants = {}
        for mode in MODES:
            for k in KS:
                for stake in STAKES:
                    r = build_target(target, dates, series, cmap, omo, mode, k, stake)
                    if r is None:
                        continue
                    variants[f"{mode}|k{k}|{stake}"] = r
                    print(f"  {target} {mode} k={k} {stake}: 交易 {r['metrics']['model']['trades']} "
                          f"累计 {r['metrics']['model']['total']} {TARGETS[target]['unit']} "
                          f"t={r['metrics']['model']['t']} 置换p={r['perm_p']} 换手={r['turnover']}")
        main_key = "|".join([MAIN[0], f"k{MAIN[1]}", MAIN[2]])
        if main_key in variants:
            mv = variants[main_key]
            # 主口径补做完整离场规则对照：需要逐日 pred/short1/short2 与 train_target_std
            main_daily = build_main_daily(target, dates, series, cmap, omo, MAIN[1])
            rules = []
            for rule_id, label, desc in EXIT_RULES:
                paths = [simulate_exit_rule(make_trades_from(main_daily, MAIN[2], p), main_daily,
                                            level_of(target, cmap), dates, rule_id)
                         for p in range(REBALANCE_EVERY)]
                pm = [exit_metrics(p, TARGETS[target]["unit"]) for p in paths]
                dp = [exit_daily_path(p, level_of(target, cmap), dates,
                                      make_trades_from(main_daily, MAIN[2], idx)[-1]["exit"])
                      for idx, p in enumerate(paths)]
                for m2, d2 in zip(pm, dp):
                    e = np.asarray([x["equity"] for x in d2])
                    m2["daily_max_drawdown"] = _round((e - np.maximum.accumulate(e)).min(), 3) if len(e) else None
                totals = [p["total"] for p in pm]
                rules.append({"id": rule_id, "label": label, "description": desc,
                              "metrics": pm[0], "positive_phases": sum(x > 0 for x in totals),
                              "phase_min": min(totals), "phase_max": max(totals),
                              "phase_metrics": pm, "trades": paths[0], "daily_path": dp[0]})
            mv["exit_comparison"] = {"weak_fraction": WEAK_FRAC, "rules": rules,
                                     "entry_policy": "全部规则使用与固定离场相同的入场日及权重；提前退出后等到下一个原定入场日"}
        targets[target] = {"id": target, **TARGETS[target], "variants": variants}
    null = null_distribution("tl3t", dates, series, cmap, omo, 15)
    real_t = targets["tl3t"]["variants"]["rolling|k15|quantile"]["metrics"]["model"]["t"]
    ts = [x for x in null["t_list"] if x is not None]
    null["real_t"] = real_t
    null["real_total"] = targets["tl3t"]["variants"]["rolling|k15|quantile"]["metrics"]["model"]["total"]
    null["real_percentile"] = _round(float(np.mean([x <= real_t for x in ts])), 3) if ts else None
    null["t_median"] = _round(float(np.median(ts)), 3) if ts else None
    null["t_max"] = _round(float(max(ts)), 3) if ts else None
    print("零假设对照（tl3t rolling k=15 quantile，整条流水线在循环移位因子上重跑 "
          f"{null['n_perm']} 次）：真实 t={real_t}，零假设 t 中位 {null['t_median']} / 最大 {null['t_max']} "
          f"⇒ 真实分位 {null['real_percentile']}")

    payload = {
        "meta": {
            "null_check": null,
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_date": dates[-1],
            "calculation_revision": "2026-10-12：Spearman按成对有效样本对双方取秩；逐笔和逐日账本均保留实际仓位权重。筛选和信号规则未改。",
            "horizon": HORIZON, "rebalance_every": REBALANCE_EVERY, "refit_every": REFIT_EVERY,
            "ridge_alpha": RIDGE_ALPHA, "cost": 0,
            "modes": list(MODES), "ks": list(KS), "stakes": list(STAKES),
            "main": {"mode": MAIN[0], "k": MAIN[1], "stake": MAIN[2]},
            "screen": {"mech_cut": MECH_CUT, "min_cov": MIN_COV, "min_same": MIN_SAME,
                       "rho_cut": RHO_CUT, "block": BLOCK, "quantile": QS,
                       "order": "拉黑与目标水平共线的因子 → |IC|、|ICIR| 组内前50% → 覆盖≥80% → 分年同号≥80% → 相关性去重",
                       "mech_note": f"|Spearman(因子, 目标水平)|≥{MECH_CUT} 即拉黑：收益率水平、期货主力价、隐含利率、"
                                    "Carry 曲线项等与目标水平共线，其预测力主要来自「水平→变动」的机械关系，"
                                    "而这些已由组合价差的 Z / MA / 动量等控制项表达"},
            "timing": "t 日收盘后得到因子；t+1 日按 t 日收盘价代理入场；t+6 日收盘平仓；每 6 日交易一次",
            "entry_caveat": "缺少真实次日开盘价/开盘收益率；代理入场未计隔夜跳空，结果不是可执行成交回测",
            "training": "长周期与1/2日离场模型每20日仅用各自已完成标签的历史样本重训；填补、缩尾、标准化只用训练数据",
            "rolling_note": "rolling 模式在每次重训时只用当时可得历史重跑筛选（无前视）；static 用全样本筛一次后固定，仅作对照",
            "stake_note": {"sign": "pred≥0 做多否则做空（v1 口径，受训练期标准化漂移影响）",
                           "quantile": "pred 在过去250次预测中的分位 q，仓位 w=2q−1（免疫分布漂移）",
                           "qband": "q≥0.7 做多、q≤0.3 做空、中间空仓"},
            "perm_note": "p = P(循环移位后的随机方向累计 ≤ 真实累计)，越大越好；>95% 才优于随机",
            "omo_verified_through": omo_policy["verified_through"],
            "omo_needs_review": dates[-1] > omo_policy["verified_through"],
        },
        "targets": targets,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    print("输出", OUT, os.path.getsize(OUT) // 1024, "KB")


def circshift_cols(A, s):
    """每列的有效段整体循环移位：保留列内自相关与 NaN 位置，只打乱与未来的对齐。"""
    B = A.copy()
    for j in range(A.shape[1]):
        m = np.isfinite(A[:, j])
        B[m, j] = np.roll(A[m, j], s)
    return B


def null_distribution(target, dates, series, cmap, omo, k, n_perm=12, seed=11):
    """零假设：因子与未来收益没有真实对齐时，同一条流水线能跑出多好的 t。

    整条链（滚动重筛 → 岭回归 → 分位数下注）在循环移位后的因子上重跑，
    得到 t 的分布；真实 t 在这个分布里的分位才是有意义的判据。
    """
    level = level_of(target, cmap)
    y = make_labels(level)
    y1, y2 = make_offset_labels(level, 1), make_offset_labels(level, 2)
    arrays = [np.asarray(a, float) for _, _, _, a in series]
    groups = [g for g, _, _, _ in series]
    labels = [l for _, _, l, _ in series]
    A0 = np.column_stack(arrays)
    level_z = _trailing_z(level)
    momentum5 = np.full(len(level), np.nan)
    momentum5[5:] = level[5:] - level[:-5]
    x_controls = np.column_stack([level_z, momentum5, _trailing_mean(level, 5), _trailing_mean(level, 20)])
    y10, y2_, y30 = (cmap[kk] for kk in ("10年国债", "2年国债", "30年国债"))
    x_market = np.column_stack([100 * (cmap["DR001-MA10"] - omo), 100 * (y10 - cmap["DR001-MA10"]),
                                100 * (y10 - y2_), 100 * (y30 - y10)])
    rng = np.random.default_rng(seed)
    ts, totals = [], []
    n = len(dates)
    for _ in range(n_perm):
        s = int(rng.integers(60, max(61, n - 60)))
        # 一次移位后再取各列，避免为每一列重复复制整张矩阵。
        shifted = circshift_cols(A0, s)
        screener = Screener([shifted[:, j] for j in range(A0.shape[1])], groups, labels)
        daily, _ = build_predictions(target, dates, level, y, y1, y2, screener, x_controls,
                                     x_market, "rolling", k, MIN_TRAIN[target])
        if not daily:
            continue
        w = stake_weights(daily, "quantile")
        trades = make_trades(daily, w, 0)
        m = _metrics(trades, "pnl", TARGETS[target]["unit"])
        if m["trades"] >= 10:
            ts.append(m["t"])
            totals.append(m["total"])
    return {"n_perm": len(ts), "t_list": [_round(x, 3) for x in ts],
            "total_list": [_round(x, 3) for x in totals]}


def level_of(target, cmap):
    return (3 * cmap["T主力"] - cmap["TL主力"]) if target == "tl3t" else 100 * (cmap["30年国债"] - cmap["10年国债"])


_MAIN_CACHE = {}


def build_main_daily(target, dates, series, cmap, omo, k):
    """为主口径重跑一次并保留逐日完整字段（含 short1/short2/train_target_std）。"""
    if (target, k) in _MAIN_CACHE:
        return _MAIN_CACHE[(target, k)]
    cached = _DAILY_CACHE.get((target, MAIN[0], k))
    if cached is not None:
        # 退出模拟只读预测，复制字典以保留原配置的 pool 字段。
        daily = [{key: value for key, value in d.items() if key != "pool"}
                 for d in cached[0]]
        _MAIN_CACHE[(target, k)] = daily
        return daily
    level = level_of(target, cmap)
    y = make_labels(level)
    y1, y2 = make_offset_labels(level, 1), make_offset_labels(level, 2)
    arrays = [np.asarray(a, float) for _, _, _, a in series]
    screener = Screener(arrays, [g for g, _, _, _ in series], [l for _, _, l, _ in series])
    level_z = _trailing_z(level)
    momentum5 = np.full(len(level), np.nan)
    momentum5[5:] = level[5:] - level[:-5]
    x_controls = np.column_stack([level_z, momentum5, _trailing_mean(level, 5), _trailing_mean(level, 20)])
    y10, y2_, y30 = (cmap[kk] for kk in ("10年国债", "2年国债", "30年国债"))
    x_market = np.column_stack([100 * (cmap["DR001-MA10"] - omo), 100 * (y10 - cmap["DR001-MA10"]),
                                100 * (y10 - y2_), 100 * (y30 - y10)])
    daily, _ = build_predictions(target, dates, level, y, y1, y2, screener, x_controls, x_market,
                                 MAIN[0], k, MIN_TRAIN[target])
    for d in daily:
        d.pop("pool", None)
    _MAIN_CACHE[(target, k)] = daily
    return daily


def make_trades_from(daily, stake, phase):
    w = stake_weights(daily, stake)
    return make_trades(daily, w, phase)


if __name__ == "__main__":
    main()
