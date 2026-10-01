# -*- coding: utf-8 -*-
"""探索性多因子时序模型：预选因子来自现有因子体检，不重新筛 IC。

在 t 日收盘后使用当日可得信息，t+1 日按 t 日收盘价代理建仓，t+6 日收盘平仓。
逐日扩展历史训练、每 20 日重训；只有已走完持有期的标签进入训练。
每 6 日取一次不重叠交易。全样本体检已被用于预选因子，故历史结果
只能称为时间顺序回放，不能称为独立样本外验证。
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

import prepare_factor_checkup as checkup

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "strategy_model.json")
HORIZON = 5
REBALANCE_EVERY = HORIZON + 1  # 次日代理开盘到 t+6 收盘须错开 6 个信号日
DR_WINDOW = 10
REFIT_EVERY = 20
RIDGE_ALPHA = 15.0
MIN_COVERAGE = 5  # 7 个预选因子中至少 5 个有值
WEAK_FRAC = 0.20  # 长周期预测幅度 / 当期训练标签标准差；事先固定，不搜索
EXIT_RULES = [
    ("fixed", "固定 t+6 收盘", "原策略基准"),
    ("long_reverse", "长周期预测反向退出", "持仓方向与每日长周期预测相反时，次日代理开盘退出"),
    ("long_weak", "长周期预测转弱退出", "|预测| < 0.20×仅用历史训练标签计算的标准差时退出"),
    ("long_both", "长周期反向或转弱退出", "满足反向或转弱任一条件即退出"),
    ("short1_reverse", "1日预测反向退出", "另训未来1日模型，仅用于离场判断"),
    ("short2_reverse", "2日预测反向退出", "另训未来2日模型，仅用于离场判断"),
]

# 体检 raw / 5日中预先固定的 14 条：高 |IC|、ICIR、分组差异、分年方向，
# 同时覆盖不同机构/期限/资金维度。不能在回放结果出炉后再替换因子。
SELECTED = {
    "tl3t": [
        ("derived", "tenor_zdiff|保险公司|7-10年−20-30年|结构·跨期限", "跨期限需求；高 |IC|、ICIR 与单调分组"),
        ("derived", "slope|证券公司|20-30年|趋势·买在一致", "超长端交易盘的持续买入"),
        ("derived", "pos_util|基金公司及产品|7-10年|极值·卖在一致", "中长端拥挤程度；分组差异"),
        ("cash", "国债|中小型银行|5-7年", "现券原始净买入；不同机构来源"),
        ("repo", "中小型银行|net", "净融出；资金供给状态"),
        ("repo", "中小型银行|rate_spread", "相对 DR001 的回购融资价格"),
        ("carry", "分解|tl3t_curve", "两腿收益率曲线状态；与机构行为互补"),
    ],
    "s3010": [
        ("derived", "rank|基金公司及产品|1-3年|截面·买在分歧", "机构横截面排名；分组差异与分年方向"),
        ("derived", "accel|保险公司|20-30年|趋势·买在一致", "保险超长端需求加速"),
        ("derived", "accel|证券公司|7-10年|趋势·买在一致", "交易盘中长端需求加速"),
        ("cash", "地方债|中小型银行|20-30年", "超长端原始净买入；独立样本检验线索"),
        ("cash", "国债|保险公司|合计", "保险现券总需求；与加速项区分水平和变化"),
        ("repo", "其他|rev_bal", "逆回购余额；资金供给状态"),
        ("derived", "tenor_zdiff|中小型银行|7-10年−20-30年|结构·跨期限", "同机构中长端相对需求"),
    ],
}
MIN_TRAIN = {"tl3t": 300, "s3010": 500}
TARGETS = {
    "tl3t": {"name": "3T−TL", "unit": "点", "formula": "3×T主力 − TL主力", "kind": "期货报价价差"},
    "s3010": {"name": "30Y−10Y 国债收益率利差", "unit": "bp", "formula": "100×(30年国债收益率 − 10年国债收益率)", "kind": "收益率利差方向分数"},
}


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
    """利率调整当日即可用于收盘信号；没有新调整时沿用历史值。"""
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
    """t 收盘为次日代理入场价；t+1 开始持有，t+1+horizon 收盘离场。"""
    out = np.full(len(level), np.nan)
    for t in range(len(level) - horizon - 1):
        if np.isfinite(level[t]) and np.isfinite(level[t + 1 + horizon]):
            out[t] = level[t + 1 + horizon] - level[t]
    return out


def make_offset_labels(level, offset):
    """短周期退出模型：t 收盘至 t+offset 收盘的变动。"""
    out = np.full(len(level), np.nan)
    if len(level) > offset:
        start, end = level[:-offset], level[offset:]
        valid = np.isfinite(start) & np.isfinite(end)
        out[np.flatnonzero(valid)] = end[valid] - start[valid]
    return out


def known_training_indices(t, valid, horizon=HORIZON):
    """预测 t 时，仅训练离场价格在 t 日收盘前已知的样本。"""
    return np.flatnonzero(valid[:max(0, t - horizon)])


def known_offset_indices(t, valid, offset):
    """短周期标签在 r+offset 收盘后方可进入 t 日训练。"""
    return np.flatnonzero(valid[:max(0, t - offset + 1)])


class PastOnlyScaler:
    """所有填补、缩尾、标准化参数只用当前训练窗口拟合。"""

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


def _metrics(trades, field, unit):
    if not trades:
        return {"trades": 0}
    pnl = np.asarray([x[field] for x in trades], dtype=float)
    equity = np.cumsum(pnl)
    drawdown = equity - np.maximum.accumulate(np.r_[0.0, equity])[1:]
    return {
        "trades": len(trades), "total": _round(equity[-1], 3),
        "mean_trade": _round(pnl.mean(), 4),
        "win_rate": _round(np.mean(pnl > 0), 4),
        "max_drawdown": _round(drawdown.min(), 3), "unit": unit,
    }


def simulate_exit_rule(fixed_trades, daily, level, dates, rule_id):
    """相同入场日/方向，仅修改平仓；收盘信号下一交易日按该收盘价代理执行。"""
    by_day = {d["asof"]: d for d in daily}
    day_index = {day: i for i, day in enumerate(dates)}
    out, equity = [], 0.0
    for base in fixed_trades:
        t0 = day_index[base["asof"]]
        position = base["position"]
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
                    "long_reverse": reverse,
                    "long_weak": weak,
                    "long_both": reverse or weak,
                    "short1_reverse": short1_reverse,
                    "short2_reverse": short2_reverse,
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
        pnl = _round(position * (level[price_t] - level[t0]), 4)
        equity += pnl
        out.append({
            "asof": base["asof"], "entry": base["entry"], "exit": exit_day,
            "exit_price_asof": dates[price_t], "exit_timing": execution,
            "position": position, "exit_reason": reason,
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
    """逐日盯市，包含持仓中的浮动结果；代理开盘退出当天已无价格敞口。"""
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


def _factor_rows(target, checkup_meta):
    path = os.path.join(BASE, "checkup_b", f"b_raw_{target}.json")
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)["blocks"][str(HORIZON)]
    ix = {name: i for i, name in enumerate(checkup_meta["cols"])}
    return {(r[ix["g"]], r[ix["k"]]): r for r in rows}, ix


def _target_level(target, cmap):
    if target == "tl3t":
        return 3 * cmap["T主力"] - cmap["TL主力"]
    return 100 * (cmap["30年国债"] - cmap["10年国债"])


def build_target(target, dates, series, cmap, omo, ck_meta):
    level = _target_level(target, cmap)
    y = make_labels(level)
    y1, y2 = make_offset_labels(level, 1), make_offset_labels(level, 2)
    factor_map = {(g, k): (label, arr) for g, k, label, arr in series}
    row_map, ix = _factor_rows(target, ck_meta)
    selected = []
    arrays = []
    for group, key, reason in SELECTED[target]:
        if (group, key) not in factor_map or (group, key) not in row_map:
            raise ValueError(f"因子库与体检不一致：{target} {group}|{key}")
        label, arr = factor_map[(group, key)]
        row = row_map[(group, key)]
        arrays.append(arr)
        selected.append({
            "group": group, "key": key, "label": label, "reason": reason,
            "n": row[ix["n"]], "ic": row[ix["ic"]], "icir": row[ix["icir"]],
            "q1": row[ix["q1"]], "q5": row[ix["q5"]], "qd": row[ix["qd"]],
            "sig": row[ix["sig"]],
        })
    x_factors = np.column_stack(arrays)
    level_z = _trailing_z(level)
    momentum5 = np.full(len(level), np.nan)
    momentum5[5:] = level[5:] - level[:-5]
    ma5, ma20 = _trailing_mean(level, 5), _trailing_mean(level, 20)
    x_controls = np.column_stack([level_z, momentum5, ma5, ma20])
    dr_ma10 = cmap["DR001-MA10"]
    y10, y2, y30 = (cmap[k] for k in ("10年国债", "2年国债", "30年国债"))
    # 收益率原序列单位为百分比；转为 bp 后与组合报价特征一起按训练集标准化。
    x_market = np.column_stack([
        100 * (dr_ma10 - omo),
        100 * (y10 - dr_ma10),
        100 * (y10 - y2),
        100 * (y30 - y10),
    ])
    x_market_base = np.column_stack([x_controls, x_market])
    x_full = np.column_stack([x_factors, x_market_base])
    coverage = np.isfinite(x_factors).sum(axis=1)
    x_ok = ((coverage >= MIN_COVERAGE) & np.isfinite(x_controls).all(axis=1)
            & np.isfinite(x_market).all(axis=1))
    train_ok = x_ok & np.isfinite(y)
    short1_ok, short2_ok = x_ok & np.isfinite(y1), x_ok & np.isfinite(y2)
    min_train = MIN_TRAIN[target]
    daily = []
    model = baseline = market_baseline = short1_model = short2_model = None
    scale_for_score = score_signs = train_target_std = None
    refit_at = -REFIT_EVERY
    for t in range(len(dates)):
        if not x_ok[t] or not np.isfinite(level[t]):
            continue
        train_idx = known_training_indices(t, train_ok)
        if len(train_idx) < min_train:
            continue
        if model is None or t - refit_at >= REFIT_EVERY:
            model = ridge_fit(x_full[train_idx], y[train_idx])
            baseline = ridge_fit(x_controls[train_idx], y[train_idx])
            market_baseline = ridge_fit(x_market_base[train_idx], y[train_idx])
            short1_idx = known_offset_indices(t, short1_ok, 1)
            short2_idx = known_offset_indices(t, short2_ok, 2)
            short1_model = ridge_fit(x_full[short1_idx], y1[short1_idx])
            short2_model = ridge_fit(x_full[short2_idx], y2[short2_idx])
            train_target_std = float(np.std(y[train_idx], ddof=1))
            scale_for_score = PastOnlyScaler().fit(x_factors[train_idx])
            z_train = scale_for_score.transform(x_factors[train_idx])
            cov = z_train.T @ (y[train_idx] - np.mean(y[train_idx]))
            score_signs = np.where(cov >= 0, 1, -1)
            refit_at = t
        pred = float(ridge_predict(model, x_full[t:t + 1])[0])
        price_pred = float(ridge_predict(baseline, x_controls[t:t + 1])[0])
        market_pred = float(ridge_predict(market_baseline, x_market_base[t:t + 1])[0])
        short1_pred = float(ridge_predict(short1_model, x_full[t:t + 1])[0])
        short2_pred = float(ridge_predict(short2_model, x_full[t:t + 1])[0])
        equal_score = float(np.mean(scale_for_score.transform(x_factors[t:t + 1])[0] * score_signs))
        actual = y[t]
        daily.append({
            "t": t, "asof": dates[t],
            "entry": dates[t + 1] if t + 1 < len(dates) else None,
            "exit": dates[t + 1 + HORIZON] if t + 1 + HORIZON < len(dates) else None,
            "pred": _round(pred), "price_pred": _round(price_pred),
            "market_pred": _round(market_pred),
            "short1_pred": _round(short1_pred), "short2_pred": _round(short2_pred),
            "train_target_std": _round(train_target_std),
            "equal_score": _round(equal_score), "actual": _round(actual),
            "train_n": len(train_idx), "coverage": int(coverage[t]),
        })

    # t+1 代理开盘至 t+6 收盘：每 6 个信号日开一笔，避免同日开/平仓重叠。
    first = daily[0]["t"] if daily else None

    def trade_path(phase):
        trades = []
        equity = {"pnl": 0.0, "price_pnl": 0.0, "market_pnl": 0.0, "equal_pnl": 0.0}
        for d in daily:
            if d["actual"] is None or (d["t"] - first) % REBALANCE_EVERY != phase:
                continue
            actual = d["actual"]
            q = 1 if d["pred"] >= 0 else -1
            bq = 1 if d["price_pred"] >= 0 else -1
            mq = 1 if d["market_pred"] >= 0 else -1
            eq = 1 if d["equal_score"] >= 0 else -1
            item = {k: d[k] for k in ("asof", "entry", "exit", "pred", "actual", "train_n")}
            item.update({"position": q, "price_position": bq, "market_position": mq, "equal_position": eq,
                         "pnl": _round(q * actual, 4),
                         "price_pnl": _round(bq * actual, 4),
                         "market_pnl": _round(mq * actual, 4),
                         "equal_pnl": _round(eq * actual, 4)})
            for key in equity:
                equity[key] += item[key]
                item["equity" if key == "pnl" else key.replace("pnl", "equity")] = _round(equity[key], 4)
            trades.append(item)
        return trades

    trades = trade_path(0)
    exit_rules = []
    for rule_id, label, description in EXIT_RULES:
        paths = [simulate_exit_rule(trade_path(p), daily, level, dates, rule_id)
                 for p in range(REBALANCE_EVERY)]
        phase_metrics = [exit_metrics(path, TARGETS[target]["unit"]) for path in paths]
        daily_paths = [exit_daily_path(path, level, dates, trade_path(p)[-1]["exit"])
                       for p, path in enumerate(paths)]
        for metrics, daily_path in zip(phase_metrics, daily_paths):
            equity = np.asarray([d["equity"] for d in daily_path])
            metrics["daily_max_drawdown"] = _round((equity - np.maximum.accumulate(equity)).min(), 3)
        totals = [p["total"] for p in phase_metrics]
        exit_rules.append({
            "id": rule_id, "label": label, "description": description,
            "metrics": phase_metrics[0], "positive_phases": sum(x > 0 for x in totals),
            "phase_min": min(totals), "phase_max": max(totals),
            "phase_metrics": phase_metrics, "trades": paths[0],
            "daily_path": daily_paths[0],
        })
    known = [d for d in daily if d["actual"] is not None]
    result = {
        **TARGETS[target], "id": target, "selected": selected,
        "controls": ["截至信号日的价差60日滚动Z", "截至信号日的价差过去5日变动",
                     "组合价差MA5", "组合价差MA20"],
        "market_factors": ["DR001-MA10 − 7天OMO", "10年国债 − DR001-MA10",
                           "10年国债 − 2年国债", "30年国债 − 10年国债"],
        "daily_predictions": daily,
        "trades": trades,
        "exit_comparison": {"weak_fraction": WEAK_FRAC, "rules": exit_rules,
                            "entry_policy": "全部规则使用与固定离场相同的入场日及方向；提前退出后等到下一个原定入场日"},
        "period": [trades[0]["entry"], trades[-1]["exit"]] if trades else None,
        "forecast_ic": _rank_corr([d["pred"] for d in known], [d["actual"] for d in known]),
        "price_forecast_ic": _rank_corr([d["price_pred"] for d in known], [d["actual"] for d in known]),
        "metrics": {
            "model": _metrics(trades, "pnl", TARGETS[target]["unit"]),
            "price_only": _metrics(trades, "price_pnl", TARGETS[target]["unit"]),
            "market_only": _metrics(trades, "market_pnl", TARGETS[target]["unit"]),
            "equal_weight": _metrics(trades, "equal_pnl", TARGETS[target]["unit"]),
        },
        "phase_robustness": [
            {"phase": p, "model": _metrics(trade_path(p), "pnl", TARGETS[target]["unit"]),
             "price_only": _metrics(trade_path(p), "price_pnl", TARGETS[target]["unit"]),
             "market_only": _metrics(trade_path(p), "market_pnl", TARGETS[target]["unit"]),
             "equal_weight": _metrics(trade_path(p), "equal_pnl", TARGETS[target]["unit"])}
            for p in range(REBALANCE_EVERY)
        ],
    }
    return result


def main():
    _, dates, series, cmap = checkup.load_library()
    omo, omo_policy = load_omo_policy(dates)
    with open(os.path.join(BASE, "factor_checkup.json"), encoding="utf-8") as f:
        ck = json.load(f)
    payload = {
        "meta": {
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_date": dates[-1], "checkup_generated": ck["meta"]["generated"],
            "horizon": HORIZON, "rebalance_every": REBALANCE_EVERY,
            "factor_count": sum(map(len, SELECTED.values())),
            "dr_window": DR_WINDOW, "omo_verified_through": omo_policy["verified_through"],
            "omo_needs_review": dates[-1] > omo_policy["verified_through"],
            "target_definition": "信号日 t 的收盘价作为 t+1 入场代理价，目标为 t+6 收盘减 t 收盘",
            "refit_every": REFIT_EVERY, "ridge_alpha": RIDGE_ALPHA,
            "exit_rules": [x[0] for x in EXIT_RULES], "weak_fraction": WEAK_FRAC,
            "min_coverage": MIN_COVERAGE, "min_train": MIN_TRAIN,
            "cost": 0, "selection": "预选自因子体检 raw/未来5日；全样本已被看过，回放不是独立样本外验证",
            "timing": "t 日收盘后得到因子；t+1 日按 t 日收盘价作为代理入场价；t+6 日收盘平仓；每 6 日交易一次以避免持仓重叠",
            "entry_caveat": "缺少真实次日开盘价/开盘收益率；代理入场未计隔夜跳空，结果不是可执行成交回测",
            "market_source": "OMO 为人民银行7天逆回购政策利率变更记录，详见 omo_policy_7d.json；DR001 使用原始库 MA10",
            "training": "长周期与1/2日离场模型每20日仅用各自已完成标签的历史样本重训；填补、缩尾、标准化及等权方向仅用训练数据",
            "strategy": "模型预测变动>=0 做多价差，否则做空；固定满额方向，不扣成本，不做仓位优化",
        },
        "targets": {key: build_target(key, dates, series, cmap, omo, ck["meta"]) for key in SELECTED},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    print("输出", OUT, "总因子", payload["meta"]["factor_count"])
    for key, r in payload["targets"].items():
        print(key, "交易", len(r["trades"]), "区间", r["period"], "预测IC", r["forecast_ic"],
              "模型", r["metrics"]["model"], "价格基线", r["metrics"]["price_only"])


if __name__ == "__main__":
    main()
