"""Frozen full-history factors and past-only forecast-percentile gates for 3T-TL.

The frozen factor set uses full-history screening (explicit selection lookahead).
Coefficient fitting uses only completed labels; percentile gates use past forecasts.
No alpha, threshold, exit-rule or feature-count search is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from prepare_direction_screen import (RULES, baseline_features, load_candidates, raw_numbers,
                                     rolling_spearman, safe_number, screen_snapshot,
                                     spearman_columns)
from prepare_strategy_model import make_offset_labels, ridge_fit, ridge_predict

ROOT = Path(__file__).resolve().parent
CONFIG = {"min_train": 300, "refit_every": 20, "ridge_alpha": 15.0,
          "horizon": 6, "percentile_window": 250, "percentile_min_history": 60,
          "entry_percentile": 60.0,
          "prediction_candidate_coverage": 0.90}
ARMS = ("baseline", "full")


def completed_indices(level, y, t, horizon=6):
    """Label at t-horizon is known after t close; label at t-horizon+1 is not."""
    first = np.flatnonzero(np.isfinite(level))[0]
    indices = np.arange(first, max(first, t - horizon + 1))
    return indices[np.isfinite(y[indices])]


def model_record(model, features):
    scaler, intercept, coefficients = model
    return {"intercept": float(intercept), "features": [
        {"id": f["id"], "label": f["label"], "unit": f.get("unit", ""),
         "coef": float(coefficients[j]), "median": float(scaler.median[j]),
         "clip_low": float(scaler.lo[j]), "clip_high": float(scaler.hi[j]),
         "mean": float(scaler.mean[j]), "std": float(scaler.std[j])}
        for j, f in enumerate(features)]}


def walk_forward(dates, level, baselines, candidates, asof=None, config=None, screen_fn=None,
                 progress=None, frozen_snapshot=None):
    config = CONFIG | (config or {})
    screen_fn = screen_fn or screen_snapshot
    asof = len(dates) - 1 if asof is None else asof
    horizon = config["horizon"]
    if horizon != RULES["horizon"]:
        raise ValueError("模型与筛选的标签期限必须一致")
    level = np.asarray(level, float)
    y = make_offset_labels(level, horizon)
    base_matrix = np.column_stack([f["values"] for f in baselines])
    candidate_map = {c["id"]: c for c in candidates}
    if len(candidate_map) != len(candidates):
        raise ValueError("候选因子ID重复")
    # Trailing ICs are causal and can be shared by all expanding-history snapshots.
    # Windows before the common TL start are deliberately unavailable.
    first = np.flatnonzero(np.isfinite(level))[0]
    cached = np.full((len(dates), len(candidates)), np.nan)
    if frozen_snapshot is None and screen_fn is screen_snapshot:
        last_known = asof - horizon
        matrix = np.column_stack([c["values"] for c in candidates])[first:last_known + 1]
        cached[first:last_known + 1] = rolling_spearman(
            matrix, y[first:last_known + 1], RULES["icir_window"], RULES["icir_min_pairs"])
    predictions = {arm: np.full(len(dates), np.nan) for arm in ARMS}
    sigma = np.full(len(dates), np.nan)
    refit_index = np.full(len(dates), -1, dtype=int)
    skipped = []
    refits, next_refit, models, full_features = [], first, None, None
    for t in range(first, asof + 1):
        train = completed_indices(level, y, t, horizon)
        if len(train) < config["min_train"]:
            continue
        if models is None or t >= next_refit:
            if frozen_snapshot is not None:
                snapshot = frozen_snapshot
            else:
                # Retained for reproducing the earlier historical re-screening experiment.
                kwargs = {"precomputed_rolling": cached} if screen_fn is screen_snapshot else {}
                snapshot, _ = screen_fn(dates, level, candidates, t, **kwargs)
            selected = [candidate_map[key] for key in snapshot["selected_ids"]]
            full_features = baselines + selected
            full_matrix = np.column_stack([f["values"] for f in full_features])
            models = {"baseline": ridge_fit(base_matrix[train], y[train], config["ridge_alpha"]),
                      "full": ridge_fit(full_matrix[train], y[train], config["ridge_alpha"])}
            train_sigma = float(y[train].std(ddof=1))
            selected_set = set(snapshot["selected_ids"])
            rows = [r for r in snapshot["rows"] if r["id"] in selected_set]
            order = {key: k for k, key in enumerate(snapshot["selected_ids"])}
            rows.sort(key=lambda r: order[r["id"]])
            refits.append({"date": dates[t], "index": t, "train_start": dates[train[0]],
                           "train_end": dates[train[-1]], "last_label_exit": dates[train[-1] + horizon],
                           "train_n": len(train), "sigma": train_sigma,
                           "selected_ids": snapshot["selected_ids"], "selected": rows,
                           "screen_summary": snapshot["summary"], "screen_window": snapshot["window"],
                           "models": {"baseline": model_record(models["baseline"], baselines),
                                      "full": model_record(models["full"], full_features)}})
            next_refit = t + config["refit_every"]
            if progress:
                progress(refits[-1])
        sigma[t] = refits[-1]["sigma"]
        refit_index[t] = len(refits) - 1
        # Evaluate both arms on the same dates. Partial candidate missingness uses train medians.
        selected_values = np.array([f["values"][t] for f in full_features[len(baselines):]])
        coverage = np.mean(np.isfinite(selected_values)) if len(selected_values) else 1.0
        if not np.isfinite(level[t]) or not np.isfinite(base_matrix[t]).all() or coverage < config["prediction_candidate_coverage"]:
            skipped.append({"date": dates[t], "candidate_coverage": float(coverage),
                            "baseline_complete": bool(np.isfinite(base_matrix[t]).all())})
            continue
        predictions["baseline"][t] = float(ridge_predict(models["baseline"], base_matrix[t]))
        predictions["full"][t] = float(ridge_predict(models["full"], full_matrix[t]))
    return {"predictions": predictions, "target": y, "sigma": sigma,
            "refit_index": refit_index, "refits": refits, "skipped": skipped, "config": config}


def forecast_metrics(pred, y):
    mask = np.isfinite(pred) & np.isfinite(y)
    p, actual = pred[mask], y[mask]
    if not len(p):
        return {"n": 0}
    err = p - actual
    denom = np.sum((actual - actual.mean()) ** 2)
    nonzero = (p != 0) & (actual != 0)
    return {"n": len(p), "mae": safe_number(np.mean(np.abs(err))),
            "rmse": safe_number(np.sqrt(np.mean(err ** 2))),
            "ic": safe_number(spearman_columns(p[:, None], actual)[0]),
            "direction_accuracy": safe_number(np.mean(np.sign(p[nonzero]) == np.sign(actual[nonzero]))) if nonzero.any() else None,
            "direction_n": int(nonzero.sum()), "r2": safe_number(1 - np.sum(err ** 2) / denom) if denom > 0 else None,
            "prediction_mean": safe_number(p.mean()), "actual_mean": safe_number(actual.mean())}


def position_for(prediction, threshold):
    if not np.isfinite(prediction) or not np.isfinite(threshold) or threshold <= 0 or abs(prediction) <= threshold:
        return 0
    return int(np.sign(prediction))


def forecast_percentile_gate(prediction, window=250, min_history=60, cut=60.0):
    """Midrank of |today's forecast| within forecasts issued on previous sessions.

    Historical values are archived forecasts from their then-current models, never
    current-model in-sample fitted values. The window is sessions, not compressed
    nonmissing observations. Percentile is in [0,100]; quantile retains forecast units.
    """
    prediction = np.asarray(prediction, float)
    percentile = np.full(len(prediction), np.nan)
    quantile = np.full(len(prediction), np.nan)
    count = np.zeros(len(prediction), int)
    signal = np.zeros(len(prediction), int)
    for t in range(len(prediction)):
        past = np.abs(prediction[max(0, t - window):t])
        past = past[np.isfinite(past)]
        count[t] = len(past)
        if len(past) < min_history or not np.isfinite(prediction[t]):
            continue
        current = abs(prediction[t])
        percentile[t] = 100 * (np.sum(past < current) + .5 * np.sum(past == current)) / len(past)
        # A descriptive reference value; the operative gate is the exact midrank above.
        quantile[t] = float(np.quantile(past, cut / 100))
        if percentile[t] > cut and prediction[t] != 0:
            signal[t] = int(np.sign(prediction[t]))
    return {"percentile": percentile, "quantile": quantile, "history_n": count, "signal": signal}


def simulate(dates, level, prediction, threshold, rule="fixed", horizon=6, phase=None,
             exit_ratio=1.0, gate=None):
    """Position for close-to-close interval (t-1,t] is decided at t-1 close.

    Fixed exits occur at entry_signal+horizon close. Dynamic exits using t's
    forecast execute at the next open proxy (=t close), never before t's P&L.
    One spread unit maximum; no overlapping trades or mark-to-market omissions.
    """
    if rule not in ("fixed", "dynamic"):
        raise ValueError(rule)
    level = np.asarray(level, float)
    eligible = np.isfinite(prediction)
    if gate is not None:
        eligible &= np.isfinite(gate["percentile"])
    valid_predictions = np.flatnonzero(eligible)
    start = int(valid_predictions[0]) if len(valid_predictions) else len(dates) - 1
    n = len(dates)
    position = np.zeros(n, dtype=int)
    daily = np.zeros(n)
    trades, decisions = [], []
    active = None

    def close_trade(t, reason):
        nonlocal active
        entry = active["entry_index"]
        trades.append({"signal_date": dates[entry], "entry_session": dates[entry + 1],
                       "entry_price": float(level[entry]), "exit_signal_date": dates[t],
                       "exit_execution": "当日收盘" if reason == "max_hold" else "下一交易日代理开盘",
                       "exit_session": dates[t] if reason == "max_hold" else (dates[t + 1] if t + 1 < n else None),
                       "exit_price": float(level[t]), "direction": active["direction"],
                       "holding_sessions": t - entry, "entry_prediction": active["prediction"],
                       "entry_threshold": active["threshold"], "entry_percentile": active["percentile"],
                       "entry_history_n": active["history_n"], "reason": reason,
                       "pnl": float(active["direction"] * (level[t] - level[entry])), "closed": True})
        active = None

    for t in range(start, n):
        if t > start and active is not None:
            position[t] = active["direction"]
            if not np.isfinite(level[t - 1:t + 1]).all():
                raise ValueError("持仓区间价格缺失，不能静默记零盈亏")
            daily[t] = position[t] * (level[t] - level[t - 1])
        expired = active is not None and t - active["entry_index"] >= horizon
        reason = None
        if expired:
            reason = "max_hold"
        elif active is not None and rule == "dynamic" and t + 1 < n:
            if not np.isfinite(prediction[t]) or not np.isfinite(threshold[t]):
                reason = "missing_signal"
            elif active["direction"] * prediction[t] <= 0:
                reason = "reverse"
            elif (gate["signal"][t] == 0 if gate is not None else abs(prediction[t]) <= threshold[t] * exit_ratio):
                reason = "weak"
        if reason:
            close_trade(t, reason)
        action = int(gate["signal"][t]) if gate is not None else position_for(prediction[t], threshold[t])
        on_grid = phase is None or (t - start) % horizon == phase
        if active is None and action != 0 and on_grid and t + 1 < n and np.isfinite(level[t]):
            active = {"entry_index": t, "direction": action, "prediction": float(prediction[t]),
                      "threshold": float(threshold[t]),
                      "percentile": float(gate["percentile"][t]) if gate is not None else None,
                      "history_n": int(gate["history_n"][t]) if gate is not None else None}
        next_position = active["direction"] if active else 0
        planned_reason = reason
        if t == n - 1:
            # The last close has a next-session decision but no realized next-session P&L.
            if active is not None and rule == "dynamic":
                if not np.isfinite(prediction[t]) or not np.isfinite(threshold[t]):
                    planned_reason = "missing_signal"
                elif active["direction"] * prediction[t] <= 0:
                    planned_reason = "reverse"
                elif (gate["signal"][t] == 0 if gate is not None else abs(prediction[t]) <= threshold[t] * exit_ratio):
                    planned_reason = "weak"
                if planned_reason:
                    next_position = action if on_grid else 0
            elif active is None:
                next_position = action if on_grid else 0
        decisions.append({"date": dates[t], "next_position": next_position,
                          "exit_reason": planned_reason, "opened": bool(active and active["entry_index"] == t)})
    if active is not None:
        entry = active["entry_index"]
        trades.append({"signal_date": dates[entry], "entry_session": dates[entry + 1],
                       "entry_price": float(level[entry]), "exit_signal_date": None,
                       "exit_execution": None, "exit_session": None, "exit_price": None,
                       "direction": active["direction"], "holding_sessions": n - 1 - entry,
                       "entry_prediction": active["prediction"], "entry_threshold": active["threshold"],
                       "entry_percentile": active["percentile"], "entry_history_n": active["history_n"],
                       "reason": "open", "pnl": float(active["direction"] * (level[-1] - level[entry])), "closed": False})
    equity = np.cumsum(daily)
    drawdown = equity - np.maximum.accumulate(np.r_[0.0, equity])[1:]
    closed = [tr for tr in trades if tr["closed"]]
    return {"position": position, "daily_pnl": daily, "equity": equity, "drawdown": drawdown,
            "trades": trades, "decisions": decisions, "metrics": {
                "total": safe_number(equity[-1]), "max_drawdown": safe_number(-drawdown.min()),
                "closed_trades": len(closed), "open_trades": len(trades) - len(closed),
                "win_rate": safe_number(np.mean([tr["pnl"] > 0 for tr in closed])) if closed else None,
                "mean_trade": safe_number(np.mean([tr["pnl"] for tr in closed])) if closed else None,
                "average_holding": safe_number(np.mean([tr["holding_sessions"] for tr in closed])) if closed else None,
                "exposure": safe_number(np.mean(position[start + 1:] != 0)) if start + 1 < n else None,
                "long_sessions": int((position == 1).sum()), "short_sessions": int((position == -1).sum()),
                "max_holding": max((tr["holding_sessions"] for tr in trades), default=0)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asof", help="只处理截至此日的数据，默认最后期货交易日")
    args = parser.parse_args()
    lib, dates, cmap, candidates = load_candidates()
    asof = max(i for i, d in enumerate(dates) if args.asof is None or d <= args.asof)
    dates = dates[:asof + 1]
    cmap = {key: a[:asof + 1] for key, a in cmap.items()}
    candidates = [c | {"values": c["values"][:asof + 1]} for c in candidates]
    level = 3 * cmap["T主力"] - cmap["TL主力"]
    baselines, policy = baseline_features(level, cmap, dates)
    frozen = json.loads((ROOT / "curve_timing_factor_set.json").read_text(encoding="utf-8"))
    frozen_ids = frozen["selected_ids"]
    if not frozen_ids or len(frozen_ids) > 20 or len(set(frozen_ids)) != len(frozen_ids):
        raise ValueError("固定因子名单必须为1到20个唯一ID")
    missing_ids = set(frozen_ids) - {c["id"] for c in candidates}
    if missing_ids:
        raise ValueError("固定名单的因子不在当前库中，不能自动重新筛选替代：" + str(sorted(missing_ids)))
    result = walk_forward(dates, level, baselines, candidates,
                          frozen_snapshot=frozen,
                          progress=lambda r: print(r["date"], "训练", r["train_n"], "固定因子", len(r["selected_ids"]), flush=True))
    if not result["refits"]:
        raise ValueError("没有足够历史训练模型")
    first = next(i for i, p in enumerate(result["predictions"]["baseline"]) if np.isfinite(p))
    gates = {arm: forecast_percentile_gate(result["predictions"][arm], CONFIG["percentile_window"],
                                          CONFIG["percentile_min_history"], CONFIG["entry_percentile"]) for arm in ARMS}
    trade_eligible = np.flatnonzero(np.isfinite(gates["baseline"]["percentile"]) & np.isfinite(gates["full"]["percentile"]))
    simulations = {rule: {arm: simulate(dates, level, result["predictions"][arm], gates[arm]["quantile"],
                                       rule=rule, gate=gates[arm])
                           for arm in ARMS} for rule in ("fixed", "dynamic")}
    metrics = {arm: forecast_metrics(result["predictions"][arm], result["target"]) for arm in ARMS}
    zero_forecast = np.where(np.isfinite(result["predictions"]["baseline"]), 0.0, np.nan)
    years = sorted({d[:4] for d in dates[first:]})
    annual = []
    for year in years:
        mask = np.array([d.startswith(year) for d in dates])
        annual.append({"year": year, "forecast": {arm: forecast_metrics(
            result["predictions"][arm][mask], result["target"][mask]) for arm in ARMS},
            "pnl": {rule: {arm: safe_number(simulations[rule][arm]["daily_pnl"][mask].sum())
                           for arm in ARMS} for rule in simulations}})
    # Non-overlapping labels by six starting phases, descriptive only; no phase selected by P&L.
    phases = [{"phase": phase, "forecast": {arm: forecast_metrics(
        result["predictions"][arm][first + phase::6], result["target"][first + phase::6]) for arm in ARMS},
        "fixed_grid": {arm: simulate(dates, level, result["predictions"][arm], gates[arm]["quantile"],
                                     phase=phase, gate=gates[arm])["metrics"] for arm in ARMS}} for phase in range(6)]
    frequency = {}
    for refit in result["refits"]:
        for row in refit["selected"]:
            rec = frequency.setdefault(row["id"], {"id": row["id"], "label": row["label"], "group": row["group"],
                                                  "count": 0, "first": refit["date"], "last": None})
            rec["count"] += 1
            rec["last"] = refit["date"]
    final = result["refits"][-1]
    latest = {"date": dates[-1], "refit_date": final["date"],
              "prediction": {arm: safe_number(result["predictions"][arm][-1]) for arm in ARMS},
              "percentile": {arm: safe_number(gates[arm]["percentile"][-1]) for arm in ARMS},
              "quantile": {arm: safe_number(gates[arm]["quantile"][-1]) for arm in ARMS},
              "history_n": {arm: int(gates[arm]["history_n"][-1]) for arm in ARMS},
              "signal": {arm: int(gates[arm]["signal"][-1]) for arm in ARMS},
              "next_position": {rule: {arm: simulations[rule][arm]["decisions"][-1]["next_position"]
                                       for arm in ARMS} for rule in simulations}}
    output = {"meta": {"generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "asof": dates[-1],
                       "library_date": lib["meta"]["date_range"][-1], "target": "3T−TL", "unit": "点",
                       "label": "S[t+6]−S[t]，S=3×T主连−TL主连", "config": CONFIG, "screen_rules": RULES,
                       "selection_mode": "fixed-full-history", "selection_asof": frozen["meta"]["selection_asof"],
                       "frozen_on": frozen["meta"]["frozen_on"],
                       "first_prediction": dates[first], "last_completed_signal": dates[len(dates) - 7],
                       "first_trade_eligible": dates[trade_eligible[0]] if len(trade_eligible) else None,
                       "refits": len(result["refits"]), "candidate_count": len(candidates),
                       "baseline_count": len(baselines), "omo_verified_through": policy["verified_through"],
                       "timing": "t收盘后决策；t+1以t收盘价代理开盘成交；持仓盈亏从t到t+1开始计算。",
                       "training": "扩展历史；仅使用r+6≤重训日的标签；两模型同训练日期；中位数填补、1/99%缩尾、标准化只拟合训练期。",
                       "screening": "固定2026-10-09全历史筛出的20条，不再重筛；每20个交易日仅更新系数与训练期预处理参数，8条基准固定保留。",
                       "execution": "固定6日：平仓后逐日等待下一合格信号；动态：反向/转弱/数据缺失时次日代理开盘离场，可反手，最长6日。单一持仓±1或0。",
                       "threshold": "各模型分别比较预测绝对值与此前250个交易日真实生成的历史预测；至少60个有值才启用；中位并列秩百分位严格>60才开仓，按预测正负确定方向；动态≤60时转弱离场。",
                       "percentile_definition": "100×(历史|预测|小于当前值的个数＋0.5×等于的个数)/历史有效个数；历史窗口不含今天。参考60%分位点使用线性插值，仅供解释；交易以经验百分位为准。",
                       "caveat": "本页固定全历史筛出的因子，早期回测包含因子选择前视，属于探索性回放；仅系数拟合与交易百分位按历史时点推进，不是独立样本外验证。未扣成本，采用昨日收盘代理开盘；6日标签重叠。百分位表示相对历史强度，不是上涨概率。",
                       "input_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                                        ("factor_library.json", "omo_policy_7d.json", "curve_timing_factor_set.json", "prepare_direction_screen.py",
                                         "prepare_strategy_model.py", "prepare_curve_timing_walkforward.py")}},
              "baselines": [{k: f[k] for k in ("id", "label", "unit")} for f in baselines],
              "forecast_metrics": metrics, "forecast_reference": forecast_metrics(zero_forecast, result["target"]),
              "annual": annual, "phases": phases, "latest": latest,
              "refits": result["refits"], "skipped": result["skipped"],
              "selection_frequency": sorted(frequency.values(), key=lambda r: (-r["count"], r["id"])),
              "daily": {"dates": dates[first:], "level": raw_numbers(level[first:]),
                        "target": raw_numbers(result["target"][first:]),
                        "percentile": {arm: raw_numbers(gates[arm]["percentile"][first:]) for arm in ARMS},
                        "quantile": {arm: raw_numbers(gates[arm]["quantile"][first:]) for arm in ARMS},
                        "history_n": {arm: gates[arm]["history_n"][first:].tolist() for arm in ARMS},
                        "entry_signal": {arm: gates[arm]["signal"][first:].tolist() for arm in ARMS},
                        "refit_index": result["refit_index"][first:].tolist(),
                        "prediction": {arm: raw_numbers(result["predictions"][arm][first:]) for arm in ARMS}},
              "strategies": {rule: {arm: {"metrics": sim["metrics"], "trades": sim["trades"],
                                          "position": sim["position"][first:].tolist(),
                                          "equity": raw_numbers(sim["equity"][first:]),
                                          "drawdown": raw_numbers(sim["drawdown"][first:]),
                                          "daily_pnl": raw_numbers(sim["daily_pnl"][first:]),
                                          "decisions": sim["decisions"]}
                                    for arm, sim in arms.items()} for rule, arms in simulations.items()}}
    (ROOT / "curve_timing_walkforward.json").write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":"),
                                                                  allow_nan=False), encoding="utf-8")
    print("预测", metrics)
    for rule, arms in simulations.items():
        print(rule, {arm: sim["metrics"] for arm, sim in arms.items()})


if __name__ == "__main__":
    main()
