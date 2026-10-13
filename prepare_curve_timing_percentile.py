"""Compare absolute-strength entry with signed forecast percentile entry.

Reuse the published once-fitted forecasts without fitting or selecting anything.
All rolling ranks use strictly earlier forecasts. No window/threshold search.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from prepare_curve_timing_walkforward import ARMS, forecast_percentile_gate, simulate
from prepare_direction_screen import raw_numbers, safe_number

ROOT = Path(__file__).resolve().parent
SOURCE = "curve_timing_walkforward.json"
OUTPUT = "curve_timing_percentile.json"


def signed_percentile_gate(prediction, window=250, min_history=60, lower=40.0, upper=60.0):
    if not 0 <= lower < upper <= 100 or not 1 <= min_history <= window:
        raise ValueError("Invalid percentile thresholds or history window")
    prediction = np.asarray(prediction, float)
    n = len(prediction)
    percentile, low_q, high_q = (np.full(n, np.nan) for _ in range(3))
    count, signal = np.zeros(n, int), np.zeros(n, int)
    for t in range(n):
        past = prediction[max(0, t - window):t]
        past = past[np.isfinite(past)]
        count[t] = len(past)
        if len(past) < min_history or not np.isfinite(prediction[t]):
            continue
        value = prediction[t]
        percentile[t] = 100 * (np.sum(past < value) + .5 * np.sum(past == value)) / len(past)
        low_q[t], high_q[t] = np.quantile(past, [lower / 100, upper / 100])
        signal[t] = -1 if percentile[t] < lower else 1 if percentile[t] > upper else 0
    return {"direction_basis": "percentile", "percentile": percentile, "quantile": high_q,
            "lower_quantile": low_q, "history_n": count, "signal": signal}


def serialize_sim(sim):
    return {k: raw_numbers(v) if k in ("daily_pnl", "equity", "drawdown") else v.tolist()
            if k == "position" else v for k, v in sim.items()}


def period_metrics(dates, level, prediction, target, gate, sim, mask):
    pnl, position = sim["daily_pnl"][mask], sim["position"][mask]
    equity = np.cumsum(pnl)
    drawdown = equity - np.maximum.accumulate(np.r_[0., equity])[1:]
    trades = [t for t in sim["trades"] if mask[dates.index(t["signal_date"])]]
    valid = mask & np.isfinite(gate["percentile"])
    opposite = valid & (gate["signal"] * prediction < 0)
    zones = []
    for direction, label in ((-1, "做空信号"), (0, "空仓信号"), (1, "做多信号")):
        paired = valid & (gate["signal"] == direction) & np.isfinite(target)
        actual = target[paired]
        zones.append({"direction": direction, "label": label, "n": int(paired.sum()),
                      "mean_change": safe_number(actual.mean()) if len(actual) else None,
                      "median_change": safe_number(np.median(actual)) if len(actual) else None,
                      "direction_hit": safe_number(np.mean(direction * actual > 0)) if direction and len(actual) else None})
    return {"days": int(mask.sum()), "eligible_days": int(valid.sum()),
            "long_signal_days": int((valid & (gate["signal"] == 1)).sum()),
            "short_signal_days": int((valid & (gate["signal"] == -1)).sum()),
            "opposite_forecast_signal_days": int(opposite.sum()),
            "total": safe_number(pnl.sum()), "max_drawdown": safe_number(-drawdown.min()),
            "new_entries": len(trades), "new_long_entries": sum(t["direction"] == 1 for t in trades),
            "new_short_entries": sum(t["direction"] == -1 for t in trades),
            "long_sessions": int((position == 1).sum()), "short_sessions": int((position == -1).sum()),
            "zones": zones}


def build(source):
    daily = source["daily"]
    dates = daily["dates"]
    level, target = np.array(daily["level"], float), np.array(daily["target"], float)
    predictions = {arm: np.array(daily["prediction"][arm], float) for arm in ARMS}
    config = source["meta"]["config"]
    window, min_history = config["percentile_window"], config["percentile_min_history"]
    gates = {
        "absolute": {arm: forecast_percentile_gate(predictions[arm], window, min_history,
                                                    config["entry_percentile"]) for arm in ARMS},
        "signed": {arm: signed_percentile_gate(predictions[arm], window, min_history) for arm in ARMS}}
    strategies, periods = {}, {}
    masks = {"all": np.ones(len(dates), bool)}
    masks.update({year: np.array([d.startswith(year) for d in dates]) for year in sorted({d[:4] for d in dates})})
    masks["2026-04+"] = np.array([d >= "2026-04-01" for d in dates])
    for method, arms in gates.items():
        strategies[method], periods[method] = {}, {}
        for rule in ("fixed", "dynamic"):
            strategies[method][rule], periods[method][rule] = {}, {}
            for arm in ARMS:
                gate = arms[arm]
                sim = simulate(dates, level, predictions[arm], gate["quantile"], rule=rule, gate=gate)
                if method == "absolute":
                    # Exact reproduction is required before comparing the alternative.
                    old = source["strategies"][rule][arm]
                    for key in ("position", "daily_pnl", "equity", "drawdown"):
                        np.testing.assert_allclose(sim[key], old[key], atol=1e-12, rtol=0)
                    if sim["trades"] != old["trades"] or sim["decisions"] != old["decisions"]:
                        raise ValueError("Original accounting was not reproduced")
                strategies[method][rule][arm] = serialize_sim(sim)
                periods[method][rule][arm] = {name: period_metrics(dates, level, predictions[arm], target,
                                                                  gate, sim, mask) for name, mask in masks.items()}
    return {"meta": {"generated": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
                     "asof": source["meta"]["asof"], "source": SOURCE,
                     "config": {"window": window, "min_history": min_history, "lower": 40, "upper": 60, "horizon": 6},
                     "train_through": source["meta"]["train_through"],
                     "model_changed": False, "preprocessing_changed": False,
                     "definition": "带符号预测与此前250个交易日预测比较；中位并列秩；严格<40做空、>60做多，含40/60的中间区间空仓。预测正负不限制交易方向。",
                     "dynamic_exit": "百分位进入中间区间或信号缺失时次日代理开盘退出；进入另一侧时退出并可反手；最长6日。原规则仍按原始预测反向/强度转弱退出。",
                     "caveat": "沿用全历史筛出的20因子，含因子选择前视。此次规则在观察2026年结果后提出，属于探索性对照。未扣成本；昨日收盘代理开盘；信号分组6日标签重叠。年度回撤以该年度起点为零重算。",
                     "input_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                                      (SOURCE, "prepare_curve_timing_percentile.py", "prepare_curve_timing_walkforward.py")}},
            "forecast_metrics": source["forecast_metrics"], "source_models": source["refits"][0]["models"],
            "daily": {"dates": dates, "level": daily["level"], "target": daily["target"],
                      "prediction": daily["prediction"]},
            "gates": {method: {arm: {k: raw_numbers(v) if k in ("percentile", "quantile", "lower_quantile")
                                      else v.tolist() for k, v in gate.items() if k != "direction_basis"}
                                for arm, gate in arms.items()} for method, arms in gates.items()},
            "strategies": strategies, "periods": periods}


def main():
    source = json.loads((ROOT / SOURCE).read_text(encoding="utf-8"))
    output = build(source)
    (ROOT / OUTPUT).write_bytes(json.dumps(output, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    for method in output["strategies"]:
        for rule in output["strategies"][method]:
            print(method, rule, "B", output["strategies"][method][rule]["full"]["metrics"], flush=True)
            print("2026", output["periods"][method][rule]["full"]["2026"], flush=True)


if __name__ == "__main__":
    main()
