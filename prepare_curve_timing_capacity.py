"""Type-aware preprocessing and nested 20/30/40 factor Ridge comparisons.

Protect prices, yields, spreads, bounded indicators and standardized indicators
from winsorization. Keep train-only 1/99% clipping for institution amount factors.
Fit once through 2024, use unchanged signed-percentile entries and two exit rules.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from prepare_curve_timing_percentile import period_metrics, serialize_sim, signed_percentile_gate
from prepare_curve_timing_walkforward import CONFIG, completed_indices, forecast_metrics, simulate
from prepare_direction_screen import baseline_features, load_candidates, raw_numbers, safe_number
from prepare_strategy_model import make_offset_labels

ROOT = Path(__file__).resolve().parent
SETS_FILE = "curve_timing_capacity_factor_sets.json"
OUTPUT = "curve_timing_capacity.json"
VARIANTS = ("baseline", "f20", "f30", "f40")
ALL_VARIANTS = ("legacy20",) + VARIANTS
LABELS = {"legacy20": "旧缩尾 · 20+8", "baseline": "修正 · 仅8基准",
          "f20": "修正 · 20+8", "f30": "修正 · 30+8", "f40": "修正 · 40+8"}


def digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def freeze_sets():
    """Extend the already accepted ranking once; never re-screen or replace it."""
    frozen = json.loads((ROOT / "curve_timing_factor_set.json").read_text(encoding="utf-8"))
    if (ROOT / SETS_FILE).exists():
        result = json.loads((ROOT / SETS_FILE).read_text(encoding="utf-8"))
    else:
        screen = json.loads((ROOT / "direction_screen.json").read_text(encoding="utf-8"))
        qualified = sorted((r for r in screen["rows"] if r["qualified"]), key=lambda r: r["qualified_rank"])
        if len(qualified) < 40 or [r["id"] for r in qualified[:20]] != frozen["selected_ids"]:
            raise ValueError("Existing frozen top20 and screening ranking do not agree")
        result = {"meta": {"selection_asof": frozen["meta"]["selection_asof"],
                           "frozen_on": "2026-10-13", "qualified_count": len(qualified),
                           "source_screen_sha256": digest("direction_screen.json"),
                           "source_frozen20_sha256": digest("curve_timing_factor_set.json"),
                           "selection": "沿用已接受的全历史87条合格名单排序，取前20/30/40；不放松规则，不重新筛选。"},
                  "sets": {f"f{n}": [r["id"] for r in qualified[:n]] for n in (20, 30, 40)},
                  "rows": qualified[:40], "window": screen["window"], "rules": screen["meta"]["rules"]}
        (ROOT / SETS_FILE).write_bytes(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    if result["sets"]["f20"] != frozen["selected_ids"]:
        raise ValueError("The original20 must stay exactly unchanged")
    for n in (20, 30, 40):
        ids = result["sets"][f"f{n}"]
        if len(ids) != n or len(set(ids)) != n or ids != result["sets"]["f40"][:n]:
            raise ValueError("Factor sets must be unique nested prefixes")
    return result


def clip_policy(feature):
    key = feature["id"]
    group = key.split("|", 1)[0]
    if group == "baseline":
        return False, "市场基准及价差水平、MA、动量、Z值保留真实变化"
    if group in ("curve", "carry", "tech"):
        return False, "价格、利率、曲线利差及技术指标不做训练分位截断"
    if group == "cash":
        return True, "机构日度净买入金额：训练1%/99%缩尾"
    if group == "repo":
        return (False, "回购利差保留真实变化") if key.endswith("|rate_spread") else (
            True, "机构回购金额/余额：训练1%/99%缩尾")
    if group == "derived":
        op = key.split("|")[1]
        if op in ("accel", "slope", "tenor_diff", "cfg_trd", "cfg_trd_accel", "act_anom"):
            return True, "机构金额衍生及异常活动比率：训练1%/99%缩尾"
        if op in ("days_ratio", "rank", "pos_util", "cons", "hhi", "tenor_zdiff"):
            return False, "有自然边界的比例/分位或标准化期限差保留原值"
    raise ValueError("Feature preprocessing must be explicitly classified: " + key)


class TypeAwareScaler:
    def __init__(self, features):
        self.policies = [clip_policy(f) for f in features]
        self.clip_mask = np.array([p[0] for p in self.policies], bool)

    def fit(self, x):
        x = np.asarray(x, float)
        self.median = np.nanmedian(x, axis=0)
        if not np.isfinite(self.median).all():
            raise ValueError("A frozen factor has no finite training observations")
        filled = np.where(np.isfinite(x), x, self.median)
        self.lo, self.hi = np.full(x.shape[1], np.nan), np.full(x.shape[1], np.nan)
        self.lo[self.clip_mask] = np.quantile(filled[:, self.clip_mask], .01, axis=0)
        self.hi[self.clip_mask] = np.quantile(filled[:, self.clip_mask], .99, axis=0)
        transformed = self.apply_limits(filled)
        self.mean, self.std = transformed.mean(axis=0), transformed.std(axis=0)
        self.std = np.where(self.std > 1e-10, self.std, 1.)
        return self

    def apply_limits(self, filled):
        out = np.array(filled, float, copy=True)
        out[..., self.clip_mask] = np.clip(out[..., self.clip_mask], self.lo[self.clip_mask], self.hi[self.clip_mask])
        return out

    def transform(self, x):
        x = np.asarray(x, float)
        filled = np.where(np.isfinite(x), x, self.median)
        return (self.apply_limits(filled) - self.mean) / self.std


def fit_model(x, y, features, alpha=15.):
    scaler = TypeAwareScaler(features).fit(x)
    z = scaler.transform(x)
    intercept = float(np.mean(y))
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - intercept))
    return scaler, intercept, coef


def predict(model, x):
    scaler, intercept, coef = model
    return intercept + scaler.transform(x) @ coef


def model_record(model, features):
    scaler, intercept, coef = model
    return {"intercept": intercept, "features": [
        {"id": f["id"], "label": f["label"], "unit": f.get("unit", ""), "group": f["id"].split("|")[0],
         "coef": float(coef[j]), "median": float(scaler.median[j]),
         "clip_enabled": bool(scaler.clip_mask[j]), "clip_reason": scaler.policies[j][1],
         "clip_low": float(scaler.lo[j]) if scaler.clip_mask[j] else None,
         "clip_high": float(scaler.hi[j]) if scaler.clip_mask[j] else None,
         "mean": float(scaler.mean[j]), "std": float(scaler.std[j])}
        for j, f in enumerate(features)]}


def fit_variants(dates, level, baselines, candidates, sets, train_through="2024-12-31"):
    cutoff = max(i for i, day in enumerate(dates) if day <= train_through)
    target = make_offset_labels(level, 6)
    train = completed_indices(level, target, cutoff, 6)
    if len(train) < 300 or cutoff == len(dates) - 1:
        raise ValueError("Insufficient train/test history")
    cmap = {f["id"]: f for f in candidates}
    features = {"baseline": baselines}
    features.update({k: baselines + [cmap[key] for key in ids] for k, ids in sets.items()})
    matrices = {k: np.column_stack([f["values"] for f in features[k]]) for k in VARIANTS}
    models = {k: fit_model(matrices[k][train], target[train], features[k], CONFIG["ridge_alpha"]) for k in VARIANTS}
    predictions = {k: np.full(len(dates), np.nan) for k in VARIANTS}
    skipped = []
    for t in range(cutoff + 1, len(dates)):
        coverage = {k: float(np.mean(np.isfinite(matrices[k][t, len(baselines):]))) for k in ("f20", "f30", "f40")}
        if not np.isfinite(level[t]) or not np.isfinite(matrices["baseline"][t]).all() or min(coverage.values()) < .9:
            skipped.append({"date": dates[t], "coverage": coverage})
            continue
        for k in VARIANTS:
            predictions[k][t] = float(predict(models[k], matrices[k][t]))
    return {"predictions": predictions, "target": target, "models": models, "features": features,
            "matrices": matrices, "train": train, "cutoff": cutoff, "skipped": skipped,
            "train_metrics": {k: forecast_metrics(predict(models[k], matrices[k][train]), target[train]) for k in VARIANTS}}


def clipping_audit(result, dates, legacy):
    old = {f["id"]: f for f in legacy["refits"][0]["models"]["full"]["features"]}
    mask = np.array([d.startswith("2026") for d in dates])
    rows = []
    for j, f in enumerate(result["features"]["f40"]):
        scaler = result["models"]["f40"][0]
        values = result["matrices"]["f40"][mask, j]
        valid = values[np.isfinite(values)]
        old_rec = old.get(f["id"])
        train = result["matrices"]["f40"][result["train"], j]
        rows.append({"id": f["id"], "label": f["label"], "clip_enabled": bool(scaler.clip_mask[j]),
                     "reason": scaler.policies[j][1], "new_low": safe_number(scaler.lo[j]), "new_high": safe_number(scaler.hi[j]),
                     "old_low": old_rec["clip_low"] if old_rec else None, "old_high": old_rec["clip_high"] if old_rec else None,
                     "2026_raw_median": safe_number(np.median(valid)) if len(valid) else None,
                     "2026_old_clip_rate": safe_number(np.mean((valid < old_rec["clip_low"]) | (valid > old_rec["clip_high"]))) if old_rec and len(valid) else None,
                     "2026_new_clip_rate": safe_number(np.mean((valid < scaler.lo[j]) | (valid > scaler.hi[j]))) if scaler.clip_mask[j] and len(valid) else 0.,
                     "train_missing_rate": safe_number(np.mean(~np.isfinite(train)))})
    return rows


def build():
    frozen = freeze_sets()
    legacy = json.loads((ROOT / "curve_timing_walkforward.json").read_text(encoding="utf-8"))
    if digest("factor_library.json") != legacy["meta"]["input_hashes"]["factor_library.json"]:
        raise ValueError("Factor library changed; cannot silently compare different data snapshots")
    lib, dates, cmap, candidates = load_candidates()
    asof = dates.index(legacy["meta"]["asof"])
    dates = dates[:asof + 1]
    cmap = {k: v[:asof + 1] for k, v in cmap.items()}
    candidates = [f | {"values": f["values"][:asof + 1]} for f in candidates]
    level = 3 * cmap["T主力"] - cmap["TL主力"]
    baselines, policy = baseline_features(level, cmap, dates)
    result = fit_variants(dates, level, baselines, candidates, frozen["sets"])
    start = result["cutoff"] + 1
    test_dates, test_level, test_target = dates[start:], level[start:], result["target"][start:]
    if test_dates != legacy["daily"]["dates"]:
        raise ValueError("Comparison dates differ")
    np.testing.assert_array_equal(test_level, np.array(legacy["daily"]["level"], float))
    np.testing.assert_array_equal(test_target, np.array(legacy["daily"]["target"], float))
    predictions = {k: p[start:] for k, p in result["predictions"].items()}
    predictions["legacy20"] = np.array(legacy["daily"]["prediction"]["full"], float)
    # Use a shared availability calendar for fair comparison of both preprocessing and factor counts.
    shared = np.isfinite(predictions["f40"])
    predictions["legacy20"] = np.where(shared, predictions["legacy20"], np.nan)
    gates = {k: signed_percentile_gate(predictions[k]) for k in ALL_VARIANTS}
    strategies, periods = {}, {}
    masks = {"all": np.ones(len(test_dates), bool),
             "2025": np.array([d.startswith("2025") for d in test_dates]),
             "2026": np.array([d.startswith("2026") for d in test_dates]),
             "2026-04+": np.array([d >= "2026-04-01" for d in test_dates])}
    for rule in ("fixed", "dynamic"):
        strategies[rule], periods[rule] = {}, {}
        for k in ALL_VARIANTS:
            sim = simulate(test_dates, test_level, predictions[k], gates[k]["quantile"], rule=rule, gate=gates[k])
            strategies[rule][k] = serialize_sim(sim)
            periods[rule][k] = {}
            for name, mask in masks.items():
                row = period_metrics(test_dates, test_level, predictions[k], test_target, gates[k], sim, mask)
                row["long_pnl"] = safe_number(sim["daily_pnl"][mask & (sim["position"] == 1)].sum())
                row["short_pnl"] = safe_number(sim["daily_pnl"][mask & (sim["position"] == -1)].sum())
                row["forecast"] = forecast_metrics(predictions[k][mask], test_target[mask])
                periods[rule][k][name] = row
    model_records = {k: model_record(result["models"][k], result["features"][k]) for k in VARIANTS}
    model_records["legacy20"] = legacy["refits"][0]["models"]["full"]
    source_names = ("factor_library.json", "curve_timing_walkforward.json", "curve_timing_factor_set.json", SETS_FILE,
                    "omo_policy_7d.json", "prepare_curve_timing_capacity.py", "prepare_curve_timing_percentile.py",
                    "prepare_curve_timing_walkforward.py", "prepare_direction_screen.py", "prepare_strategy_model.py")
    rows_by_id = {f["id"]: f for f in candidates}
    return {"meta": {"generated": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
                     "asof": dates[-1], "train_through": "2024-12-31", "train_n": len(result["train"]),
                     "train_start": dates[result["train"][0]], "train_end": dates[result["train"][-1]],
                     "last_train_label_exit": dates[result["train"][-1] + 6], "test_n": len(test_dates),
                     "variants": list(ALL_VARIANTS), "labels": LABELS,
                     "config": {"alpha": 15, "horizon": 6, "window": 250, "min_history": 60, "lower": 40, "upper": 60},
                     "fit_count": len(VARIANTS), "target_clipped": False, "level_clipped": False,
                     "baseline_clipped": False, "selection_asof": frozen["meta"]["selection_asof"],
                     "preprocessing": "仅机构金额及指定金额衍生做训练1%/99%缩尾。价差原值、目标变动、价差MA/动量/Z、收益率、曲线利差及技术指标不缩尾；全部填补与标准化参数只拟合训练集并冻结。",
                     "availability": "8基准完整，20/30/40各池有值比例均≥90%才共同预测；使用相同训练标签与共同测试日期。",
                     "caveat": "沿用已接受的全历史筛选名单，含选择前视；预处理和因子数量比较是在观察测试结果后提出，属于探索性研究。昨日收盘代理次日开盘，不扣成本；6日标签重叠。区间回撤从各区间起点重算。",
                     "input_hashes": {name: digest(name) for name in source_names}},
            "factor_sets": frozen, "models": model_records,
            "group_counts": {k: dict(Counter(rows_by_id[x]["group"] for x in ids)) for k, ids in frozen["sets"].items()},
            "forecast": {k: {"test": forecast_metrics(predictions[k], test_target),
                              "train": result["train_metrics"][k] if k in VARIANTS else legacy["train_forecast_metrics"]["full"]} for k in ALL_VARIANTS},
            "daily": {"dates": test_dates, "level": raw_numbers(test_level), "target": raw_numbers(test_target),
                      "prediction": {k: raw_numbers(p) for k, p in predictions.items()}},
            "gates": {k: {name: raw_numbers(v) if name in ("percentile", "quantile", "lower_quantile") else v.tolist()
                          for name, v in gate.items() if name != "direction_basis"} for k, gate in gates.items()},
            "strategies": strategies, "periods": periods,
            "clipping_audit": clipping_audit(result, dates, legacy), "skipped": result["skipped"]}


def main():
    output = build()
    (ROOT / OUTPUT).write_bytes(json.dumps(output, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    print("train", output["meta"]["train_n"], "test", output["meta"]["test_n"], "skipped", len(output["skipped"]))
    for k in ALL_VARIANTS:
        print(k, "test forecast", output["forecast"][k]["test"])
        for rule in ("fixed", "dynamic"):
            s = output["strategies"][rule][k]["metrics"]
            y = output["periods"][rule][k]["2026"]
            print(rule, "total", s["total"], "DD", s["max_drawdown"], "2026", y["total"], "entries", y["new_entries"])


if __name__ == "__main__":
    main()
