"""3T−TL direction-only screening snapshot, with explicit reasons for every candidate.

This is a current research candidate list, not a historical out-of-sample backtest.
No correlation deduplication, target-level blacklist, or model-increment filter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import prepare_factor_checkup as library_loader
from prepare_strategy_model import load_omo_policy, make_offset_labels, _trailing_mean, _trailing_z

ROOT = Path(__file__).resolve().parent
RULES = {
    "horizon": 6, "min_coverage": 0.90, "min_obs": 300, "min_abs_ic": 0.05,
    "rank_cut": 50, "icir_window": 60, "icir_min_pairs": 54,
    "quintile_min_obs": 30, "quintile_min_steps": 3,
    "stability_blocks": 4, "stability_block_sessions": 63,
    "stability_min_pairs": 30, "stability_min_same": 3, "max_selected": 20,
}
CHECK_LABELS = {
    "data": "数据质量", "ic_floor": "|IC|≥0.05", "ic_rank": "|IC|全库前50%",
    "icir_rank": "|ICIR|全库前50%", "sign_agree": "IC与ICIR同号",
    "group_count": "五组各≥30观测", "group_trend": "五分组同向",
    "stability": "四段至少三段同向",
}
DERIVED_FORMULAS = {
    "days_ratio": "100×过去20日净买入为正的天数/有效天数",
    "accel": "过去5日净买入均值−过去20日均值",
    "slope": "过去10日累计净买入路径对时间的OLS斜率；10日历史要求完整",
    "rank": "同期限各机构过去20日净买入均值的截面百分位；并列平均秩",
    "pos_util": "20日均值×20在过去250日的最小/最大区间位置；不等于真实机构仓位",
    "act_anom": "MA20(|净买入|)/MA250(MA20(|净买入|))；不是总成交量",
    "cons": "100×(20日均值为正的机构数−为负的机构数)/(为正或负的机构数)",
    "hhi": "100×Σ(各机构|20日均值|占比²)",
    "cfg_trd": "配置盘机构20日净买入均值之和−交易盘机构20日均值之和；归属见因子库",
    "cfg_trd_accel": "配置盘与交易盘的MA5−MA20之差；归属见因子库",
    "tenor_diff": "同机构两期限严格5日累计净买入额之差",
    "tenor_zdiff": "两端各自的5日累计净买入做Z60后相减；Z60含当前值",
}


def safe_number(value, decimals=6):
    return round(float(value), decimals) if value is not None and np.isfinite(value) else None


def numbers(values):
    return [safe_number(v) for v in values]


def raw_numbers(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def spearman_columns(a, y, min_pairs=2):
    """Exact paired ranks; missing observations retain their original calendar positions."""
    a, y = np.asarray(a, float), np.asarray(y, float)
    mask = np.isfinite(a) & np.isfinite(y)[:, None]
    rx = pd.DataFrame(np.where(mask, a, np.nan)).rank().to_numpy()
    ry = pd.DataFrame(np.where(mask, y[:, None], np.nan)).rank().to_numpy()
    rx, ry = np.where(mask, rx, 0), np.where(mask, ry, 0)
    n = mask.sum(axis=0)
    denom = np.maximum(n, 1)
    cov = (rx * ry).sum(axis=0) / denom - rx.sum(axis=0) * ry.sum(axis=0) / denom ** 2
    vx = (rx * rx).sum(axis=0) / denom - (rx.sum(axis=0) / denom) ** 2
    vy = (ry * ry).sum(axis=0) / denom - (ry.sum(axis=0) / denom) ** 2
    ok = (n >= min_pairs) & (vx > 1e-12) & (vy > 1e-12)
    result = np.full(a.shape[1], np.nan)
    result[ok] = cov[ok] / np.sqrt(vx[ok] * vy[ok])
    return np.clip(result, -1, 1)


def rolling_spearman(a, y, window=60, min_pairs=54):
    result = np.full((len(y), a.shape[1]), np.nan)
    for end in range(window - 1, len(y)):
        begin = end - window + 1
        result[end] = spearman_columns(a[begin:end + 1], y[begin:end + 1], min_pairs)
    return result


def global_percentiles(ic, icir, eligible):
    out = []
    for values in (ic, icir):
        p = np.full(len(values), np.nan)
        idx = np.flatnonzero(eligible)
        if len(idx) == 1:
            p[idx] = 50
        elif len(idx) > 1:
            p[idx] = 100 * (rankdata(np.abs(values[idx]), method="average") - 1) / (len(idx) - 1)
        out.append(p)
    return out


def rank_gate(ic_pct, icir_pct):
    return bool(ic_pct >= RULES["rank_cut"] and icir_pct >= RULES["rank_cut"])


def quintiles(x, y, sign):
    mask = np.isfinite(x) & np.isfinite(y)
    xx, yy = x[mask], y[mask]
    if len(xx) == 0:
        return {"edges": [], "counts": [0] * 5, "means": [None] * 5,
                "difference": None, "same_steps": 0, "count_pass": False, "trend_pass": False}
    edges = np.quantile(xx, [0.2, 0.4, 0.6, 0.8])
    # Equal values remain in the same group; do not split ties according to date or outcome.
    bins = np.searchsorted(edges, xx, side="left")
    counts = [int(np.sum(bins == k)) for k in range(5)]
    means = np.array([np.mean(yy[bins == k]) if counts[k] else np.nan for k in range(5)])
    steps = int(np.sum(sign * np.diff(means) > 0)) if np.isfinite(means).all() else 0
    difference = means[-1] - means[0]
    count_pass = min(counts) >= RULES["quintile_min_obs"]
    trend_pass = bool(np.isfinite(difference) and sign * difference > 0 and
                      steps >= RULES["quintile_min_steps"])
    return {"edges": numbers(edges), "counts": counts, "means": numbers(means),
            "difference": safe_number(difference), "same_steps": steps,
            "count_pass": count_pass, "trend_pass": trend_pass}


def quarter_blocks(a, y, dates, sign):
    n, w = len(y), RULES["stability_block_sessions"]
    rows = []
    for k in range(4):
        begin, end = n - 4 * w + k * w, n - 4 * w + (k + 1) * w
        if begin < 0:
            rows.append({"start": None, "end": None, "n": 0, "ic": None, "same": False})
            continue
        xx, yy = a[begin:end], y[begin:end]
        mask = np.isfinite(xx) & np.isfinite(yy)
        ic = spearman_columns(xx[:, None], yy, RULES["stability_min_pairs"])[0]
        rows.append({"start": dates[begin], "end": dates[end - 1], "n": int(mask.sum()),
                     "ic": safe_number(ic), "same": bool(np.isfinite(ic) and sign * ic > 0)})
    return rows


def trailing_pct(a, window=250):
    out = np.full(len(a), np.nan)
    for t in range(len(a)):
        hist = a[max(0, t - window + 1):t + 1]
        hist = hist[np.isfinite(hist)]
        if np.isfinite(a[t]) and len(hist) >= max(5, int(window * 0.7)):
            out[t] = np.mean(hist <= a[t]) * 100
    return out


def technical_candidates(cmap):
    candidates = []
    for contract in ("T", "TF", "TS", "TL"):
        p = np.asarray(cmap[contract + "主力"], float)
        s = pd.Series(p)
        ret = {}
        for n in (1, 20, 60):
            previous = s.shift(n)
            ret[n] = ((s / previous - 1) * 100).where(previous != 0).to_numpy()
        ma5 = s.rolling(5, min_periods=4).mean().where(s.notna()).to_numpy()
        ma20 = s.rolling(20, min_periods=14).mean().where(s.notna()).to_numpy()
        sd20 = s.rolling(20).std(ddof=0).to_numpy()
        rsi = np.full(len(p), np.nan)
        gains, losses, g, loss = [], [], None, None
        for t in range(1, len(p)):
            if not np.isfinite(p[t - 1:t + 1]).all():
                gains, losses, g, loss = [], [], None, None
                continue
            delta = p[t] - p[t - 1]
            up, down = max(delta, 0), max(-delta, 0)
            if g is None:
                gains.append(up); losses.append(down)
                if len(gains) < 14:
                    continue
                g, loss = np.mean(gains), np.mean(losses)
            else:
                g, loss = (13 * g + up) / 14, (13 * loss + down) / 14
            rsi[t] = 100 if loss == 0 else 100 - 100 / (1 + g / loss)
        fast = s.ewm(span=12, adjust=False, ignore_na=True).mean().where(s.notna())
        slow = s.ewm(span=26, adjust=False, ignore_na=True).mean().where(s.notna())
        diff = fast - slow
        macd = (diff - diff.ewm(span=9, adjust=False, ignore_na=True).mean()).to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            boll = np.where(sd20 > 0, (p - ma20) / (2 * sd20), np.nan)
        definitions = [
            ("mom20", "动量20日", "%", ret[20], "100×(P_t/P_{t−20}−1)"),
            ("mom60", "动量60日", "%", ret[60], "100×(P_t/P_{t−60}−1)"),
            ("ma_gap", "MA5−MA20", "点", ma5 - ma20, "MA5(P)−MA20(P)"),
            ("rsi14", "RSI(14)", "", rsi, "Wilder RSI，14个期货交易日"),
            ("macd", "MACD柱", "点", macd, "EMA12−EMA26−EMA9(EMA12−EMA26)"),
            ("boll", "布林位置", "", boll, "(P−MA20)/(2×STD20)，STD使用ddof=0"),
            ("vol20", "20日波动率", "%", pd.Series(ret[1]).rolling(20).std(ddof=0).to_numpy() * np.sqrt(242),
             "过去20日百分比变动标准差×√242，ddof=0"),
            ("pct250", "250日分位", "%", trailing_pct(p), "当前价格在过去250个期货交易日中的经验分位，至少175个有效值"),
        ]
        for key, label, unit, values, formula in definitions:
            candidates.append({"id": f"tech|{contract}|{key}", "group": "tech", "key": f"{contract}|{key}",
                               "label": f"技术·{label}·{contract}主力", "unit": unit,
                               "formula": formula, "values": values})
    return candidates


def screen_snapshot(dates, level, candidates, asof_index, lookback=None, precomputed_rolling=None):
    y = make_offset_labels(np.asarray(level, float), RULES["horizon"])
    # Only labels whose exit close is known at the screening close may be used.
    end = asof_index - RULES["horizon"]
    first = np.flatnonzero(np.isfinite(level))[0]
    # Retain the session axis even if an interior price/label is missing.
    idx = np.arange(first, end + 1)
    if lookback:
        idx = idx[-lookback:]
    if len(idx) < RULES["min_obs"]:
        raise ValueError("共同历史不足300个已完成标签，不能执行正式筛选")
    a = np.column_stack([c["values"] for c in candidates])[idx]
    yy, window_dates = y[idx], [dates[i] for i in idx]
    ic = spearman_columns(a, yy)
    if precomputed_rolling is None:
        ric = rolling_spearman(a, yy, RULES["icir_window"], RULES["icir_min_pairs"])
    else:
        if lookback is not None:
            raise ValueError("预计算滚动IC仅用于从共同历史起点扩展的窗口")
        ric = precomputed_rolling[idx]
    ir, ric_counts = np.full(len(candidates), np.nan), np.isfinite(ric).sum(axis=0)
    for j in range(len(candidates)):
        valid = ric[np.isfinite(ric[:, j]), j]
        if len(valid) >= 2 and valid.std(ddof=1) > 1e-12:
            ir[j] = valid.mean() / valid.std(ddof=1)
    count = (np.isfinite(a) & np.isfinite(yy)[:, None]).sum(axis=0)
    coverage = count / len(idx)
    eligible = ((count >= RULES["min_obs"]) & (coverage >= RULES["min_coverage"])
                & np.isfinite(ic) & np.isfinite(ir))
    pic, pir = global_percentiles(ic, ir, eligible)
    # Four paired-rank passes across all columns, rather than four passes per factor.
    block_data = []
    w = RULES["stability_block_sessions"]
    for k in range(4):
        begin, finish = len(yy) - 4 * w + k * w, len(yy) - 4 * w + (k + 1) * w
        if begin < 0:
            block_data.append(None)
        else:
            block_data.append((window_dates[begin], window_dates[finish - 1],
                               (np.isfinite(a[begin:finish]) & np.isfinite(yy[begin:finish, None])).sum(axis=0),
                               spearman_columns(a[begin:finish], yy[begin:finish], RULES["stability_min_pairs"])))
    rows = []
    for j, c in enumerate(candidates):
        sign = int(np.sign(ic[j])) if np.isfinite(ic[j]) else 0
        q = quintiles(a[:, j], yy, sign)
        quarters = [{"start": b[0], "end": b[1], "n": int(b[2][j]), "ic": safe_number(b[3][j]),
                     "same": bool(np.isfinite(b[3][j]) and sign * b[3][j] > 0)} if b is not None else
                    {"start": None, "end": None, "n": 0, "ic": None, "same": False} for b in block_data]
        same = sum(b["same"] for b in quarters)
        mask = np.isfinite(a[:, j]) & np.isfinite(yy)
        pearson = np.corrcoef(a[mask, j], yy[mask])[0, 1] if mask.sum() >= 2 and np.isfinite(ic[j]) else np.nan
        checks = {
            "data": bool(eligible[j]), "ic_floor": bool(abs(ic[j]) >= RULES["min_abs_ic"]),
            "ic_rank": bool(pic[j] >= RULES["rank_cut"]), "icir_rank": bool(pir[j] >= RULES["rank_cut"]),
            "sign_agree": bool(ic[j] * ir[j] > 0), "group_count": q["count_pass"],
            "group_trend": q["trend_pass"], "stability": same >= RULES["stability_min_same"],
        }
        reasons = [CHECK_LABELS[key] + "未通过" for key, passed in checks.items() if not passed]
        if not eligible[j]:
            detail = []
            if count[j] < RULES["min_obs"]: detail.append("有效观测不足300")
            if coverage[j] < RULES["min_coverage"]: detail.append("覆盖不足90%")
            if not np.isfinite(ic[j]): detail.append("IC不可计算（样本或常数问题）")
            if not np.isfinite(ir[j]): detail.append("ICIR不可计算（滚动样本或标准差问题）")
            reasons[0] = "；".join(detail)
        rows.append({key: c[key] for key in ("id", "group", "key", "label", "unit", "formula")} | {
            "n": int(count[j]), "coverage": safe_number(coverage[j]), "ic": safe_number(ic[j]),
            "pearson_ic": safe_number(pearson), "icir": safe_number(ir[j]),
            "rolling_ic_n": int(ric_counts[j]), "ic_percentile": safe_number(pic[j], 3),
            "icir_percentile": safe_number(pir[j], 3), "score": safe_number((pic[j] + pir[j]) / 2, 3),
            "direction": sign, "quintiles": q, "quarters": quarters, "same_blocks": same,
            "checks": checks, "reasons": reasons, "qualified": all(checks.values()),
            "status": "rejected" if eligible[j] else "insufficient",
        })
    qualified = sorted((r for r in rows if r["qualified"]),
                       key=lambda r: (-r["score"], -abs(r["ic"]), -abs(r["icir"]), r["id"]))
    for rank, row in enumerate(qualified, 1):
        row["qualified_rank"] = rank
        row["status"] = "selected" if rank <= RULES["max_selected"] else "capped"
        row["reasons"] = [] if row["status"] == "selected" else ["全部条件通过，但超过20条上限"]
    stages, active = [], np.ones(len(rows), dtype=bool)
    for key, label in CHECK_LABELS.items():
        active &= np.array([row["checks"][key] for row in rows])
        stages.append({"key": key, "label": label, "remaining": int(active.sum())})
    return {"rows": rows, "selected_ids": [r["id"] for r in qualified[:20]],
            "window": {"start": window_dates[0], "end": window_dates[-1], "sessions": len(idx),
                       "last_label_exit": dates[idx[-1] + 6], "rank_universe": int(eligible.sum())},
            "summary": {"total": len(rows), "data_qualified": int(eligible.sum()),
                        "qualified": len(qualified), "selected": min(len(qualified), 20), "stages": stages}}, y


def baseline_features(level, cmap, dates):
    omo, policy = load_omo_policy(dates)
    dr = cmap["DR001-MA10"]
    definitions = [
        ("dr_omo", "DR001(MA10)−7天OMO", "bp", 100 * (dr - omo)),
        ("y10_dr", "10年国债−DR001(MA10)", "bp", 100 * (cmap["10年国债"] - dr)),
        ("y10_y2", "10年国债−2年国债", "bp", 100 * (cmap["10年国债"] - cmap["2年国债"])),
        ("y30_y10", "30年国债−10年国债", "bp", 100 * (cmap["30年国债"] - cmap["10年国债"])),
        ("spread_ma5", "3T−TL的MA5", "点", _trailing_mean(level, 5)),
        ("spread_ma20", "3T−TL的MA20", "点", _trailing_mean(level, 20)),
        ("spread_mom5", "3T−TL过去5日变动", "点", np.r_[np.full(5, np.nan), level[5:] - level[:-5]]),
        ("spread_z60", "3T−TL的Z60", "标准差", _trailing_z(level, 60)),
    ]
    return [{"id": "baseline|" + key, "label": label, "unit": unit, "values": values}
            for key, label, unit, values in definitions], policy


def load_candidates():
    """Shared full-pool construction for the snapshot and historical re-screening."""
    lib, dates, series, cmap = library_loader.load_library()
    units = {"cash": "亿元", "curve": "百分点/期货报价", "carry": "bp"}
    candidates = []
    for group, key, label, values in series:
        unit = units.get(group, "亿元/百分点" if group == "repo" else "见因子库定义")
        formula = "因子库原值；不追加MA或百分位变换"
        if group == "derived":
            op, inst, tenor, cls = key.split("|")
            s = next(s for s in lib["derived"] if (s["op"], s["inst"], s["tenor"], s["cls"]) == (op, inst, tenor, cls))
            unit = s.get("unit", unit)
            formula = DERIVED_FORMULAS[op]
        if group == "carry":
            cat, source_key = key.split("|")
            formula = next(s["formula"] for s in lib["carry"] if s["key"] == source_key and s["cat"] == cat)
        if group == "curve" and key.endswith("主力"): unit = "点"
        if group == "repo": unit = "百分点" if key.endswith("rate_spread") else "亿元"
        candidates.append({"id": group + "|" + key, "group": group, "key": key, "label": label,
                           "unit": unit, "formula": formula, "values": values})
    candidates.extend(technical_candidates(cmap))
    return lib, dates, cmap, candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asof", help="筛选收盘日；默认最后一个期货交易日")
    parser.add_argument("--lookback", type=int, help="仅用于显式指定窗口；默认全部共同历史")
    args = parser.parse_args()
    if args.lookback is not None and args.lookback < RULES["min_obs"]:
        parser.error("lookback至少300个期货交易日")
    lib, dates, cmap, candidates = load_candidates()
    asof = max(i for i, d in enumerate(dates) if args.asof is None or d <= args.asof)
    level = 3 * cmap["T主力"] - cmap["TL主力"]
    result, target = screen_snapshot(dates, level, candidates, asof, args.lookback)
    baselines, policy = baseline_features(level, cmap, dates)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["meta"] = {
        "generated": generated, "asof": dates[asof], "library_date": lib["meta"]["date_range"][-1],
        "history": "all" if args.lookback is None else args.lookback, "rules": RULES,
        "target": "3T−TL", "target_definition": "S[t+6]−S[t]，S=3×T主连−TL主连",
        "timing": "t收盘后观察因子；t+1以t收盘代理入场；原始因子不再次统一套MA/百分位",
        "rank_definition": "通过数据质量检查的全库因子，按|IC|和|ICIR|分别计算分位，两项都≥50%；并列取平均秩",
        "quintile_definition": "同一已完成标签区间内，以因子20/40/60/80%分位分组；相同取值不强行拆组",
        "stability_definition": "截至最后已完成标签的信号日，向前四段各63个期货交易日；每段≥30配对值才计算IC",
        "technical_definition": "网页32条技术指标按期货交易日重算；不搜索新的指标窗口",
        "selection_order": "通过全部条件后，按IC与ICIR全库分位平均值降序；同分依次按|IC|、|ICIR|、ID",
        "deduplication": False, "model_increment": False,
        "caveat": "本页是截至当前的研究候选名单，不是样本外回测；不能把名单回填整个历史后宣称独立样本外有效。预测标签重叠，ICIR与分组均值均非显著性证明。",
        "omo_verified_through": policy["verified_through"],
        "input_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                         for name in ("factor_library.json", "omo_policy_7d.json", "prepare_direction_screen.py")},
    }
    result["baselines"] = [{"id": b["id"], "label": b["label"], "unit": b["unit"],
                            "latest": safe_number(b["values"][asof]), "role": "固定基准，另列、不参加全库排名"}
                           for b in baselines]
    selected_map = {c["id"]: c for c in candidates if c["id"] in result["selected_ids"]}
    selected = [selected_map[key] for key in result["selected_ids"]]
    input_dates = np.flatnonzero(np.isfinite(level) & (np.arange(len(level)) <= asof))
    inputs = {
        "meta": {"generated": generated, "asof": dates[asof], "target_definition": result["meta"]["target_definition"],
                 "purpose": "当前筛选快照输入，不能回填历史；历史推进模型在各时点独立重筛。原始缺失保留null，预处理只能在训练期拟合。"},
        "dates": [dates[i] for i in input_dates], "target_h6": raw_numbers(target[input_dates]),
        "features": [{"id": c["id"], "label": c["label"], "unit": c["unit"],
                      "role": "candidate" if c["id"] in selected_map else "baseline", "values": raw_numbers(c["values"][input_dates])}
                     for c in selected + baselines],
    }
    # Export only completed labels, including for explicit historical as-of snapshots.
    inputs["target_h6"] = [v if raw_i + 6 <= asof else None
                           for raw_i, v in zip(input_dates, inputs["target_h6"])]
    for name, payload in (("direction_screen.json", result), ("direction_screen_inputs.json", inputs)):
        (ROOT / name).write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print("筛选日", dates[asof], "共同标签区间", result["window"])
    print("统计", result["summary"])
    rows_by_id = {row["id"]: row for row in result["rows"]}
    for key in result["selected_ids"]:
        row = rows_by_id[key]
        print(row["qualified_rank"], row["label"], row["ic"], row["icir"], row["same_blocks"])


if __name__ == "__main__":
    main()
