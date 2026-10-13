"""Daily inference with frozen Ridge models; no training or factor selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

from jsonio import write_json_skip_unchanged
from prepare_curve_timing_percentile import serialize_sim, signed_percentile_gate
from prepare_curve_timing_walkforward import simulate
from prepare_direction_screen import baseline_features, load_candidates, safe_number
from prepare_factor_library import build as build_library

ROOT = Path(__file__).resolve().parent
MODELS = "ridge_daily_models.json"
OUTPUT = "ridge_daily_signals.json"
ARMS = ("f20", "f30")
CONFIG = {"horizon": 6, "window": 250, "min_history": 60,
          "lower": 40, "upper": 60, "min_coverage": .9}


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def frozen_models():
    if (ROOT / MODELS).exists():
        result = read(MODELS)
    else:
        source = read("curve_timing_capacity.json")
        result = {"meta": {"version": "ridge-fixed-2024-typeaware-20261013",
                           "source": "curve_timing_capacity.json", "source_sha256": sha("curve_timing_capacity.json"),
                           "train_through": source["meta"]["train_through"], "train_n": source["meta"]["train_n"],
                           "selection_asof": source["meta"]["selection_asof"], "alpha": 15,
                           "main": "f20", "comparison": "f30", "refit": False},
                  "config": CONFIG, "models": {k: source["models"][k] for k in ARMS}}
        validate_models(result)
        write_json_skip_unchanged(str(ROOT / MODELS), result)
    validate_models(result)
    return result


def validate_models(result):
    if result["config"] != CONFIG:
        raise ValueError("Frozen daily trading configuration changed; discuss a new version first")
    sets = read("curve_timing_capacity_factor_sets.json")["sets"]
    for arm in ARMS:
        fs = result["models"][arm]["features"]
        if len(fs) != int(arm[1:]) + 8 or [f["id"] for f in fs[8:]] != sets[arm]:
            raise ValueError("Frozen factor membership changed")
        if any(f["clip_enabled"] for f in fs[:8]) or any(not f["id"].startswith("baseline|") for f in fs[:8]):
            raise ValueError("All eight baselines must be un-clipped")
        for f in fs:
            if not all(np.isfinite(f[n]) for n in ("median", "mean", "std", "coef")) or f["std"] <= 0:
                raise ValueError("Invalid frozen preprocessing parameter")


def infer(model, values):
    """Apply only saved train medians, clipping limits, scaling and coefficients."""
    result = float(model["intercept"])
    details = []
    for f, value in zip(model["features"], values, strict=True):
        missing = not np.isfinite(value)
        filled = f["median"] if missing else float(value)
        processed = float(np.clip(filled, f["clip_low"], f["clip_high"])) if f["clip_enabled"] else filled
        z = (processed - f["mean"]) / f["std"]
        contribution = z * f["coef"]
        result += contribution
        details.append({"id": f["id"], "label": f["label"], "unit": f["unit"],
                        "raw": safe_number(value), "filled": filled, "processed": processed,
                        "z": z, "contribution": contribution, "missing": missing,
                        "clipped": processed != filled, "clip_enabled": f["clip_enabled"]})
    return result, details


def seed_archive():
    source = read("curve_timing_capacity.json")
    daily = source["daily"]
    return [{"date": day, "level": level, "origin": "research-backfill",
             "models": {k: {"prediction": daily["prediction"][k][i], "status": "backfill"} for k in ARMS}}
            for i, (day, level) in enumerate(zip(daily["dates"], daily["level"], strict=True))]


def sources():
    bond, repo, curve, policy = (read(n) for n in ("bond_trading_data.json", "repo_trading_data.json",
                                                "yield_curve_data.json", "omo_policy_7d.json"))
    # Dates must have actual institution detail, rather than just a metadata end date.
    bond_dates = {bond["idx"]["dates"][r[0]] for r in bond["detail"]}
    repo_dates = set(repo["dates"])
    return {"institution": max(bond_dates), "repo": max(repo_dates),
            "curve": curve["meta"]["date_range"][-1], "omo_verified": policy["verified_through"],
            "omo_source": policy.get("latest_verification"), "omo_rate": policy.get("latest_verified_rate")}, bond_dates, repo_dates


def quality(day, level, baselines, selected, source_info, bond_dates, repo_dates):
    problems = []
    if not np.isfinite(level): problems.append("T或TL收盘价缺失")
    if not np.isfinite(baselines).all(): problems.append("8个市场基准不完整")
    coverage = float(np.mean(np.isfinite(selected)))
    if coverage < CONFIG["min_coverage"]: problems.append("固定因子覆盖率不足90%")
    if day not in bond_dates: problems.append("当日期限机构日报缺失")
    if day not in repo_dates: problems.append("当日回购日报缺失")
    if source_info["omo_verified"] < day: problems.append("7天OMO利率尚未核验至信号日")
    return coverage, problems


def enrich(rows):
    for arm in ARMS:
        pred = np.array([r["models"][arm]["prediction"] for r in rows], float)
        gate = signed_percentile_gate(pred, window=250, min_history=60, lower=40, upper=60)
        for i, row in enumerate(rows):
            rec = row["models"][arm]
            current = {"percentile": safe_number(gate["percentile"][i]), "history_n": int(gate["history_n"][i]),
                       "signal": int(gate["signal"][i]), "lower_reference": safe_number(gate["lower_quantile"][i]),
                       "upper_reference": safe_number(gate["quantile"][i])}
            # All previously issued signals, including percentile gates, are immutable.
            if rec.get("status") == "issued" and "signal" in rec:
                continue
            rec.update(current)


def archive_simulation(rows, arm):
    days = [r["date"] for r in rows]
    level = np.array([r["level"] for r in rows], float)
    pred = np.array([r["models"][arm]["prediction"] for r in rows], float)
    gate = {"direction_basis": "percentile", "percentile": np.array([r["models"][arm]["percentile"] for r in rows], float),
            "signal": np.array([r["models"][arm]["signal"] for r in rows], int),
            "history_n": np.array([r["models"][arm]["history_n"] for r in rows], int),
            "quantile": np.array([r["models"][arm]["upper_reference"] for r in rows], float)}
    return simulate(days, level, pred, gate["quantile"], rule="dynamic", horizon=6, gate=gate)


def position_summary(rows, arm, sim=None):
    sim = archive_simulation(rows, arm) if sim is None else sim
    decision = sim["decisions"][-1]
    open_trade = next((t for t in reversed(sim["trades"]) if not t["closed"]), None)
    return {"position_during_signal_day": int(sim["position"][-1]), "next_position": decision["next_position"],
            "exit_reason": decision["exit_reason"], "open_trade": open_trade,
            "metrics": sim["metrics"], "label": "按已归档信号模拟的仓位；不代表账户实际持仓"}


def build(max_date=None):
    frozen = frozen_models()
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    max_date = max_date or (now.date() - timedelta(days=1)).isoformat()
    lib = build_library(prefer_compact=True)
    # Validate current reconstruction before publishing the factor library.
    with tempfile.TemporaryDirectory() as temp:
        p = Path(temp) / "factor_library.json"
        p.write_bytes(json.dumps(lib, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        with patch("prepare_factor_checkup.LIB_PATH", str(p)):
            _, dates, cmap, candidates = load_candidates()
    level = 3 * cmap["T主力"] - cmap["TL主力"]
    baselines, _ = baseline_features(level, cmap, dates)
    features = {f["id"]: f for f in candidates + baselines}
    for model in frozen["models"].values():
        missing = [f["id"] for f in model["features"] if f["id"] not in features]
        if missing: raise ValueError("Fixed features absent from updated library: " + ", ".join(missing))
    info, bond_dates, repo_dates = sources()
    raw_curve = read("yield_curve_data.json")["series"]
    latest_future_date = max(d for name in ("T主力", "TL主力") for d in raw_curve[name]["dates"] if d <= max_date)
    old = read(OUTPUT) if (ROOT / OUTPUT).exists() else None
    if old and old["meta"]["model_sha256"] != sha(MODELS):
        raise ValueError("Frozen model changed; cannot append to old signal archive")
    rows = old["records"] if old else seed_archive()
    known = {r["date"]: r for r in rows}
    first = rows[0]["date"]
    new_dates, revisions, latest_issue = [], [], None
    for i, day in enumerate(dates):
        if day < first or day > max_date: continue
        if day in known and np.isfinite(level[i]) and abs(known[day]["level"] - level[i]) > 1e-8:
            revisions.append({"date": day, "issued_level": known[day]["level"], "current_level": float(level[i])})
        if not np.isfinite(level[i]):
            if day >= rows[-1]["date"]: latest_issue = {"date": day, "problems": ["T或TL价格缺失，无法计算当日新信号"]}
            continue
        if day in known and all(known[day]["models"][k]["status"] in ("issued", "backfill") for k in ARMS):
            continue
        # Do not manufacture a historical signal for an earlier missed session.
        if day in known and day < rows[-1]["date"]:
            continue
        rec = known.get(day) or {"date": day, "level": float(level[i]), "origin": "live-issued", "models": {}}
        for arm in ARMS:
            if rec["models"].get(arm, {}).get("status") in ("issued", "backfill"): continue
            model = frozen["models"][arm]
            values = np.array([features[f["id"]]["values"][i] for f in model["features"]], float)
            coverage, problems = quality(day, level[i], values[:8], values[8:], info, bond_dates, repo_dates)
            prediction, details = (None, []) if problems else infer(model, values)
            proposed = {"prediction": prediction, "status": "blocked" if problems else "issued",
                                  "coverage": coverage, "problems": problems, "inputs": details,
                                  "issued_at": now.isoformat(timespec="seconds"),
                                  "input_sha256": {n: sha(n) for n in ("bond_trading_data.json", "repo_trading_data.json",
                                                                      "yield_curve_data.json", "omo_policy_7d.json")}}
            previous = rec["models"].get(arm)
            if previous and proposed["status"] == "blocked" and all(previous.get(n) == proposed[n] for n in
                    ("prediction", "status", "coverage", "problems", "input_sha256")):
                continue
            rec["models"][arm] = proposed
        if day not in known:
            rows.append(rec)
            known[day] = rec
        if any(rec["models"][k]["status"] == "issued" for k in ARMS): new_dates.append(day)
    rows.sort(key=lambda r: r["date"])
    if latest_future_date > rows[-1]["date"]:
        latest_issue = {"date": latest_future_date, "problems": ["较新期货数据日T或TL价格缺失，不能把旧预测视为新信号"]}
    enrich(rows)
    latest = rows[-1]
    statuses = [latest["models"][k]["status"] for k in ARMS]
    healthy = not latest_issue and all(s in ("issued", "backfill") for s in statuses)
    simulations = {k: archive_simulation(rows, k) for k in ARMS}
    summaries = {k: position_summary(rows, k, simulations[k]) for k in ARMS}
    report = {"meta": {"generated": now.isoformat(timespec="seconds"), "model_version": frozen["meta"]["version"],
                       "model_sha256": sha(MODELS), "signal_date": latest["date"], "main": "f20",
                       "comparison": "f30", "healthy": healthy, "source_dates": info,
                       "unpriced_latest": latest_issue, "price_revisions": revisions,
                       "timing": "中午更新与复核；信号日收盘后观察，下一期货交易日开盘执行，代理开盘价=信号日收盘价。",
                       "archive": "历史行标记research-backfill；首次上线后的预测逐日归档且不覆盖。仅模拟持仓，不自动交易。",
                       "research_caveat": "固定因子来自全历史筛选，历史业绩含选择前视；不扣交易成本。"},
              "config": CONFIG, "latest": latest, "positions": summaries, "records": rows,
              "strategies": {k: serialize_sim(simulations[k]) for k in ARMS}}
    library_changed = write_json_skip_unchanged(str(ROOT / "factor_library.json"), lib)
    signal_changed = write_json_skip_unchanged(str(ROOT / OUTPUT), report)
    update_versions()
    return report, {"new_signal_dates": new_dates, "library_changed": library_changed, "signal_changed": signal_changed}


def update_versions():
    manifest = read("data_version.json")
    for name in ("factor_library.json", "omo_policy_7d.json", MODELS, OUTPUT):
        p = ROOT / name
        obj = read(name)
        stamp = obj.get("meta", {}).get("last_updated") or obj.get("meta", {}).get("generated")
        stamp = stamp or datetime.fromtimestamp(p.stat().st_mtime, ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
        manifest["files"][name] = {"last_updated": stamp, "size": p.stat().st_size}
    manifest["generated"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    write_json_skip_unchanged(str(ROOT / "data_version.json"), manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-date", help="Optional historical cutoff; normal noon run excludes today's unfinished close")
    args = parser.parse_args()
    report, changes = build(args.max_date)
    print(json.dumps({"signal_date": report["meta"]["signal_date"], "healthy": report["meta"]["healthy"],
                      **changes, "signals": {k: {n: report["latest"]["models"][k][n]
                                                  for n in ("prediction", "percentile", "signal", "status")}
                                             for k in ARMS}}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
