"""Frozen Ridge versus shallow trees and constrained XGBoost for 3T-TL.

Nonlinear parameter selection uses only a chronological holdout inside the 2024
training period. Purge crossing six-session labels; refit on all completed train
labels once. Reuse the corrected factor sets, preprocessing and trading engine.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
DEPENDENCIES = ROOT.parent / "ficc_strategy_research" / "python_deps"
if DEPENDENCIES.is_dir():
    sys.path.insert(0, str(DEPENDENCIES))
import xgboost
from xgboost import XGBRegressor

from prepare_curve_timing_capacity import TypeAwareScaler, freeze_sets
from prepare_curve_timing_percentile import period_metrics, serialize_sim, signed_percentile_gate
from prepare_curve_timing_walkforward import completed_indices, forecast_metrics, simulate
from prepare_direction_screen import baseline_features, load_candidates, raw_numbers, safe_number
from prepare_strategy_model import make_offset_labels

SOURCE = "curve_timing_capacity.json"
OUTPUT = "curve_timing_nonlinear.json"
MODELS_OUTPUT = "curve_timing_nonlinear_models.json"
SIZES = (20, 30, 40)
FAMILIES = ("ridge", "tree", "xgb")
TREE_CONFIG = {"criterion": "squared_error", "min_samples_leaf": 40, "random_state": 42}
XGB_CONFIG = {"objective": "reg:squarederror", "eval_metric": "rmse", "tree_method": "hist",
              "max_depth": 2, "learning_rate": .03, "min_child_weight": 30,
              "subsample": .8, "colsample_bytree": .8, "reg_lambda": 10., "reg_alpha": 0.,
              "n_jobs": 2, "random_state": 42, "verbosity": 0}


def chronological_split(signal_indices, horizon=6, fraction=.8):
    indices = np.asarray(signal_indices, int)
    split = int(len(indices) * fraction)
    if len(indices) < 300 or split == len(indices):
        raise ValueError("Insufficient training/validation history")
    validation = np.arange(split, len(indices))
    fitting = np.flatnonzero((np.arange(len(indices)) < split) & (indices + horizon < indices[split]))
    purged = np.setdiff1d(np.arange(split), fitting)
    return fitting, validation, purged


def fit_nonlinear(x, y, train, features):
    """Test inputs and test labels are deliberately absent from model selection."""
    train = np.asarray(train, int)
    tx, ty = np.asarray(x, float)[train], np.asarray(y, float)[train]
    inner, validation, purged = chronological_split(train)
    inner_scaler = TypeAwareScaler(features).fit(tx[inner])
    zx, zv = inner_scaler.transform(tx[inner]), inner_scaler.transform(tx[validation])
    # Fixed Ridge alpha is an inner-validation reference, never tuned here.
    mean = float(ty[inner].mean())
    coef = np.linalg.solve(zx.T @ zx + 15 * np.eye(zx.shape[1]), zx.T @ (ty[inner] - mean))
    ridge_validation = forecast_metrics(mean + zv @ coef, ty[validation])
    choices = []
    for depth in (2, 3):
        model = DecisionTreeRegressor(max_depth=depth, **TREE_CONFIG).fit(zx, ty[inner])
        pred = model.predict(zv)
        choices.append({"depth": depth, "metrics": forecast_metrics(pred, ty[validation]),
                        "rmse_exact": float(np.sqrt(np.mean((pred - ty[validation]) ** 2)))})
    chosen_depth = min(choices, key=lambda c: (c["rmse_exact"], c["depth"]))["depth"]
    tuning_xgb = XGBRegressor(n_estimators=300, early_stopping_rounds=20,
                              base_score=mean, **XGB_CONFIG)
    tuning_xgb.fit(zx, ty[inner], eval_set=[(zv, ty[validation])], verbose=False)
    rounds = int(tuning_xgb.best_iteration) + 1
    xgb_validation = forecast_metrics(tuning_xgb.predict(zv), ty[validation])
    # Only after parameter selection do we fit preprocessing and models on all train data.
    scaler = TypeAwareScaler(features).fit(tx)
    z = scaler.transform(tx)
    tree = DecisionTreeRegressor(max_depth=chosen_depth, **TREE_CONFIG).fit(z, ty)
    xgb = XGBRegressor(n_estimators=rounds, base_score=float(ty.mean()), **XGB_CONFIG).fit(z, ty, verbose=False)
    selection = {"inner_positions": inner.tolist(), "validation_positions": validation.tolist(),
                 "purged_positions": purged.tolist(), "tree_candidates": choices,
                 "selected_depth": chosen_depth, "ridge_validation": ridge_validation,
                 "xgb_best_iteration": int(tuning_xgb.best_iteration), "xgb_rounds": rounds,
                 "xgb_validation": xgb_validation,
                 "xgb_validation_rmse_path": [float(v) for v in tuning_xgb.evals_result()["validation_0"]["rmse"]],
                 "selection_rule": "树深2/3按训练内验证RMSE取优，同分选浅；XGBoost最多300轮、20轮早停，只用训练内验证集。"}
    return {"tree": tree, "xgb": xgb}, scaler, selection


def preprocessing_record(scaler, features):
    return [{"id": f["id"], "label": f["label"], "unit": f.get("unit", ""),
             "median": float(scaler.median[j]), "mean": float(scaler.mean[j]), "std": float(scaler.std[j]),
             "clip_enabled": bool(scaler.clip_mask[j]),
             "clip_low": float(scaler.lo[j]) if scaler.clip_mask[j] else None,
             "clip_high": float(scaler.hi[j]) if scaler.clip_mask[j] else None}
            for j, f in enumerate(features)]


def serialize_tree(model, scaler, features):
    tree, nodes = model.tree_, []
    for i in range(tree.node_count):
        j = int(tree.feature[i])
        leaf = int(tree.children_left[i]) == -1
        nodes.append({"id": i, "left": int(tree.children_left[i]), "right": int(tree.children_right[i]),
                      "leaf": leaf, "feature_index": None if leaf else j,
                      "feature_id": None if leaf else features[j]["id"],
                      "label": None if leaf else features[j]["label"],
                      "threshold_scaled": None if leaf else float(tree.threshold[i]),
                      "threshold_raw": None if leaf else float(tree.threshold[i] * scaler.std[j] + scaler.mean[j]),
                      "value": float(tree.value[i, 0, 0]), "n_train": int(tree.n_node_samples[i])})
    return {"nodes": nodes, "depth": int(model.get_depth()), "leaves": int(model.get_n_leaves())}


def importances(model, features):
    order = np.argsort(-model.feature_importances_, kind="stable")
    return [{"id": features[j]["id"], "label": features[j]["label"], "weight": float(model.feature_importances_[j])}
            for j in order if model.feature_importances_[j] > 0]


def summarize_path(dates, level, target, pred):
    gate = signed_percentile_gate(pred)
    paths, periods = {}, {}
    masks = {"all": np.ones(len(dates), bool), "2025": np.array([d.startswith("2025") for d in dates]),
             "2026": np.array([d.startswith("2026") for d in dates]),
             "2026-04+": np.array([d >= "2026-04-01" for d in dates])}
    for rule in ("fixed", "dynamic"):
        sim = simulate(dates, level, pred, gate["quantile"], rule=rule, gate=gate)
        paths[rule], periods[rule] = serialize_sim(sim), {}
        for name, mask in masks.items():
            row = period_metrics(dates, level, pred, target, gate, sim, mask)
            row["forecast"] = forecast_metrics(pred[mask], target[mask])
            row["long_pnl"] = safe_number(sim["daily_pnl"][mask & (sim["position"] == 1)].sum())
            row["short_pnl"] = safe_number(sim["daily_pnl"][mask & (sim["position"] == -1)].sum())
            periods[rule][name] = row
    serial_gate = {k: raw_numbers(v) if k in ("percentile", "quantile", "lower_quantile") else v.tolist()
                   for k, v in gate.items() if k != "direction_basis"}
    return paths, periods, serial_gate


def build():
    source = json.loads((ROOT / SOURCE).read_text(encoding="utf-8"))
    expected_hash = source["meta"]["input_hashes"]["factor_library.json"]
    if hashlib.sha256((ROOT / "factor_library.json").read_bytes()).hexdigest() != expected_hash:
        raise ValueError("The factor library must match the Ridge comparison snapshot")
    frozen = freeze_sets()
    _, dates, cmap, candidates = load_candidates()
    last = dates.index(source["meta"]["asof"])
    dates = dates[:last + 1]
    cmap = {k: v[:last + 1] for k, v in cmap.items()}
    level = 3 * cmap["T主力"] - cmap["TL主力"]
    target = make_offset_labels(level, 6)
    baselines, _ = baseline_features(level, cmap, dates)
    candidate_map = {f["id"]: f | {"values": f["values"][:last + 1]} for f in candidates}
    cutoff = max(i for i, d in enumerate(dates) if d <= "2024-12-31")
    train = completed_indices(level, target, cutoff, 6)
    start = cutoff + 1
    test_dates, test_level, test_target = dates[start:], level[start:], target[start:]
    if test_dates != source["daily"]["dates"]:
        raise ValueError("Test dates do not match")
    np.testing.assert_array_equal(test_level, np.array(source["daily"]["level"], float))
    np.testing.assert_array_equal(test_target, np.array(source["daily"]["target"], float))
    models, predictions, tuning, forecasts, labels = {}, {}, {}, {}, {}
    shared = np.isfinite(np.array(source["daily"]["prediction"]["f40"], float))
    for n in SIZES:
        key = f"f{n}"
        features = baselines + [candidate_map[x] for x in frozen["sets"][key]]
        matrix = np.column_stack([f["values"] for f in features])
        fitted, scaler, selection = fit_nonlinear(matrix, target, train, features)
        prep = preprocessing_record(scaler, features)
        for j, record in enumerate(source["models"][key]["features"]):
            for field in ("median", "mean", "std", "clip_low", "clip_high"):
                if prep[j][field] != record[field]:
                    raise ValueError("Final preprocessing differs from corrected Ridge")
        inner, validation = selection["inner_positions"], selection["validation_positions"]
        selection["dates"] = {"inner_train_start": dates[train[inner[0]]], "inner_train_end": dates[train[inner[-1]]],
                              "inner_last_label_exit": dates[train[inner[-1]] + 6],
                              "validation_start": dates[train[validation[0]]], "validation_end": dates[train[validation[-1]]],
                              "last_validation_label_exit": dates[train[validation[-1]] + 6],
                              "inner_n": len(inner), "validation_n": len(validation),
                              "purged_n": len(selection["purged_positions"])}
        tuning[key] = selection
        ridge_key = f"ridge{n}"
        predictions[ridge_key] = np.array(source["daily"]["prediction"][key], float)
        labels[ridge_key] = f"岭回归 · {n}+8"
        models[ridge_key] = {"family": "ridge", "params": {"alpha": 15}, "preprocessing": prep,
                            "linear_model": source["models"][key]}
        forecasts[ridge_key] = source["forecast"][key]
        for family in ("tree", "xgb"):
            model_key, model = f"{family}{n}", fitted[family]
            predictions[model_key] = np.where(shared, model.predict(scaler.transform(matrix[start:])), np.nan)
            labels[model_key] = f"{'浅层树' if family == 'tree' else 'XGBoost'} · {n}+8"
            config = TREE_CONFIG | {"max_depth": selection["selected_depth"]} if family == "tree" else (
                XGB_CONFIG | {"n_estimators": selection["xgb_rounds"], "base_score": float(target[train].mean())})
            models[model_key] = {"family": family, "params": config, "preprocessing": prep,
                                 "importances": importances(model, features)}
            if family == "tree":
                models[model_key]["tree"] = serialize_tree(model, scaler, features)
            else:
                models[model_key]["booster"] = json.loads(model.get_booster().save_raw(raw_format="json"))
            forecasts[model_key] = {"test": forecast_metrics(predictions[model_key], test_target),
                                   "train": forecast_metrics(model.predict(scaler.transform(matrix[train])), target[train])}
        print(n, "tree depth", selection["selected_depth"], "XGBoost rounds", selection["xgb_rounds"], flush=True)
    paths, periods, gates = {r: {} for r in ("fixed", "dynamic")}, {r: {} for r in ("fixed", "dynamic")}, {}
    diagnostics = {}
    for key, pred in predictions.items():
        path, period, gate = summarize_path(test_dates, test_level, test_target, pred)
        gates[key] = gate
        diagnostics[key] = {}
        for year in ("all", "2025", "2026", "2026-04+"):
            mask = np.ones(len(test_dates), bool) if year == "all" else np.array([
                d >= "2026-04-01" if year == "2026-04+" else d.startswith(year) for d in test_dates])
            finite = pred[mask & np.isfinite(pred)]
            diagnostics[key][year] = {"days": len(finite), "unique_predictions": len(np.unique(finite)),
                                      "mean": safe_number(finite.mean()), "std": safe_number(finite.std()),
                                      "min": safe_number(finite.min()), "max": safe_number(finite.max())}
        for rule in paths:
            paths[rule][key], periods[rule][key] = path[rule], period[rule]
        if key.startswith("ridge"):
            n = key.removeprefix("ridge")
            for rule in paths:
                if paths[rule][key] != source["strategies"][rule][f"f{n}"]:
                    raise ValueError("Ridge accounting changed")
    model_archive = {"meta": {"train_through": "2024-12-31", "xgboost_version": xgboost.__version__,
                               "sklearn_version": sklearn.__version__, "source": SOURCE}, "models": models}
    (ROOT / MODELS_OUTPUT).write_bytes(json.dumps(model_archive, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    source_names = (SOURCE, MODELS_OUTPUT, "factor_library.json", "curve_timing_capacity_factor_sets.json",
                    "prepare_curve_timing_nonlinear.py", "prepare_curve_timing_capacity.py",
                    "prepare_curve_timing_percentile.py", "prepare_curve_timing_walkforward.py",
                    "prepare_direction_screen.py", "prepare_strategy_model.py", "omo_policy_7d.json")
    return {"meta": {"generated": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
                     "asof": dates[-1], "train_through": "2024-12-31", "train_n": len(train), "test_n": len(test_dates),
                     "families": list(FAMILIES), "sizes": list(SIZES), "labels": labels,
                     "xgboost_version": xgboost.__version__, "sklearn_version": sklearn.__version__,
                     "config": {"horizon": 6, "window": 250, "min_history": 60, "lower": 40, "upper": 60},
                     "final_new_model_count": 6, "model_archive": MODELS_OUTPUT,
                     "validation": "训练期按时间前80%/后20%划分；前段再隔离6日标签。预处理只拟合前段。树深2/3按验证RMSE选择；XGBoost固定保守参数、最多300轮、20轮早停。选定后用完整406标签重新拟合，测试期冻结。",
                     "preprocessing": source["meta"]["preprocessing"],
                     "caveat": "固定名单仍来自全历史筛选，含选择前视。树深与早停未使用2025/2026标签，但本研究仍属探索性对照，不能称为独立样本外证明。未扣成本、昨日收盘代理开盘、6日标签重叠。树预测并列值取中位秩，不强制凑交易频率。",
                     "input_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names}},
            "factor_sets": frozen["sets"], "tuning": tuning, "forecast": forecasts,
            "forecast_reference": {
                "zero": forecast_metrics(np.where(shared, 0., np.nan), test_target),
                "train_mean": forecast_metrics(np.where(shared, float(target[train].mean()), np.nan), test_target)},
            "prediction_diagnostics": diagnostics,
            "model_summary": {key: {k: v for k, v in record.items() if k not in ("booster", "linear_model")}
                              for key, record in models.items()},
            "daily": {"dates": test_dates, "level": raw_numbers(test_level), "target": raw_numbers(test_target),
                      "prediction": {key: raw_numbers(pred) for key, pred in predictions.items()}},
            "gates": gates, "strategies": paths, "periods": periods}


def main():
    result = build()
    (ROOT / OUTPUT).write_bytes(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    for key in result["forecast"]:
        metric, sim = result["forecast"][key]["test"], result["strategies"]["dynamic"][key]["metrics"]
        print(key, "IC", metric["ic"], "RMSE", metric["rmse"], "dynamic", sim["total"], "DD", sim["max_drawdown"],
              "2026", result["periods"]["dynamic"][key]["2026"]["total"], flush=True)


if __name__ == "__main__":
    main()
