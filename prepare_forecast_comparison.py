"""Six frozen model versions: prediction-only comparison, no trading simulation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from jsonio import write_json_skip_unchanged
from prepare_bayes_study import dynamic_forecast, load_data
from prepare_curve_timing_capacity import TypeAwareScaler, fit_model, model_record
from prepare_curve_timing_nonlinear import XGB_CONFIG, XGBRegressor, xgboost
from prepare_curve_timing_walkforward import completed_indices
from prepare_direction_screen import raw_numbers
from prepare_strategy_model import make_offset_labels
from prepare_temporal_study import PROTECTED as TEMPORAL_PROTECTED, restore_xgb, transform_record

ROOT = Path(__file__).resolve().parent
OUTPUT = 'forecast_model_comparison.json'
MODELS = 'forecast_model_comparison_models.json'
SHORTLIST = 'forecast_model_shortlist.json'
KEYS = ('ridge_original', 'dlm_original', 'xgb_original', 'ridge_temporal', 'dlm_temporal', 'xgb_temporal')
LABELS = ('原Ridge', '原动态贝叶斯', '原XGBoost', '时序Ridge', '时序贝叶斯递推', '时序XGBoost')
PROTECTED = TEMPORAL_PROTECTED + ('curve_timing_temporal.json', 'curve_timing_temporal_models.json')
BOOTSTRAP = {'block': 12, 'draws': 2000, 'seed': 42}


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def rank_ic(p, y):
    if len(p) < 3 or np.ptp(p) == 0 or np.ptp(y) == 0:
        return None
    a, b = rankdata(p), rankdata(y)
    return float(np.corrcoef(a, b)[0, 1])


def metrics(prediction, actual):
    p, y = np.asarray(prediction, float), np.asarray(actual, float)
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if not len(y):
        return {'n': 0}
    ps, ys = np.sign(p), np.sign(y)
    up, down, nz = y > 0, y < 0, y != 0
    up_recall = float(np.mean(p[up] > 0)) if up.any() else None
    down_recall = float(np.mean(p[down] < 0)) if down.any() else None
    predicted_up, predicted_down = (p > 0) & nz, (p < 0) & nz
    correct = int(np.sum((ps == ys) & nz))
    error = p - y
    return {'n': len(y), 'mae': float(np.mean(np.abs(error))), 'rmse': float(np.sqrt(np.mean(error ** 2))),
            'bias': float(error.mean()), 'rank_ic': rank_ic(p, y),
            'direction_n': int(nz.sum()), 'direction_correct': correct,
            'direction_accuracy': correct / int(nz.sum()) if nz.any() else None,
            'up_n': int(up.sum()), 'down_n': int(down.sum()), 'actual_zero_n': int((y == 0).sum()),
            'predicted_zero_n': int((p == 0).sum()), 'predicted_up_n': int((p > 0).sum()),
            'predicted_down_n': int((p < 0).sum()),
            'up_recall': up_recall, 'down_recall': down_recall,
            'balanced_accuracy': (up_recall + down_recall) / 2 if up_recall is not None and down_recall is not None else None,
            'up_precision': float(np.mean(y[predicted_up] > 0)) if predicted_up.any() else None,
            'down_precision': float(np.mean(y[predicted_down] < 0)) if predicted_down.any() else None,
            'prediction_mean': float(p.mean()), 'actual_mean': float(y.mean()),
            'confusion': [[int(np.sum((ys == a) & (ps == b))) for b in (-1, 0, 1)] for a in (-1, 1)]}


def xgb_fingerprint(x, y, dates):
    h = hashlib.sha256(json.dumps({'dates': dates, 'config': XGB_CONFIG,
                                  'max_rounds': 300, 'early_stopping': 20}, sort_keys=True).encode())
    for a in (x, y):
        b = np.array(a, dtype='<f8', copy=True)
        b[~np.isfinite(b)] = np.nan
        h.update(b.tobytes())
    return h.hexdigest()


def fit_original_xgb(dates, level, x, features):
    if dates[-1] > '2025-12-31':
        raise ValueError('No 2026 input may enter XGBoost selection or fit')
    y = make_offset_labels(level, 6)
    cut = len(dates) - 1
    inner_cut = max(i for i, d in enumerate(dates) if d <= '2025-06-30')
    inner = completed_indices(level, y, inner_cut, 6)
    train = completed_indices(level, y, cut, 6)
    validation = np.array([i for i in train if i > inner_cut], int)
    scaler = TypeAwareScaler(features).fit(x[inner])
    z = scaler.transform(x)
    tuning = XGBRegressor(n_estimators=300, early_stopping_rounds=20,
                           base_score=float(y[inner].mean()), **XGB_CONFIG)
    tuning.fit(z[inner], y[inner], eval_set=[(z[validation], y[validation])], verbose=False)
    rounds = int(tuning.best_iteration) + 1
    # Use shared type-aware preprocessing fitted only on the final train sample.
    preprocessing = fit_model(x[train], y[train], features, alpha=15.)
    zt = preprocessing[0].transform(x[train])
    fitted = XGBRegressor(n_estimators=rounds, base_score=float(y[train].mean()), **XGB_CONFIG)
    fitted.fit(zt, y[train], verbose=False)
    record = {'meta': {'train_through': dates[-1], 'train_n': len(train),
                        'last_train_signal': dates[train[-1]], 'last_train_label_exit': dates[train[-1] + 6],
                        'input_sha256': xgb_fingerprint(x, y, dates), 'version': xgboost.__version__,
                        'config': XGB_CONFIG, 'rounds': rounds},
              'validation': {'initial_train_through': dates[inner_cut], 'train_n': len(inner),
                             'last_inner_signal': dates[inner[-1]], 'last_inner_exit': dates[inner[-1] + 6],
                             'start': dates[validation[0]], 'last_signal': dates[validation[-1]],
                             'last_label_exit': dates[validation[-1] + 6], 'n': len(validation),
                             'metrics': metrics(tuning.predict(z[validation]), y[validation]),
                             'rmse_path': [float(v) for v in tuning.evals_result()['validation_0']['rmse']]},
              'model': {'preprocessing': model_record(preprocessing, features),
                         'booster': json.loads(fitted.get_booster().save_raw(raw_format='json')),
                         'rounds': rounds}}
    np.testing.assert_array_equal(fitted.predict(zt), restore_xgb(record['model']).predict(xgboost.DMatrix(zt)))
    return record


def quarter_metrics(dates, prediction, actual):
    quarters = [d[:4] + '-Q' + str((int(d[5:7]) - 1) // 3 + 1) for d in dates]
    result = {}
    for q in sorted(set(quarters)):
        ix = np.flatnonzero(np.array(quarters) == q)
        m = metrics(np.asarray(prediction)[ix], np.asarray(actual)[ix])
        result[q] = m | {'start': dates[ix[0]], 'end': dates[ix[-1]], 'sufficient': m['n'] >= 30}
    return result


def rolling_metrics(prediction, actual, window=63):
    """Place each metric on label completion day, never on the earlier origin."""
    p, y = np.asarray(prediction), np.asarray(actual)
    names = ('direction_accuracy', 'balanced_accuracy', 'rank_ic', 'mae', 'rmse')
    result = {name: [None] * len(p) for name in names}
    for t in range(window - 1, len(p) - 6):
        yy = y[t - window + 1:t + 1]
        if not np.isfinite(yy).all():
            continue
        m = metrics(p[t - window + 1:t + 1], yy)
        for name in names:
            result[name][t + 6] = m[name]
    return result


def bootstrap_indices(n, block, draws, seed):
    if n < block:
        raise ValueError('Insufficient history for moving block bootstrap')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(draws, int(np.ceil(n / block))))
    return (starts[..., None] + np.arange(block)).reshape(draws, -1)[:, :n]


def boot_metrics(p, y):
    e = p - y
    up, down, nz = y > 0, y < 0, y != 0
    acc = np.sum((np.sign(p) == np.sign(y)) & nz, axis=1) / np.sum(nz, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        bal = .5 * (np.sum((p > 0) & up, axis=1) / up.sum(axis=1) +
                    np.sum((p < 0) & down, axis=1) / down.sum(axis=1))
        a, b = rankdata(p, axis=1), rankdata(y, axis=1)
        a -= a.mean(axis=1, keepdims=True)
        b -= b.mean(axis=1, keepdims=True)
        ic = np.sum(a * b, axis=1) / np.sqrt(np.sum(a*a, axis=1) * np.sum(b*b, axis=1))
    return {'direction_accuracy': acc, 'balanced_accuracy': bal, 'rank_ic': ic,
            'mae': np.mean(np.abs(e), axis=1), 'rmse': np.sqrt(np.mean(e*e, axis=1))}


def paired_intervals(predictions, actual, config=BOOTSTRAP):
    y = np.asarray(actual, float)
    if not np.isfinite(y).all():
        raise ValueError('Bootstrap requires one common, completed sample')
    ix = bootstrap_indices(len(y), **config)
    yy = y[ix]
    base = boot_metrics(np.asarray(predictions['ridge_original'])[ix], yy)
    zero = boot_metrics(np.zeros_like(yy), yy)
    output = {}
    for key, p in predictions.items():
        bm = boot_metrics(np.asarray(p)[ix], yy)
        row = {}
        for name, value in bm.items():
            delta = value - base[name]
            valid = delta[np.isfinite(delta)]
            row[name] = {'low': float(np.quantile(valid, .025)), 'high': float(np.quantile(valid, .975)),
                         'valid_draws': len(valid)} if len(valid) else {'low': None, 'high': None, 'valid_draws': 0}
        row['vs_zero'] = {name: {'low': float(np.quantile(bm[name] - zero[name], .025)),
                                'high': float(np.quantile(bm[name] - zero[name], .975))} for name in ('mae', 'rmse')}
        output[key] = row
    return output


def select_candidates(rows, benchmarks):
    """Pure metrics; this function has no access to forecasts, dates or PnL."""
    eligible = [k for k in KEYS if rows[k]['balanced_accuracy'] is not None and
                rows[k]['balanced_accuracy'] > .5 and rows[k]['rank_ic'] is not None and rows[k]['rank_ic'] > 0]
    orient = {'direction_accuracy': 1, 'balanced_accuracy': 1, 'rank_ic': 1, 'mae': -1, 'rmse': -1}
    dominated = {}
    for k in eligible:
        dominated[k] = [j for j in eligible if j != k and
                        all(orient[f] * rows[j][f] >= orient[f] * rows[k][f] for f in orient) and
                        any(orient[f] * rows[j][f] > orient[f] * rows[k][f] for f in orient)]
    front = [k for k in eligible if not dominated[k]]
    if not front:
        return {'eligible': eligible, 'pareto': [], 'dominated_by': dominated, 'selected': [],
                'direction_candidate': None, 'amplitude_candidate': None, 'ranking_candidate': None}
    direction = min(front, key=lambda k: (-rows[k]['balanced_accuracy'], -rows[k]['rank_ic'], rows[k]['mae'], KEYS.index(k)))
    ranking = min(front, key=lambda k: (-rows[k]['rank_ic'], -rows[k]['balanced_accuracy'], KEYS.index(k)))
    amplitude = min(front, key=lambda k: (rows[k]['rmse'], rows[k]['mae'], KEYS.index(k)))
    if not all(rows[amplitude]['rmse'] < benchmarks[b]['rmse'] for b in ('zero', 'train_mean')):
        amplitude = None
    chosen = [direction]
    if amplitude is not None and amplitude != direction:
        chosen.append(amplitude)
    elif ranking != direction:
        chosen.append(ranking)
    return {'eligible': eligible, 'pareto': front, 'dominated_by': dominated, 'selected': chosen,
            'direction_candidate': direction, 'amplitude_candidate': amplitude, 'ranking_candidate': ranking}


def build():
    protected = {n: sha(n) for n in PROTECTED}
    dates, level, xb, y, fs, cut, train = load_data()
    td, ty = dates[cut + 1:], y[cut + 1:]
    old, temporal = read('curve_timing_bayes.json'), read('curve_timing_temporal.json')
    if old['dates'] != td or temporal['dates'] != td:
        raise ValueError('Research snapshots must share the latest complete calendar before comparing')
    train_y = make_offset_labels(level[:cut + 1], 6)
    fingerprint = xgb_fingerprint(xb[:cut + 1], train_y, dates[:cut + 1])
    if (ROOT / MODELS).exists():
        xgb = read(MODELS)
        if xgb['meta']['input_sha256'] != fingerprint:
            raise ValueError('Frozen original XGBoost inputs changed; do not overwrite')
    else:
        xgb = fit_original_xgb(dates[:cut + 1], level[:cut + 1], xb[:cut + 1], fs)
    z = transform_record(xgb['model']['preprocessing'], xb[cut + 1:])
    xp = restore_xgb(xgb['model']).predict(xgboost.DMatrix(z))
    predictions = {'ridge_original': np.array(old['variants']['ridge20']['prediction']),
                   'dlm_original': np.array(old['variants']['dlm20']['prediction']), 'xgb_original': xp,
                   **{k: np.array(temporal['variants'][k]['prediction']) for k in KEYS[3:]}}
    # Reproduce the archived original DLM from its frozen initial state and the
    # observed outcomes, verifying the same delayed online update policy.
    prior = read('bayes_study_models_202512.json')
    bz = transform_record(prior['scaler_model'], xb)
    d = dynamic_forecast(bz, y, cut + 1, int(train[-1]), np.array(prior['mean']),
                          np.array(prior['covariance']), prior['noise_variance'], prior['discount'], dates)
    np.testing.assert_allclose(d['mean'], predictions['dlm_original'], atol=1e-12, rtol=0)
    common = np.isfinite(ty)
    for p in predictions.values():
        if not np.isfinite(p).all():
            raise ValueError('Missing forecast; cannot quietly give models different evaluation dates')
    eval_dates = [day for day, ok in zip(td, common) if ok]
    actual = ty[common]
    benchmarks = {'zero': metrics(np.zeros(len(actual)), actual),
                   'train_mean': metrics(np.full(len(actual), y[train].mean()), actual),
                   'always_up': metrics(np.ones(len(actual)), actual),
                   'always_down': metrics(-np.ones(len(actual)), actual)}
    models, rows = {}, {}
    for key, label in zip(KEYS, LABELS):
        p = predictions[key]
        m = metrics(p[common], actual)
        rows[key] = m
        quarters = quarter_metrics(td, p, ty)
        sufficient = [q for q in quarters.values() if q['sufficient'] and q['balanced_accuracy'] is not None]
        models[key] = {'label': label, 'features_n': 92 if key.endswith('temporal') else 28,
                       'prediction': raw_numbers(p), 'metrics': m, 'quarters': quarters,
                       'rolling': rolling_metrics(p, ty),
                       'phases': [{'phase': phase, **metrics(p[np.arange(len(p)) % 6 == phase],
                                                            ty[np.arange(len(p)) % 6 == phase])} for phase in range(6)],
                       'stability': {'sufficient_quarters': len(sufficient),
                                     'quarters_balanced_above_half': sum(q['balanced_accuracy'] > .5 for q in sufficient),
                                     'quarters_positive_ic': sum(q['rank_ic'] is not None and q['rank_ic'] > 0 for q in sufficient)},
                       'fitting_mode': '已兑现标签递推' if key.startswith('dlm') else '系数冻结'}
    selection = select_candidates(rows, benchmarks)
    retained = read(SHORTLIST)
    if len(retained['selected']) != len(set(retained['selected'])) or not set(retained['selected']) <= set(KEYS):
        raise ValueError('User shortlist contains duplicate or unknown model versions')
    if not set(retained['user_added']) <= set(retained['selected']):
        raise ValueError('User-added models must be retained')
    if not all(k in retained['roles'] for k in retained['selected']):
        raise ValueError('Each retained model requires an explicit role')
    selected_pairs = {}
    for a in selection['selected']:
        for b in selection['selected']:
            if a != b:
                ca = np.sign(predictions[a][common]) == np.sign(actual)
                cb = np.sign(predictions[b][common]) == np.sign(actual)
                nz = actual != 0
                selected_pairs[a + '_vs_' + b] = {'a_only_correct': int(np.sum(ca & ~cb & nz)),
                                                  'b_only_correct': int(np.sum(cb & ~ca & nz)),
                                                  'both_correct': int(np.sum(ca & cb & nz)),
                                                  'both_wrong': int(np.sum(~ca & ~cb & nz))}
    intervals = paired_intervals({k: p[common] for k, p in predictions.items()}, actual)
    meta = {'asof': td[-1], 'train_through': '2025-12-31', 'train_n': len(train),
             'last_train_signal': dates[train[-1]], 'last_train_label_exit': dates[train[-1] + 6],
             'test_start': td[0], 'forecast_n': len(td), 'completed_n': len(actual),
             'evaluation_start': eval_dates[0], 'last_evaluation_signal': eval_dates[-1],
             'last_evaluation_label_exit': td[-1], 'horizon': 6, 'bootstrap': BOOTSTRAP,
             'prediction_only': True, 'factor_selection_lookahead': True,
             'caveat': '2026年已反复观察，候选为探索选择，不能再把同一区间当作选中后的独立测试。六日标签重叠，区块区间不能纠正筛选前视。',
             'protected_hashes': protected,
             'settings': {'ridge_original': '28输入；alpha=15，冻结',
                          'dlm_original': '28输入；alpha=15，discount=.995，延迟递推',
                          'xgb_original': f'28输入；深度2，2025验证早停{xgb["meta"]["rounds"]}轮，冻结',
                          'ridge_temporal': '92输入；2025验证选alpha=500，冻结',
                          'dlm_temporal': '92输入；alpha=500，2025验证选discount=1，延迟递推无额外漂移',
                          'xgb_temporal': '92输入；深度2，2025验证早停1轮，冻结'}}
    result = {'meta': meta, 'dates': td, 'target': raw_numbers(ty),
               'label_exit_dates': [td[i + 6] if i + 6 < len(td) else None for i in range(len(td))],
               'benchmarks': benchmarks, 'models': models, 'selection': selection, 'retained': retained,
               'benchmark_rolling': {'zero': rolling_metrics(np.zeros(len(td)), ty)},
               'paired_intervals': intervals, 'selected_pairs': selected_pairs,
               'xgb_original_validation': xgb['validation'],
               'controls': {k: {'label': v['label'], 'metrics': metrics(np.array(v['prediction'])[common], actual)}
                            for k, v in temporal['controls'].items()},
               'update_audit': {'dlm_original': d['updates'],
                                'dlm_temporal': temporal['diagnostics']['dlm']['updates']}}
    if protected != {n: sha(n) for n in PROTECTED}:
        raise ValueError('Protected models or research archives changed')
    json.dumps(xgb, allow_nan=False)
    write_json_skip_unchanged(str(ROOT / MODELS), xgb)
    result['meta']['supplement_models_sha256'] = sha(MODELS)
    json.dumps(result, allow_nan=False)
    write_json_skip_unchanged(str(ROOT / OUTPUT), result)
    print(json.dumps({'selection': selection, 'metrics': rows, 'paired': selected_pairs}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    build()
