"""Isolated lag-feature study; never changes production factors or models.

Fixed original 20+8 versus 92 time-aware inputs. All tuning ends in 2025;
only the explicitly labelled DLM assimilates matured 2026 outcomes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from jsonio import write_json_skip_unchanged
from prepare_bayes_study import (PROTECTED as BAYES_PROTECTED, dynamic_forecast,
                                  initial_state, load_data, summarize_variant)
from prepare_curve_timing_capacity import TypeAwareScaler, fit_model, model_record
from prepare_curve_timing_nonlinear import XGB_CONFIG, XGBRegressor, xgboost
from prepare_curve_timing_walkforward import completed_indices, forecast_metrics
from prepare_direction_screen import raw_numbers
from prepare_strategy_model import make_offset_labels
from prepare_threshold_study import trade_stats

ROOT = Path(__file__).resolve().parent
OUTPUT = 'curve_timing_temporal.json'
MODELS = 'curve_timing_temporal_models.json'
CUTOFF = '2025-12-31'
INNER_CUTOFF = '2025-06-30'
ALPHAS = (15., 50., 150., 500.)
DISCOUNTS = (1., .995, .98)
KEYS = ('ridge_original', 'ridge_temporal', 'dlm_temporal', 'xgb_temporal')
PROTECTED = BAYES_PROTECTED + ('curve_timing_bayes.json', 'bayes_study_models_202512.json',
                              'curve_timing_thresholds.json', 'factor_library.json')
CONFIG = {'cutoff': CUTOFF, 'inner_cutoff': INNER_CUTOFF, 'alphas': list(ALPHAS),
          'discounts': list(DISCOUNTS), 'xgb': XGB_CONFIG, 'max_rounds': 300,
          'early_stopping': 20, 'horizon': 6, 'history_factor_start': 8,
          'lags': [1, 5], 'change': 5, 'spread_changes': [1, 3, 10],
          'vol_window': 20, 'vol_ddof': 0, 'feature_version': 1}


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def lag(values, periods):
    values = np.asarray(values, float)
    if periods < 1:
        raise ValueError('Lag must be positive')
    result = np.full(values.shape, np.nan)
    if periods < len(values):
        result[periods:] = values[:-periods]
    return result


def temporal_features(level, original):
    """No forward filling, calendar shifts or use of any future observations."""
    level = np.asarray(level, float)
    result = [dict(f, block='original', source_id=f['id'], operation='current') for f in original]
    for h in (1, 3, 10):
        result.append({'id': f'baseline|spread_mom{h}', 'label': f'3T−TL过去{h}日变动',
                       'unit': '点', 'values': level - lag(level, h), 'block': 'spread_history',
                       'source_id': '3T−TL', 'operation': f'change{h}',
                       'formula': f'S[t]−S[t−{h}]'})
    changes = level - lag(level, 1)
    vol = np.full(len(level), np.nan)
    for t in range(20, len(level)):
        sample = changes[t - 19:t + 1]
        if np.isfinite(sample).all():
            vol[t] = np.std(sample, ddof=0)
    result.append({'id': 'baseline|spread_vol20', 'label': '3T−TL过去20日波动率',
                   'unit': '点/日', 'values': vol, 'block': 'spread_history',
                   'source_id': '3T−TL', 'operation': 'vol20',
                   'formula': '过去20个日度价差变动的标准差，ddof=0，不年化'})
    # Only the accepted 20 candidates get lag expansions; the 8 market baselines
    # remain intact. A change5 is algebraically redundant with current and lag5
    # before clipping/imputation; this is disclosed rather than called new data.
    for f in original[8:]:
        for op in ('lag1', 'lag5', 'change5'):
            v = lag(f['values'], 1 if op == 'lag1' else 5)
            if op == 'change5':
                v = np.asarray(f['values'], float) - v
            result.append(dict(f, id=f['id'] + '|history_' + op,
                               label=f['label'] + ' · ' + {'lag1': '滞后1日', 'lag5': '滞后5日',
                                                          'change5': '5日变动'}[op],
                               values=v, block='factor_history', source_id=f['id'], operation=op,
                               formula='X[t]−X[t−5]' if op == 'change5' else 'X[t−1]' if op == 'lag1' else 'X[t−5]'))
    if len({f['id'] for f in result}) != len(result):
        raise ValueError('Duplicate temporal feature IDs')
    return result


def matrix(features):
    return np.column_stack([f['values'] for f in features])


def definitions(features):
    return [{k: v for k, v in f.items() if k != 'values'} for f in features]


def training_digest(xb, xt, y, dates):
    h = hashlib.sha256(json.dumps({'config': CONFIG, 'dates': dates}, sort_keys=True).encode())
    for a in (xb, xt, y):
        canonical = np.array(a, dtype='<f8', copy=True)
        canonical[~np.isfinite(canonical)] = np.nan
        h.update(str(canonical.shape).encode())
        h.update(canonical.tobytes())
    return h.hexdigest()


def exact_rmse(pred, target):
    mask = np.isfinite(pred) & np.isfinite(target)
    return float(np.sqrt(np.mean((pred[mask] - target[mask]) ** 2)))


def choose(rows, parameter):
    # Prefer greater shrinkage / less drift in exact ties. No test PnL is used.
    return min(rows, key=lambda r: (r['rmse_exact'], -r[parameter]))[parameter]


def transform_record(record, x):
    fs = record['features']
    filled = np.where(np.isfinite(x), x, [f['median'] for f in fs])
    z = filled.copy()
    for j, f in enumerate(fs):
        if f['clip_enabled']:
            z[:, j] = np.clip(z[:, j], f['clip_low'], f['clip_high'])
    return (z - [f['mean'] for f in fs]) / [f['std'] for f in fs]


def linear_prediction(record, x):
    return record['intercept'] + transform_record(record, x) @ [f['coef'] for f in record['features']]


def restore_xgb(record):
    booster = xgboost.Booster()
    booster.load_model(bytearray(json.dumps(record['booster']).encode()))
    return booster


def fit_training(dates, level, xb, xt, base_features, time_features):
    """This API refuses dates beyond 2025; no 2026 values can tune a parameter."""
    if dates[-1] > CUTOFF or not (len(dates) == len(level) == len(xb) == len(xt)):
        raise ValueError('Training/tuning input must end by 2025-12-31')
    y = make_offset_labels(level, 6)
    cut = len(dates) - 1
    inner_cut = max(i for i, d in enumerate(dates) if d <= INNER_CUTOFF)
    inner = completed_indices(level, y, inner_cut, 6)
    train = completed_indices(level, y, cut, 6)
    validation = np.array([i for i in train if i > inner_cut], int)
    if len(inner) < 300 or len(validation) < 60:
        raise ValueError('Insufficient causal internal validation')
    scaler = TypeAwareScaler(time_features).fit(xt[inner])
    z = scaler.transform(xt)
    mean_y = float(y[inner].mean())
    centered = y[inner] - mean_y
    gram = z[inner].T @ z[inner]
    rhs = z[inner].T @ centered
    alphas = []
    for alpha in ALPHAS:
        coef = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)
        pred = mean_y + z[validation] @ coef
        alphas.append({'alpha': alpha, 'rmse_exact': exact_rmse(pred, y[validation]),
                       **forecast_metrics(pred, y[validation])})
    alpha = choose(alphas, 'alpha')
    state = initial_state(z[inner], y[inner], alpha)
    discounts = []
    for discount in DISCOUNTS:
        forecast = dynamic_forecast(z, y, inner_cut + 1, inner[-1], *state, discount)
        pred = forecast['mean'][validation - inner_cut - 1]
        discounts.append({'discount': discount, 'rmse_exact': exact_rmse(pred, y[validation]),
                          **forecast_metrics(pred, y[validation])})
    discount = choose(discounts, 'discount')
    tuning = XGBRegressor(n_estimators=300, early_stopping_rounds=20,
                           base_score=mean_y, **XGB_CONFIG)
    tuning.fit(z[inner], y[inner], eval_set=[(z[validation], y[validation])], verbose=False)
    rounds = int(tuning.best_iteration) + 1
    xgb_val = forecast_metrics(tuning.predict(z[validation]), y[validation])
    inner_original = fit_model(xb[inner], y[inner], base_features, alpha=15.)
    original_val = forecast_metrics(inner_original[1] + inner_original[0].transform(xb[validation]) @ inner_original[2],
                                    y[validation])
    original = fit_model(xb[train], y[train], base_features, alpha=15.)
    temporal = fit_model(xt[train], y[train], time_features, alpha=alpha)
    tz = temporal[0].transform(xt[train])
    mean, cov, noise = initial_state(tz, y[train], alpha)
    np.testing.assert_allclose(mean, np.r_[temporal[1], temporal[2]], atol=1e-12)
    xgb = XGBRegressor(n_estimators=rounds, base_score=float(y[train].mean()), **XGB_CONFIG)
    xgb.fit(tz, y[train], verbose=False)
    xgb_record = {'preprocessing': model_record(temporal, time_features),
                  'booster': json.loads(xgb.get_booster().save_raw(raw_format='json')),
                  'importances': raw_numbers(xgb.feature_importances_), 'rounds': rounds}
    restored = restore_xgb(xgb_record)
    np.testing.assert_array_equal(xgb.predict(tz), restored.predict(xgboost.DMatrix(tz)))
    return {'meta': {'train_through': CUTOFF, 'train_n': len(train), 'train_start': dates[train[0]],
                     'last_train_signal': dates[train[-1]], 'last_train_label_exit': dates[train[-1] + 6],
                     'last_train_index': int(train[-1]), 'alpha_original': 15., 'alpha_temporal': alpha,
                     'discount': discount, 'rounds': rounds, 'xgboost_version': xgboost.__version__,
                     'input_sha256': training_digest(xb, xt, y, dates), 'config': CONFIG},
            'validation': {'initial_train_through': INNER_CUTOFF, 'train_n': len(inner),
                           'last_inner_train_signal': dates[inner[-1]],
                           'last_inner_label_exit': dates[inner[-1] + 6],
                           'start': dates[validation[0]], 'last_signal': dates[validation[-1]],
                           'last_label_exit': dates[validation[-1] + 6], 'n': len(validation),
                           'alphas': alphas, 'discounts': discounts, 'xgb': xgb_val,
                           'ridge_original': original_val,
                           'xgb_rmse_path': [float(v) for v in tuning.evals_result()['validation_0']['rmse']]},
            'models': {'ridge_original': model_record(original, base_features),
                       'ridge_temporal': model_record(temporal, time_features),
                       'dlm_temporal': {'mean': mean.tolist(), 'covariance': cov.tolist(),
                                        'noise_variance': noise}, 'xgb_temporal': xgb_record},
            'features': definitions(time_features),
            'quality': {'train_missing_cells_original': int(np.sum(~np.isfinite(xb[train]))),
                        'train_missing_cells_temporal': int(np.sum(~np.isfinite(xt[train]))),
                        'train_missing_by_feature': np.sum(~np.isfinite(xt[train]), axis=0).tolist()}}


def build():
    protected = {n: sha(n) for n in PROTECTED}
    dates, level, xb, y, base, cut, train = load_data()
    features = temporal_features(level, base)
    xt = matrix(features)
    # Pass only the pre-2026 slice into parameter selection and fitting.
    train_y = make_offset_labels(level[:cut + 1], 6)
    fingerprint = training_digest(xb[:cut + 1], xt[:cut + 1], train_y, dates[:cut + 1])
    if (ROOT / MODELS).exists():
        bundle = read(MODELS)
        if bundle['meta']['input_sha256'] != fingerprint or bundle['features'] != definitions(features):
            raise ValueError('Frozen temporal training inputs changed; do not overwrite the study model')
    else:
        bundle = fit_training(dates[:cut + 1], level[:cut + 1], xb[:cut + 1], xt[:cut + 1], base, features)
    assert bundle['meta']['input_sha256'] == fingerprint
    start = cut + 1
    td, tl, target = dates[start:], level[start:], y[start:]
    if not np.isfinite(xt[start:]).all():
        raise ValueError('Expanded features missing in test; stop rather than silently alter comparison dates')
    models = bundle['models']
    op = linear_prediction(models['ridge_original'], xb)
    tp = linear_prediction(models['ridge_temporal'], xt)
    z = transform_record(models['ridge_temporal'], xt)
    state = models['dlm_temporal']
    dyn = dynamic_forecast(z, y, start, bundle['meta']['last_train_index'], np.array(state['mean']),
                           np.array(state['covariance']), state['noise_variance'], bundle['meta']['discount'], dates)
    booster = restore_xgb(models['xgb_temporal'])
    xp = booster.predict(xgboost.DMatrix(z))
    seeds = {'ridge_original': op[start - 250:start], 'ridge_temporal': tp[start - 250:start],
             'dlm_temporal': tp[start - 250:start], 'xgb_temporal': xp[start - 250:start]}
    variants = {
        'ridge_original': summarize_variant('原Ridge · 20＋8', op[start:], seeds['ridge_original'], td, tl, target),
        'ridge_temporal': summarize_variant('时序Ridge · 92输入', tp[start:], seeds['ridge_temporal'], td, tl, target),
        'dlm_temporal': summarize_variant('时序贝叶斯递推 · 92输入', dyn['mean'], seeds['dlm_temporal'], td, tl, target,
                                          dyn['std'], (dyn['mean'] - 1.281551565545 * dyn['std'],
                                                       dyn['mean'] + 1.281551565545 * dyn['std']),
                                          ndtr(dyn['mean'] / dyn['std'])),
        'xgb_temporal': summarize_variant('时序XGBoost · 92输入', xp[start:], seeds['xgb_temporal'], td, tl, target)}
    old = read('curve_timing_bayes.json')
    if old['dates'] != td:
        raise ValueError('Original Ridge reference is not on the same calendar')
    np.testing.assert_allclose(op[start:], old['variants']['ridge20']['prediction'], atol=1e-12, rtol=0)
    # Attribution controls, not additional selected production candidates. Freeze
    # both records from the same 649 completed training labels. This prevents
    # conflating the effect of lag inputs with changing alpha from 15 to 500.
    if 'controls' not in bundle:
        bundle['controls'] = {
            'original_matched_alpha': model_record(fit_model(xb[train], y[train], base,
                                                            alpha=bundle['meta']['alpha_temporal']), base),
            'temporal_original_alpha': model_record(fit_model(xt[train], y[train], features, alpha=15.), features)}
    controls = {}
    for key, label, xx in (('original_matched_alpha', '原28输入 · 与时序Ridge同收缩', xb),
                            ('temporal_original_alpha', '时序92输入 · 原alpha=15', xt)):
        pred = linear_prediction(bundle['controls'][key], xx)
        controls[key] = summarize_variant(label, pred[start:], pred[start - 250:start], td, tl, target)
        for row in controls[key]['modes'].values():
            row['trade_stats'] = trade_stats(row['sim']['trades'])
    for key, variant in variants.items():
        variant['train_n'] = len(train)
        variant['features_n'] = 28 if key == 'ridge_original' else len(features)
        variant['fitting_mode'] = 'delayed-online-update' if key == 'dlm_temporal' else 'frozen'
        for mode, row in variant['modes'].items():
            row['trade_stats'] = trade_stats(row['sim']['trades'])
            row['sides'] = {str(d): trade_stats([t for t in row['sim']['trades'] if t['direction'] == d]) for d in (-1, 1)}
            if key == 'ridge_original':
                for field in ('daily_pnl', 'position', 'equity', 'drawdown'):
                    np.testing.assert_allclose(row['sim'][field], old['variants']['ridge20']['modes'][mode]['sim'][field],
                                               atol=1e-10, rtol=0)
    feature_audit = []
    for j, f in enumerate(features):
        feature_audit.append({k: v for k, v in f.items() if k != 'values'} |
                             {'latest': float(xt[-1, j]), 'train_missing_n': bundle['quality']['train_missing_by_feature'][j],
                              'clip_enabled': models['ridge_temporal']['features'][j]['clip_enabled'],
                              'ridge_coef': models['ridge_temporal']['features'][j]['coef'],
                              'dlm_initial_coef': state['mean'][j + 1],
                              'dlm_latest_coef': float(dyn['coefficients'][-1, j + 1]),
                              'xgb_gain_share': models['xgb_temporal']['importances'][j]})
    result = {'meta': {'asof': dates[-1], 'train_through': CUTOFF, 'test_start': td[0],
                       'test_n': len(td), 'completed_n': int(np.sum(np.isfinite(target))),
                       'train_n': len(train), 'last_train_signal': bundle['meta']['last_train_signal'],
                       'last_train_label_exit': bundle['meta']['last_train_label_exit'],
                       'features_original': 28, 'features_temporal': len(features),
                       'blocks': {'original': 28, 'spread_history': 4, 'factor_history': 60},
                       'default_mode': 'cold', 'alpha_original': 15.,
                       'alpha_temporal': bundle['meta']['alpha_temporal'], 'discount': bundle['meta']['discount'],
                       'xgb_rounds': bundle['meta']['rounds'], 'horizon': 6,
                       'cold': '仅使用2026年依次产生的历史预测，至少60条后交易；所有账户从共同首日空仓开始。',
                       'seeded': '用2025年底模型对此前250日回溯评分初始化；不是当时发布的预测，属于探索口径。',
                       'caveat': '固定20因子来自全历史筛选，含选择前视；2026已反复观察，本轮为探索研究。未替换每日策略。',
                       'preprocessing': '同一训练样本649条；仅按原类型对指定机构金额及其历史项做训练1%/99%缩尾，价格、价差、利率及技术指标不缩尾。缺失只用训练中位数填补，参数冻结。',
                       'change_note': '原值、滞后5日与5日变动在线性代数上冗余；保留便于统一输入，Ridge收缩处理共线，不能把变动项称为额外独立信息。',
                       'protected_hashes': protected, 'models_sha256': None},
              'dates': td, 'level': raw_numbers(tl), 'target': raw_numbers(target), 'variants': variants,
              'controls': controls,
              'forecast_references': {'zero': forecast_metrics(np.zeros(len(td)), target),
                                       'train_mean': forecast_metrics(np.full(len(td), y[train].mean()), target)},
              'validation': bundle['validation'], 'features': feature_audit,
              'quality': bundle['quality'] | {'test_missing_cells': int(np.sum(~np.isfinite(xt[start:])))},
              'diagnostics': {'dlm': {'updates': dyn['updates'], 'coefficients': dyn['coefficients'].tolist(),
                                       'coefficient_names': ['截距'] + [f['label'] for f in features]},
                              'xgb': {'config': XGB_CONFIG, 'rounds': bundle['meta']['rounds'],
                                      'version': bundle['meta']['xgboost_version']}}}
    if protected != {n: sha(n) for n in PROTECTED}:
        raise ValueError('Production or prior research changed during the experiment')
    json.dumps(bundle, allow_nan=False)
    write_json_skip_unchanged(str(ROOT / MODELS), bundle)
    result['meta']['models_sha256'] = sha(MODELS)
    json.dumps(result, allow_nan=False)
    write_json_skip_unchanged(str(ROOT / OUTPUT), result)
    print(json.dumps({'meta': {k: result['meta'][k] for k in ('asof', 'train_n', 'features_temporal',
                                                           'alpha_temporal', 'discount', 'xgb_rounds')},
                      'results': {k: {'forecast': v['forecast'], 'pnl': {m: r['sim']['metrics'] for m, r in v['modes'].items()}}
                                  for k, v in variants.items()}}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    build()
