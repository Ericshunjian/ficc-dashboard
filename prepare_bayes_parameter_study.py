"""Bounded Bayesian sensitivity study. No production or prior archive is mutated."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jsonio import write_json_skip_unchanged
from prepare_bayes_study import dynamic_forecast, load_data
from prepare_curve_timing_walkforward import completed_indices
from prepare_direction_screen import raw_numbers
from prepare_entry_rules_study import entry_gate, read, sha
from prepare_exit_rules_study import simulate_reverse
from prepare_forecast_comparison import metrics, bootstrap_indices, boot_metrics
from prepare_strategy_model import make_offset_labels
from prepare_strategy_stability import PROTECTED as OLD_PROTECTED, quarter
from prepare_temporal_study import temporal_features, matrix, transform_record, training_digest
from prepare_threshold_study import trade_stats

ROOT = Path(__file__).resolve().parent
OUTPUT = 'bayes_parameter_study.json'
MODELS = 'bayes_parameter_models.json'
PROTECTED = tuple(dict.fromkeys(OLD_PROTECTED + ('strategy_stability.json',)))
CONFIG = {'version': 1, 'inner_cutoff': '2025-06-30', 'train_through': '2025-12-31',
          'alpha': 500., 'discounts': [1., .995, .98], 'covariance_multipliers': [1., 6., 18.],
          'horizon': 6, 'update_spacing': 6, 'noise': 'fixed_training_residual_variance',
          'selection': '2025 validation minimum RMSE; ties prefer higher discount then lower covariance',
          'bootstrap': {'block': 12, 'draws': 2000, 'seed': 42}}


def identifier(discount, multiplier):
    return f'd{discount:g}_c{multiplier:g}'


def run(z, y, start, last_train, state, discount, multiplier, dates):
    if not (0 < discount <= 1 and multiplier > 0):
        raise ValueError('Invalid discount or covariance multiplier')
    # Old covariance = 6 * noise * inverse(precision). This parameter replaces
    # the factor 6, not the noise variance or ridge penalty. It is sensitivity
    # analysis of a working prior, not an estimated dependence correction.
    return dynamic_forecast(z, y, start, last_train, np.array(state['mean']),
                            np.array(state['covariance']) * multiplier / 6.,
                            state['noise_variance'], discount, dates)


def select(rows):
    return min(rows, key=lambda k: (rows[k]['validation']['rmse'], -rows[k]['discount'],
                                   rows[k]['covariance_multiplier']))


def tune(dates, level, x, calibration):
    if dates[-1] > CONFIG['train_through'] or len(dates) != len(x) or len(dates) != len(level):
        raise ValueError('Tuning input must end by 2025-12-31 and be aligned')
    if calibration['meta']['config']['temporal_alpha'] != CONFIG['alpha']:
        raise ValueError('Frozen calibration shrinkage differs')
    y = make_offset_labels(level, 6)
    cut = max(i for i, d in enumerate(dates) if d <= CONFIG['inner_cutoff'])
    inner = completed_indices(level, y, cut, 6)
    valid = np.array([i for i in completed_indices(level, y, len(dates)-1, 6) if i > cut], int)
    if len(inner) != calibration['meta']['train_n'] or len(valid) < 60:
        raise ValueError('Calibration history differs or insufficient labels')
    z = transform_record(calibration['models']['ridge_temporal'], x)
    rows = {}
    for discount in CONFIG['discounts']:
        for multiplier in CONFIG['covariance_multipliers']:
            out = run(z, y, cut+1, int(inner[-1]), calibration['models']['dlm_temporal'],
                      discount, multiplier, dates)
            prediction = out['mean'][valid-cut-1]
            rows[identifier(discount, multiplier)] = {
                'discount': discount, 'covariance_multiplier': multiplier,
                'validation': metrics(prediction, y[valid]),
                'validation_prediction': raw_numbers(out['mean']), 'validation_updates': out['updates']}
    np.testing.assert_allclose(rows['d1_c6']['validation_prediction'],
                               calibration['predictions']['dlm_temporal'], atol=1e-11, rtol=0)
    return {'initial_train_through': dates[cut], 'initial_train_n': len(inner),
            'last_train_signal': dates[inner[-1]], 'last_train_label_exit': dates[inner[-1]+6],
            'dates': dates[cut+1:], 'target': raw_numbers(y[cut+1:]), 'n': len(valid),
            'last_complete_signal': dates[valid[-1]], 'last_complete_exit': dates[valid[-1]+6],
            'rows': rows, 'selected': select(rows)}


def intervals(prediction, reference, target):
    good = np.isfinite(target)
    p, r, y = prediction[good], reference[good], target[good]
    cfg = CONFIG['bootstrap']
    ix = bootstrap_indices(len(y), cfg['block'], cfg['draws'], cfg['seed'])
    first, second = boot_metrics(p[ix], y[ix]), boot_metrics(r[ix], y[ix])
    names = ('direction_accuracy', 'balanced_accuracy', 'rank_ic', 'mae', 'rmse')
    return {name: {'low': float(np.nanquantile(first[name]-second[name], .025)),
                   'high': float(np.nanquantile(first[name]-second[name], .975))} for name in names}


def build():
    protected = {n: sha(n) for n in PROTECTED}
    dates, level, xb, y, base, cut, train = load_data()
    features = temporal_features(level, base)
    x = matrix(features)
    bundle, calibration, source = (read(n) for n in (
        'curve_timing_temporal_models.json', 'forecast_signal_calibration_2025.json',
        'forecast_model_comparison.json'))
    if training_digest(xb[:cut+1], x[:cut+1], make_offset_labels(level[:cut+1], 6), dates[:cut+1]) != bundle['meta']['input_sha256']:
        raise ValueError('Frozen training data changed')
    tuning = tune(dates[:cut+1], level[:cut+1], x[:cut+1], calibration)
    selected = tuning['selected']
    chosen = tuning['rows'][selected]
    initial = bundle['models']['dlm_temporal']
    freeze = {'config': CONFIG, 'selected': selected, 'training_input_sha256': bundle['meta']['input_sha256'],
              'calibration_sha256': sha('forecast_signal_calibration_2025.json'),
              'train_n': len(train), 'last_train_signal': dates[train[-1]],
              'last_train_label_exit': dates[train[-1]+6],
              'preprocessing': bundle['models']['ridge_temporal'],
              'state': {**initial, 'covariance': (np.array(initial['covariance']) * chosen['covariance_multiplier']/6.).tolist()},
              'discount': chosen['discount']}
    if (ROOT/MODELS).exists() and read(MODELS) != freeze:
        raise ValueError('New study model already frozen; use a new version')
    td = dates[cut+1:]
    if td != source['dates']:
        raise ValueError('Evaluation dates differ from archived model comparison')
    target, prices = y[cut+1:], level[cut+1:]
    z = transform_record(bundle['models']['ridge_temporal'], x)
    rows = tuning['rows']
    for key, row in rows.items():
        forecast = run(z, y, cut+1, int(train[-1]), initial, row['discount'],
                       row['covariance_multiplier'], dates)
        row.update(prediction=raw_numbers(forecast['mean']), std=raw_numbers(forecast['std']),
                   metrics=metrics(forecast['mean'], target), updates=forecast['updates'])
    np.testing.assert_allclose(rows['d1_c6']['prediction'], source['models']['dlm_temporal']['prediction'], atol=1e-11, rtol=0)
    names = {'ridge': '时序Ridge·冻结', 'old': '时序贝叶斯·旧参数', 'selected': '时序贝叶斯·2025验证选参'}
    predictions = {'ridge': np.array(source['models']['ridge_temporal']['prediction']),
                   'old': np.array(rows['d1_c6']['prediction']), 'selected': np.array(rows[selected]['prediction'])}
    common_cut = read('entry_rules_study.json')['meta']['negative_cuts']['ridge_temporal']
    models = {}
    for key, prediction in predictions.items():
        periods = {}
        for q in sorted({quarter(d) for d in td}):
            mask = np.array([quarter(d)==q for d in td])
            complete = np.flatnonzero(mask & np.isfinite(target))
            periods[q] = {'metrics': metrics(prediction[mask], target[mask]),
                          'last_complete_signal': td[complete[-1]] if len(complete) else None}
        trades = {}
        for rule, policy in [('sign','reverse'), ('asymmetric','neutral')]:
            gate = entry_gate(prediction, rule, common_cut)
            sim = simulate_reverse(td, prices, prediction, gate, rule, common_cut, exit_policy=policy)
            trades[rule] = {'exit': policy, 'sim': sim, 'trade_stats': trade_stats(sim['trades'])}
        models[key] = {'label': names[key], 'prediction': raw_numbers(prediction),
                       'metrics': metrics(prediction, target), 'quarters': periods, 'trading': trades}
    result = {'meta': {'asof': td[-1], 'train_through': CONFIG['train_through'], 'train_n': len(train),
                       'input_n': len(features), 'prediction_n': len(td), 'completed_n': int(np.isfinite(target).sum()),
                       'last_complete_signal': td[np.flatnonzero(np.isfinite(target))[-1]],
                       'selected': selected, 'old': 'd1_c6', 'config': CONFIG,
                       'shared_negative_cut': common_cut, 'protected_hashes': protected,
                       'caveat': '2026已反复观察、固定20因子含全历史筛选前视；本轮仅2025验证选参，2026仍为探索评估，未扣成本、不替换生产。',
                       'trading_note': '纯方向开仓／反向退出；另附非对称开仓／中性退出，共用既有时序Ridge做空阈值，最长6日。次日代理开盘=t收盘；末日浮动保留，不加调参策略。'},
              'dates': td, 'level': raw_numbers(prices), 'target': raw_numbers(target),
              'validation': {k:v for k,v in tuning.items() if k!='rows'},
              'grid': rows, 'models': models,
              'paired95': {key: intervals(predictions['selected'], predictions[key], target) for key in ('old','ridge')}}
    if protected != {n: sha(n) for n in PROTECTED}:
        raise ValueError('Prior models, factors or archived strategies changed')
    json.dumps(result, allow_nan=False)
    write_json_skip_unchanged(str(ROOT/MODELS), freeze)
    write_json_skip_unchanged(str(ROOT/OUTPUT), result)
    print(json.dumps({'selected': selected, 'grid': {k: {'validation_rmse': v['validation']['rmse'],
                       'test_rmse': v['metrics']['rmse'], 'direction': v['metrics']['direction_accuracy'],
                       'rank_ic': v['metrics']['rank_ic']} for k,v in rows.items()},
                       'comparison': {k: {'metrics': v['metrics'], 'trading': {r: t['sim']['metrics'] for r,t in v['trading'].items()}} for k,v in models.items()}}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    build()
