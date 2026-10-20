"""Fixed 2025 calibration bins, 2026 prediction-only diagnostics for three models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from jsonio import write_json_skip_unchanged
from prepare_bayes_study import dynamic_forecast, initial_state, load_data
from prepare_curve_timing_capacity import fit_model, model_record
from prepare_curve_timing_walkforward import completed_indices
from prepare_direction_screen import raw_numbers
from prepare_forecast_comparison import PROTECTED as COMPARISON_PROTECTED, bootstrap_indices
from prepare_strategy_model import make_offset_labels
from prepare_temporal_study import matrix, temporal_features

ROOT = Path(__file__).resolve().parent
OUTPUT = 'forecast_signal_diagnostics.json'
CALIBRATION = 'forecast_signal_calibration_2025.json'
KEYS = ('ridge_original', 'ridge_temporal', 'dlm_temporal')
PROTECTED = COMPARISON_PROTECTED + ('forecast_model_comparison.json', 'forecast_model_comparison_models.json',
                                   'forecast_model_shortlist.json')
CONFIG = {'train_through': '2025-06-30', 'calibration_end': '2025-12-31', 'horizon': 6,
          'original_alpha': 15., 'temporal_alpha': 500., 'discount': 1.,
          'quantiles': [1/3, 2/3], 'quantile_method': 'linear', 'bootstrap_block': 12,
          'bootstrap_draws': 2000, 'bootstrap_seed': 42, 'small_sample': 20}
GROUPS = [(direction * strength, direction, strength) for direction in (1, -1) for strength in (1, 2, 3)]


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def fingerprint(dates, level, xb, xt):
    h = hashlib.sha256(json.dumps({'dates': dates, 'config': CONFIG}, sort_keys=True).encode())
    for a in (level, xb, xt):
        v = np.array(a, dtype='<f8', copy=True)
        v[~np.isfinite(v)] = np.nan
        h.update(str(v.shape).encode())
        h.update(v.tobytes())
    return h.hexdigest()


def fit_boundaries(prediction):
    p = np.asarray(prediction, float)
    if not np.isfinite(p).all():
        raise ValueError('Calibration predictions must be finite')
    result = {}
    for direction in (1, -1):
        v = np.abs(p[p * direction > 0])
        cuts = np.quantile(v, CONFIG['quantiles'], method='linear').tolist() if len(v) >= 3 else None
        result[str(direction)] = {'n': len(v), 'cuts': cuts,
                                   'min_abs': float(v.min()) if len(v) else None,
                                   'max_abs': float(v.max()) if len(v) else None}
    return result


def classify(prediction, boundaries):
    p = np.asarray(prediction, float)
    if not np.isfinite(p).all():
        raise ValueError('Missing forecast cannot be silently classified')
    result = np.zeros(len(p), int)
    for direction in (1, -1):
        mask = p * direction > 0
        cuts = boundaries[str(direction)]['cuts']
        if cuts is None:
            result[mask] = 99 * direction  # explicit unsupported direction, not a weak group
        else:
            result[mask] = direction * (1 + np.searchsorted(cuts, np.abs(p[mask]), side='left'))
    return result


def calibrate(dates, level, xb, xt, base, features):
    if dates[-1] > CONFIG['calibration_end']:
        raise ValueError('Calibration cannot receive 2026 data')
    y = make_offset_labels(level, 6)
    cut = max(i for i, day in enumerate(dates) if day <= CONFIG['train_through'])
    train = completed_indices(level, y, cut, 6)
    if len(train) < 300:
        raise ValueError('Insufficient calibration training labels')
    original = fit_model(xb[train], y[train], base, alpha=CONFIG['original_alpha'])
    temporal = fit_model(xt[train], y[train], features, alpha=CONFIG['temporal_alpha'])
    z = temporal[0].transform(xt)
    state = initial_state(z[train], y[train], CONFIG['temporal_alpha'])
    start = cut + 1
    dyn = dynamic_forecast(z, y, start, train[-1], *state, CONFIG['discount'], dates)
    predictions = {'ridge_original': original[1] + original[0].transform(xb[start:]) @ original[2],
                   'ridge_temporal': temporal[1] + z[start:] @ temporal[2], 'dlm_temporal': dyn['mean']}
    yy = y[start:]
    complete = np.isfinite(yy)
    return {'meta': {'config': CONFIG, 'input_sha256': fingerprint(dates, level, xb, xt),
                     'train_n': len(train), 'last_train_signal': dates[train[-1]],
                     'last_train_label_exit': dates[train[-1] + 6],
                     'start': dates[start], 'end': dates[-1], 'prediction_n': len(yy),
                     'completed_n': int(complete.sum()),
                     'last_complete_signal': dates[start + np.flatnonzero(complete)[-1]]},
            'dates': dates[start:], 'target': raw_numbers(yy),
            'models': {'ridge_original': model_record(original, base),
                       'ridge_temporal': model_record(temporal, features),
                       'dlm_temporal': {'mean': state[0].tolist(), 'covariance': state[1].tolist(),
                                        'noise_variance': state[2], 'discount': CONFIG['discount']}},
            'predictions': {k: raw_numbers(v) for k, v in predictions.items()},
            'boundaries': {k: fit_boundaries(v[complete]) for k, v in predictions.items()},
            'updates': dyn['updates']}


def interval(values, minimum_valid):
    v = np.asarray(values, float)
    valid = v[np.isfinite(v)]
    return {'low': float(np.quantile(valid, .025)) if len(valid) >= minimum_valid else None,
            'high': float(np.quantile(valid, .975)) if len(valid) >= minimum_valid else None,
            'valid_draws': len(valid)}


def bootstrap_groups(prediction, actual, codes):
    """Sample the whole dated sequence, never independent rows inside a bucket."""
    y, c = np.asarray(actual, float), np.asarray(codes, int)
    complete = np.isfinite(y)
    if np.any(np.diff(np.flatnonzero(complete)) > 1):
        raise ValueError('Interior target gaps cannot be compressed for dated block resampling')
    y, c = y[complete], c[complete]
    draws = CONFIG['bootstrap_draws']
    if len(y) < CONFIG['bootstrap_block']:
        return {}, {}
    ix = bootstrap_indices(len(y), CONFIG['bootstrap_block'], draws, CONFIG['bootstrap_seed'])
    yy, cc = y[ix], c[ix]
    raw, exported = {}, {}
    for code, direction, _ in GROUPS:
        mask = cc == code
        den = mask.sum(axis=1)
        direction_den = (mask & (yy != 0)).sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            signed_mean = np.sum(np.where(mask, direction * yy, 0), axis=1) / den
            hit = np.sum(mask & (direction * yy > 0), axis=1) / direction_den
        raw[code] = {'signed_mean': signed_mean, 'direction_accuracy': hit}
        exported[str(code)] = {k: interval(v, int(.9 * draws)) for k, v in raw[code].items()}
    contrasts = {}
    for direction in (1, -1):
        contrasts[str(direction)] = {name: interval(raw[direction * 3][name] - raw[direction][name], int(.9 * draws))
                                     for name in ('signed_mean', 'direction_accuracy')}
    return exported, contrasts


def summarize_period(dates, prediction, actual, boundaries):
    p, y = np.asarray(prediction, float), np.asarray(actual, float)
    codes = classify(p, boundaries)
    ci, contrasts = bootstrap_groups(p, y, codes)
    rows = []
    for code, direction, strength in GROUPS:
        mask = codes == code
        complete = mask & np.isfinite(y)
        yp, pp = y[complete], p[complete]
        n = len(yp)
        nonzero = yp != 0
        correct = int(np.sum(direction * yp > 0))
        error = pp - yp
        rows.append({'code': code, 'direction': direction, 'strength': strength,
                     'forecast_n': int(mask.sum()), 'n': n, 'pending_n': int(mask.sum()) - n,
                     'direction_n': int(nonzero.sum()), 'correct': correct, 'actual_zero_n': int(np.sum(~nonzero)),
                     'direction_accuracy': correct / int(nonzero.sum()) if nonzero.any() else None,
                     'actual_mean': float(yp.mean()) if n else None,
                     'actual_median': float(np.median(yp)) if n else None,
                     'signed_mean': float(direction * yp.mean()) if n else None,
                     'signed_median': float(np.median(direction * yp)) if n else None,
                     'prediction_mean': float(pp.mean()) if n else None,
                     'mae': float(np.mean(np.abs(error))) if n else None,
                     'rmse': float(np.sqrt(np.mean(error ** 2))) if n else None,
                     'max_absolute_error': float(np.max(np.abs(error))) if n else None,
                     'small_sample': n < CONFIG['small_sample'],
                     'interval95': ci.get(str(code), {})})
    diagnostics = {}
    for direction in (1, -1):
        group = [row for row in rows if row['direction'] == direction]
        nonempty = all(row['direction_n'] > 0 for row in group)
        diagnostics[str(direction)] = {
            'n': sum(row['n'] for row in group), 'all_groups_at_least_20': all(row['n'] >= 20 for row in group),
            'accuracy_monotonic': all(a['direction_accuracy'] <= b['direction_accuracy'] for a, b in zip(group, group[1:])) if nonempty else None,
            'signed_mean_monotonic': all(a['signed_mean'] <= b['signed_mean'] for a, b in zip(group, group[1:])) if nonempty else None,
            'strong_minus_weak_accuracy': group[-1]['direction_accuracy'] - group[0]['direction_accuracy'] if nonempty else None,
            'strong_minus_weak_mean': group[-1]['signed_mean'] - group[0]['signed_mean'] if nonempty else None,
            'contrast_interval95': contrasts.get(str(direction), {})}
    unsupported = np.abs(codes) == 99
    zero = codes == 0
    outside = np.zeros(len(p), bool)
    for direction in (1, -1):
        source = boundaries[str(direction)]
        if source['n']:
            outside |= (p * direction > 0) & ((np.abs(p) < source['min_abs']) | (np.abs(p) > source['max_abs']))
    return {'dates': dates, 'prediction': raw_numbers(p), 'target': raw_numbers(y), 'codes': codes.tolist(),
            'rows': rows, 'diagnostics': diagnostics,
            'quality': {'forecast_n': len(p), 'completed_n': int(np.isfinite(y).sum()),
                        'pending_n': int(np.sum(~np.isfinite(y))), 'zero_prediction_n': int(zero.sum()),
                        'zero_completed_n': int(np.sum(zero & np.isfinite(y))),
                        'unsupported_n': int(unsupported.sum()),
                        'unsupported_completed_n': int(np.sum(unsupported & np.isfinite(y))),
                        'outside_calibration_range_n': int(outside.sum())}}


def build():
    protected = {n: sha(n) for n in PROTECTED}
    comparison = read('forecast_model_comparison.json')
    if set(read('forecast_model_shortlist.json')['selected']) != set(KEYS):
        raise ValueError('Confirm a new diagnostic version if the retained model set changes')
    dates, level, xb, _, base, year_end, _ = load_data()
    fs = temporal_features(level, base)
    xt = matrix(fs)
    dd = dates[:year_end + 1]
    ll, bx, tx = level[:year_end + 1], xb[:year_end + 1], xt[:year_end + 1]
    expected = fingerprint(dd, ll, bx, tx)
    if (ROOT / CALIBRATION).exists():
        calibration = read(CALIBRATION)
        if calibration['meta']['input_sha256'] != expected:
            raise ValueError('Frozen 2025 calibration inputs changed; refusing to overwrite')
    else:
        calibration = calibrate(dd, ll, bx, tx, base, fs)
    cy = np.array(calibration['target'], float)
    complete = np.isfinite(cy)
    cd = [day for day, ok in zip(calibration['dates'], complete) if ok]
    models = {}
    for key in KEYS:
        source = comparison['models'][key]
        boundaries = calibration['boundaries'][key]
        cp = np.array(calibration['predictions'][key], float)
        models[key] = {'label': source['label'], 'boundaries': boundaries,
                       'calibration2025': summarize_period(cd, cp[complete], cy[complete], boundaries),
                       'test2026': summarize_period(comparison['dates'], np.array(source['prediction']),
                                                    np.array(comparison['target'], float), boundaries)}
    result = {'meta': {'asof': comparison['meta']['asof'], 'train_through': '2025-12-31',
                       'initial_calibration_train_through': CONFIG['train_through'],
                       'calibration_train_n': calibration['meta']['train_n'],
                       'calibration_start': cd[0], 'calibration_end_signal': cd[-1],
                       'calibration_complete_n': len(cd), 'test_start': comparison['meta']['test_start'],
                       'test_prediction_n': comparison['meta']['forecast_n'], 'test_complete_n': comparison['meta']['completed_n'],
                       'last_test_complete_signal': comparison['meta']['last_evaluation_signal'],
                       'config': CONFIG, 'prediction_only': True, 'protected_hashes': protected,
                       'caveat': '2025是内部开发校准，参数也参考过该验证期；2026已反复研究，固定名单有选择前视。分层不是独立交易回测，不自动选开仓门槛。'},
              'calibration_audit': {k: v for k, v in calibration['meta'].items() if k != 'input_sha256'},
              'models': models}
    if protected != {n: sha(n) for n in PROTECTED}:
        raise ValueError('Existing forecasts or models changed during diagnostics')
    json.dumps(calibration, allow_nan=False)
    write_json_skip_unchanged(str(ROOT / CALIBRATION), calibration)
    result['meta']['calibration_sha256'] = sha(CALIBRATION)
    json.dumps(result, allow_nan=False)
    write_json_skip_unchanged(str(ROOT / OUTPUT), result)
    print(json.dumps({k: {'diagnostics': v['test2026']['diagnostics'],
                          'groups': [{f: row[f] for f in ('code', 'n', 'direction_accuracy', 'signed_mean', 'mae', 'rmse')}
                                     for row in v['test2026']['rows']]} for k, v in models.items()}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    build()
