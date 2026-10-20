"""Extend all frozen comparison models, without fitting or changing old forecasts."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from prepare_curve_timing_percentile import serialize_sim, signed_percentile_gate
from prepare_curve_timing_walkforward import forecast_metrics, simulate
from prepare_direction_screen import raw_numbers, safe_number

ROOT = Path(__file__).resolve().parent
MODELS = 'curve_timing_nonlinear_models.json'
FAMILIES = ('ridge', 'tree', 'xgb', 'logistic')
KEYS = tuple(f'{family}{n}' for n in (20, 30, 40) for family in FAMILIES)
AUTHORITIES = (MODELS, 'ridge_daily_models.json', 'curve_timing_logistic_models.json')
SOURCES = ('curve_timing_nonlinear.json', 'curve_timing_logistic.json')


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def validate_models(frozen, ridge):
    sets = read('curve_timing_capacity_factor_sets.json')['sets']
    for key, model in frozen['models'].items():
        fs = model['preprocessing']
        n = int(key.removeprefix(model['family']))
        if len(fs) != n + 8 or [f['id'] for f in fs[8:]] != sets[f'f{n}']:
            raise ValueError('Comparison factor membership changed: ' + key)
        if any(f['clip_enabled'] for f in fs[:8]):
            raise ValueError('Comparison baselines must not be clipped')
        if any(not all(np.isfinite(f[k]) for k in ('median', 'mean', 'std')) or f['std'] <= 0 for f in fs):
            raise ValueError('Invalid frozen comparison preprocessing')
    for n in (20, 30):
        if frozen['models'][f'ridge{n}']['linear_model'] != ridge['models'][f'f{n}']:
            raise ValueError('Comparison Ridge differs from daily authority')


def transform(model, values):
    result = []
    for f, value in zip(model['preprocessing'], values, strict=True):
        x = float(value) if np.isfinite(value) else f['median']
        if f['clip_enabled']:
            x = float(np.clip(x, f['clip_low'], f['clip_high']))
        result.append((x - f['mean']) / f['std'])
    return np.asarray(result, float)


def predict(model, values, boosters=None, key=None):
    if model['family'] == 'ridge':
        from prepare_ridge_daily import infer
        return infer(model['linear_model'], values)[0]
    # sklearn trees and XGBoost both consume float32 feature matrices.
    z = transform(model, values).astype(np.float32)
    if model['family'] == 'tree':
        nodes, i = model['tree']['nodes'], 0
        while not nodes[i]['leaf']:
            node = nodes[i]
            i = node['left'] if float(z[node['feature_index']]) <= node['threshold_scaled'] else node['right']
        return float(nodes[i]['value'])
    if boosters is None:
        boosters = {}
    if key not in boosters:
        deps = ROOT.parent / 'ficc_strategy_research' / 'python_deps'
        if deps.is_dir() and str(deps) not in sys.path:
            sys.path.insert(0, str(deps))
        import xgboost
        booster = xgboost.Booster()
        booster.load_model(bytearray(json.dumps(model['booster']).encode('utf-8')))
        booster.set_param({'nthread': 2})
        boosters[key] = booster
    return float(boosters[key].inplace_predict(z.reshape(1, -1))[0])


def simulate_records(rows, key, rule):
    pred = np.array([r['models'][key]['prediction'] for r in rows], float)
    records = [r['models'][key] for r in rows]
    logistic = key.startswith('logistic')
    score = np.array([100 * r['prediction'] if logistic and r['prediction'] is not None and
                      r['percentile'] is not None else r['percentile'] if not logistic else np.nan
                      for r in records], float)
    g = {'direction_basis': 'percentile', 'percentile': score,
         'signal': np.array([r['signal'] for r in records], int),
         'history_n': np.array([r['history_n'] for r in records], int),
         'quantile': np.array([r['upper_reference'] for r in records], float)}
    return serialize_sim(simulate([r['date'] for r in rows], [r['level'] for r in rows], pred,
                                 g['quantile'], rule=rule, horizon=6, gate=g))


def metrics(pred, target, logistic):
    if not logistic:
        return forecast_metrics(np.array(pred, float), target)
    p = np.asarray(pred, float)
    valid = np.isfinite(p) & np.isfinite(target)
    p, change = p[valid], target[valid]
    binary = change > 0
    return {'n': len(p), 'ic': safe_number(spearmanr(p, change).statistic) if len(p) > 1 else None,
            'rmse': None, 'auc': safe_number(roc_auc_score(binary, p)) if len(np.unique(binary)) > 1 else None,
            'accuracy': safe_number(np.mean((p > .5) == binary)) if len(p) else None}


def build_archive(report, old, features, dates, source_info, bond_dates, repo_dates, now):
    from prepare_ridge_daily import quality, sha
    frozen, ridge = read(MODELS), read('ridge_daily_models.json')
    validate_models(frozen, ridge)
    hashes = {n: sha(n) for n in AUTHORITIES + SOURCES}
    if old and old['meta']['frozen_hashes'] != hashes:
        raise ValueError('Frozen comparison model or research snapshot changed; cannot continue archive')
    base, logbase = (read(n) for n in SOURCES)
    if base['daily']['dates'] != logbase['daily']['dates']:
        raise ValueError('Comparison research calendars differ')
    history_axis = {d: i for i, d in enumerate(base['daily']['dates'])}
    axis = {d: i for i, d in enumerate(dates)}
    logrows = {r['date']: r for r in report['logistic']['records']}
    known = {r['date']: r for r in old['records']} if old else {}
    rows, new_dates, boosters = [], [], {}
    backfill_through = old['meta']['supplemental_backfill_through'] if old else report['meta']['signal_date']
    for rr in report['records']:
        day = rr['date']
        if day in known and all(v['status'] != 'blocked' for v in known[day]['models'].values()):
            rows.append(copy.deepcopy(known[day]))
            continue
        if day in known and day < old['meta']['asof']:
            rows.append(copy.deepcopy(known[day]))
            continue
        row = {'date': day, 'level': rr['level'], 'models': {}}
        for key in KEYS:
            arm = 'f' + ''.join(c for c in key if c.isdigit())
            existing = rr['models'].get(arm) if key in ('ridge20', 'ridge30') else \
                logrows[day]['models'][arm] if key.startswith('logistic') else None
            if existing is not None:
                row['models'][key] = {k: copy.deepcopy(v) for k, v in existing.items() if k != 'inputs'}
                row['models'][key]['origin'] = logrows[day]['origin'] if key.startswith('logistic') else rr['origin']
                continue
            if day in history_axis:
                row['models'][key] = {'prediction': base['daily']['prediction'][key][history_axis[day]],
                                      'status': 'backfill', 'origin': 'research-backfill'}
                continue
            model = frozen['models'][key]
            values = np.array([features[f['id']]['values'][axis[day]] for f in model['preprocessing']], float)
            coverage, problems = quality(day, rr['level'], values[:8], values[8:], source_info, bond_dates, repo_dates)
            row['models'][key] = {
                'prediction': None if problems else predict(model, values, boosters, key),
                'status': 'blocked' if problems else 'deployment-backfill' if day <= backfill_through else 'issued',
                'origin': 'supplemental-backfill' if day <= backfill_through else 'live-issued',
                'coverage': coverage, 'problems': problems, 'issued_at': now.isoformat(timespec='seconds'),
                'input_sha256': {n: sha(n) for n in ('bond_trading_data.json', 'repo_trading_data.json',
                                                  'yield_curve_data.json', 'omo_policy_7d.json')}}
        rows.append(row)
        if day not in history_axis:
            new_dates.append(day)
    for key in KEYS:
        g = signed_percentile_gate([r['models'][key]['prediction'] for r in rows])
        for i, row in enumerate(rows):
            rec = row['models'][key]
            if 'signal' in rec:
                continue
            rec.update({'percentile': safe_number(g['percentile'][i]), 'history_n': int(g['history_n'][i]),
                        'signal': int(g['signal'][i]), 'lower_reference': safe_number(g['lower_quantile'][i]),
                        'upper_reference': safe_number(g['quantile'][i])})
    strategies = {rule: {key: simulate_records(rows, key, rule) for key in KEYS} for rule in ('dynamic', 'fixed')}
    for n in (20, 30):
        if strategies['dynamic'][f'ridge{n}'] != report['strategies'][f'f{n}']:
            raise ValueError('Comparison changed daily Ridge accounting')
    for n in (20, 30, 40):
        if strategies['dynamic'][f'logistic{n}'] != report['logistic']['strategies'][f'f{n}']:
            raise ValueError('Comparison changed daily Logistic accounting')
    level = np.array([r['level'] for r in rows], float)
    target = np.full(len(rows), np.nan)
    target[:-6] = level[6:] - level[:-6]
    predictions = {key: [r['models'][key]['prediction'] for r in rows] for key in KEYS}
    forecast = {key: {'train': (logbase if key.startswith('logistic') else base)['forecast'][key].get('train'),
                      'test': metrics(predictions[key], target, key.startswith('logistic'))} for key in KEYS}
    return {'meta': {**base['meta'], 'generated': now.isoformat(timespec='seconds'), 'asof': rows[-1]['date'],
                     'research_asof': base['meta']['asof'], 'test_n': len(rows), 'families': list(FAMILIES),
                     'frozen_hashes': hashes, 'supplemental_backfill_through': backfill_through,
                     'healthy': report['meta']['healthy'] and report['logistic']['meta']['healthy'] and
                        all(v['status'] != 'blocked' for v in rows[-1]['models'].values()),
                     'archive': '原研究快照及既有岭回归/Logistic预测原样保留。新增对照首次补算标记supplemental-backfill，之后逐日归档，不训练、不重筛。'},
            'records': rows, 'daily': {'dates': [r['date'] for r in rows], 'level': raw_numbers(level),
                                     'target': raw_numbers(target), 'prediction': predictions},
            'forecast': forecast, 'strategies': strategies}, new_dates
