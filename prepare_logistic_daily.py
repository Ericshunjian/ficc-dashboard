"""Frozen L2 Logistic inference and immutable supplemental daily archive.

Invoked by prepare_ridge_daily.build. No fitting, selection, or source Excel.
Probability 60/40 replaces only the score used for the direction gate; execution,
position size, dynamic exits, warmup and six-session expiry use the same engine.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from prepare_curve_timing_percentile import serialize_sim, signed_percentile_gate
from prepare_curve_timing_walkforward import simulate

ROOT = Path(__file__).resolve().parent
MODELS = 'curve_timing_logistic_models.json'
SOURCE = 'curve_timing_logistic.json'
ARMS = ('f20', 'f30', 'f40')
CONFIG = {'horizon': 6, 'window': 250, 'min_history': 60, 'lower': 40, 'upper': 60,
          'direction_basis': 'absolute_probability', 'min_coverage': .9}


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def probability(score):
    return 1 / (1 + math.exp(-score)) if score >= 0 else math.exp(score) / (1 + math.exp(score))


def gate(predictions):
    pred = np.asarray(predictions, float)
    rank = signed_percentile_gate(pred, window=250, min_history=60, lower=40, upper=60)
    valid = np.isfinite(rank['percentile']) & np.isfinite(pred)
    return {**rank, 'history_percentile': rank['percentile'],
            'percentile': np.where(valid, 100 * pred, np.nan),
            'signal': np.where(valid & (pred < .4), -1, np.where(valid & (pred > .6), 1, 0)),
            'quantile': np.where(valid, .6, np.nan), 'lower_quantile': np.where(valid, .4, np.nan)}


def validate_models(frozen):
    if frozen['config'] != CONFIG:
        raise ValueError('Frozen Logistic configuration changed; use a new strategy version')
    sets = read('curve_timing_capacity_factor_sets.json')['sets']
    for arm in ARMS:
        model = frozen['models'][arm]
        fs = model['features']
        if len(fs) != int(arm[1:]) + 8 or [f['id'] for f in fs[8:]] != sets[arm]:
            raise ValueError('Frozen Logistic factors differ from the accepted lists')
        if any(f['clip_enabled'] for f in fs[:8]):
            raise ValueError('Logistic baselines must never be clipped')
        if not np.isfinite(model['intercept']):
            raise ValueError('Invalid Logistic intercept')
        for f in fs:
            if not all(np.isfinite(f[k]) for k in ('coef', 'median', 'mean', 'std')) or f['std'] <= 0:
                raise ValueError('Invalid frozen Logistic coefficient or preprocessing')


def enrich(rows):
    for arm in ARMS:
        g = gate([r['models'][arm]['prediction'] for r in rows])
        for i, row in enumerate(rows):
            rec = row['models'][arm]
            if rec.get('status') in ('issued', 'deployment-backfill') and 'signal' in rec:
                continue
            rec.update({'percentile': float(g['history_percentile'][i]) if np.isfinite(g['history_percentile'][i]) else None,
                        'history_n': int(g['history_n'][i]), 'signal': int(g['signal'][i]),
                        'lower_reference': .4, 'upper_reference': .6})


def archive_simulation(rows, arm, rule='dynamic'):
    pred = np.array([r['models'][arm]['prediction'] for r in rows], float)
    score = [100 * r['models'][arm]['prediction'] if r['models'][arm]['percentile'] is not None and
             r['models'][arm]['prediction'] is not None else np.nan for r in rows]
    g = {'direction_basis': 'percentile', 'percentile': np.array(score, float),
         'signal': np.array([r['models'][arm]['signal'] for r in rows], int),
         'history_n': np.array([r['models'][arm]['history_n'] for r in rows], int),
         'quantile': np.repeat(.6, len(rows))}
    return simulate([r['date'] for r in rows], [r['level'] for r in rows], pred, g['quantile'],
                    rule=rule, horizon=6, gate=g)


def build_archive(ridge_report, old, features, dates, source_info, bond_dates, repo_dates, now):
    # Import common production quality, preprocessing and summaries without fitting.
    from prepare_ridge_daily import infer, position_summary, quality
    frozen = read(MODELS)
    validate_models(frozen)
    source = read(SOURCE)
    if old and old['meta']['model_sha256'] != sha(MODELS):
        raise ValueError('Frozen Logistic model changed; cannot append to its archive')
    if old:
        rows = old['records']
    else:
        rows = [{'date': day, 'level': value, 'origin': 'research-backfill',
                 'models': {arm: {'prediction': source['daily']['prediction']['logistic' + arm[1:]][i],
                                  'status': 'backfill'} for arm in ARMS}}
                for i, (day, value) in enumerate(zip(source['daily']['dates'], source['daily']['level'], strict=True))]
    known, axis = {r['date']: r for r in rows}, {day: i for i, day in enumerate(dates)}
    new_dates = []
    for rr in ridge_report['records']:
        day = rr['date']
        if day < rows[0]['date'] or day not in axis:
            continue
        if day in known and all(known[day]['models'][k]['status'] in ('issued', 'backfill', 'deployment-backfill') for k in ARMS):
            continue
        if day in known and day < rows[-1]['date']:
            continue
        i = axis[day]
        origin = 'deployment-backfill' if old is None else 'live-issued'
        rec = known.get(day) or {'date': day, 'level': rr['level'], 'origin': origin, 'models': {}}
        for arm in ARMS:
            if rec['models'].get(arm, {}).get('status') in ('issued', 'backfill', 'deployment-backfill'):
                continue
            model = frozen['models'][arm]
            missing = [f['id'] for f in model['features'] if f['id'] not in features]
            if missing:
                raise ValueError('Fixed Logistic features missing from library: ' + str(missing))
            values = np.array([features[f['id']]['values'][i] for f in model['features']], float)
            coverage, problems = quality(day, rr['level'], values[:8], values[8:], source_info, bond_dates, repo_dates)
            score, inputs = (None, []) if problems else infer(model, values)
            proposed = {'prediction': None if problems else probability(score), 'logit': score,
                        'status': 'blocked' if problems else ('deployment-backfill' if old is None else 'issued'),
                        'coverage': coverage, 'problems': problems, 'inputs': inputs,
                        'issued_at': now.isoformat(timespec='seconds'),
                        'input_sha256': {n: sha(n) for n in ('bond_trading_data.json', 'repo_trading_data.json',
                                                          'yield_curve_data.json', 'omo_policy_7d.json')}}
            previous = rec['models'].get(arm)
            if previous and proposed['status'] == 'blocked' and all(previous.get(n) == proposed[n] for n in
                    ('prediction', 'status', 'coverage', 'problems', 'input_sha256')):
                continue
            rec['models'][arm] = proposed
        if day not in known:
            rows.append(rec)
            known[day] = rec
        if any(rec['models'][k]['status'] in ('issued', 'deployment-backfill') for k in ARMS):
            new_dates.append(day)
    rows.sort(key=lambda r: r['date'])
    enrich(rows)
    simulations = {k: archive_simulation(rows, k) for k in ARMS}
    summaries = {k: position_summary(rows, k, simulations[k]) for k in ARMS}
    latest = rows[-1]
    return {'meta': {'generated': now.isoformat(timespec='seconds'), 'model_version': frozen['meta']['version'],
                     'model_sha256': sha(MODELS), 'signal_date': latest['date'], 'main': 'f20',
                     'healthy': ridge_report['meta']['unpriced_latest'] is None and latest['models']['f20']['status'] != 'blocked',
                     'direction_basis': 'absolute_probability', 'deployment_backfill_through':
                         old['meta'].get('deployment_backfill_through') if old else latest['date'],
                     'archive': '研究回填及部署回填不是此前已发布的信号；部署后的每日预测不可覆盖。'},
            'config': CONFIG, 'records': rows, 'latest': latest, 'positions': summaries,
            'strategies': {k: serialize_sim(s) for k, s in simulations.items()}}, new_dates
