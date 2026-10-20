"""Nine predeclared entry variants, fixed-six exits, unchanged model forecasts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jsonio import write_json_skip_unchanged
from prepare_curve_timing_percentile import signed_percentile_gate, serialize_sim
from prepare_curve_timing_walkforward import simulate
from prepare_direction_screen import raw_numbers
from prepare_forecast_signal_diagnostics import PROTECTED as DIAGNOSTIC_PROTECTED, sha
from prepare_threshold_study import trade_stats, signal_stats

ROOT = Path(__file__).resolve().parent
OUTPUT = 'entry_rules_study.json'
KEYS = ('ridge_original', 'ridge_temporal', 'dlm_temporal')
RULES = ('sign', 'asymmetric', 'rank')
LABELS = {'sign': '纯方向', 'asymmetric': '非对称过滤', 'rank': '滚动40／60排名'}
PROTECTED = DIAGNOSTIC_PROTECTED + ('forecast_signal_calibration_2025.json', 'forecast_signal_diagnostics.json',
                                    'curve_timing_thresholds.json')


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def entry_gate(prediction, rule, negative_cut):
    p = np.asarray(prediction, float)
    if not np.isfinite(p).all():
        raise ValueError('Missing forecast; stop rather than silently change comparison dates')
    if rule == 'rank':
        g = signed_percentile_gate(p, window=250, min_history=60, lower=40., upper=60.)
        g['eligible'] = np.isfinite(g['percentile'])
        return g
    if not np.isfinite(negative_cut) or negative_cut <= 0:
        raise ValueError('A positive frozen negative-magnitude boundary is required')
    if rule == 'sign':
        signal = np.sign(p).astype(int)
    elif rule == 'asymmetric':
        signal = np.where(p > 0, 1, np.where(p < -negative_cut, -1, 0))
    else:
        raise ValueError(rule)
    # The legacy simulator checks finite percentile as eligibility. These are
    # internal placeholders only; exported sign/asymmetric rows never claim a rank.
    return {'direction_basis': 'forecast_sign', 'signal': signal, 'eligible': np.ones(len(p), bool),
            'percentile': np.zeros(len(p)), 'quantile': np.zeros(len(p)), 'history_n': np.zeros(len(p), int)}


def slice_gate(gate, start):
    return {k: v[start:] if isinstance(v, np.ndarray) else v for k, v in gate.items()}


def evaluate(dates, level, prediction, target, gate, rule, negative_cut):
    sim = simulate(dates, level, prediction, gate['quantile'], rule='fixed', horizon=6, gate=gate)
    # Clear all unitless placeholder thresholds from the public ledger. Explicit
    # entry_cutoff carries the actual boundary in its own declared units.
    for trade in sim['trades']:
        trade['entry_rule'] = rule
        trade['entry_threshold'] = None
        if rule == 'rank':
            trade['entry_cutoff'] = 60. if trade['direction'] == 1 else 40.
            trade['entry_cutoff_unit'] = '预测百分位%'
        else:
            trade['entry_percentile'] = None
            trade['entry_history_n'] = None
            trade['entry_cutoff'] = -negative_cut if rule == 'asymmetric' and trade['direction'] == -1 else 0.
            trade['entry_cutoff_unit'] = '预测价差点'
    decisions = {d['date']: d for d in sim['decisions']}
    sim['decisions'] = [decisions.get(day, {'date': day, 'next_position': 0, 'exit_reason': None, 'opened': False})
                        for day in dates]
    sim['metrics']['exposure'] = float(np.mean(sim['position'][1:] != 0)) if len(dates) > 1 else None
    stats = trade_stats(sim['trades'])
    monthly = {}
    for month in sorted({day[:7] for day in dates}):
        mask = np.array([day.startswith(month) for day in dates])
        monthly[month] = {name: float(np.sum(sim['daily_pnl'][mask & side])) for name, side in
                          [('total', np.ones(len(dates), bool)), ('long', sim['position'] == 1),
                           ('short', sim['position'] == -1)]}
    eligible = gate['eligible']
    return {'sim': serialize_sim(sim), 'trade_stats': stats,
            'sides': {str(d): trade_stats([t for t in sim['trades'] if t['direction'] == d]) for d in (-1, 1)},
            'months': monthly,
            'signal': gate['signal'].tolist(), 'eligible': eligible.tolist(),
            'percentile': raw_numbers(gate['percentile']) if rule == 'rank' else [None] * len(dates),
            'history_n': gate['history_n'].tolist() if rule == 'rank' else [None] * len(dates),
            'signal_stats': signal_stats(gate['signal'], target),
            'signal_sides': {str(d): signal_stats(gate['signal'], target, d) for d in (-1, 1)},
            'counts': {'eligible': int(eligible.sum()), 'waiting': int(np.sum(~eligible)),
                       'long': int(np.sum(gate['signal'] == 1)), 'short': int(np.sum(gate['signal'] == -1)),
                       'neutral': int(np.sum(eligible & (gate['signal'] == 0))),
                       'opposes_prediction': int(np.sum(gate['signal'] * prediction < 0))},
            'latest': {'prediction': float(prediction[-1]),
                        'percentile': float(gate['percentile'][-1]) if rule == 'rank' and eligible[-1] else None,
                        'signal': int(gate['signal'][-1]), 'position': int(sim['position'][-1]),
                        'next_position': int(sim['decisions'][-1]['next_position'])}}


def run_paths(dates, level, predictions, target, cuts):
    gates = {k: {rule: entry_gate(predictions[k], rule, cuts[k]) for rule in RULES} for k in KEYS}
    common = np.logical_and.reduce([gates[k]['rank']['eligible'] for k in KEYS])
    if not common.any():
        raise ValueError('No common date has sufficient prior ranking history')
    common_start = int(np.flatnonzero(common)[0])
    periods = {}
    for period, start in (('full', 0), ('common', common_start)):
        td, tl, ty = dates[start:], level[start:], target[start:]
        models = {}
        for key in KEYS:
            models[key] = {rule: evaluate(td, tl, predictions[key][start:], ty, slice_gate(gates[key][rule], start),
                                           rule, cuts[key]) for rule in RULES}
        periods[period] = {'start_index': start, 'dates': td, 'level': raw_numbers(tl), 'target': raw_numbers(ty),
                           'models': models}
    return periods


def build():
    protected = {n: sha(n) for n in PROTECTED}
    source, asset, calibration = read('forecast_model_comparison.json'), read('curve_timing_bayes.json'), read('forecast_signal_calibration_2025.json')
    if set(read('forecast_model_shortlist.json')['selected']) != set(KEYS):
        raise ValueError('Retained model set changed; confirm a new study version')
    dates = source['dates']
    if dates != asset['dates'] or source['target'] != asset['target']:
        raise ValueError('Forecast and spread snapshots must share the same dates and labels')
    level, target = np.array(asset['level'], float), np.array(source['target'], float)
    if not np.isfinite(level).all():
        raise ValueError('Cannot mark missing spread prices as zero PnL')
    predictions = {key: np.array(source['models'][key]['prediction'], float) for key in KEYS}
    cuts = {key: float(calibration['boundaries'][key]['-1']['cuts'][1]) for key in KEYS}
    periods = run_paths(dates, level, predictions, target, cuts)
    # Existing rank/fixed/cold 40/60 Ridge path is an independent accounting check.
    previous = read('curve_timing_thresholds.json')
    ref = next(row['sim'] for row in previous['models']['ridge20']['periods']['test2026']['cold']['fixed']
               if row['band'] == '40_60')
    now = periods['full']['models']['ridge_original']['rank']['sim']
    for field in ('daily_pnl', 'position', 'equity', 'drawdown'):
        np.testing.assert_allclose(now[field], ref[field], atol=1e-10, rtol=0)
    result = {'meta': {'asof': dates[-1], 'train_through': '2025-12-31', 'test_start': dates[0],
                       'forecast_n': len(dates), 'completed_n': int(np.isfinite(target).sum()),
                       'common_start': periods['common']['dates'][0], 'horizon': 6,
                       'default_period': 'common', 'default_model': 'ridge_temporal', 'default_rule': 'asymmetric',
                       'ranking': {'window': 250, 'min_history': 60, 'lower': 40, 'upper': 60},
                       'rules': LABELS, 'negative_cuts': cuts,
                       'full_note': '完整2026区间：所有账户首日空仓；纯方向／过滤立即可交易，排名先等待60条预测。总收益差异含等待效应。',
                       'common_note': '共同起点：三模型均有60条历史后，全部9组重新从空仓开始；排名沿用此前预测历史，不重置历史、不携带旧持仓。',
                       'caveat': '9种组合均为观察2026年后提出的探索回放；分档阈值虽来自2025年，规则选择仍参考2026。因子名单含前视；未扣成本，不自动替换生产。',
                       'protected_hashes': protected},
              'models': {key: {'label': source['models'][key]['label'], 'prediction': raw_numbers(predictions[key])} for key in KEYS},
              'periods': periods}
    if protected != {n: sha(n) for n in PROTECTED}:
        raise ValueError('Old forecasts, factors or production files changed')
    json.dumps(result, allow_nan=False)
    write_json_skip_unchanged(str(ROOT / OUTPUT), result)
    print(json.dumps({period: {key: {rule: row['sim']['metrics'] for rule, row in model.items()}
                               for key, model in p['models'].items()} for period, p in periods.items()}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    build()
