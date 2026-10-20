"""Four predeclared percentile bands, 2025 validation, frozen 2026 forecasts.

No final model is refit and no 2026 labels are used to choose a band. Fixed-six
exits isolate entry sensitivity; matched dynamic exits are a separate experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jsonio import write_json_skip_unchanged
from prepare_bayes_study import (PROTECTED as BAYES_PROTECTED, digest, dynamic_forecast,
                                  initial_state, load_data)
from prepare_curve_timing_capacity import fit_model
from prepare_curve_timing_percentile import signed_percentile_gate, serialize_sim
from prepare_curve_timing_walkforward import completed_indices, simulate
from prepare_direction_screen import raw_numbers
from prepare_strategy_model import make_offset_labels

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'output' / 'bayes' / 'threshold_validation_2025'
OUTPUT = 'curve_timing_thresholds.json'
BANDS = ((45, 55), (40, 60), (35, 65), (30, 70))
KEYS = ('ridge20', 'dlm20', 'bayes20', 'bart20')
MIN_TRADES = 10
PROTECTED = BAYES_PROTECTED + ('curve_timing_bayes.json', 'bayes_study_models_202512.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare_inputs():
    """Calibration model is trained through June 2025, never through 2026."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dates, level, x, y, features, year_end, _ = load_data()
    cut = max(i for i, d in enumerate(dates) if d <= '2025-06-30')
    train = completed_indices(level, y, cut, 6)
    model = fit_model(x[train], y[train], features, alpha=15)
    z = model[0].transform(x[:year_end + 1])
    mean, covariance, noise = initial_state(z[train], y[train])
    start = cut + 1
    np.savez_compressed(CACHE / 'bart_inputs.npz', x_train=z[train], y_train=y[train],
                        x_score=z[start - 250:], train_indices=train, n_seed=np.array(250),
                        train_signal_dates=np.array([dates[i] for i in train]),
                        train_label_exit_dates=np.array([dates[i + 6] for i in train]),
                        score_dates=np.array(dates[start - 250:year_end + 1]))
    np.savez_compressed(CACHE / 'linear_inputs.npz', z=z,
                        y=make_offset_labels(level[:year_end + 1], 6),
                        level=level[:year_end + 1], dates=np.array(dates[:year_end + 1]),
                        mean=mean, covariance=covariance)
    meta = {'train_through': '2025-06-30', 'train_n': len(train), 'validation_start': dates[start],
            'validation_end': dates[year_end], 'start_index': start, 'last_train_index': int(train[-1]),
            'last_train_signal': dates[train[-1]], 'last_train_label_exit': dates[train[-1] + 6],
            'noise_variance': noise, 'discount': read(ROOT / 'bayes_study_models_202512.json')['discount'],
            'inputs_sha256': digest(CACHE / 'bart_inputs.npz'),
            'protected_hashes': {n: digest(ROOT / n) for n in PROTECTED}}
    write_json_skip_unchanged(str(CACHE / 'study_inputs.json'), meta)
    print(json.dumps(meta, ensure_ascii=False))


def mapped_gate(source_gate, lower, upper):
    """Reuse archived causal ranks; thresholds cannot change any past forecast.

    Price-score quantiles are not used for signed-rank execution. The simulator
    receives a finite placeholder; the exported ledger explicitly clears it.
    """
    percentile = np.array(source_gate['percentile'], float)
    signal = np.where(percentile < lower, -1, np.where(percentile > upper, 1, 0))
    return {'direction_basis': 'percentile', 'percentile': percentile,
            'signal': signal.astype(int), 'history_n': np.array(source_gate['history_n'], int),
            'quantile': np.zeros(len(percentile))}


def trade_stats(trades):
    closed = [t for t in trades if t['closed']]
    values = np.array([t['pnl'] for t in closed], float)
    wins, losses = values[values > 0], values[values < 0]
    average_win = float(wins.mean()) if len(wins) else None
    average_loss = float(-losses.mean()) if len(losses) else None
    return {'closed_n': len(closed), 'open_n': len(trades) - len(closed),
            'wins': len(wins), 'losses': len(losses), 'breakeven': int((values == 0).sum()),
            'win_rate': float(np.mean(values > 0)) if len(values) else None,
            'mean_pnl': float(values.mean()) if len(values) else None,
            'closed_pnl': float(values.sum()), 'floating_pnl': float(sum(t['pnl'] for t in trades if not t['closed'])),
            'average_win': average_win, 'average_loss': average_loss,
            'payoff_ratio': average_win / average_loss if average_win is not None and average_loss else None,
            'profit_factor': float(wins.sum() / -losses.sum()) if len(losses) else None}


def signal_stats(signal, target, direction=None):
    mask = np.isfinite(target) & (signal != 0)
    if direction is not None:
        mask &= signal == direction
    outcome = signal[mask] * target[mask]
    return {'n': len(outcome), 'direction_hit': float(np.mean(outcome > 0)) if len(outcome) else None,
            'mean_signed_six_day_change': float(outcome.mean()) if len(outcome) else None}


def evaluate(dates, level, prediction, target, source_gate, lower, upper, rule):
    gate = mapped_gate(source_gate, lower, upper)
    sim = simulate(dates, level, prediction, gate['quantile'], rule=rule, gate=gate, horizon=6)
    for t in sim['trades']:
        t['entry_threshold'] = None
        t['entry_percentile_lower'], t['entry_percentile_upper'] = lower, upper
    by_side = {label: trade_stats([t for t in sim['trades'] if t['direction'] == direction])
               for label, direction in (('long', 1), ('short', -1))}
    stats = trade_stats(sim['trades'])
    stats.update({k: sim['metrics'][k] for k in ('total', 'max_drawdown', 'average_holding', 'long_sessions', 'short_sessions')})
    months = {month: float(sum(p for d, p in zip(dates, sim['daily_pnl']) if d.startswith(month)))
              for month in sorted({d[:7] for d in dates})}
    return {'metrics': stats, 'by_side': by_side,
            'signals': {'all': signal_stats(gate['signal'], target),
                        'long': signal_stats(gate['signal'], target, 1),
                        'short': signal_stats(gate['signal'], target, -1)},
            'signal': gate['signal'].tolist(), 'months': months, 'sim': serialize_sim(sim)}


def choose_band(validation_rows):
    """Predeclared selection: fixed exit, cold ranks, positive expectancy, n>=10.

    Never force a winner or loosen the trade-count floor after seeing results.
    """
    eligible = [r for r in validation_rows if r['metrics']['closed_n'] >= MIN_TRADES and
                r['metrics']['mean_pnl'] is not None and r['metrics']['mean_pnl'] > 0]
    if not eligible:
        return {'band': None, 'status': 'no-qualified-band', 'fallback_reference': '40_60',
                'reason': '没有同时满足至少10笔已平交易且每笔平均盈利为正的门槛，不宣布已选出有效组合。'}
    eligible.sort(key=lambda r: (-r['metrics']['mean_pnl'], -r['metrics']['closed_pnl'],
                                 r['metrics']['max_drawdown'], r['band']))
    return {'band': eligible[0]['band'], 'status': 'validation-selected',
            'reason': '仅使用2025年冷启动、固定六日结果；每笔平均盈利优先，其次已实现总盈亏，再次回撤。未用2026年收益或胜率选择。'}


def validation_data():
    meta = read(CACHE / 'study_inputs.json')
    diag = read(CACHE / 'bart_summary.json')
    if meta['inputs_sha256'] != diag['inputs_sha256']:
        raise ValueError('Validation BART inputs changed')
    linear, bart = np.load(CACHE / 'linear_inputs.npz'), np.load(CACHE / 'bart_predictions.npz')
    dates = linear['dates'].tolist()
    start = meta['start_index']
    z, y, mean, covariance = linear['z'], linear['y'], linear['mean'], linear['covariance']
    level = linear['level'][start:]
    target = make_offset_labels(level, 6)  # Last six validation labels cannot reach 2026.
    base_args = (z, y, start, meta['last_train_index'], mean, covariance, meta['noise_variance'])
    dyn = dynamic_forecast(*base_args, meta['discount'], dates)
    stationary = dynamic_forecast(*base_args, 1., dates)
    ridge = np.column_stack([np.ones(len(level)), z[start:]]) @ mean
    linear_seed = np.column_stack([np.ones(250), z[start - 250:start]]) @ mean
    predictions = {'ridge20': ridge, 'dlm20': dyn['mean'], 'bayes20': stationary['mean'], 'bart20': bart['mean'][250:]}
    seeds = {k: bart['mean'][:250] if k == 'bart20' else linear_seed for k in KEYS}
    gates = {}
    for key, pred in predictions.items():
        gates[key] = {}
        for mode in ('cold', 'seeded'):
            seed = seeds[key] if mode == 'seeded' else np.array([])
            g = signed_percentile_gate(np.r_[seed, pred])
            gates[key][mode] = {k: raw_numbers(v[len(seed):]) if np.issubdtype(v.dtype, np.floating)
                               else v[len(seed):].tolist()
                               for k, v in g.items() if isinstance(v, np.ndarray)}
    return meta, dates[start:], level, target, predictions, gates, diag


def build_report():
    protected = {n: digest(ROOT / n) for n in PROTECTED}
    source = read(ROOT / 'curve_timing_bayes.json')
    val_meta, vd, vl, vy, vp, vg, bart_diag = validation_data()
    if val_meta['protected_hashes'] != protected:
        raise ValueError('A protected study or model changed during threshold experiment')
    td, tl = source['dates'], np.array(source['level'], float)
    ty = np.array(source['target'], float)
    results, selected = {}, {}
    for key in KEYS:
        variant = source['variants'][key]
        results[key] = {'label': variant['label'], 'periods': {},
                        'percentiles': {'validation2025': {m: vg[key][m]['percentile'] for m in ('cold', 'seeded')},
                                        'test2026': {m: variant['modes'][m]['gate']['percentile'] for m in ('cold', 'seeded')}}}
        for period, dates, level, target, prediction, gates in (
            ('validation2025', vd, vl, vy, vp[key], vg[key]),
            ('test2026', td, tl, ty, np.array(variant['prediction'], float),
             {m: variant['modes'][m]['gate'] for m in ('cold', 'seeded')})):
            records = {}
            for mode in ('cold', 'seeded'):
                records[mode] = {}
                for rule in ('fixed', 'dynamic'):
                    rows = []
                    for lower, upper in BANDS:
                        row = evaluate(dates, level, prediction, target, gates[mode], lower, upper, rule)
                        row.update({'band': f'{lower}_{upper}', 'lower': lower, 'upper': upper})
                        if period == 'test2026' and rule == 'dynamic' and (lower, upper) == (40, 60):
                            reference = variant['modes'][mode]['sim']
                            for field in ('position', 'daily_pnl', 'equity', 'drawdown'):
                                np.testing.assert_allclose(row['sim'][field], reference[field], atol=1e-12)
                        rows.append(row)
                    records[mode][rule] = rows
            results[key]['periods'][period] = records
        selected[key] = choose_band(results[key]['periods']['validation2025']['cold']['fixed'])
    report = {'meta': {'asof': source['meta']['asof'], 'train_through': '2025-12-31',
                       'test_start': td[0], 'validation_train_through': val_meta['train_through'],
                       'validation_start': vd[0], 'validation_end': vd[-1], 'validation_train_n': val_meta['train_n'],
                       'window': 250, 'min_history': 60, 'horizon': 6, 'bands': [list(b) for b in BANDS],
                       'default_model': 'bart20', 'default_period': 'test2026', 'default_mode': 'cold',
                       'default_exit': 'fixed', 'minimum_validation_trades': MIN_TRADES,
                       'selection_basis': '2025年冷启动固定六日，至少10笔已平交易且平均每笔盈利为正；均值优先、已实现总盈亏其次、回撤再次。',
                       'dynamic_note': '动态对照随门槛同步调整中性离场区，改变了开仓与离场；不能当作纯开仓优化。',
                       'caveat': '固定名单仍含全历史筛选前视；2026已反复观察，属于探索对照。2026最优行只展示，不用于选门槛。信号六日标签重叠，命中率不是独立交易胜率；未扣成本。',
                       'protected_hashes': protected},
              'periods': {'validation2025': {'dates': vd, 'level': raw_numbers(vl), 'target': raw_numbers(vy)},
                          'test2026': {'dates': td, 'level': raw_numbers(tl), 'target': raw_numbers(ty)}},
              'models': results, 'selection': selected, 'validation_bart_diagnostics': bart_diag}
    if protected != {n: digest(ROOT / n) for n in PROTECTED}:
        raise ValueError('Threshold study modified a protected model or forecast archive')
    json.dumps(report, allow_nan=False)  # Browser JSON.parse must accept every exported value.
    write_json_skip_unchanged(str(ROOT / OUTPUT), report)
    print(json.dumps({'selection': selected, 'bart_test_cold_fixed': [r['metrics'] | {'band': r['band']}
                      for r in results['bart20']['periods']['test2026']['cold']['fixed']]}, ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('inputs', 'report'))
    args = parser.parse_args()
    prepare_inputs() if args.stage == 'inputs' else build_report()
