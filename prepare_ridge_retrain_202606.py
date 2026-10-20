"""Independent expanded-training Ridge study. Production models are never modified."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from jsonio import write_json_skip_unchanged
from prepare_curve_timing_capacity import fit_model, model_record
from prepare_curve_timing_percentile import serialize_sim, signed_percentile_gate
from prepare_curve_timing_walkforward import completed_indices, forecast_metrics, simulate
from prepare_direction_screen import baseline_features, load_candidates, raw_numbers
from prepare_ridge_daily import infer
from prepare_strategy_model import make_offset_labels

ROOT = Path(__file__).resolve().parent
MODELS = 'ridge_retrain_202606_models.json'
OUTPUT = 'ridge_retrain_202606.json'
CUTOFF, TEST_START = '2026-06-30', '2026-07-01'
ARMS = ('f20', 'f30', 'f40')
PROTECTED = ('ridge_daily_models.json', 'curve_timing_nonlinear_models.json',
             'curve_timing_logistic_models.json', 'ridge_daily_signals.json',
             'curve_timing_capacity_factor_sets.json', 'direction_screen.json')


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def gate_and_simulate(dates, level, predictions, seed=None):
    prior = np.array([] if seed is None else seed, float)
    full = signed_percentile_gate(np.r_[prior, predictions])
    g = {k: v[len(prior):] if isinstance(v, np.ndarray) else v for k, v in full.items()}
    sim = simulate(dates, level, predictions, g['quantile'], rule='dynamic', horizon=6, gate=g)
    return {k: raw_numbers(v) if k in ('percentile', 'quantile', 'lower_quantile') else v.tolist()
            for k, v in g.items() if k != 'direction_basis'}, serialize_sim(sim)


def build():
    protected = {n: sha(n) for n in PROTECTED}
    library, dates, curve, candidates = load_candidates()
    live = read('ridge_daily_signals.json')
    last = dates.index(live['meta']['signal_date'])
    dates = dates[:last + 1]
    level = (3 * curve['T主力'] - curve['TL主力'])[:last + 1]
    baselines, _ = baseline_features(level, {k: v[:last + 1] for k, v in curve.items()}, dates)
    candidate_map = {f['id']: f for f in candidates}
    sets = read('curve_timing_capacity_factor_sets.json')
    fs = {arm: baselines + [candidate_map[k] for k in sets['sets'][arm]] for arm in ARMS}
    matrix = {arm: np.column_stack([f['values'][:last + 1] for f in fs[arm]]) for arm in ARMS}
    cut = max(i for i, d in enumerate(dates) if d <= CUTOFF)
    target = make_offset_labels(level, 6)
    train = completed_indices(level, target, cut, 6)
    if len(train) < 300:
        raise ValueError('Insufficient completed training labels')
    frozen = read(MODELS) if (ROOT / MODELS).exists() else None
    if frozen is None:
        models = {arm: model_record(fit_model(matrix[arm][train], target[train], fs[arm], alpha=15), fs[arm])
                  for arm in ARMS}
        frozen = {'meta': {'train_through': CUTOFF, 'train_start': dates[train[0]],
                           'last_train_signal': dates[train[-1]], 'last_label_exit': dates[train[-1] + 6],
                           'train_n': len(train), 'alpha': 15, 'factor_set_sha256': sha('curve_timing_capacity_factor_sets.json'),
                           'selection_asof': sets['meta']['selection_asof'], 'refit': False}, 'models': models}
        write_json_skip_unchanged(str(ROOT / MODELS), frozen)
    if frozen['meta']['train_through'] != CUTOFF or frozen['meta']['factor_set_sha256'] != sha('curve_timing_capacity_factor_sets.json'):
        raise ValueError('Frozen study factor list or training cutoff changed')
    original = read('curve_timing_nonlinear_models.json')['models']
    old = {arm: original['ridge' + arm[1:]]['linear_model'] for arm in ARMS}
    start = next(i for i, d in enumerate(dates) if d >= TEST_START)
    td, tl, ty = dates[start:], np.array(level[start:]), target[start:]
    # Both accounts start flat on the first test signal date. No pre-July positions are carried in.
    shared = np.isfinite(tl) & np.isfinite(matrix['f20'][start:, :8]).all(axis=1)
    for arm in ARMS:
        shared &= np.mean(np.isfinite(matrix[arm][start:, 8:]), axis=1) >= .9
    variants, coefficients, latest = {}, {}, {}
    for arm in ARMS:
        coefficients[arm] = []
        for a, b in zip(old[arm]['features'], frozen['models'][arm]['features'], strict=True):
            if a['id'] != b['id']:
                raise ValueError('Old and expanded models use different factors')
            if b['id'].startswith('baseline|') and b['clip_enabled']:
                raise ValueError('Spread and baselines must not be clipped')
            coefficients[arm].append({'id': b['id'], 'label': b['label'], 'unit': b['unit'],
                                      'old_coef': a['coef'], 'new_coef': b['coef'],
                                      'old_raw_coef': a['coef'] / a['std'], 'new_raw_coef': b['coef'] / b['std']})
        for version, model in [('old', old[arm]), ('expanded', frozen['models'][arm])]:
            predictions = np.array([infer(model, x)[0] if ok else np.nan for x, ok in zip(matrix[arm][start:], shared)], float)
            # This is an explicitly labelled train-time rescore, NOT a past issued forecast.
            history = matrix[arm][max(0, start - 250):start]
            seed = np.array([infer(model, x)[0] if np.isfinite(x[:8]).all() and np.mean(np.isfinite(x[8:])) >= .9
                             else np.nan for x in history], float)
            key = version + arm[1:]
            modes = {}
            for mode in ('cold', 'seeded'):
                g, sim = gate_and_simulate(td, tl, predictions, seed if mode == 'seeded' else None)
                pnl, pos = np.array(sim['daily_pnl']), np.array(sim['position'])
                periods = {}
                for month in sorted({d[:7] for d in td}):
                    mask = np.array([d.startswith(month) for d in td])
                    periods[month] = {'pnl': float(pnl[mask].sum()), 'long_pnl': float(pnl[mask & (pos == 1)].sum()),
                                      'short_pnl': float(pnl[mask & (pos == -1)].sum()),
                                      'long_days': int((mask & (pos == 1)).sum()), 'short_days': int((mask & (pos == -1)).sum())}
                eligible = np.flatnonzero(np.isfinite(np.array(g['percentile'], float)))
                modes[mode] = {'gate': g, 'sim': sim, 'months': periods,
                               'first_eligible_signal': td[eligible[0]] if len(eligible) else None,
                               'first_entry_session': sim['trades'][0]['entry_session'] if sim['trades'] else None}
            variants[key] = {'prediction': raw_numbers(predictions), 'forecast': forecast_metrics(predictions, ty), 'modes': modes}
            latest[key] = {mode: {'prediction': float(predictions[-1]), 'percentile': modes[mode]['gate']['percentile'][-1],
                                 'signal': modes[mode]['gate']['signal'][-1],
                                 'position': modes[mode]['sim']['position'][-1],
                                 'next_position': modes[mode]['sim']['decisions'][-1]['next_position']}
                           for mode in modes}
    result = {'meta': {**frozen['meta'], 'generated': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                       'asof': td[-1], 'test_start': TEST_START, 'test_n': len(td), 'test_completed_n': int(np.isfinite(ty).sum()),
                       'first_test_signal': td[0], 'horizon': 6, 'window': 250, 'min_history': 60,
                       'lower': 40, 'upper': 60, 'default_mode': 'cold',
                       'cold': '两版都只用7月1日后的预测计算历史排名，满60条后交易。7月及大部分8/9月可能无交易，不将其误称为策略有效。',
                       'seeded': '以6月底已知的此前250日因子，用各自冻结系数回溯评分初始化排名。新模型的参考值是训练期评分，不是当时发布的预测。之后排除当日、滚动更新。',
                       'timing': '7月1日收盘为首个测试信号，两版均从空仓起步，最早7月2日代理开盘入场。动态退出、固定±1/0，最长6交易日，未扣成本。',
                       'caveat': '因子名单仍来自截至2026年10月的全历史筛选，含7月后信息；这只是回归拟合的时间外比较，不是端到端独立样本外。训练扩大方案也在观察10月结果后提出。',
                       'input_hashes': {n: sha(n) for n in ('factor_library.json', 'omo_policy_7d.json', MODELS) + PROTECTED}},
              'dates': td, 'level': raw_numbers(tl), 'target': raw_numbers(ty), 'variants': variants,
              'coefficients': coefficients, 'latest': latest}
    if protected != {n: sha(n) for n in PROTECTED}:
        raise ValueError('Production archive or model was modified')
    write_json_skip_unchanged(str(ROOT / OUTPUT), result)
    return result


if __name__ == '__main__':
    d = build()
    print(json.dumps({'meta': {k: d['meta'][k] for k in ('asof', 'train_n', 'last_train_signal', 'last_label_exit', 'test_n', 'test_completed_n')},
                      'results': {k: {m: v['sim']['metrics'] | {'first_entry': v['first_entry_session']} for m, v in a['modes'].items()}
                                  for k, a in d['variants'].items()}}, ensure_ascii=False))
