"""Isolated Bayesian curve-timing experiment; never changes production models.

Stages: inputs (base Python), bart (isolated StochTree environment), report (base Python).
The BART worker receives no post-cutoff labels. All hyperparameter selection ends
at June 2026. Artifacts under output/bayes/ are local training caches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from jsonio import write_json_skip_unchanged
from prepare_curve_timing_capacity import TypeAwareScaler, fit_model, model_record
from prepare_curve_timing_walkforward import completed_indices, forecast_metrics
from prepare_direction_screen import baseline_features, load_candidates, raw_numbers
from prepare_ridge_retrain_202606 import gate_and_simulate
from prepare_strategy_model import make_offset_labels

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'output' / 'bayes'
CUTOFF = '2026-06-30'
VALIDATION_CUTOFF = '2025-12-31'
OUTPUT = 'curve_timing_bayes.json'
PROTECTED = ('ridge_daily_models.json', 'ridge_daily_signals.json',
             'curve_timing_capacity_factor_sets.json', 'ridge_retrain_202606_models.json',
             'curve_timing_nonlinear_models.json', 'curve_timing_logistic_models.json')


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data():
    _, dates, curves, candidates = load_candidates()
    end = dates.index(read('ridge_daily_signals.json')['meta']['signal_date'])
    dates = dates[:end + 1]
    level = (3 * curves['T主力'] - curves['TL主力'])[:end + 1]
    base, _ = baseline_features(level, {k: v[:end + 1] for k, v in curves.items()}, dates)
    cmap = {f['id']: f for f in candidates}
    features = base + [cmap[i] for i in read('curve_timing_capacity_factor_sets.json')['sets']['f20']]
    x = np.column_stack([f['values'][:end + 1] for f in features])
    y = make_offset_labels(level, 6)
    cut = max(i for i, d in enumerate(dates) if d <= CUTOFF)
    train = completed_indices(level, y, cut, 6)
    valid = np.isfinite(x[:, :8]).all(axis=1) & (np.isfinite(x[:, 8:]).mean(axis=1) >= .9)
    valid &= np.isfinite(level)
    if not valid[cut + 1:].all():
        raise ValueError('Research requires complete common test availability')
    return dates, np.asarray(level), x, y, features, cut, train


def initial_state(z, y, alpha=15.):
    """Ridge-centered empirical Bayes prior, tempered for six-day overlap.

    The factor-six covariance inflation is a working dependence adjustment,
    not a claim that serial residual covariance has been fully estimated.
    """
    mean = np.r_[np.mean(y), np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - y.mean()))]
    design = np.column_stack([np.ones(len(z)), z])
    noise = max(float(np.mean((y - design @ mean) ** 2)), 1e-6)
    precision = design.T @ design + np.diag(np.r_[0., np.full(z.shape[1], alpha)])
    covariance = 6 * noise * np.linalg.inv(precision)
    return mean, covariance, noise


def dynamic_forecast(z, y, start, last_train, mean, covariance, noise, discount, dates=None):
    """Filter delayed outcomes, six sessions apart; never smooth with future data.

    State is dated by label *origin*. At close t, label t-6 may be assimilated.
    Then project its state forward to t. No current/future label is accessed.
    """
    mean, covariance = mean.copy(), covariance.copy()
    state_time = last_train
    last_update = last_train
    means, stds, coefficients, updates = [], [], [], []
    eye = np.eye(len(mean))
    for t in range(start, len(z)):
        origin = t - 6
        if origin >= last_update + 6 and np.isfinite(y[origin]) and np.isfinite(z[origin]).all():
            prior_cov = covariance / discount ** (origin - state_time)
            row = np.r_[1., z[origin]]
            variance = float(noise + row @ prior_cov @ row)
            innovation = float(y[origin] - row @ mean)
            gain = prior_cov @ row / variance
            mean = mean + gain * innovation
            # Joseph form preserves positive semidefiniteness numerically.
            residual = eye - np.outer(gain, row)
            covariance = residual @ prior_cov @ residual.T + noise * np.outer(gain, gain)
            covariance = (covariance + covariance.T) / 2
            state_time = last_update = origin
            updates.append({'asof_index': t, 'origin_index': origin,
                            'asof': dates[t] if dates else t,
                            'origin': dates[origin] if dates else origin,
                            'label_exit': dates[origin + 6] if dates else origin + 6,
                            'innovation': innovation})
        row = np.r_[1., z[t]]
        projected = covariance / discount ** (t - state_time)
        means.append(float(row @ mean))
        stds.append(float(np.sqrt(max(noise + row @ projected @ row, 1e-12))))
        coefficients.append(mean.copy())
    return {'mean': np.array(means), 'std': np.array(stds), 'coefficients': np.array(coefficients),
            'updates': updates, 'final_mean': mean, 'final_covariance': covariance,
            'state_time': state_time}


def prepare_inputs():
    CACHE.mkdir(parents=True, exist_ok=True)
    dates, level, x, y, features, cut, train = load_data()
    # One causal validation period, wholly before July; no test-based tuning.
    val_cut = max(i for i, d in enumerate(dates) if d <= VALIDATION_CUTOFF)
    inner_train = completed_indices(level, y, val_cut, 6)
    inner_scaler = TypeAwareScaler(features).fit(x[inner_train])
    iz = inner_scaler.transform(x[:cut + 1])
    inner_state = initial_state(iz[inner_train], y[inner_train])
    tuning = []
    for discount in (1., .995, .98):
        forecast = dynamic_forecast(iz, y[:cut + 1], val_cut + 1, inner_train[-1], *inner_state, discount)
        indices = np.arange(val_cut + 1, cut + 1)
        mask = (indices + 6 <= cut) & np.isfinite(y[indices])
        metrics = forecast_metrics(forecast['mean'][mask], y[indices][mask])
        tuning.append({'discount': discount, **metrics})
    # Select among genuine time-varying candidates; retain discount=1 as an
    # explicit no-state-drift Bayesian updating control in the report.
    discount = min((r for r in tuning if r['discount'] < 1), key=lambda r: r['rmse'])['discount']
    model = fit_model(x[train], y[train], features, alpha=15)
    scaler = model[0]
    z = scaler.transform(x)
    mean, cov, noise = initial_state(z[train], y[train])
    # Check same center as the already published June Ridge experiment.
    old_record = read('ridge_retrain_202606_models.json')['models']['f20']
    expected = np.r_[old_record['intercept'], [f['coef'] for f in old_record['features']]]
    if not np.allclose(mean, expected, atol=1e-10):
        raise ValueError('June Ridge initialization differs from frozen reference')
    seed_start = cut + 1 - 250
    bart_path = CACHE / 'bart_inputs.npz'
    # BART receives ONLY training Y. No test targets in this file.
    np.savez_compressed(bart_path, x_train=z[train], y_train=y[train],
                        x_score=z[seed_start:], train_indices=train,
                        n_seed=np.array(250), score_dates=np.array(dates[seed_start:]))
    meta = {'train_through': CUTOFF, 'validation_initial_through': VALIDATION_CUTOFF,
            'validation': tuning, 'discount': discount, 'train_n': len(train),
            'last_train_signal': dates[train[-1]], 'last_train_label_exit': dates[train[-1] + 6],
            'asof': dates[-1], 'features': [{'id': f['id'], 'label': f['label']} for f in features],
            'scaler_model': model_record(model, features), 'initial_mean': mean.tolist(),
            'initial_covariance': cov.tolist(), 'noise_variance': noise,
            'initial_state_index': int(train[-1]), 'score_start': dates[seed_start],
            'inputs_sha256': digest(bart_path),
            'protected_hashes': {n: digest(ROOT / n) for n in PROTECTED}}
    write_json_skip_unchanged(str(CACHE / 'study_inputs.json'), meta)
    print(json.dumps({k: meta[k] for k in ('asof', 'train_n', 'discount', 'validation')}, ensure_ascii=False))
    return meta


def summarize_variant(name, prediction, seed, td, level, target, std=None, interval=None, probability=None):
    modes = {}
    for mode in ('cold', 'seeded'):
        gate, sim = gate_and_simulate(td, level, prediction, seed if mode == 'seeded' else None)
        pnl, pos = np.array(sim['daily_pnl']), np.array(sim['position'])
        months = {}
        for month in sorted({d[:7] for d in td}):
            mask = np.array([d.startswith(month) for d in td])
            months[month] = {'pnl': float(pnl[mask].sum()),
                             'long_pnl': float(pnl[mask & (pos == 1)].sum()),
                             'short_pnl': float(pnl[mask & (pos == -1)].sum())}
        modes[mode] = {'gate': gate, 'sim': sim, 'months': months,
                       'latest': {'prediction': float(prediction[-1]),
                                  'percentile': gate['percentile'][-1], 'signal': gate['signal'][-1],
                                  'position': sim['position'][-1],
                                  'next_position': sim['decisions'][-1]['next_position']}}
    result = {'label': name, 'prediction': raw_numbers(prediction),
              'forecast': forecast_metrics(prediction, target), 'modes': modes}
    if std is not None:
        result['predictive_std'] = raw_numbers(std)
        result['interval80'] = {'low': raw_numbers(interval[0]), 'high': raw_numbers(interval[1])}
        result['probability_positive'] = raw_numbers(probability)
        mask = np.isfinite(target)
        result['forecast']['coverage80'] = float(np.mean((target[mask] >= interval[0][mask]) &
                                                        (target[mask] <= interval[1][mask])))
        result['forecast']['interval80_width'] = float(np.mean(interval[1][mask] - interval[0][mask]))
        result['forecast']['brier'] = float(np.mean((probability[mask] - (target[mask] > 0)) ** 2))
    return result


def build_report():
    meta = json.loads((CACHE / 'study_inputs.json').read_text(encoding='utf-8'))
    bart_meta = json.loads((CACHE / 'bart_summary.json').read_text(encoding='utf-8'))
    if bart_meta['inputs_sha256'] != meta['inputs_sha256']:
        raise ValueError('BART predictions and study inputs differ')
    if meta['protected_hashes'] != {n: digest(ROOT / n) for n in PROTECTED}:
        raise ValueError('Production or comparison model changed during experiment')
    dates, level, x, y, features, cut, train = load_data()
    if dates[-1] != meta['asof']:
        raise ValueError('Rebuild explicitly after data date changes')
    rec = meta['scaler_model']
    z = np.column_stack([(np.clip(np.where(np.isfinite(x[:, j]), x[:, j], f['median']),
                                  f['clip_low'], f['clip_high']) if f['clip_enabled'] else
                          np.where(np.isfinite(x[:, j]), x[:, j], f['median']))
                         for j, f in enumerate(rec['features'])])
    z = (z - [f['mean'] for f in rec['features']]) / [f['std'] for f in rec['features']]
    start = cut + 1
    original_inputs = np.load(CACHE / 'bart_inputs.npz')
    np.testing.assert_array_equal(z[train], original_inputs['x_train'])
    np.testing.assert_array_equal(y[train], original_inputs['y_train'])
    np.testing.assert_array_equal(z[start - 250:], original_inputs['x_score'])
    td, tl, target = dates[start:], level[start:], y[start:]
    dyn = dynamic_forecast(z, y, start, meta['initial_state_index'], np.array(meta['initial_mean']),
                           np.array(meta['initial_covariance']), meta['noise_variance'], meta['discount'], dates)
    seed = np.column_stack([np.ones(250), z[start - 250:start]]) @ meta['initial_mean']
    sd, dm = dyn['std'], dyn['mean']
    dlm = summarize_variant('动态贝叶斯线性', dm, seed, td, tl, target, sd,
                            (dm - 1.281551565545 * sd, dm + 1.281551565545 * sd), ndtr(dm / sd))
    bp = np.load(CACHE / 'bart_predictions.npz')
    bm, bs = bp['mean'][250:], bp['std'][250:]
    bart = summarize_variant('BART贝叶斯树', bm, bp['mean'][:250], td, tl, target, bs,
                             (bp['low80'][250:], bp['high80'][250:]), bp['probability_positive'][250:])
    chain_sensitivity = []
    for i, chain in enumerate(bp['chain_mean']):
        evaluations = {}
        for mode in ('cold', 'seeded'):
            gate, sim = gate_and_simulate(td, tl, chain[250:], chain[:250] if mode == 'seeded' else None)
            evaluations[mode] = {'total': sim['metrics']['total'],
                                 'october': float(sum(p for day, p in zip(td, sim['daily_pnl']) if day.startswith('2026-10'))),
                                 'last_signal': gate['signal'][-1]}
        chain_sensitivity.append({'chain': i + 1, 'forecast': forecast_metrics(chain[250:], target),
                                  'modes': evaluations})
    bart_meta['chain_sensitivity'] = chain_sensitivity
    bart_meta['posterior_artifact_sha256'] = digest(CACHE / 'bart_model.json.gz')
    bart_meta['convergence_caution'] = bool(bart_meta['latent_rhat_p95'] > 1.05)
    prior_study = read('ridge_retrain_202606.json')
    variants = {'old20': prior_study['variants']['old20'], 'expanded20': prior_study['variants']['expanded20'],
                'dlm20': dlm, 'bart20': bart}
    variants['old20']['label'] = '岭回归·训练至2024年底'
    variants['expanded20']['label'] = '岭回归·训练至2026年6月'
    if prior_study['dates'] != td:
        raise ValueError('Ridge reference comparison dates differ')
    static_dyn = dynamic_forecast(z, y, start, meta['initial_state_index'], np.array(meta['initial_mean']),
                                  np.array(meta['initial_covariance']), meta['noise_variance'], 1., dates)
    sm, ss = static_dyn['mean'], static_dyn['std']
    variants['bayes20'] = summarize_variant('贝叶斯递推·无状态漂移对照', sm, seed, td, tl, target, ss,
                                           (sm - 1.281551565545 * ss, sm + 1.281551565545 * ss), ndtr(sm / ss))
    momentum_index = next(i for i, f in enumerate(features) if f['id'] == 'baseline|spread_mom5')
    diagnostics = {'bart': bart_meta, 'dlm': {'discount': meta['discount'], 'validation': meta['validation'],
                     'updates': dyn['updates'], 'update_count': len(dyn['updates']),
                     'coefficient_names': ['截距'] + [f['label'] for f in features],
                     'coefficients': dyn['coefficients'].tolist(),
                     'initial_coefficients': meta['initial_mean'], 'noise_variance': meta['noise_variance'],
                     'momentum_coefficient_index': momentum_index + 1}}
    result = {'meta': {'asof': dates[-1], 'train_through': CUTOFF, 'train_n': len(train),
                       'test_start': td[0], 'test_n': len(td), 'completed_n': int(np.isfinite(target).sum()),
                       'horizon': 6, 'features_n': len(features), 'default_mode': 'cold',
                       'factor_selection_lookahead': True,
                       'caveat': '固定因子名单来自截至10月的全历史筛选，含选择前视；方案在观察10月亏损后提出，属于探索研究。',
                       'cold': '仅用7月后实际按时间产生的预测，至少60条后计算排名；前期无交易。',
                       'seeded': '以6月底冻结模型对此前250日的训练期历史评分初始化；DLM使用6月底初始系数评分。这些不是当时发布的预测。',
                       'interval': '80%区间为未来六日实际价差变动的模型预测区间，不是均值区间；概率仅观察，不用于交易。',
                       'inputs_sha256': meta['inputs_sha256'], 'protected_hashes': meta['protected_hashes'],
                       'last_train_signal': meta['last_train_signal'],
                       'last_train_label_exit': meta['last_train_label_exit'],
                       'preprocessing': meta['scaler_model']['features']},
              'dates': td, 'target': raw_numbers(target), 'level': raw_numbers(tl),
              'variants': variants, 'diagnostics': diagnostics}
    write_json_skip_unchanged(str(ROOT / OUTPUT), result)
    print(json.dumps({k: {'forecast': v['forecast'], 'pnl': {m: d['sim']['metrics'] for m, d in v['modes'].items()}}
                      for k, v in variants.items()}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('inputs', 'report'))
    args = p.parse_args()
    prepare_inputs() if args.stage == 'inputs' else build_report()
