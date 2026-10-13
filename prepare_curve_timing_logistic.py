"""Reproduce frozen Logistic research; never fit models or select factors."""
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from jsonio import write_json_skip_unchanged
from prepare_direction_screen import baseline_features, load_candidates, raw_numbers, safe_number
from prepare_strategy_model import make_offset_labels
from prepare_ridge_daily import infer
from prepare_logistic_daily import CONFIG, MODELS, gate, probability, validate_models
from prepare_curve_timing_walkforward import simulate
from prepare_curve_timing_percentile import serialize_sim
from research_inputs import research_input

ROOT = Path(__file__).resolve().parent


def build():
    frozen = json.loads((ROOT / MODELS).read_text(encoding='utf-8'))
    validate_models(frozen)
    snapshot = research_input('factor_library.json', frozen['meta']['source_snapshot_sha256'])
    with patch('prepare_factor_checkup.LIB_PATH', str(snapshot)), patch('prepare_strategy_model.BASE', str(snapshot.parent)):
        _, dates, cmap, candidates = load_candidates()
        end = dates.index('2026-10-09') + 1
        dates, cmap = dates[:end], {k: v[:end] for k, v in cmap.items()}
        candidates = [f | {'values': f['values'][:end]} for f in candidates]
        level = 3 * cmap['T主力'] - cmap['TL主力']
        baselines, _ = baseline_features(level, cmap, dates)
    first = next(i for i, d in enumerate(dates) if d > '2024-12-31')
    y = make_offset_labels(level, 6)[first:]
    td, tl = dates[first:], level[first:]
    fs = {f['id']: f for f in candidates + baselines}
    predictions, forecasts, gates = {}, {}, {}
    strategies = {'dynamic': {}, 'fixed': {}}
    for arm, model in frozen['models'].items():
        key = 'logistic' + arm[1:]
        pred = np.array([probability(infer(model, [fs[f['id']]['values'][i] for f in model['features']])[0])
                         for i in range(first, len(dates))])
        predictions[key] = raw_numbers(pred)
        g = gate(pred)
        gates[key] = {k: raw_numbers(v) if k not in ('signal', 'history_n') else v.tolist()
                      for k, v in g.items() if k != 'direction_basis'}
        for rule in strategies:
            strategies[rule][key] = serialize_sim(simulate(td, tl, pred, g['quantile'], rule=rule, gate=g))
        forecasts[key] = {}
        for year in ('all', '2025', '2026'):
            valid = np.isfinite(y) & (np.ones(len(td), bool) if year == 'all' else np.array([d.startswith(year) for d in td]))
            p, change = pred[valid], y[valid]
            binary = (change > 0).astype(int)
            calibration = []
            for lo in np.arange(0., 1., .1):
                mask = (p >= lo) & (p < lo + .1)
                if mask.any():
                    calibration.append({'lower': safe_number(lo), 'upper': safe_number(lo + .1), 'n': int(mask.sum()),
                                        'mean_probability': safe_number(p[mask].mean()),
                                        'actual_positive_rate': safe_number(binary[mask].mean())})
            forecasts[key]['test' if year == 'all' else year] = {
                'n': int(valid.sum()), 'ic': safe_number(spearmanr(p, change).statistic), 'rmse': None,
                'auc': safe_number(roc_auc_score(binary, p)), 'log_loss': safe_number(log_loss(binary, p, labels=[0, 1])),
                'brier': safe_number(brier_score_loss(binary, p)), 'accuracy': safe_number(np.mean((p > .5) == binary)),
                'calibration': calibration}
    result = {'meta': {'asof': td[-1], 'train_through': '2024-12-31', 'train_n': 406, 'test_n': len(td), 'sizes': [20, 30, 40],
                       'family': 'L2 logistic', 'config': CONFIG, 'label': 'P(S[t+6]−S[t]>0)',
                       'caveat': '全历史筛选含选择前视；模型概率尚未完全校准；前收代理开盘、未扣成本。'},
              'daily': {'dates': td, 'level': raw_numbers(tl), 'target': raw_numbers(y), 'prediction': predictions},
              'forecast': forecasts, 'gates': gates, 'strategies': strategies}
    write_json_skip_unchanged(str(ROOT / 'curve_timing_logistic.json'), result)
    return result


if __name__ == '__main__':
    report = build()
    print(json.dumps({key: value['metrics'] for key, value in report['strategies']['dynamic'].items()}, ensure_ascii=False))
