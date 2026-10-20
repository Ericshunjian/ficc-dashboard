"""Frozen-model backcasts, explicitly separated from original 2026 testing.

No historical dynamic Bayesian predictions are invented: before 2026 use the
frozen initial posterior mean (equal to temporal Ridge); after it copy the
already archived delayed-update predictions exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jsonio import write_json_skip_unchanged
from prepare_entry_rules_study import KEYS, RULES, PROTECTED as ENTRY_PROTECTED, entry_gate, evaluate, read, sha
from prepare_temporal_study import (MODELS, load_data, temporal_features, matrix, linear_prediction,
                                    training_digest, make_offset_labels)

ROOT = Path(__file__).resolve().parent
OUTPUT = 'entry_rules_history.json'
PROTECTED = ENTRY_PROTECTED + ('entry_rules_study.json', MODELS)


def interval_summary(dates, row, mask):
    pnl = np.array(row['sim']['daily_pnl'])[mask]
    eq = np.cumsum(pnl)
    dd = eq - np.maximum.accumulate(np.r_[0., eq])[1:]
    position = np.array(row['sim']['position'])[mask]
    return {'start': np.array(dates)[mask][0], 'end': np.array(dates)[mask][-1],
            'total': float(pnl.sum()), 'max_drawdown': float(-dd.min()),
            'long_pnl': float(pnl[position == 1].sum()), 'short_pnl': float(pnl[position == -1].sum()),
            'note': '连续账户按每日盯市切片；不重置持仓，段内回撤按该段盈亏从0累计。'}


def replay(dates, level, predictions, cuts, name):
    target = make_offset_labels(level, 6)
    rows = {}
    for key in KEYS:
        rows[key] = {}
        for rule in RULES:
            gate = entry_gate(predictions[key], rule, cuts[key])
            row = evaluate(dates, level, predictions[key], target, gate, rule, cuts[key])
            row['years'] = {year: interval_summary(dates,row,np.array([d.startswith(year) for d in dates]))
                            for year in sorted({d[:4] for d in dates})}
            row['stages'] = {label: interval_summary(dates,row,mask) for label,mask in
                             [('train',np.array([d<='2025-12-31' for d in dates])),
                              ('test',np.array([d>'2025-12-31' for d in dates]))] if mask.any()}
            rows[key][rule] = row
    return {'start_index': 0, 'dates': dates, 'level': level.tolist(),
            'target': [float(v) if np.isfinite(v) else None for v in target],
            'prediction': {k:p.tolist() for k,p in predictions.items()}, 'models':rows,
            'note': name}


def build():
    protected = {n:sha(n) for n in PROTECTED}
    dates, level, xb, _, base, cut, train = load_data()
    bundle = read(MODELS)
    xt = matrix(temporal_features(level,base))
    fingerprint = training_digest(xb[:cut+1],xt[:cut+1],make_offset_labels(level[:cut+1],6),dates[:cut+1])
    if fingerprint != bundle['meta']['input_sha256']:
        raise ValueError('Historical training snapshot differs from frozen model')
    first = int(np.flatnonzero(np.isfinite(level))[0])
    if not np.isfinite(level[first:]).all():
        raise ValueError('Historical price missing; never fill zero PnL')
    source, previous = read('forecast_model_comparison.json'),read('entry_rules_study.json')
    if dates[cut+1:] != source['dates']:
        raise ValueError('Current calendar differs from archived test dates')
    predictions = {k:linear_prediction(bundle['models'][k],x) for k,x in
                   [('ridge_original',xb),('ridge_temporal',xt)]}
    for key in predictions:
        np.testing.assert_allclose(predictions[key][cut+1:],source['models'][key]['prediction'],rtol=0,atol=1e-12)
        # Preserve serialized test forecasts byte-for-value; never replace archives.
        predictions[key][cut+1:] = source['models'][key]['prediction']
    initial = bundle['models']['dlm_temporal']['mean']
    temporal = bundle['models']['ridge_temporal']
    np.testing.assert_allclose(initial,np.r_[temporal['intercept'],[f['coef'] for f in temporal['features']]],atol=1e-12)
    predictions['dlm_temporal'] = predictions['ridge_temporal'].copy()
    predictions['dlm_temporal'][cut+1:] = source['models']['dlm_temporal']['prediction']
    td, tl = dates[first:],level[first:]
    pp = {k:v[first:] for k,v in predictions.items()}
    cuts = previous['meta']['negative_cuts']
    training_n = cut-first+1
    periods = {'history_full':replay(td,tl,pp,cuts,
                '全历史连续回放：2023—2025样本内；2026沿用原预测，但持仓及排名继承2025，不等于2026空仓冷启动。'),
               'history_train':replay(td[:training_n],tl[:training_n],{k:p[:training_n] for k,p in pp.items()},cuts,
                '训练期拟合回放：2025年底才得到的模型回看2023—2025，含模型、预处理及门槛前视；期末持仓仅盯市，不强制平仓。')}
    out={'meta':{'asof':td[-1],'start':td[0],'train_through':'2025-12-31',
                 'test_start':source['dates'][0], 'forecast_n':len(td),'train_price_n':training_n,
                 'train_label_n':len(train),'train_last_signal':dates[train[-1]],
                 'negative_cuts':cuts,'rules':previous['meta']['rules'],
                 'missing_original_cells':int(np.sum(~np.isfinite(xb[first:cut+1]))),
                 'missing_temporal_cells':int(np.sum(~np.isfinite(xt[first:cut+1]))),
                 'bayes_backcast':'2025年底以前使用冻结初始后验均值，等于时序Ridge；2026使用原有延迟递推预测，未向前伪造贝叶斯更新。',
                 'ranking':'历史排名含此前的样本内回溯评分，2023年起先等待60条；2026不重置。',
                 'preprocessing':'完全复用2025训练预处理；早期缺失以训练中位数填补，因此存在未来信息，不能当作当时可交易信号。',
                 'caveat':'全历史拟合+测试混合展示，不能作为纯样本外业绩。固定因子名单、2025校准门槛也包含历史前视。原2026测试与生产归档不变。',
                 'protected_hashes':protected},
         'models':{k:{'label':source['models'][k]['label']} for k in KEYS},'periods':periods}
    if protected != {n:sha(n) for n in PROTECTED}:
        raise ValueError('Original archives changed')
    json.dumps(out,allow_nan=False)
    write_json_skip_unchanged(str(ROOT/OUTPUT),out)
    print(json.dumps({name:{k:{rule:row['sim']['metrics'] for rule,row in m.items()}
                            for k,m in p['models'].items()} for name,p in periods.items()},ensure_ascii=False))
    return out


if __name__=='__main__':build()
