"""Neutral-zone plus reversal exits, preserving the already archived 18 variants."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jsonio import write_json_skip_unchanged
from prepare_exit_rules_study import PROTECTED as EXIT_PROTECTED, summarize
from prepare_entry_rules_study import KEYS,RULES,read,sha,entry_gate,slice_gate

ROOT=Path(__file__).resolve().parent
OUTPUT='neutral_exit_study.json'
PROTECTED=EXIT_PROTECTED+('exit_rules_study.json',)


def build():
    protected={n:sha(n) for n in PROTECTED}
    fixed=read('entry_rules_study.json');history=read('entry_rules_history.json');reverse=read('exit_rules_study.json')
    if not fixed['meta']['asof']==history['meta']['asof']==reverse['meta']['asof']:
        raise ValueError('All comparisons must share a frozen date')
    cuts=fixed['meta']['negative_cuts'];periods={};report={}
    for period,p in {**fixed['periods'],**history['periods']}.items():
        dates=p['dates'];start=p['start_index'];level=np.array(p['level'])
        predictions=p.get('prediction') or {k:v['prediction'][start:] for k,v in fixed['models'].items()}
        rows={};duplicates=[];report[period]={}
        for key in KEYS:
            rows[key]={};report[period][key]={}
            for rule in RULES:
                pred=np.array(predictions[key])
                all_p=np.array(fixed['models'][key]['prediction']) if period in fixed['periods'] else pred
                gate=slice_gate(entry_gate(all_p,rule,cuts[key]),start)
                old=p['models'][key][rule];prior=reverse['periods'][period]['models'][key][rule]
                np.testing.assert_array_equal(gate['signal'],old['signal'])
                row=summarize(dates,level,pred,gate,rule,cuts[key],old,exit_policy='neutral')
                # Ledger exit_rule differs by design; compare executable path and
                # trade times/reasons separately rather than treating a name as a change.
                old_trades=[{k:v for k,v in t.items() if k!='exit_rule'} for t in prior['sim']['trades']]
                new_trades=[{k:v for k,v in t.items() if k!='exit_rule'} for t in row['sim']['trades']]
                same=all(row['sim'][f]==prior['sim'][f] for f in ('position','daily_pnl')) and new_trades==old_trades
                row['same_as_reverse']=same
                if same:duplicates.append(key+'|'+rule)
                rows[key][rule]=row
                report[period][key][rule]={name:v['sim']['metrics']['total'] for name,v in
                                           [('fixed',old),('reverse',prior),('neutral',row)]}
        periods[period]={'dates':dates,'models':rows,'same_as_reverse':duplicates}
    out={'meta':{'asof':fixed['meta']['asof'],'nominal_strategies':27,'added_configurations':9,
                  'horizon':6,'exit_name':'中性或反向退出·最长6日',
                  'zones':'非对称：−c≤预测≤0为中性；百分位：40%≤P≤60%为中性；纯方向：仅预测=0为中性，不增加新门槛。',
                  'exit':'持仓方向与当前开仓信号不同就次日退出；中性信号只平不开，相反方向满足原开仓条件才反手。',
                  'timing':'t收盘判断，t+1代理开盘=t收盘价执行；同向不重置，最多6日，第6日收盘到期优先。末日只记计划，不伪造次日成交。',
                  'scope':'采用完整中性区，不使用55／45缓冲，不搜索参数，不重训或重筛因子。原18套与生产策略保留。',
                  'duplicates':'27是参数组合数，不是27个独立信号；纯方向没有实质中性带，当前零预测不存在时，新版与仅反向退出相同。',
                  'caveat':'截至同一旧研究快照的探索回放；2023—2025为样本内拟合，2026已反复观察，未扣成本，含期末浮动。',
                  'protected_hashes':protected},'periods':periods}
    if protected!={n:sha(n) for n in PROTECTED}:raise ValueError('Archived strategies or model inputs changed')
    json.dumps(out,allow_nan=False)
    write_json_skip_unchanged(str(ROOT/OUTPUT),out)
    print(json.dumps(report,ensure_ascii=False))
    return out


if __name__=='__main__':build()
