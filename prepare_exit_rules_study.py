"""Add reversal-only exits to nine frozen entry variants. No fitting or tuning."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from jsonio import write_json_skip_unchanged
from prepare_direction_screen import safe_number
from prepare_entry_rules_study import KEYS, RULES, read, sha, entry_gate, slice_gate
from prepare_entry_rules_history import PROTECTED as HISTORY_PROTECTED, interval_summary
from prepare_threshold_study import trade_stats

ROOT=Path(__file__).resolve().parent
OUTPUT='exit_rules_study.json'
PROTECTED=HISTORY_PROTECTED+('entry_rules_history.json',)


def reverses(direction, prediction, action, entry_rule):
    # Neutral rank and zero forecasts do not exit; weak-signal exits are NOT added.
    return direction*action<0 if entry_rule=='rank' else direction*prediction<0


def early_exit_reason(direction, prediction, action, entry_rule, exit_policy):
    if exit_policy=='neutral':
        return 'neutral' if action==0 else 'reverse' if direction*action<0 else None
    return 'reverse' if reverses(direction,prediction,action,entry_rule) else None


def simulate_reverse(dates, level, prediction, gate, entry_rule, negative_cut, horizon=6, exit_policy='reverse'):
    if exit_policy not in ('reverse','neutral'):
        raise ValueError('Unknown exit policy')
    level,prediction=np.asarray(level,float),np.asarray(prediction,float)
    n=len(dates)
    if n==0 or len(level)!=n or len(prediction)!=n or not np.isfinite(level).all() or not np.isfinite(prediction).all():
        raise ValueError('Complete aligned prices and forecasts required')
    position=np.zeros(n,int); daily=np.zeros(n); trades=[]; decisions=[]; active=None

    def finish(t,reason):
        nonlocal active
        tr=copy.deepcopy(active); entry=tr.pop('entry_index')
        tr.update(exit_signal_date=dates[t],exit_session=dates[t] if reason=='max_hold' else dates[t+1],
                  exit_execution='当日收盘' if reason=='max_hold' else '下一交易日代理开盘',
                  exit_price=float(level[t]),holding_sessions=t-entry,reason=reason,closed=True,
                  pnl=float(tr['direction']*(level[t]-tr['entry_price'])))
        trades.append(tr);active=None

    for t,day in enumerate(dates):
        if t and active is not None:
            position[t]=active['direction']
            daily[t]=position[t]*(level[t]-level[t-1])
        action=int(gate['signal'][t]) if gate['eligible'][t] else 0
        reason=None
        if active is not None:
            early=early_exit_reason(active['direction'],prediction[t],action,entry_rule,exit_policy)
            if t-active['entry_index']>=horizon:
                reason='max_hold';finish(t,reason)
            elif early and t+1<n:
                reason=early;finish(t,reason)
        opened=False
        if active is None and action and t+1<n:
            ranked=entry_rule=='rank'
            active={'entry_index':t,'signal_date':day,'entry_session':dates[t+1],
                    'entry_price':float(level[t]),'direction':action,'entry_prediction':float(prediction[t]),
                    'entry_rule':entry_rule,'exit_rule':exit_policy,'entry_threshold':None,
                    'entry_percentile':float(gate['percentile'][t]) if ranked else None,
                    'entry_history_n':int(gate['history_n'][t]) if ranked else None,
                    'entry_cutoff':(60. if action==1 else 40.) if ranked else
                        (-negative_cut if entry_rule=='asymmetric' and action==-1 else 0.),
                    'entry_cutoff_unit':'预测百分位%' if ranked else '预测价差点'}
            opened=True
        next_position=active['direction'] if active else 0
        if t==n-1:
            early=early_exit_reason(active['direction'],prediction[t],action,entry_rule,exit_policy) if active else None
            if early:
                reason=early;next_position=action
            elif active is None:
                next_position=action
        decisions.append({'date':day,'next_position':next_position,'exit_reason':reason,'opened':opened})
    if active is not None:
        tr=copy.deepcopy(active);entry=tr.pop('entry_index')
        tr.update(exit_signal_date=None,exit_session=None,exit_execution=None,exit_price=None,
                  holding_sessions=n-1-entry,reason='open',closed=False,
                  pnl=float(tr['direction']*(level[-1]-tr['entry_price'])))
        trades.append(tr)
    equity=np.cumsum(daily);dd=equity-np.maximum.accumulate(np.r_[0.,equity])[1:]
    closed=[t for t in trades if t['closed']]
    metrics={'total':safe_number(equity[-1]),'max_drawdown':safe_number(-dd.min()),
             'closed_trades':len(closed),'open_trades':len(trades)-len(closed),
             'win_rate':safe_number(np.mean([t['pnl']>0 for t in closed])) if closed else None,
             'mean_trade':safe_number(np.mean([t['pnl'] for t in closed])) if closed else None,
             'average_holding':safe_number(np.mean([t['holding_sessions'] for t in closed])) if closed else None,
             'exposure':float(np.mean(position[1:]!=0)) if n>1 else None,
             'long_sessions':int((position==1).sum()),'short_sessions':int((position==-1).sum()),
             'max_holding':max((t['holding_sessions'] for t in trades),default=0)}
    return {'position':position.tolist(),'daily_pnl':daily.tolist(),'equity':equity.tolist(),
            'drawdown':dd.tolist(),'trades':trades,'decisions':decisions,'metrics':metrics}


def summarize(dates, level, prediction, gate, entry_rule, cut, fixed, exit_policy='reverse'):
    # Entry diagnostics are identical to the original report; replace only the
    # simulated execution and execution-dependent summaries in this new artifact.
    row=copy.deepcopy(fixed)
    sim=simulate_reverse(dates,level,prediction,gate,entry_rule,cut,exit_policy=exit_policy)
    row.update(sim=sim,trade_stats=trade_stats(sim['trades']),
               sides={str(d):trade_stats([t for t in sim['trades'] if t['direction']==d]) for d in (-1,1)})
    pnl=np.array(sim['daily_pnl']);position=np.array(sim['position'])
    row['months']={month:{name:float(pnl[mask&side].sum()) for name,side in
                    [('total',np.ones(len(dates),bool)),('long',position==1),('short',position==-1)]}
                   for month in sorted({day[:7] for day in dates})
                   for mask in [np.array([day.startswith(month) for day in dates])]}
    row['latest'].update(position=sim['position'][-1],next_position=sim['decisions'][-1]['next_position'])
    row['exit_counts']={reason:sum(t['closed'] and t['reason']==reason for t in sim['trades'])
                        for reason in (('max_hold','reverse','neutral') if exit_policy=='neutral' else ('max_hold','reverse'))}
    if 'years' in row:
        row['years']={year:interval_summary(dates,row,np.array([d.startswith(year) for d in dates]))
                       for year in sorted({d[:4] for d in dates})}
        row['stages']={label:interval_summary(dates,row,mask) for label,mask in
                        [('train',np.array([d<='2025-12-31' for d in dates])),
                         ('test',np.array([d>'2025-12-31' for d in dates]))] if mask.any()}
    return row


def build():
    protected={n:sha(n) for n in PROTECTED}
    fixed=read('entry_rules_study.json'); history=read('entry_rules_history.json')
    cuts=fixed['meta']['negative_cuts'];periods={}
    for period,p in {**fixed['periods'],**history['periods']}.items():
        dates=p['dates'];start=p['start_index'];level=np.array(p['level'])
        predictions=p.get('prediction') or {k:v['prediction'][start:] for k,v in fixed['models'].items()}
        rows={}
        for key in KEYS:
            rows[key]={}
            for rule in RULES:
                prediction=np.array(predictions[key])
                # Common-start ranking must retain the previous 60 forecasts.
                all_p=np.array(fixed['models'][key]['prediction']) if period in fixed['periods'] else prediction
                gate=slice_gate(entry_gate(all_p,rule,cuts[key]),start)
                np.testing.assert_array_equal(gate['signal'],p['models'][key][rule]['signal'])
                rows[key][rule]=summarize(dates,level,prediction,gate,rule,cuts[key],p['models'][key][rule])
        periods[period]={'dates':dates,'models':rows}
    result={'meta':{'asof':fixed['meta']['asof'],'strategies':18,'new_reverse_strategies':9,
                    'horizon':6,'exits':{'fixed':'固定6日','reverse':'反向离场·最长6日'},
                    'timing':'t收盘判断反转，t+1代理开盘=t收盘价执行；先计入t当日持仓盈亏，再决定下一交易日仓位。到期在第6日收盘退出。',
                    'reversal':'纯方向／过滤：持多时预测<0退出、持空时预测>0退出；预测=0不退出。排名：持多时P<40退出、持空时P>60退出；中性及边界不提前退出。',
                    'reentry':'离场后仅当当日满足对应开仓规则才反手。弱负预测可平多但未必开空；同向不重置持有期。',
                    'last_day':'末日反转只记录下一交易日计划，未平交易仍按末日收盘盯市，不伪造成交或删掉浮动。',
                    'caveat':'只新增反向离场，不含转弱退出；保持原预测、因子、门槛和成本口径。仍为探索研究，全历史训练期不是样本外。',
                    'protected_hashes':protected},'periods':periods}
    if protected!={n:sha(n) for n in PROTECTED}:raise ValueError('Frozen artifacts changed')
    json.dumps(result,allow_nan=False)
    write_json_skip_unchanged(str(ROOT/OUTPUT),result)
    print(json.dumps({period:{k:{q:{'fixed':p['models'][k][q]['sim']['metrics']['total'],
                                  'reverse':r['sim']['metrics']['total'],
                                  'reverse_dd':r['sim']['metrics']['max_drawdown']}
                               for q,r in models.items()} for k,models in new['models'].items()}
                      for period,new in periods.items()
                      for p in [{**fixed['periods'],**history['periods']}[period]]},ensure_ascii=False))
    return result


if __name__=='__main__':build()
