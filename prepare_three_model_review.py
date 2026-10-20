"""Three frozen predictors x three existing entries x three existing exits.

The 2026 selection is exploratory. No prior forecasts or production are changed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from jsonio import write_json_skip_unchanged
from prepare_bayes_parameter_study import PROTECTED as PARAMETER_PROTECTED
from prepare_entry_rules_study import read, sha, entry_gate, slice_gate, evaluate
from prepare_exit_rules_study import summarize, simulate_reverse
from prepare_strategy_stability import slice_stats, concentration, quarter
from prepare_forecast_comparison import bootstrap_indices

ROOT=Path(__file__).resolve().parent
OUTPUT='three_model_strategy_review.json'
MANIFEST='curve_timing_live_candidates.json'
KEYS=('ridge_original','ridge_temporal','bayes_temporal_new')
RULES=('sign','asymmetric','rank')
EXITS=('fixed','reverse','neutral')
LABELS={'ridge_original':'原Ridge','ridge_temporal':'时序Ridge','bayes_temporal_new':'新时序贝叶斯'}
ENTRY_LABELS={'sign':'纯方向','asymmetric':'非对称过滤','rank':'历史排名40／60'}
EXIT_LABELS={'fixed':'固定6日','reverse':'反向退出','neutral':'中性退出'}
PROTECTED=tuple(dict.fromkeys(PARAMETER_PROTECTED+('bayes_parameter_study.json','bayes_parameter_models.json')))
CONFIG={'version':1,'entries':list(RULES),'exits':list(EXITS),'horizon':6,
        'selected_on':'2026-10-21',
        'negative_cut_source':'frozen temporal Ridge 2025 calibration, shared by all models',
        'selection':{'full_min_closed':20,'common_min_closed':10,'profitable_complete_quarters':2,
                     'total_positive_both':True,'both_sides_positive_both':True,
                     'after_top3_positive_both':True,'six_starts_positive_both':True,
                     'recent60_positive_both':True,'neighbor_exits_positive_at_least':2,
                     'score':'minimum(full total/maxDD, common total/maxDD)',
                     'max_primary_models':2,'one_primary_per_model':True,
                     'tie_exit_preference':['reverse','neutral','fixed']}}


def fingerprint(position):
    return hashlib.sha256(np.array(position,dtype=np.int8).tobytes()).hexdigest()


def path(dates,level,pred,gate,rule,policy,cut):
    fixed=evaluate(dates,level,pred,np.full(len(dates),np.nan),gate,rule,cut)
    return fixed if policy=='fixed' else summarize(dates,level,pred,gate,rule,cut,fixed,exit_policy=policy)


def diagnostics(row, dates, pred, all_gate, rule, policy, cut, start, all_dates, all_level):
    sim=row['sim'];n=len(dates)
    months={m:slice_stats(dates,sim,[i for i,d in enumerate(dates) if d.startswith(m)]) for m in sorted({d[:7] for d in dates})}
    quarters={q:slice_stats(dates,sim,[i for i,d in enumerate(dates) if quarter(d)==q]) for q in sorted({quarter(d) for d in dates})}
    starts=[]
    for offset in range(6):
        j=start+offset
        gate=slice_gate(all_gate,j)
        if policy=='fixed':
            s=path(all_dates[j:],all_level[j:],pred[j:],gate,rule,policy,cut)['sim']
        else:
            s=simulate_reverse(all_dates[j:],all_level[j:],pred[j:],gate,rule,cut,exit_policy=policy)
        starts.append({'offset':offset,'start':all_dates[j],'metrics':s['metrics']})
    return {'months':months,'quarters':quarters,
            'recent':{str(w):slice_stats(dates,sim,np.arange(max(0,n-w),n)) for w in (30,60)},
            'concentration':concentration(row),'starts':starts,
            'start_min':min(s['metrics']['total'] for s in starts),
            'positive_month_n':sum(v['total']>0 for v in months.values()),'month_n':len(months),
            'profitable_complete_quarters':sum(v['total']>0 for q,v in quarters.items() if q!='2026-Q4' and v['n']>=30),
            'complete_quarters':sum(q!='2026-Q4' and v['n']>=30 for q,v in quarters.items()),
            'side_pnl':{str(d):float(np.array(sim['daily_pnl'])[np.array(sim['position'])==d].sum()) for d in (-1,1)},
            'path_sha256':fingerprint(sim['position'])}


def shortlist(periods):
    cfg=CONFIG['selection'];review={}
    for model in KEYS:
        for rule in RULES:
            for policy in EXITS:
                key=f'{model}|{rule}|{policy}'
                rows={p:periods[p]['models'][model][rule][policy] for p in ('full','common')}
                failures=[]
                for p,r in rows.items():
                    s,d=r['sim']['metrics'],r['diagnostics']
                    checks={'收益不为正':s['total']>0,
                            '交易笔数偏少':s['closed_trades']>=cfg[p+'_min_closed'],
                            '多空两侧未同时盈利':min(d['side_pnl'].values())>0,
                            '剔除最好三笔后不盈利':d['concentration']['total_after_top3']>0,
                            '相邻起点不稳':d['start_min']>0,
                            '近60日不盈利':d['recent']['60']['total']>0,
                            '盈利足量季度不足2个':d['profitable_complete_quarters']>=cfg['profitable_complete_quarters']}
                    for reason,ok in checks.items():
                        if not ok:failures.append(p+':'+reason)
                    neighbors=periods[p]['models'][model][rule]
                    if sum(v['sim']['metrics']['total']>0 for v in neighbors.values())<cfg['neighbor_exits_positive_at_least']:
                        failures.append(p+':相邻离场规则盈利不足2组')
                score=min(r['sim']['metrics']['total']/max(r['sim']['metrics']['max_drawdown'],1e-6) for r in rows.values())
                review[key]={'model':model,'entry':rule,'exit':policy,'label':f'{LABELS[model]}·{ENTRY_LABELS[rule]}·{EXIT_LABELS[policy]}',
                             'eligible':not failures,'failures':failures,'score':score,
                             'paths':{p:r['diagnostics']['path_sha256'] for p,r in rows.items()}}
    ordered=sorted(review,key=lambda k:(-review[k]['score'],CONFIG['selection']['tie_exit_preference'].index(review[k]['exit']),k))
    selected=[];models=set();paths=set()
    for key in ordered:
        r=review[key]
        pair=(r['paths']['full'],r['paths']['common'])
        if r['eligible'] and r['model'] not in models and pair not in paths:
            selected.append(key);models.add(r['model']);paths.add(pair)
        if len(selected)==cfg['max_primary_models']:break
    return {'rows':review,'ordered':ordered,'selected':selected,
            'rule':'固定描述性稳健检查后，按两区间较低收益回撤比排序；每模型最多一组，相同路径去重，最多2个主跟踪。',
            'not_independent':'同一因子池和同一价差，模型和策略高度相关；不将主跟踪名单解释为独立资金组合。'}


def build():
    protected={n:sha(n) for n in PROTECTED}
    source,params=read('forecast_model_comparison.json'),read('bayes_parameter_study.json')
    if source['dates']!=params['dates'] or source['target']!=params['target']:
        raise ValueError('Frozen predictor dates differ')
    dates=source['dates'];level=np.array(params['level']);target=np.array(source['target'],float)
    predictions={k:np.array(source['models'][k]['prediction']) for k in KEYS[:2]}
    predictions['bayes_temporal_new']=np.array(params['models']['selected']['prediction'])
    cut=params['meta']['shared_negative_cut']
    gates={k:{r:entry_gate(p,r,cut) for r in RULES} for k,p in predictions.items()}
    common=int(np.flatnonzero(np.logical_and.reduce([gates[k]['rank']['eligible'] for k in KEYS]))[0])
    periods={}
    for period,start in [('full',0),('common',common)]:
        td,tl,ty=dates[start:],level[start:],target[start:];models={}
        for key in KEYS:
            models[key]={}
            for rule in RULES:
                gate=slice_gate(gates[key][rule],start);p=predictions[key][start:]
                fixed=evaluate(td,tl,p,ty,gate,rule,cut)
                models[key][rule]={}
                for policy in EXITS:
                    row=fixed if policy=='fixed' else summarize(td,tl,p,gate,rule,cut,fixed,exit_policy=policy)
                    row['diagnostics']=diagnostics(row,td,predictions[key],gates[key][rule],rule,policy,cut,start,dates,level)
                    models[key][rule][policy]=row
        periods[period]={'dates':td,'level':tl.tolist(),'start_index':start,'models':models}
    selection=shortlist(periods)
    relationships=[]
    for i,a in enumerate(selection['selected']):
        aa=selection['rows'][a]
        for b in selection['selected'][i+1:]:
            bb=selection['rows'][b];pair={}
            for period in periods:
                get=lambda v:periods[period]['models'][v['model']][v['entry']][v['exit']]['sim']
                x,y=get(aa),get(bb)
                delta=np.array(x['daily_pnl'])[1:]-np.array(y['daily_pnl'])[1:]
                ix=bootstrap_indices(len(delta),12,2000,42)
                interval=np.quantile(delta[ix].sum(axis=1),[.025,.975])
                pair[period]={'pnl_correlation':float(np.corrcoef(x['daily_pnl'],y['daily_pnl'])[0,1]),
                              'same_position':float(np.mean(np.array(x['position'])==np.array(y['position']))),
                              'pnl_difference':float(delta.sum()),
                              'difference95':{'low':float(interval[0]),'high':float(interval[1])}}
            relationships.append({'a':a,'b':b,'periods':pair})
    result={'meta':{'asof':dates[-1],'train_through':'2025-12-31','train_n':649,
                    'forecast_n':len(dates),'completed_n':int(np.isfinite(target).sum()),
                    'selected_on':CONFIG['selected_on'],
                    'unique_paths':{p:len({r['diagnostics']['path_sha256'] for m in v['models'].values() for q in m.values() for r in q.values()}) for p,v in periods.items()},
                    'common_start':dates[common],'config':CONFIG,'shared_negative_cut':cut,
                    'models':LABELS,'entries':ENTRY_LABELS,'exits':EXIT_LABELS,
                    'bayes_parameters':{'discount':.995,'initial_covariance_multiplier':1.,'alpha':500.},
                    'protected_hashes':protected,
                    'caveat':'基于反复观察过的2026探索筛选，固定20因子含全历史筛选前视；不计成本、使用次日开盘代理价。不是独立样本外证明，不确认账户持仓或成交。'},
            'prediction_metrics':{**{k:source['models'][k]['metrics'] for k in KEYS[:2]},
                                  'bayes_temporal_new':params['models']['selected']['metrics']},
            'predictions':{k:p.tolist() for k,p in predictions.items()},
            'periods':periods,'selection':selection,'relationships':relationships}
    manifest={'version':1,'selected_on':CONFIG['selected_on'],'research_asof':dates[-1],'research_commit_base':'4937ec8',
              'selection_config':CONFIG['selection'],'source_hashes':{n:sha(n) for n in ('curve_timing_temporal_models.json','bayes_parameter_models.json')},
              'candidates':[{**{k:v for k,v in selection['rows'][key].items() if k not in ('paths','failures')},
                             'role':'主交易参考' if i==0 else '交叉复核'} for i,key in enumerate(selection['selected'])],
              'shared_negative_cut':cut,'horizon':6,'execution':'t收盘观察，t+1开盘代理价=t收盘价；固定±1/0，不自动下单',
              'disagreement':'主交易参考保持自身规则；交叉复核只提示分歧，不追加投票、共识过滤或叠加仓位。',
              'exposure':'纯方向策略预测非零即有方向，历史几乎始终持仓，不等同于强信号才交易。',
              'status':'主跟踪名单；选取日之后才可开始前瞻验证。此文件未切换旧每日自动报告，不是实际持仓或成交。'}
    if (ROOT/MANIFEST).exists() and read(MANIFEST)!=manifest:
        raise ValueError('Candidate selection already frozen; create a new version')
    if protected!={n:sha(n) for n in PROTECTED}:raise ValueError('Prior archives changed')
    json.dumps(result,allow_nan=False)
    write_json_skip_unchanged(str(ROOT/OUTPUT),result)
    write_json_skip_unchanged(str(ROOT/MANIFEST),manifest)
    print(json.dumps({'selected':selection['selected'],'rows':{
        p:{key:{'pnl':r['sim']['metrics']['total'],'dd':r['sim']['metrics']['max_drawdown'],
                'long':r['diagnostics']['side_pnl']['1'],'short':r['diagnostics']['side_pnl']['-1'],
                'recent60':r['diagnostics']['recent']['60']['total'],
                'top3_removed':r['diagnostics']['concentration']['total_after_top3'],
                'pass':selection['rows'][key]['eligible'],'reasons':selection['rows'][key]['failures']}
           for k,m in v['models'].items() for q,x in m.items() for e,r in x.items() for key in [f'{k}|{q}|{e}']}
           for p,v in periods.items()}},ensure_ascii=False))
    return result


if __name__=='__main__':build()
