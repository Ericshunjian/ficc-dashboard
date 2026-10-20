"""Diagnostics for two retained strategies and Bayesian updates, without tuning."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jsonio import write_json_skip_unchanged
from prepare_entry_rules_study import read,sha,entry_gate,slice_gate
from prepare_exit_rules_study import simulate_reverse
from prepare_neutral_exit_study import PROTECTED as NEUTRAL_PROTECTED
from prepare_forecast_comparison import metrics,bootstrap_indices,boot_metrics
from prepare_temporal_study import load_data,temporal_features,matrix,transform_record

ROOT=Path(__file__).resolve().parent
OUTPUT='strategy_stability.json'
PROTECTED=tuple(dict.fromkeys(NEUTRAL_PROTECTED+('neutral_exit_study.json','forecast_model_comparison.json',
    'curve_timing_temporal.json','curve_timing_temporal_models.json')))
CANDIDATES={
    'direction':{'label':'时序Ridge·纯方向·反向退出','rule':'sign','exit':'reverse'},
    'filtered':{'label':'时序Ridge·非对称·中性退出','rule':'asymmetric','exit':'neutral'}}


def quarter(day):
    return day[:4]+'-Q'+str((int(day[5:7])-1)//3+1)


def slice_stats(dates,sim,indices):
    ix=np.asarray(indices,int);pnl=np.asarray(sim['daily_pnl'])[ix];pos=np.asarray(sim['position'])[ix]
    eq=np.cumsum(pnl);dd=eq-np.maximum.accumulate(np.r_[0.,eq])[1:]
    return {'start':dates[ix[0]],'end':dates[ix[-1]],'n':len(ix),'total':float(pnl.sum()),
            'max_drawdown':float(-dd.min()),'long_pnl':float(pnl[pos==1].sum()),
            'short_pnl':float(pnl[pos==-1].sum()),'position_days':int((pos!=0).sum()),
            'first_interval_is_carry':bool(pos[0]!=0),
            'note':'原连续账户每日盯市切片，保留跨期持仓；段内回撤从0累计，不是重新开仓回测。'}


def concentration(row):
    positive=sorted([t for t in row['sim']['trades'] if t['closed'] and t['pnl']>0],key=lambda t:t['pnl'],reverse=True)
    top=positive[:3];total=float(sum(row['sim']['daily_pnl']));gross=sum(t['pnl'] for t in positive)
    amount=sum(t['pnl'] for t in top)
    return {'top_n':len(top),'top3_pnl':amount,'total_after_top3':total-amount,
            'share_of_net':amount/total if total>0 else None,'share_of_gross_wins':amount/gross if gross>0 else None,
            'trades':top,'note':'仅从总盈亏减去最好3笔已平盈利，其他交易及浮动保留；不重跑账户，不代表事前可以剔除这些交易。'}


def start_sensitivity(dates,level,prediction,cut,base,rule,exit_policy):
    gate=entry_gate(prediction,rule,cut);out=[]
    for offset in range(6):
        i=base+offset
        sim=simulate_reverse(dates[i:],level[i:],prediction[i:],slice_gate(gate,i),rule,cut,exit_policy=exit_policy)
        out.append({'offset':offset,'start':dates[i],'metrics':sim['metrics']})
    values=[r['metrics']['total'] for r in out]
    return {'rows':out,'min':min(values),'max':max(values),'median':float(np.median(values)),
            'positive_n':sum(v>0 for v in values),'note':'6个相邻信号日分别空仓重启，保留各自此前排名历史；不挑最好起点，不视为6个独立样本。'}


def strategy_diagnostics(fixed,reverse,neutral):
    dates=fixed['periods']['full']['dates'];level=np.array(fixed['periods']['full']['level'])
    prediction=np.array(fixed['models']['ridge_temporal']['prediction']);cut=fixed['meta']['negative_cuts']['ridge_temporal']
    periods={}
    for period in ('full','common'):
        p=fixed['periods'][period];td=p['dates'];base=p['start_index'];rows={}
        for key,c in CANDIDATES.items():
            row=(reverse if c['exit']=='reverse' else neutral)['periods'][period]['models']['ridge_temporal'][c['rule']]
            sim=row['sim'];n=len(td)
            months={m:slice_stats(td,sim,[i for i,d in enumerate(td) if d.startswith(m)]) for m in sorted({d[:7] for d in td})}
            quarters={q:slice_stats(td,sim,[i for i,d in enumerate(td) if quarter(d)==q]) for q in sorted({quarter(d) for d in td})}
            rows[key]={'label':c['label'],'metrics':sim['metrics'],'trade_stats':row['trade_stats'],
                       'sides':row['sides'],'months':months,'quarters':quarters,
                       'recent':{str(w):slice_stats(td,sim,np.arange(max(0,n-w),n)) for w in (30,60)},
                       'concentration':concentration(row),
                       'starts':start_sensitivity(dates,level,prediction,cut,base,c['rule'],c['exit']),
                       'rolling':{str(w):[None if i+1<w else float(np.sum(sim['daily_pnl'][i-w+1:i+1])) for i in range(n)] for w in (30,60)},
                       'positive_months':sum(v['total']>0 for v in months.values()),'month_n':len(months)}
        a=(reverse)['periods'][period]['models']['ridge_temporal']['sign']['sim']
        b=neutral['periods'][period]['models']['ridge_temporal']['asymmetric']['sim']
        fields=('signal_date','entry_session','exit_signal_date','exit_session','direction','pnl')
        longs=lambda sim:[{k:t[k] for k in fields} for t in sim['trades'] if t['direction']==1]
        relationship={'long_trades_equal':longs(a)==longs(b),'long_trade_n':len(longs(a)),
                      'same_position_fraction':float(np.mean(np.array(a['position'])==np.array(b['position']))),
                      'daily_pnl_correlation':float(np.corrcoef(a['daily_pnl'],b['daily_pnl'])[0,1])}
        periods[period]={'dates':td,'models':rows,'relationship':relationship}
    return periods


def bayes_diagnostics(source,temporal,bundle):
    dates=source['dates'];target=np.array(source['target'],float);valid=np.isfinite(target)
    keys=('ridge_temporal','dlm_temporal');pp={k:np.array(source['models'][k]['prediction']) for k in keys}
    ridge,bayes=pp['ridge_temporal'],pp['dlm_temporal']
    changed=valid&(np.sign(ridge)!=np.sign(bayes))
    helped=changed&(np.sign(bayes)==np.sign(target));hurt=changed&(np.sign(ridge)==np.sign(target))
    change_rows=[{'date':dates[i],'ridge':float(ridge[i]),'bayes':float(bayes[i]),'actual':float(target[i]),
                   'effect':'helped' if helped[i] else 'hurt' if hurt[i] else 'neither'} for i in np.flatnonzero(changed)]
    # Paired blocks preserve dates and six-day dependence; uncertainty is not a
    # post-selection guarantee. This is NOT a new strategy optimization.
    ix=bootstrap_indices(int(valid.sum()),12,2000,42);yy=target[valid][ix]
    boot={k:boot_metrics(p[valid][ix],yy) for k,p in pp.items()}
    intervals={name:{'low':float(np.nanquantile(boot[keys[1]][name]-boot[keys[0]][name],.025)),
                      'high':float(np.nanquantile(boot[keys[1]][name]-boot[keys[0]][name],.975))}
               for name in ('direction_accuracy','rank_ic','mae','rmse')}
    periods={}
    for name,mask in [('all',np.ones(len(dates),bool))]+[(q,np.array([quarter(d)==q for d in dates])) for q in sorted({quarter(d) for d in dates})]:
        complete=mask&valid;indices=np.flatnonzero(complete)
        periods[name]={'n':int(complete.sum()),'last_complete_signal':dates[indices[-1]] if len(indices) else None,
                       'models':{k:metrics(p[complete],target[complete]) for k,p in pp.items()}}
    recent={}
    for w in (30,60):
        indices=np.flatnonzero(valid)[-w:]
        recent[str(w)]={'start':dates[indices[0]],'end':dates[indices[-1]],
                       'models':{k:metrics(p[indices],target[indices]) for k,p in pp.items()}}
    updates=temporal['diagnostics']['dlm']['updates'];coef=np.array(temporal['diagnostics']['dlm']['coefficients'])
    initial=np.array(bundle['models']['dlm_temporal']['mean'])
    record=bundle['models']['ridge_temporal']
    np.testing.assert_allclose(initial,np.r_[record['intercept'],[f['coef'] for f in record['features']]],atol=1e-12)
    # Reconstruct contribution changes using only frozen preprocessing and already
    # archived coefficient paths, without replaying updates or fitting models.
    all_dates,level,_,_,base,cut,_=load_data();features=temporal_features(level,base)
    z=transform_record(record,matrix(features))[cut+1:]
    if all_dates[cut+1:]!=dates:raise ValueError('Coefficient audit calendar mismatch')
    design=np.column_stack([np.ones(len(z)),z])
    np.testing.assert_allclose(design@initial,ridge,atol=1e-10)
    np.testing.assert_allclose(np.sum(design*coef,axis=1),bayes,atol=1e-10)
    contribution=(coef-initial)*design
    blocks=['intercept']+[f['block'] for f in features]
    attribution={block:{'mean':float(contribution[:,np.array(blocks)==block].sum(axis=1).mean()),
                        'mean_absolute':float(np.abs(contribution[:,np.array(blocks)==block].sum(axis=1)).mean())}
                   for block in dict.fromkeys(blocks)}
    last_terms=contribution[-1];names=['截距']+[f['label'] for f in features]
    top=np.argsort(np.abs(last_terms))[-5:][::-1]
    variant=temporal['variants']['dlm_temporal'];lo=np.array(variant['interval80']['low']);hi=np.array(variant['interval80']['high'])
    return {'dates':dates,'ridge_prediction':ridge.tolist(),'bayes_prediction':bayes.tolist(),
            'prediction_delta':(bayes-ridge).tolist(),'coefficient_delta_l2':np.linalg.norm(coef-initial,axis=1).tolist(),
            'config':{'initial_train_labels':bundle['meta']['train_n'],'input_n':len(features),
                      'alpha':bundle['meta']['alpha_temporal'],'discount':bundle['meta']['discount'],
                      'updates':len(updates),'first_update':updates[0]['asof'],'last_update':updates[-1]['asof'],
                      'horizon':6,'validation_discounts':bundle['validation']['discounts']},
            'periods':periods,'recent':recent,'paired95':intervals,
            'sign_changes':{'n':int(changed.sum()),'helped':int(helped.sum()),'hurt':int(hurt.sum()),'rows':change_rows},
            'attribution':attribution,'latest_top_contributions':[{'label':names[i],'contribution':float(last_terms[i])} for i in top],
            'prediction_correlation':float(np.corrcoef(ridge,bayes)[0,1]),
            'interval80_contains_zero_n':int(np.sum((lo<=0)&(hi>=0))),
            'interval_n':len(lo),'updates':updates,
            'last_prediction':{'date':dates[-1],'ridge':float(ridge[-1]),'bayes':float(bayes[-1]),
                               'actual_complete':bool(valid[-1])},
            'notes':['discount=1仍更新后验，但没有额外状态漂移或主动遗忘旧样本；该值由2025内部验证选择，不因2026表现更换。',
                     '同一92输入及相同初始均值，纯方向开仓和平仓的门槛一致，可作为在线更新效果对照；非对称模型各自c不同，收益差不能全部归因模型。',
                     f'{len(updates)}次更新不表示只训练{len(updates)}条，初始{bundle["meta"]["train_n"]}标签仍在起作用。更新仅用六日后已兑现、间隔六日的标签，不把重叠标签当独立新增。',
                     '预测置信区间和概率是模型内估计；当前交易只用预测均值/排名，没有用不确定性过滤，不能推断加过滤一定改善。',
                     '贡献分解是同一标准化输入上系数变化的代数分解，非经济因果；滞后项共线时不解释为资金权重。']}


def build():
    protected={n:sha(n) for n in PROTECTED}
    fixed,reverse,neutral=read('entry_rules_study.json'),read('exit_rules_study.json'),read('neutral_exit_study.json')
    source,temporal,bundle=read('forecast_model_comparison.json'),read('curve_timing_temporal.json'),read('curve_timing_temporal_models.json')
    if len({v['meta']['asof'] for v in (fixed,reverse,neutral,source,temporal)})!=1:
        raise ValueError('Research snapshots differ')
    periods=strategy_diagnostics(fixed,reverse,neutral);bayes=bayes_diagnostics(source,temporal,bundle)
    bayes['pure_strategy_control']={p:{k:reverse['periods'][p]['models'][k]['sign']['sim']['metrics']
                                     for k in ('ridge_temporal','dlm_temporal')} for p in ('full','common')}
    bayes['negative_cuts']={k:fixed['meta']['negative_cuts'][k] for k in ('ridge_temporal','dlm_temporal')}
    out={'meta':{'asof':source['meta']['asof'],'train_through':'2025-12-31','last_complete_signal':source['dates'][np.flatnonzero(np.isfinite(np.array(source['target'],float)))[-1]],
                  'candidates':CANDIDATES,'protected_hashes':protected,
                  'reference':{'title':'Mike West, Bayesian Dynamic Modelling','url':'https://www2.stat.duke.edu/~mw/MWextrapubs/WestAFMSbook2012.pdf'},
                  'caveat':'只做既定两候选的描述性稳定性检查，不搜索新门槛，不挑最好起点，不自动确认胜出或替换生产。2026已反复观察，固定因子含前视；训练期不用于美化稳定性。'},
         'periods':periods,'bayes':bayes}
    if protected!={n:sha(n) for n in PROTECTED}:raise ValueError('Frozen artifacts changed')
    json.dumps(out,allow_nan=False);write_json_skip_unchanged(str(ROOT/OUTPUT),out)
    print(json.dumps({'strategies':{p:{k:{'total':r['metrics']['total'],'after_top3':r['concentration']['total_after_top3'],
                                           'starts_min':r['starts']['min'],'starts_max':r['starts']['max'],'sides':r['sides'],
                                           'recent':r['recent']} for k,r in v['models'].items()} for p,v in periods.items()},
                      'bayes':{'sign_changes':{k:v for k,v in bayes['sign_changes'].items() if k!='rows'},'paired95':bayes['paired95'],
                               'zero_intervals':bayes['interval80_contains_zero_n'],'attribution':bayes['attribution']}},ensure_ascii=False))
    return out


if __name__=='__main__':build()
