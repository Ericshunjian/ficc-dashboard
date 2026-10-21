"""Daily review of the selected NEW Bayesian model and frozen Ridge controls.

Append-only forecasts; Bayesian state uses archived inputs and realized prices.
Probability is posterior predictive, not a percentile or a new trading gate.
"""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
from datetime import datetime,timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
from scipy.special import ndtr

from jsonio import write_json_skip_unchanged
from prepare_bayes_study import dynamic_forecast
from prepare_direction_screen import baseline_features,load_candidates,raw_numbers
from prepare_factor_library import build as build_library
from prepare_strategy_model import make_offset_labels
from prepare_temporal_study import temporal_features,matrix,transform_record
from prepare_three_model_review import PROTECTED as REVIEW_PROTECTED,KEYS,LABELS
from prepare_entry_rules_study import read,sha,entry_gate
from prepare_exit_rules_study import simulate_reverse

ROOT=Path(__file__).resolve().parent
OUTPUT='curve_signal_daily.json'
DEPLOYMENT='2026-10-21'
Z80=1.2815515655446004
MODEL_FILES=('curve_timing_temporal_models.json','bayes_parameter_models.json','curve_timing_live_candidates.json')
PROTECTED=tuple(dict.fromkeys(tuple(n for n in REVIEW_PROTECTED if n!='factor_library.json')+
                            ('three_model_strategy_review.json',)+MODEL_FILES))


def probability(mean,std):
    if not np.isfinite(mean) or not np.isfinite(std) or std<=0:
        raise ValueError('Predictive mean and positive standard deviation required')
    p=float(ndtr(mean/std))
    return {'up_probability':p,'down_probability':1-p,'predictive_std':float(std),
            'interval80':[float(mean-Z80*std),float(mean+Z80*std)]}


def probability_check(p,y,means,stds):
    p,y,means,stds=map(lambda a:np.asarray(a,float),(p,y,means,stds))
    valid=np.isfinite(y)&np.isfinite(p);pp=p[valid];yy=(y[valid]>0).astype(float)
    if not len(pp):return {'n':0,'bins':[]}
    bins=[]
    for j in range(10):
        mask=(pp>=j/10)&((pp<(j+1)/10) if j<9 else (pp<=1))
        bins.append({'low':j/10,'high':(j+1)/10,'n':int(mask.sum()),
                     'mean_probability':float(pp[mask].mean()) if mask.any() else None,
                     'actual_up_rate':float(yy[mask].mean()) if mask.any() else None})
    clipped=np.clip(pp,1e-12,1-1e-12)
    return {'n':len(pp),'brier':float(np.mean((pp-yy)**2)),'coin_brier':.25,
            'log_loss':float(-np.mean(yy*np.log(clipped)+(1-yy)*np.log(1-clipped))),
            'coverage80':float(np.mean(np.abs(y[valid]-means[valid])<=Z80*stds[valid])),
            'mean_probability':float(pp.mean()),'actual_up_rate':float(yy.mean()),'bins':bins,
            'note':'仅已兑现6日标签；固定10个百分点分档，不据2026重新校准概率或改交易门槛。标签重叠，分档频率不是独立样本显著性检验。'}


def input_data(max_date):
    lib=build_library(prefer_compact=True)
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'factor_library.json';path.write_text(json.dumps(lib,ensure_ascii=False),encoding='utf8')
        with patch('prepare_factor_checkup.LIB_PATH',str(path)):
            _,dates,curves,candidates=load_candidates()
    end=max(i for i,d in enumerate(dates) if d<=max_date)
    dates=dates[:end+1];curves={k:v[:end+1] for k,v in curves.items()}
    level=3*curves['T主力']-curves['TL主力']
    base,_=baseline_features(level,curves,dates)
    cmap={f['id']:f for f in candidates+base}
    model=read('curve_timing_temporal_models.json')['models']['ridge_temporal']
    originals=[dict(cmap[f['id']],values=cmap[f['id']]['values'][:end+1]) for f in model['features'][:28]]
    fs=temporal_features(level,originals)
    if [f['id'] for f in fs]!=[f['id'] for f in model['features']]:raise ValueError('Frozen feature order changed')
    x=matrix(fs)
    return lib,dates,level,x


def row(day,price,raw,z,predictions,info,origin,now,sd):
    models={k:{'prediction':float(v),'signal':int(np.sign(v))} for k,v in predictions.items()}
    models['bayes_temporal_new'].update(probability(models['bayes_temporal_new']['prediction'],sd))
    return {'date':day,'level':float(price),'origin':origin,'recorded_at':now,
            'raw_inputs':raw_numbers(raw),'bayes_z':z.tolist(),'source_hashes':info,'models':models}


def posterior(out,dates,noise):
    return {'mean':out['final_mean'].tolist(),'covariance':out['final_covariance'].tolist(),
            'state_origin':dates[out['state_time']],'noise_variance':noise,'discount':.995}


def seed(dates,level,x,z,bundle,frozen,now,info):
    source=read('three_model_strategy_review.json');params=read('bayes_parameter_study.json')
    sd=params['grid'][params['meta']['selected']]['std'];td=source['periods']['full']['dates']
    start=dates.index(td[0]);stop=dates.index(td[-1])+1
    if dates[start:stop]!=td:raise ValueError('Research seed calendar changed')
    np.testing.assert_allclose(level[start:stop],source['periods']['full']['level'],atol=1e-10,rtol=0)
    state=frozen['state'];last=dates.index(bundle['meta']['last_train_signal'])
    out=dynamic_forecast(z[:stop],make_offset_labels(level[:stop],6),start,last,np.array(state['mean']),
                         np.array(state['covariance']),state['noise_variance'],.995,dates[:stop])
    np.testing.assert_allclose(out['mean'],source['predictions']['bayes_temporal_new'],atol=1e-11,rtol=0)
    np.testing.assert_allclose(out['std'],sd,atol=1e-11,rtol=0)
    for key,xx in [('ridge_original',x[:,:28]),('ridge_temporal',x)]:
        rec=bundle['models'][key]
        pred=rec['intercept']+transform_record(rec,xx)@np.array([f['coef'] for f in rec['features']])
        np.testing.assert_allclose(pred[start:stop],source['predictions'][key],atol=1e-11,rtol=0)
    records=[row(d,level[start+i],x[start+i],z[start+i],{k:source['predictions'][k][i] for k in KEYS},
                 info,'research-backfill',now,sd[i]) for i,d in enumerate(td)]
    return records,posterior(out,dates,state['noise_variance']),out['updates']


def advance(records,state,updates,dates,level,x,z,bundle,available,now,info):
    """Resume from the saved posterior; never replay revised old inputs/prices."""
    first=dates.index(records[0]['date']);end=dates.index(records[-1]['date'])
    if dates[first:end+1]!=[r['date'] for r in records]:raise ValueError('Archived futures calendar changed')
    price=np.array(level,copy=True);zz=np.array(z,copy=True);revisions=[]
    for r in records:
        i=dates.index(r['date'])
        if not np.isfinite(price[i]) or abs(price[i]-r['level'])>1e-8:
            revisions.append({'date':r['date'],'archived':r['level'],'current':float(price[i]) if np.isfinite(price[i]) else None})
        price[i]=r['level'];zz[i]=r['bayes_z']
    new=[];issue=None
    for t in range(end+1,len(dates)):
        day=dates[t];problems=[]
        if not np.isfinite(level[t]):problems.append('T或TL收盘价缺失')
        if not np.isfinite(x[t,:8]).all():problems.append('市场基准不完整')
        if np.isfinite(x[t,8:28]).mean()<.9 or np.isfinite(x[t]).mean()<.9:problems.append('固定因子覆盖不足90%')
        if day not in available['bond']:problems.append('机构日报缺失')
        if day not in available['repo']:problems.append('回购日报缺失')
        if available['policy_verified']<day:problems.append('7天OMO尚未核验至信号日')
        if problems:issue={'date':day,'problems':problems};break
        origin=dates.index(state['state_origin'])
        out=dynamic_forecast(zz[:t+1],make_offset_labels(price[:t+1],6),t,origin,np.array(state['mean']),
                             np.array(state['covariance']),state['noise_variance'],.995,dates[:t+1])
        predictions={'bayes_temporal_new':float(out['mean'][0])}
        for key,xx in [('ridge_original',x[t:t+1,:28]),('ridge_temporal',x[t:t+1])]:
            rec=bundle['models'][key]
            predictions[key]=float(rec['intercept']+transform_record(rec,xx)[0]@np.array([f['coef'] for f in rec['features']]))
        r=row(day,level[t],x[t],zz[t],predictions,info,'daily-review' if day>=DEPLOYMENT else 'deployment-backfill',now,out['std'][0])
        records.append(r);new.append(day);state=posterior(out,dates,state['noise_variance']);updates.extend(out['updates'])
    return records,state,updates,new,issue,revisions


def simulations(records):
    td=[r['date'] for r in records];price=np.array([r['level'] for r in records]);models={}
    history=read('entry_rules_history.json')['periods']['history_train'];hd=history['dates']+td;hp=np.r_[history['level'],price]
    for key in KEYS:
        pred=np.array([r['models'][key]['prediction'] for r in records]);gate=entry_gate(pred,'sign',.16956145865272215)
        test=simulate_reverse(td,price,pred,gate,'sign',.16956145865272215,exit_policy='reverse')
        hpred=np.r_[history['prediction']['ridge_temporal' if key=='bayes_temporal_new' else key],pred]
        back=simulate_reverse(hd,hp,hpred,entry_gate(hpred,'sign',.16956145865272215),'sign',.16956145865272215,exit_policy='reverse')
        models[key]={'label':LABELS[key],'ytd':test,'history':back}
    return {'dates':td,'history_dates':hd,'history_train_n':len(history['dates']),'models':models}


def build(max_date=None,now=None):
    now=now or datetime.now(ZoneInfo('Asia/Shanghai'));stamp=now.isoformat(timespec='seconds')
    max_date=max_date or (now.date()-timedelta(days=1)).isoformat()
    if max_date>=now.date().isoformat():raise ValueError('Do not publish an unfinished current-day close')
    hashes={n:sha(n) for n in MODEL_FILES};protected={n:sha(n) for n in PROTECTED}
    frozen,bundle=read(MODEL_FILES[1]),read(MODEL_FILES[0])
    if frozen['selected']!='d0.995_c1' or frozen['discount']!=.995:raise ValueError('New Bayesian model identity differs')
    for n,digest in read(MODEL_FILES[2])['source_hashes'].items():
        if sha(n)!=digest:raise ValueError('Selected frozen model changed: '+n)
    lib,dates,level,x=input_data(max_date);z=transform_record(bundle['models']['ridge_temporal'],x)
    source_names=('bond_trading_data.json','repo_trading_data.json','yield_curve_data.json','omo_policy_7d.json')
    info={n:sha(n) for n in source_names}
    bond,repo,curve,policy=(read(n) for n in source_names)
    available={'bond':{bond['idx']['dates'][r[0]] for r in bond['detail']},'repo':set(repo['dates']),
               'policy_verified':policy['verified_through']}
    old=read(OUTPUT) if (ROOT/OUTPUT).exists() else None
    if old and old['meta']['frozen_hashes']!=hashes:raise ValueError('Cannot append a different model to old predictions')
    if old:
        records,state,updates=copy.deepcopy((old['records'],old['posterior'],old['updates']))
    else:records,state,updates=seed(dates,level,x,z,bundle,frozen,stamp,info)
    records,state,updates,new,issue,revisions=advance(records,state,updates,dates,level,x,z,bundle,available,stamp,info)
    newest=max(d for name in ('T主力','TL主力') for d in curve['series'][name]['dates'] if d<=max_date)
    if newest>records[-1]['date'] and not issue:issue={'date':newest,'problems':['较新期货日期缺少完整输入，不发布新信号']}
    if policy['verified_through']<records[-1]['date']:issue={'date':records[-1]['date'],'problems':['政策利率核验不足，当前信号暂停使用']}
    sims=simulations(records);target=make_offset_labels(np.array([r['level'] for r in records]),6)
    bayes=[r['models']['bayes_temporal_new'] for r in records]
    check=probability_check([b['up_probability'] for b in bayes],target,[b['prediction'] for b in bayes],[b['predictive_std'] for b in bayes])
    report={'meta':{'generated':stamp,'asof':records[-1]['date'],'deployment_date':DEPLOYMENT,
                    'healthy':issue is None,'issue':issue,'latest_source_date':newest,'new_dates':new,
                    'frozen_hashes':hashes,'model_version':'new-temporal-bayes-d0995-c1-alpha500',
                    'primary':'bayes_temporal_new','reference':'ridge_temporal',
                    'probability':'P(S[t+6]−S[t]>0)=Phi(mean/sqrt(noise+x·C·x)); Gaussian empirical Bayes posterior predictive, uncalibrated',
                    'price_revisions':revisions,'research_until':'2026-10-19',
                    'forward_record_n':sum(r['origin']=='daily-review' for r in records),
                    'caveat':'2023—2025为训练拟合回放；2026反复观察且因子筛选含前视；概率为模型假设下未校准值，不是保证。中午复核、不确认账户成交，不自动下单。'},
            'latest':records[-1],'records':records,'posterior':state,'updates':updates,
            'simulation':sims,'probability_check':check}
    if protected!={n:sha(n) for n in PROTECTED}:raise ValueError('Prior models or research archives changed')
    json.dumps(report,allow_nan=False)
    library_changed=write_json_skip_unchanged(str(ROOT/'factor_library.json'),lib)
    # A transient new_dates list is returned separately; do not create a second
    # timestamp-only archive rewrite on the next run without new observations.
    report['meta'].pop('new_dates')
    changed=write_json_skip_unchanged(str(ROOT/OUTPUT),report)
    versions=read('data_version.json')
    for name in ('factor_library.json','omo_policy_7d.json',OUTPUT):
        obj=read(name);meta=obj.get('meta',{})
        fallback=datetime.fromtimestamp((ROOT/name).stat().st_mtime,ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        versions['files'][name]={'last_updated':meta.get('generated') or meta.get('last_updated') or fallback,
                                'size':(ROOT/name).stat().st_size}
    write_json_skip_unchanged(str(ROOT/'data_version.json'),versions)
    print(json.dumps({'asof':report['meta']['asof'],'healthy':report['meta']['healthy'],'issue':issue,
                      'new_dates':new,'changed':changed,'library_changed':library_changed,
                      'bayes':report['latest']['models']['bayes_temporal_new'],'probability_check':check},ensure_ascii=False))
    return report,{'new_dates':new,'changed':changed,'library_changed':library_changed}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--max-date')
    build(parser.parse_args().max_date)
