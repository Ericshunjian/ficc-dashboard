"""IRR additions and a separate, six-day, full-pool screening audit.

Does not replace any frozen factor set, production model, or old screening archive.
"""
from pathlib import Path
import json
import tempfile
from unittest.mock import patch
import numpy as np
from scipy.stats import rankdata,t as student,spearmanr

from prepare_factor_library import build as build_library
from prepare_direction_screen import load_candidates,screen_snapshot,RULES
from factor_irr import FOCUS_NAMES,DIFF
from jsonio import write_json_skip_unchanged

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'irr_factor_checkup.json'


def digest(p):
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rank_hac(x,y):
    """Rank-OLS slope t with calendar-preserving Bartlett HAC; IC is unchanged."""
    x,y=np.asarray(x,float),np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y);n=int(ok.sum())
    if n<30:return {'n':n,'lag':None,'t':None,'p':None}
    a,b=rankdata(x[ok]),rankdata(y[ok]);a=a-a.mean();b=b-b.mean();sxx=float(a@a)
    if sxx<=0:return {'n':n,'lag':None,'t':None,'p':None}
    beta=float(a@b/sxx);u=np.zeros(len(x));u[ok]=a*(b-beta*a)
    lag=max(5,int(np.floor(4*(n/100)**(2/9))));v=float(u@u)
    for k in range(1,lag+1):v+=2*(1-k/(lag+1))*float(u[k:]@u[:-k])
    v*=n/(n-2)
    if v<=0:return {'n':n,'lag':lag,'t':None,'p':None}
    stat=beta/(np.sqrt(v)/sxx)
    return {'n':n,'lag':lag,'t':float(stat),'p':float(2*student.sf(abs(stat),n-2))}


def bootstrap_ic(x,y,draws=500,block=12):
    rng=np.random.default_rng(42);n=len(y);out=[]
    starts=rng.integers(0,n-block+1,size=(draws,int(np.ceil(n/block))))
    for s in starts:
        ix=(s[:,None]+np.arange(block)).reshape(-1)[:n];a=x[ix];b=y[ix]
        good=np.isfinite(a)&np.isfinite(b)
        if good.sum()>30 and np.std(a[good])>0:out.append(spearmanr(a[good],b[good]).statistic)
    return np.quantile(out,[.025,.975]).tolist() if out else [None,None]


def fdr_adjust(values):
    p=np.asarray(values,float);out=np.full(len(p),np.nan);ok=np.flatnonzero(np.isfinite(p))
    order=ok[np.argsort(p[ok])];n=len(order)
    if n:out[order]=np.minimum(1,np.minimum.accumulate((p[order]*n/np.arange(1,n+1))[::-1])[::-1])
    return out


def quality(dates,candidates):
    rows=[]
    for c in candidates:
        if c['id'] not in {'curve|'+n for n in FOCUS_NAMES}:continue
        x=np.asarray(c['values'],float);ix=np.flatnonzero(np.isfinite(x));j=np.diff(x)
        top=np.flatnonzero(np.isfinite(j));top=sorted(top,key=lambda i:abs(j[i]),reverse=True)[:8]
        rows.append({'id':c['id'],'first':dates[ix[0]],'last':dates[ix[-1]],'n':len(ix),
          'min':float(np.nanmin(x)),'max':float(np.nanmax(x)),
          'latest':float(x[ix[-1]]),'largest_changes':[{'date':dates[i+1],'previous':float(x[i]),
            'current':float(x[i+1]),'change':float(j[i])} for i in top]})
    return rows


def audit_window(dates,level,candidates,asof):
    end=max(i for i,d in enumerate(dates) if d<=asof)
    result,target=screen_snapshot(dates,level,candidates,end)
    begin=np.flatnonzero(np.isfinite(level))[0];ix=np.arange(begin,end-5)
    yy=target[ix];rows={r['id']:r for r in result['rows']};focus={}
    stats={c['id']:rank_hac(c['values'][ix],yy) for c in candidates}
    adjusted=fdr_adjust(np.array([s['p'] if s['p'] is not None else np.nan for s in stats.values()]))
    fdr=dict(zip(stats,adjusted))
    source=next(c for c in candidates if c['id']=='curve|'+DIFF)['values'][ix]
    step=np.r_[np.nan,np.diff(source)];good=np.isfinite(step)
    bound=float(np.quantile(np.abs(step[good]),.99));bad=np.flatnonzero(good&(np.abs(step)>bound))
    safe=np.ones(len(ix),bool)
    for j in bad:safe[max(0,j-6):min(len(ix),j+6)]=False
    for c in candidates:
        if c['id'] not in {'curve|'+n for n in FOCUS_NAMES}:continue
        key=c['id'];x=c['values'][ix];row=dict(rows[key]);s=stats[key]
        yearly=[]
        for year in sorted(set(d[:4] for d in np.array(dates)[ix])):
            ok=np.array([dates[i].startswith(year) for i in ix])&np.isfinite(x)&np.isfinite(yy)
            yearly.append({'year':year,'n':int(ok.sum()),'ic':float(spearmanr(x[ok],yy[ok]).statistic) if ok.sum()>=30 else None})
        mask=safe&np.isfinite(x)&np.isfinite(yy)
        row.update(hac=s,fdr_p=float(fdr[key]) if np.isfinite(fdr[key]) else None,
          ic_ci95=bootstrap_ic(x,yy),yearly=yearly,
          jump_sensitivity={'retained_n':int(mask.sum()),
            'ic':float(spearmanr(x[mask],yy[mask]).statistic) if mask.sum()>=30 else None,
            'jump_threshold':bound,'jump_dates':[dates[ix[i]] for i in bad]},
          latest=float(c['values'][end]) if np.isfinite(c['values'][end]) else None)
        focus[key]=row
    return {'asof':dates[end],'window':result['window'],'summary':result['summary'],
            'focus':[focus['curve|'+n] for n in FOCUS_NAMES],
            'top20':[rows[k] for k in result['selected_ids']],
            'full_pool_n':len(candidates),
            'note':'正常筛选规则不加入HAC/FDR硬门槛；显著性另列，不据此自动改旧名单'}


def build():
    allowed={'factor_library.json',OUT.name,'data_version.json'}
    protected={p.name:digest(p) for p in ROOT.glob('*.json') if p.name not in allowed}
    library=build_library(prefer_compact=True)
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'library.json';p.write_text(json.dumps(library,ensure_ascii=False),encoding='utf8')
        with patch('prepare_factor_checkup.LIB_PATH',str(p)):
            _,dates,curves,candidates=load_candidates()
    level=3*curves['T主力']-curves['TL主力']
    latest=dates[-1]
    windows={k:audit_window(dates,level,candidates,cut) for k,cut in
             [('latest',latest),('train2025','2025-12-31'),('train202606','2026-06-30')]}
    audit_path=ROOT/'output'/'bayes'/'irr_source_audit.json'
    source_audit=json.loads(audit_path.read_text(encoding='utf8')) if audit_path.exists() else None
    if source_audit:
        source_audit.pop('excel_path',None)
        source_audit['excel_file']='FICC原始数据（衍生品、收益率曲线）.xlsx'
    report={'meta':{'asof':latest,'target':'S[t+6]−S[t]，S=3T−TL','rules':RULES,
        'source':'yield_curve_data.json：用户确认主力合约IRR，Excel期货页H/I列',
        'source_validation':source_audit,
        'focus_definitions':[{k:c[k] for k in ('id','label','formula','unit')} for c in candidates
                             if c['id'] in {'curve|'+n for n in FOCUS_NAMES}],
        'hac':'Spearman IC保持原值；另以配对秩OLS及Bartlett HAC给出近似t/p，滞后至少5，缺日保留日历轴',
        'fdr':'BH调整覆盖同一窗口全部可检验因子，不新增为原筛选硬门槛',
        'bootstrap':'500次12交易日移动区块重抽样，seed42，IC区间是探索性估计',
        'jump_sensitivity':'按窗口IRR差绝对日变化的99%分位标记跳点，剔除前6后5交易日复算IC；仅事后诊断，不作为因子清洗或交易规则',
        'roll_limit':'目前缺合约代码、CTD代码及交割天数；最大跳点不能直接认定为换月或CTD切换',
        'frozen':'旧direction_screen快照、20因子名单、8基准及全部策略模型均保留；本报告不自动入模',
        'protected_sha256':protected},'windows':windows,'quality':quality(dates,candidates)}
    if protected!={p.name:digest(p) for p in ROOT.glob('*.json') if p.name not in allowed}:
        raise AssertionError('Frozen model or archive changed')
    write_json_skip_unchanged(str(ROOT/'factor_library.json'),library)
    write_json_skip_unchanged(str(OUT),report)
    version_path=ROOT/'data_version.json'
    if version_path.exists():
        from datetime import datetime
        from zoneinfo import ZoneInfo
        versions=json.loads(version_path.read_text(encoding='utf8'))
        for name in ('factor_library.json',OUT.name):
            path=ROOT/name;previous=versions['files'].get(name,{})
            if previous.get('size')!=path.stat().st_size:
                versions['files'][name]={'last_updated':datetime.fromtimestamp(path.stat().st_mtime,ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S'),'size':path.stat().st_size}
        write_json_skip_unchanged(str(version_path),versions)
    print('IRR_SCREEN',json.dumps({k:[{'factor':r['label'],'IC':r['ic'],'ICIR':r['icir'],
          'status':r['status'],'reasons':r['reasons']} for r in v['focus']] for k,v in windows.items()},ensure_ascii=False))
    return report


if __name__=='__main__':build()
