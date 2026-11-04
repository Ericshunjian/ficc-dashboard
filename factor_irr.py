"""Main-contract IRR relative-value features, on the futures-session calendar."""
import math

T_NAME='10年期货IRR'
TL_NAME='30年期货IRR'
DIFF='T−TL主力IRR差'
MA5=DIFF+'·MA5'
CHANGE5=DIFF+'·5日变化'
EXTRA_NAMES=(DIFF,MA5,CHANGE5)
FOCUS_NAMES=(T_NAME,TL_NAME)+EXTRA_NAMES
CATEGORY='IRR相对定价'


def mapping(data,name):
    s=data['series'].get(name,{})
    if isinstance(s,dict):return dict(zip(s.get('dates',[]),s.get('values',[])))
    return dict(zip(data.get('dates',[]),s))


def finite(v):
    return isinstance(v,(int,float)) and math.isfinite(v)


def build(curve,library_dates):
    t,tl,price,tl_price=(mapping(curve,n) for n in (T_NAME,TL_NAME,'T主力','TL主力'))
    sessions=sorted(d for d,v in price.items() if finite(v) and v>0)
    diff=[t[d]-tl[d] if finite(t.get(d)) and finite(tl.get(d)) and finite(tl_price.get(d)) and tl_price[d]>0
          else None for d in sessions]
    ma=[sum(diff[i-4:i+1])/5 if i>=4 and all(finite(v) for v in diff[i-4:i+1]) else None
        for i in range(len(diff))]
    change=[diff[i]-diff[i-5] if i>=5 and finite(diff[i]) and finite(diff[i-5]) else None
            for i in range(len(diff))]
    definitions=[(DIFF,diff,'IRR_T[t]−IRR_TL[t]；主力合约原始年化百分数相减，不乘3'),
                 (MA5,ma,'过去5个期货交易日IRR差的算术均值，含当日，5日必须完整'),
                 (CHANGE5,change,'IRR差[t]−IRR差[t−5]；按期货交易日滞后，两端必须有效')]
    index={d:i for i,d in enumerate(library_dates)};rows=[]
    for name,values,formula in definitions:
        arr=[(index[d],v) for d,v in zip(sessions,values) if d in index and finite(v)]
        if not arr:continue
        first,last=arr[0][0],arr[-1][0];v=[None]*(last-first+1)
        for i,value in arr:v[i-first]=round(value,4)
        rows.append({'name':name,'cat':CATEGORY,'unit':'百分点','formula':formula,
                     'source_fields':[T_NAME,TL_NAME],'i0':first,'v':v,
                     'note':'0.01个百分点=1bp；缺失不补零，不用后复权价格重新计算IRR'})
    return rows
