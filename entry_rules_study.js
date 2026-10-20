(()=>{
'use strict';
const el=id=>document.getElementById('entry-'+id),keys=['ridge_original','ridge_temporal','dlm_temporal'],rules=['sign','asymmetric','rank'],colors=['#5470c6','#16847b','#c78b42'],charts={};let data,history;
const fmt=(v,n=4)=>v==null||!Number.isFinite(v)?'—':v.toFixed(n),pct=v=>v==null?'—':fmt(v*100,1)+'%',side=v=>v===1?'多':v===-1?'空':'空仓';
function graph(id,dates,series,bounds){
 if(charts[id])charts[id].destroy();
 const cut=dates.findIndex(d=>d>'2025-12-31');
 const marker={id:'entryTrainBoundary',afterDraw(c){if(cut<=0)return;const x=c.scales.x.getPixelForValue(cut),a=c.chartArea;if(x<a.left||x>a.right)return;const ctx=c.ctx;ctx.save();ctx.strokeStyle='#8c99a9';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(x,a.top);ctx.lineTo(x,a.bottom);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#68798b';ctx.font='10px Microsoft YaHei';ctx.fillText('2026开始',Math.min(x+4,a.right-48),a.top+12);ctx.restore();}};
 charts[id]=new Chart(el(id),{type:'line',plugins:[marker],data:{labels:dates,datasets:series.map((s,i)=>({label:s.label,data:s.data,borderColor:s.color||colors[i],backgroundColor:s.color||colors[i],borderDash:s.dash||[],borderWidth:1.7,pointRadius:0,stepped:s.stepped||false,spanGaps:false}))},options:{responsive:true,maintainAspectRatio:false,animation:false,interaction:{mode:'index',intersect:false},plugins:{legend:{position:'bottom',labels:{boxWidth:12,font:{size:10}}},zoom:{pan:{enabled:true,mode:'x'},zoom:{mode:'x',wheel:{enabled:true},pinch:{enabled:true}}}},scales:{x:{grid:{display:false},ticks:{maxRotation:0,autoSkip:false,font:{size:10},callback:function(v){const a=Math.ceil(this.min),b=Math.floor(this.max),n=this.width<450?3:5,keep=Array.from({length:n},(_,i)=>Math.round(a+(b-a)*i/(n-1)));return keep.includes(Number(v))?this.getLabelForValue(v):'';}}},y:bounds||{}}}});
 el(id).ondblclick=()=>charts[id].resetZoom();
}
function render(){
 if(!data)return;
 const period=el('period').value,key=el('model').value,rule=el('rule').value,isHistory=period.startsWith('history_'),p=(isHistory?history:data).periods[period],models=p.models,m=models[key],r=m[rule],dates=p.dates;
 const prediction=p.prediction?.[key]||data.models[key].prediction.slice(p.start_index),names=data.meta.rules;
 el('state').textContent=isHistory?`全历史拟合回放：${dates[0]}至${dates.at(-1)}，${dates.length}个价格日期。模型初始训练截至2025-12-31；此前是样本内回看，2026沿用原已归档预测。不是从2023年起真实发布的信号，也不是全期样本外业绩。`:`研究快照截至${data.meta.asof}；初始训练截至${data.meta.train_through}，2026年共${data.meta.forecast_n}条预测、${data.meta.completed_n}条六日标签完成。沿用既定预测，贝叶斯保留原有延迟递推。`;
 el('state').classList.toggle('warning',isHistory);
 el('period-note').textContent=`${dates[0]}至${dates.at(-1)}，共${dates.length}个收盘日期。`+(isHistory?p.note:period==='common'?data.meta.common_note:data.meta.full_note);
 el('download').href=isHistory?'entry_rules_history.json':'entry_rules_study.json';
 el('rank-note').textContent=isHistory?'仅排名规则使用；不含当日，从2023年起先等待60条评分。2026继承样本内回溯排名，不是当时发布的预测历史。':'仅排名规则使用；不含当日，2026冷启动，先等待60条。';
 for(const id of ['history-note','stage-panel','year-panel'])el(id).hidden=!isHistory;
 if(isHistory){
  el('history-note').textContent=history.meta.bayes_backcast+' 两者早期预测相同，但过滤规则的做空边界各自不同，仍可能产生不同交易。'+history.meta.preprocessing+' '+history.meta.ranking;
  el('stages').innerHTML=keys.flatMap(k=>rules.map(q=>{const v=models[k][q];return `<tr><td>${data.models[k].label}</td><td>${names[q]}</td><td class="num">${fmt(v.stages.train?.total)}</td><td class="num">${fmt(v.stages.test?.total)}</td><td class="num">${fmt(v.sim.metrics.total)}</td></tr>`;})).join('');
  el('years').innerHTML=Object.keys(r.years).map(year=>`<tr><td>${year}${year<'2026'?'（样本内）':'（继承账户）'}</td>${rules.map(q=>`<td class="num">${fmt(m[q].years[year].total)}</td>`).join('')}<td class="num">${fmt(r.years[year].long_pnl)}</td><td class="num">${fmt(r.years[year].short_pnl)}</td></tr>`).join('');
 }
 el('comparison').innerHTML=keys.flatMap(k=>rules.map(q=>{const v=models[k][q],s=v.sim.metrics,t=v.trade_stats;return `<tr${k===key&&q===rule?' class="selected"':''}><td>${data.models[k].label}</td><td>${names[q]}</td><td class="num">${fmt(s.total)}</td><td class="num">${fmt(s.max_drawdown)}</td><td class="num">${t.closed_n}／${t.open_n}</td><td class="num">${pct(t.win_rate)}</td><td class="num">${fmt(t.mean_pnl)}</td><td class="num">${fmt(t.payoff_ratio,2)}</td><td class="num">${fmt(t.floating_pnl)}</td><td class="num">${pct(s.exposure)}</td></tr>`;})).join('');
 const a=data.periods.full.models.ridge_temporal,b=data.periods.common.models.ridge_temporal,df=a.asymmetric.sim.metrics.total-a.sign.sim.metrics.total,dc=b.asymmetric.sim.metrics.total-b.sign.sim.metrics.total;
 el('finding').textContent=isHistory?'此处总盈利混合了2023—2025的训练期拟合效果，不能据此选模型。请对照下表的阶段拆分，并切换“完整2026”查看独立重启账户。全历史2026段与旧测试不同，来自持仓节奏和排名历史衔接，预测本身没有改变。':`2026重启账户观察：时序Ridge过滤相对纯方向，完整2026改善${fmt(df)}点，共同起点改善${fmt(dc)}点。原Ridge过滤在两种起点都弱于纯方向；时序贝叶斯过滤在完整2026落后、共同起点领先，起点与节奏影响较大。当前9组中，40／60排名的固定六日回放没有占优；不能据此推断所有排名或动态离场策略都无效。`;
 for(const field of ['equity','drawdown'])graph(field,dates,rules.map((q,i)=>({label:names[q],data:m[q].sim[field],color:colors[i]})));
 graph('position',dates,[{label:'收盘新开仓信号',data:r.signal,color:'#c78b42',stepped:true},{label:'实际区间持仓',data:r.sim.position,color:'#16847b',stepped:true}],{min:-1.2,max:1.2,ticks:{stepSize:1}});
 graph('rank',dates,[{label:'历史预测百分位',data:m.rank.percentile,color:'#5470c6'},{label:'60%',data:dates.map(()=>60),color:'#a6afbd',dash:[4,4]},{label:'40%',data:dates.map(()=>40),color:'#a6afbd',dash:[4,4]}],{min:0,max:100});
 el('months').innerHTML=Object.keys(r.months).map(month=>`<tr><td>${month}</td>${rules.map(q=>`<td class="num">${fmt(m[q].months[month].total)}</td>`).join('')}<td class="num">${fmt(r.months[month].long)}</td><td class="num">${fmt(r.months[month].short)}</td></tr>`).join('');
 el('sides').innerHTML=[1,-1].map(d=>{const t=r.sides[d];return `<tr><td>${side(d)}头</td><td class="num">${t.closed_n}／${t.open_n}</td><td class="num">${pct(t.win_rate)}</td><td class="num">${fmt(t.closed_pnl)}</td><td class="num">${fmt(t.floating_pnl)}</td><td class="num">${fmt(t.payoff_ratio,2)}</td></tr>`;}).join('');
 el('latest').innerHTML=keys.flatMap(k=>rules.map(q=>{const l=models[k][q].latest;return `<tr${k===key&&q===rule?' class="selected"':''}><td>${data.models[k].label}</td><td>${names[q]}</td><td class="num">${fmt(l.prediction)}</td><td class="num">${l.percentile==null?'不适用':fmt(l.percentile,1)+'%'}</td><td>${side(l.signal)}</td><td>${side(l.position)}</td><td>${side(l.next_position)}</td></tr>`;})).join('');
 el('ledger').innerHTML=[...r.sim.trades].reverse().map(t=>`<tr><td>${t.signal_date}</td><td>${t.entry_session}</td><td>${side(t.direction)}</td><td class="num">${fmt(t.entry_price)}</td><td class="num">${fmt(t.entry_prediction)}</td><td class="num">${fmt(t.entry_cutoff,t.entry_rule==='rank'?0:4)}${t.entry_rule==='rank'?'%':'点'}</td><td class="num">${t.entry_percentile==null?'—':fmt(t.entry_percentile,1)+'%'}</td><td>${t.exit_session||'尚未退出'}</td><td class="num">${t.holding_sessions}</td><td class="num">${fmt(t.pnl)}</td><td>${t.closed?'满6日已平':'未平浮动'}</td></tr>`).join('');
 el('daily').innerHTML=dates.map((day,i)=>`<tr><td>${day}</td><td class="num">${fmt(p.level[i])}</td><td class="num">${fmt(prediction[i])}</td><td class="num">${rule==='rank'?(r.percentile[i]==null?'历史不足':fmt(r.percentile[i],1)+'%'):'不适用'}</td><td>${r.eligible[i]?side(r.signal[i]):'等待历史'}</td><td>${side(r.sim.position[i])}</td><td>${side(r.sim.decisions[i].next_position)}</td><td class="num">${fmt(r.sim.daily_pnl[i])}</td><td class="num">${fmt(r.sim.equity[i])}</td></tr>`).reverse().join('');
}
function init(){
 const m=data.meta;
 el('cuts').textContent='固定做空边界（预测须严格低于）：'+keys.map(k=>data.models[k].label+' '+fmt(-m.negative_cuts[k],6)+'点').join('；')+'。来自2025年负预测绝对幅度的2/3分位。校准模型训练至2025年6月，最终模型训练至年底，两者预测分布可能变化；2026年不重划边界。';
 const requested=new URLSearchParams(location.search).get('entry_period');
 el('period').value=requested&&({...data.periods,...history.periods})[requested]?requested:m.default_period;el('model').value=m.default_model;el('rule').value=m.default_rule;render();
}
for(const id of ['model','period','rule'])el(id).onchange=render;
const reset=()=>Object.values(charts).forEach(c=>c.resetZoom());
el('reset').onclick=reset;document.getElementById('reset').addEventListener('click',reset);
Promise.all(['entry_rules_study.json','entry_rules_history.json'].map(file=>fetch(file+'?v='+Date.now(),{cache:'no-store'}).then(r=>{if(!r.ok)throw Error(file+' '+r.status);return r.json();}))).then(([d,h])=>{data=d;history=h;init();}).catch(e=>{el('state').textContent='交易比较读取失败：'+e.message;});
})();
