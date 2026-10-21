(()=>{
'use strict';const U=FiccUI,el=id=>document.getElementById(id),keys=['ridge_original','ridge_temporal','bayes_temporal_new'];let data,daily;
const row=(p,k,q,e)=>data.periods[p].models[k][q][e];
function render(){
 const period=el('period').value,p=data.periods[period],entry=el('entry').value,exit=el('exit').value,m=data.meta;
 el('period-note').textContent=`${p.dates[0]}至${p.dates.at(-1)}；当前曲线：${m.entries[entry]}＋${m.exits[exit]}。${period==='full'?'排名前60日热身，含等待效应。':'各组从共同起点空仓重启，排名保留此前历史。'}`;
 el('matrix').innerHTML=['sign','asymmetric','rank'].flatMap(q=>['fixed','reverse','neutral'].map(e=>`<tr><td>${m.entries[q]}</td><td>${m.exits[e]}</td>${keys.map(k=>{const s=row(period,k,q,e).sim.metrics;return `<td class="num"${data.selection.selected.includes([k,q,e].join('|'))?' style="background:#f0f8f5;font-weight:bold"':''}>${U.fmt(s.total)}／${U.fmt(s.max_drawdown)}</td>`;}).join('')}</tr>`)).join('');
 for(const field of ['equity','drawdown'])U.chart(field,p.dates,keys.map(k=>({key:k,label:m.models[k],data:row(period,k,entry,exit).sim[field]})));
 el('stability').innerHTML=data.selection.selected.map(id=>{const a=data.selection.rows[id],r=row(period,a.model,a.entry,a.exit),d=r.diagnostics;return `<tr><td>${m.models[a.model]}</td><td class="num">${U.fmt(d.side_pnl['1'])}</td><td class="num">${U.fmt(d.side_pnl['-1'])}</td><td class="num">${U.fmt(d.recent['60'].total)}</td><td class="num">${U.fmt(d.concentration.total_after_top3)}</td><td class="num">${U.fmt(d.start_min)}</td><td class="num">${d.positive_month_n}／${d.month_n}</td></tr>`;}).join('');
 const rel=data.relationships[0].periods[period],ci=rel.difference95;
 el('relationship').textContent=`两主跟踪同仓${U.pct(rel.same_position)}，日盈亏相关${U.fmt(rel.pnl_correlation,3)}。新贝叶斯减时序Ridge盈亏差${U.fmt(rel.pnl_difference)}点，配对区块95%参考区间[${U.fmt(ci.low)}, ${U.fmt(ci.high)}]点，跨0，不能确认稳定胜出。块长12、2000次，未校正多轮选择。`;
 const selected=data.selection.selected.map(id=>data.selection.rows[id]),months=Object.keys(row(period,selected[0].model,selected[0].entry,selected[0].exit).diagnostics.months);
 U.chart('months',months,selected.map(a=>({key:a.model,label:m.models[a.model]+'月度盈亏（点）',data:months.map(month=>row(period,a.model,a.entry,a.exit).diagnostics.months[month].total)})),{type:'bar'});
}
function probability(){
 const c=daily.probability_check,b=c.bins;
 el('probability-note').textContent=`截至${daily.meta.asof}，${c.n}条已完成六日标签。Brier误差${U.fmt(c.brier)}（始终报50%为0.2500，越低越好）；80%预测区间实际覆盖${U.pct(c.coverage80)}。各概率档实际频率仍不稳定，尤其高概率小样本档；暂不能把模型概率直接当作可靠胜率。`;
 U.chart('calibration',b.map(v=>`${Math.round(v.low*100)}–${Math.round(v.high*100)}%`),[{label:'平均预测走扩概率',data:b.map(v=>v.mean_probability==null?null:v.mean_probability*100),color:'#5470c6'},{label:'实际走扩频率',data:b.map(v=>v.actual_up_rate==null?null:v.actual_up_rate*100),color:'#16847b'}],{type:'bar',y:{min:0,max:100}});
 el('probability-bins').innerHTML=b.map(v=>`<tr><td>${Math.round(v.low*100)}–${Math.round(v.high*100)}%</td><td class="num">${v.n}</td><td class="num">${U.pct(v.mean_probability)}</td><td class="num">${U.pct(v.actual_up_rate)}</td><td>${v.n<20?'少于20条，仅观察':'标签仍有重叠'}</td></tr>`).join('');
}
function init(){const m=data.meta;
 el('state').textContent=`固定研究截至${m.asof}；初始训练至${m.train_through}，${m.train_n}条完成标签。模型预测评价${m.completed_n}条；27组交易曲线按末日价格盯市。这里不混入旧参数贝叶斯。`;
 el('prediction-metrics').innerHTML=keys.map(k=>{const v=data.prediction_metrics[k];return `<tr><td>${m.models[k]}</td><td class="num">${U.pct(v.direction_accuracy)}</td><td class="num">${U.fmt(v.rank_ic,3)}</td><td class="num">${U.fmt(v.mae)}</td><td class="num">${U.fmt(v.rmse)}</td></tr>`;}).join('');
 el('screen').innerHTML=data.selection.ordered.map(id=>{const r=data.selection.rows[id],picked=data.selection.selected.includes(id);return `<tr${picked?' class="selected"':''}><td>${r.label}</td><td class="num">${U.fmt(r.score,3)}</td><td>${picked?'主跟踪':r.eligible?'通过，未列主跟踪':'未通过'}</td><td class="wrap">${r.failures.join('；')||(picked?'入选':'每模型最多一组；同路径去重；最多2组')}</td></tr>`;}).join('');
 render();probability();
}
for(const id of ['period','entry','exit'])el(id).onchange=render;el('reset').onclick=U.reset;
Promise.all([U.read('three_model_strategy_review.json'),U.read('curve_signal_daily.json')]).then(([a,b])=>{data=a;daily=b;init();}).catch(e=>{el('state').classList.add('warning');el('state').textContent='研究数据读取失败：'+e.message;});
})();
