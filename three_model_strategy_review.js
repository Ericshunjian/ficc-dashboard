(()=>{
'use strict';
const el=id=>document.getElementById('review-'+id),keys=['ridge_original','ridge_temporal','bayes_temporal_new'],colors=['#a6afbd','#5470c6','#16847b'],charts={};let data;
const fmt=(v,n=4)=>v==null||!Number.isFinite(v)?'—':v.toFixed(n),pct=v=>v==null?'—':fmt(v*100,1)+'%',side=v=>v===1?'多（价差走扩）':v===-1?'空（价差收窄）':'空仓';
const row=(p,k,q,e)=>data.periods[p].models[k][q][e];
function chart(id,dates,series,position=false){
 if(charts[id])charts[id].destroy();
 charts[id]=new Chart(el(id),{type:'line',data:{labels:dates,datasets:series.map((s,i)=>({label:s.label,data:s.data,borderColor:s.color||colors[i],backgroundColor:s.color||colors[i],borderWidth:1.5,pointRadius:0,stepped:position,borderDash:s.dash||[]}))},options:{responsive:true,maintainAspectRatio:false,animation:false,interaction:{mode:'index',intersect:false},plugins:{legend:{position:'bottom',labels:{boxWidth:12,font:{size:10}}},zoom:{pan:{enabled:true,mode:'x'},zoom:{mode:'x',wheel:{enabled:true},pinch:{enabled:true}}}},scales:{x:{grid:{display:false},ticks:{autoSkip:false,maxRotation:0,font:{size:10},callback:function(v){const a=Math.ceil(this.min),b=Math.floor(this.max),n=Math.min(b-a+1,this.width<450?3:5),keep=n<2?[a]:Array.from({length:n},(_,i)=>Math.round(a+(b-a)*i/(n-1)));return keep.includes(Number(v))?this.getLabelForValue(v):'';}}},y:position?{min:-1.2,max:1.2,ticks:{stepSize:1}}:{}}}});
 el(id).ondblclick=()=>charts[id].resetZoom();
}
function render(){
 const period=el('period').value,p=data.periods[period],q=el('entry').value,e=el('exit').value,k=el('model').value,m=data.meta;
 el('period-note').textContent=`${p.dates[0]}至${p.dates.at(-1)}，${p.dates.length}个价格日期；当前为${m.entries[q]}＋${m.exits[e]}。三模型均从本区间首日空仓起步；共同起点保留此前60条排名历史，不继承旧持仓。${period==='full'?'排名前60日等待，含等待效应。':'此区间三模型所有开仓规则均可立即比较。'} 27组对应${m.unique_paths[period]}条不同持仓路径。`;
 el('matrix').innerHTML=['sign','asymmetric','rank'].flatMap(entry=>['fixed','reverse','neutral'].map(exit=>{const cells=keys.map(model=>{const r=row(period,model,entry,exit),id=[model,entry,exit].join('|'),selected=data.selection.selected.includes(id);return `<td class="num"${selected?' style="background:#f2faf7;font-weight:bold"':''}>${fmt(r.sim.metrics.total)}／${fmt(r.sim.metrics.max_drawdown)}</td>`;}).join(''),pass=keys.filter(model=>data.selection.rows[[model,entry,exit].join('|')].eligible).map(model=>m.models[model]);return `<tr><td>${m.entries[entry]}</td><td>${m.exits[exit]}</td>${cells}<td>${pass.join('、')||'未通过'}</td></tr>`;})).join('');
 el('stability').innerHTML=data.selection.selected.map(id=>{const a=data.selection.rows[id],r=row(period,a.model,a.entry,a.exit),s=r.sim.metrics,d=r.diagnostics;return `<tr><td>${a.label}</td><td class="num">${s.closed_trades}／${s.open_trades}</td><td class="num">${pct(s.win_rate)}</td><td class="num">${fmt(d.side_pnl['1'])}</td><td class="num">${fmt(d.side_pnl['-1'])}</td><td class="num">${fmt(d.recent['60'].total)}</td><td class="num">${fmt(d.concentration.total_after_top3)}</td><td class="num">${fmt(d.start_min)}</td><td class="num">${d.positive_month_n}／${d.month_n}</td></tr>`;}).join('');
 for(const field of ['equity','drawdown','position'])chart(field,p.dates,keys.map(model=>({label:m.models[model],data:row(period,model,q,e).sim[field]})),field==='position');
 chart('prediction',p.dates,[...keys.map(model=>({label:m.models[model],data:data.predictions[model].slice(p.start_index)})),{label:'零预测',data:p.dates.map(()=>0),color:'#c8ced7',dash:[4,4]}]);
 const ledger=row(period,k,q,e),last=ledger.latest;
 el('latest').textContent=`历史研究末日${m.asof}：${m.models[k]}六日预测${fmt(last.prediction)}点，开仓信号${side(last.signal)}${last.percentile==null?'':'，历史排名'+fmt(last.percentile,1)+'%'}；当日模拟持仓${side(last.position)}，其后下一交易日计划${side(last.next_position)}。这不是今天新产生的信号，不确认账户成交。`;
 el('ledger-note').textContent=`${m.models[k]} · ${m.entries[q]} · ${m.exits[e]}；入场代理价为信号日3T−TL收盘价。未平交易按研究末日盯市，浮盈亏计入曲线。`;
 const reasons={max_hold:'持有6日到期',reverse:'反向退出',neutral:'进入中性',open:'尚未平仓'};
 el('ledger').innerHTML=[...ledger.sim.trades].reverse().map(t=>`<tr><td>${t.signal_date}</td><td>${t.entry_session}</td><td>${side(t.direction)}</td><td class="num">${fmt(t.entry_price)}</td><td>${t.exit_session||'未平'}</td><td>${reasons[t.reason]||t.reason}</td><td class="num">${t.holding_sessions}</td><td class="num">${fmt(t.pnl)}</td></tr>`).join('');
}
function init(){
 const m=data.meta;
 el('state').textContent=`名单选取日${m.selected_on}，研究截至${m.asof}，训练截止${m.train_through}。完整2026有${m.forecast_n}条预测、${m.completed_n}条已完成六日标签。结果用于确定主跟踪，尚未切换旧每日自动报告。`;
 el('candidates').innerHTML=data.selection.selected.map((id,i)=>{const a=data.selection.rows[id],r=row('full',a.model,a.entry,a.exit),c=row('common',a.model,a.entry,a.exit);return `<div><div class="muted">${i===0?'主交易参考':'交叉复核'}</div><b>${m.models[a.model]}</b><p>${m.entries[a.entry]}＋${m.exits[a.exit]}</p><div>完整2026：${fmt(r.sim.metrics.total)}点／回撤${fmt(r.sim.metrics.max_drawdown)}</div><div>共同起点：${fmt(c.sim.metrics.total)}点／回撤${fmt(c.sim.metrics.max_drawdown)}</div></div>`;}).join('');
 el('conclusion').textContent='保留新时序贝叶斯为主交易参考，时序Ridge作交叉复核。原Ridge保留基准观察；历史排名40／60版本在剔除最好三笔后都转亏，本轮不列主跟踪。入选纯方向版本几乎始终持仓，未增加强信号过滤；不是只在明显优势时才交易的规则。';
 const rel=data.relationships[0].periods.full,ci=rel.difference95;
 el('relationship').textContent=`两主跟踪完整2026持仓相同占${pct(rel.same_position)}，日盈亏相关${fmt(rel.pnl_correlation,3)}；新贝叶斯减时序Ridge盈亏差${fmt(rel.pnl_difference)}点，配对移动区块95%参考区间[${fmt(ci.low)}, ${fmt(ci.high)}]点（块长12、2000次，未经探索选择校正）。区间跨0，当前不能确认新贝叶斯稳定胜出，主次仅为跟踪安排。两者高度相关，不应叠加成两倍仓位；分歧只提示，不临时改变规则。`;
 el('screen').innerHTML=data.selection.ordered.map(id=>{const a=data.selection.rows[id],selected=data.selection.selected.includes(id);return `<tr${selected?' class="selected"':''}><td>${a.label}</td><td class="num">${fmt(a.score,3)}</td><td>${selected?'主跟踪':a.eligible?'通过，未列主跟踪':'未通过'}</td><td class="wrap">${selected?'按两区间较低收益回撤比入选':a.failures.join('；')||'每模型最多一组；同路径去重；最多2组'}</td></tr>`;}).join('');
 render();
}
const reset=()=>Object.values(charts).forEach(c=>c.resetZoom());
for(const id of ['period','model','entry','exit'])el(id).onchange=render;
el('reset').onclick=reset;document.getElementById('reset').addEventListener('click',reset);
const a=document.createElement('a');a.href='#three-model-review';a.textContent='最新三模型开平仓比较与主跟踪 ↓';document.querySelector('a[href="#prediction-strata"]').before(a,document.createTextNode('　'));
fetch('three_model_strategy_review.json?v='+Date.now(),{cache:'no-store'}).then(r=>{if(!r.ok)throw Error(r.status);return r.json();}).then(d=>{data=d;init();}).catch(e=>{el('state').textContent='三模型比较读取失败：'+e.message;});
})();
