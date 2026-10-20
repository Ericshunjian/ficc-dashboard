(()=>{
'use strict';
const el=id=>document.getElementById('stability-'+id),keys=['direction','filtered'],mk=['ridge_temporal','dlm_temporal'],labels={ridge_temporal:'时序Ridge',dlm_temporal:'时序贝叶斯'},charts={};let data;
const fmt=(v,n=4)=>v==null||!Number.isFinite(v)?'—':v.toFixed(n),pct=v=>v==null?'—':fmt(v*100,1)+'%',side=d=>d===1?'多':'空';
function chart(id,dates,series,type='bar',bounds={}){
 if(charts[id])charts[id].destroy();
 charts[id]=new Chart(el(id),{type,data:{labels:dates,datasets:series.map((s,i)=>({label:s.label,data:s.data,borderColor:s.color||['#5470c6','#16847b'][i],backgroundColor:s.color||['#5470c6','#16847b'][i],borderWidth:type==='line'?1.7:1,pointRadius:0,borderDash:s.dash||[],spanGaps:false}))},options:{responsive:true,maintainAspectRatio:false,animation:false,interaction:{mode:'index',intersect:false},plugins:{legend:{position:'bottom',labels:{boxWidth:12,font:{size:10}}},zoom:{pan:{enabled:true,mode:'x'},zoom:{mode:'x',wheel:{enabled:true},pinch:{enabled:true}}}},scales:{x:{grid:{display:false},ticks:{autoSkip:false,maxRotation:0,font:{size:10},callback:function(v){const a=Math.ceil(this.min),b=Math.floor(this.max),n=Math.min(b-a+1,this.width<450?3:5),keep=n<2?[a]:Array.from({length:n},(_,i)=>Math.round(a+(b-a)*i/(n-1)));return keep.includes(Number(v))?this.getLabelForValue(v):'';}}},y:bounds}}});
 el(id).ondblclick=()=>charts[id].resetZoom();
}
function render(){
 const period=el('period').value,p=data.periods[period],models=p.models;
 el('period-note').textContent=`当前原账户：${p.dates[0]}至${p.dates.at(-1)}，${p.dates.length}个价格日期；下方分期继承该账户持仓，6起点检查另行从空仓重跑。`;
 el('summary').innerHTML=keys.map(k=>{const r=models[k],s=r.metrics;return `<tr><td>${r.label}</td><td class="num">${fmt(s.total)}</td><td class="num">${fmt(s.max_drawdown)}</td><td class="num">${r.positive_months}／${r.month_n}</td><td class="num">${fmt(r.recent['30'].total)}</td><td class="num">${fmt(r.recent['60'].total)}</td><td class="num">${fmt(r.concentration.total_after_top3)}</td><td class="num">${fmt(r.starts.min)}～${fmt(r.starts.max)}</td></tr>`;}).join('');
 const rel=p.relationship;
 el('overlap').textContent=`两候选${rel.long_trades_equal?'多头账本完全相同':'多头账本有差异'}，共${rel.long_trade_n}笔多头记录；持仓方向相同占${pct(rel.same_position_fraction)}，日盈亏相关${fmt(rel.daily_pnl_correlation,3)}。主要是在比较空头过滤，不是两份独立信号，不应仅因保留两套就认为得到分散化。`;
 const months=Object.keys(models.direction.months),quarters=Object.keys(models.direction.quarters);
 chart('months',months,keys.map(k=>({label:models[k].label,data:months.map(m=>models[k].months[m].total)})));
 chart('starts',models.direction.starts.rows.map(v=>v.start),keys.map(k=>({label:models[k].label,data:models[k].starts.rows.map(v=>v.metrics.total)})));
 el('quarters').innerHTML=quarters.flatMap(q=>keys.map(k=>{const r=models[k],v=r.quarters[q];return `<tr><td>${q}</td><td>${r.label}</td><td class="num">${v.n}</td><td class="num">${fmt(v.total)}</td><td class="num">${fmt(v.max_drawdown)}</td><td class="num">${fmt(v.long_pnl)}</td><td class="num">${fmt(v.short_pnl)}</td><td>${v.n<30?'少于30日，仅观察':'仍为探索结果'}</td></tr>`;})).join('');
 el('sides').innerHTML=keys.flatMap(k=>[1,-1].map(d=>{const r=models[k],s=r.sides[d];return `<tr><td>${r.label}</td><td>${side(d)}头</td><td class="num">${s.closed_n}／${s.open_n}</td><td class="num">${fmt(s.closed_pnl+s.floating_pnl)}</td><td class="num">${pct(s.win_rate)}</td><td class="num">${fmt(s.mean_pnl)}</td><td class="num">${fmt(s.floating_pnl)}</td><td>${s.closed_n<20?'少于20笔，样本少':'仍需前瞻观察'}</td></tr>`;})).join('');
 el('recent').innerHTML=keys.flatMap(k=>[30,60].map(w=>{const r=models[k],v=r.recent[w];return `<tr><td>${r.label}</td><td>${w}日</td><td>${v.start}至${v.end}</td><td class="num">${fmt(v.total)}</td><td class="num">${fmt(v.max_drawdown)}</td><td class="num">${fmt(v.long_pnl)}</td><td class="num">${fmt(v.short_pnl)}</td><td class="num">${v.position_days}</td></tr>`;})).join('');
 el('concentration').textContent=keys.map(k=>{const r=models[k],c=r.concentration;return `${r.label}：最好${c.top_n}笔已平盈利合计${fmt(c.top3_pnl)}点，占净盈利${pct(c.share_of_net)}、已平总盈利${pct(c.share_of_gross_wins)}；减去后剩余${fmt(c.total_after_top3)}点。`;}).join(' ');
 el('top-trades').innerHTML=keys.flatMap(k=>models[k].concentration.trades.map(t=>`<tr><td>${models[k].label}</td><td>${t.signal_date}</td><td>${t.entry_session}</td><td>${t.exit_session}</td><td>${side(t.direction)}</td><td class="num">${fmt(t.pnl)}</td></tr>`)).join('');
 el('start-rows').innerHTML=Array.from({length:6},(_,i)=>keys.map(k=>{const r=models[k],v=r.starts.rows[i],m=v.metrics;return `<tr><td>${v.start}</td><td>${r.label}</td><td class="num">${fmt(m.total)}</td><td class="num">${fmt(m.max_drawdown)}</td><td class="num">${m.closed_trades}</td><td class="num">${pct(m.win_rate)}</td></tr>`;}).join('')).join('');
}
function bayes(){
 const b=data.bayes,c=b.config,periods=b.periods,all=periods.all.models,s=b.sign_changes;
 el('bayes-config').textContent=`初始训练${c.initial_train_labels}条标签、同${c.input_n}项输入，alpha=${c.alpha}，初始均值等于时序Ridge。discount=${c.discount}：仍更新后验，但没有额外状态漂移或主动遗忘。${c.first_update}至${c.last_update}共${c.updates}次，间隔6个交易日吸收刚兑现的六日标签。原649条训练信息仍保留。`;
 el('bayes-metrics').innerHTML=mk.map(k=>{const m=all[k];return `<tr><td>${labels[k]}</td><td class="num">${m.direction_correct}／${m.direction_n}</td><td class="num">${pct(m.direction_accuracy)}</td><td class="num">${fmt(m.rank_ic,3)}</td><td class="num">${fmt(m.mae)}</td><td class="num">${fmt(m.rmse)}</td><td class="num">${fmt(m.bias)}</td></tr>`;}).join('');
 const ci=b.paired95.direction_accuracy;
 el('bayes-uncertainty').textContent=`上述${periods.all.n}条完整标签截至信号${data.meta.last_complete_signal}；末6日预测尚未完成。贝叶斯减Ridge同向率差${fmt((all.dlm_temporal.direction_accuracy-all.ridge_temporal.direction_accuracy)*100,2)}个百分点，配对区块95%区间[${fmt(ci.low*100,2)}, ${fmt(ci.high*100,2)}]个百分点。块长12、2000次；差异区间跨0，不能确认总体能力显著更差，且未纠正探索选择。`;
 el('bayes-finding').textContent=`递推改变${s.n}次已完成预测方向：${s.helped}次改对、${s.hurt}次改错，净少${s.hurt-s.helped}次命中。差距主要在一季度；三季度方向和排序反而较好。平均偏差有所减小，但MAE和排序未改善。模型内80%预测区间有${b.interval80_contains_zero_n}／${b.interval_n}条覆盖0，当前交易并没有使用这些不确定性信息；这也不表示加概率过滤一定更好。`;
 const qs=Object.keys(periods).filter(q=>q!=='all');
 chart('bayes-quarters',qs,mk.map(k=>({label:labels[k],data:qs.map(q=>periods[q].n<30?null:periods[q].models[k].direction_accuracy*100)})),'bar',{min:0,max:100});
 chart('bayes-delta',b.dates,[{label:'贝叶斯减冻结Ridge',data:b.prediction_delta},{label:'没有变化',data:b.dates.map(()=>0),color:'#a6afbd',dash:[4,4]}],'line');
 el('bayes-quarter-rows').innerHTML=qs.flatMap(q=>mk.map(k=>{const p=periods[q],m=p.models[k];return `<tr${p.n<30?' style="background:#fff8ed"':''}><td>${q}</td><td>${labels[k]}</td><td class="num">${p.n}</td><td>${p.last_complete_signal||'—'}</td><td class="num">${pct(m.direction_accuracy)}</td><td class="num">${fmt(m.rank_ic,3)}</td><td class="num">${fmt(m.mae)}</td><td class="num">${fmt(m.rmse)}</td><td>${p.n<30?'不足30条，不判断季度优劣':'仍有六日标签重叠'}</td></tr>`;})).join('');
 el('bayes-control').innerHTML=Object.entries(b.pure_strategy_control).flatMap(([p,v])=>mk.map(k=>{const m=v[k];return `<tr><td>${p==='full'?'完整2026':'共同起点'}</td><td>${labels[k]}</td><td class="num">${fmt(m.total)}</td><td class="num">${fmt(m.max_drawdown)}</td><td class="num">${m.closed_trades}</td><td class="num">${pct(m.win_rate)}</td></tr>`;})).join('');
 const blockNames={intercept:'截距',original:'原28项',spread_history:'价差历史4项',factor_history:'因子历史60项'};
 el('bayes-blocks').innerHTML=Object.entries(b.attribution).map(([k,v])=>`<tr><td>${blockNames[k]}</td><td class="num">${fmt(v.mean)}</td><td class="num">${fmt(v.mean_absolute)}</td></tr>`).join('');
 el('bayes-attribution').textContent=`2025内部验证RMSE：${c.validation_discounts.map(v=>'discount='+v.discount+'为'+fmt(v.rmse_exact,3)).join('；')}。因此不能简单认为调低discount就会改善。非对称做空边界：Ridge ${fmt(-b.negative_cuts.ridge_temporal,4)}，贝叶斯 ${fmt(-b.negative_cuts.dlm_temporal,4)}；该组收益混合了预测和门槛差异。`;
 el('bayes-flips').innerHTML=s.rows.map(v=>`<tr><td>${v.date}</td><td class="num">${fmt(v.ridge)}</td><td class="num">${fmt(v.bayes)}</td><td class="num">${fmt(v.actual)}</td><td>${v.effect==='helped'?'改对':v.effect==='hurt'?'改错':'无净变化'}</td></tr>`).join('');
}
const reset=()=>Object.values(charts).forEach(c=>c.resetZoom());
el('period').onchange=render;el('reset').onclick=reset;document.getElementById('reset').addEventListener('click',reset);
fetch('strategy_stability.json?v='+Date.now(),{cache:'no-store'}).then(r=>{if(!r.ok)throw Error(r.status);return r.json();}).then(d=>{data=d;el('state').textContent=`研究快照截至${d.meta.asof}。两候选来自既有探索结果；本轮仅检查稳定性和更新机制，不重训、不重筛因子、不调交易门槛。`;render();bayes();}).catch(e=>{el('state').textContent='稳定性检查读取失败：'+e.message;});
})();
