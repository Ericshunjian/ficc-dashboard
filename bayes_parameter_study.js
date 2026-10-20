(()=>{
'use strict';
const el=id=>document.getElementById('bayestune-'+id),keys=['ridge','old','selected'],colors=['#5470c6','#bd8642','#16847b'],charts={};let data;
const fmt=(v,n=4)=>v==null||!Number.isFinite(v)?'—':v.toFixed(n),pct=v=>v==null?'—':fmt(v*100,1)+'%';
function chart(id,series){
 if(charts[id])charts[id].destroy();
 charts[id]=new Chart(el(id),{type:'line',data:{labels:data.dates,datasets:series.map((s,i)=>({label:s.label,data:s.data,borderColor:s.color||colors[i],backgroundColor:s.color||colors[i],borderWidth:1.6,pointRadius:0,borderDash:s.dash||[],spanGaps:false}))},options:{responsive:true,maintainAspectRatio:false,animation:false,interaction:{mode:'index',intersect:false},plugins:{legend:{position:'bottom',labels:{boxWidth:12,font:{size:10}}},zoom:{pan:{enabled:true,mode:'x'},zoom:{mode:'x',wheel:{enabled:true},pinch:{enabled:true}}}},scales:{x:{grid:{display:false},ticks:{autoSkip:false,maxRotation:0,font:{size:10},callback:function(v){const a=Math.ceil(this.min),b=Math.floor(this.max),n=Math.min(b-a+1,this.width<450?3:5),keep=n<2?[a]:Array.from({length:n},(_,i)=>Math.round(a+(b-a)*i/(n-1)));return keep.includes(Number(v))?this.getLabelForValue(v):'';}}}}}});
 el(id).ondblclick=()=>charts[id].resetZoom();
}
function trading(){
 const rule=el('rule').value;
 el('rule-note').textContent=rule==='sign'?'预测>0做多、<0做空；持仓遇反向预测，次交易日平仓并按相反方向开仓；同向不重置持有期，最长6日。三组门槛均为0。':`预测>0做多、<−${fmt(data.meta.shared_negative_cut,6)}做空；其余为中性，持仓遇中性则平仓，遇相反开仓信号则反手，最长6日。三组共用已冻结的时序Ridge阈值，未为新参数另调门槛。`;
 el('trades').innerHTML=keys.map(k=>{const r=data.models[k],m=r.trading[rule].sim.metrics;return `<tr><td>${r.label}</td><td class="num">${fmt(m.total)}</td><td class="num">${fmt(m.max_drawdown)}</td><td class="num">${m.closed_trades}／${m.open_trades}</td><td class="num">${pct(m.win_rate)}</td><td class="num">${fmt(m.average_holding,2)}</td></tr>`;}).join('');
 chart('equity',keys.map(k=>({label:data.models[k].label,data:data.models[k].trading[rule].sim.equity})));
}
function render(){
 const m=data.meta,v=data.validation,g=data.grid[m.selected],old=data.grid[m.old],near=data.grid.d1_c1;
 el('state').textContent=`独立研究截至${m.asof}；2026年${m.prediction_n}条预测，${m.completed_n}条完整六日标签，最后完成信号${m.last_complete_signal}。生产模型与历史归档均保持不变。`;
 el('selection').textContent=`2025-06-30前先拟合${v.initial_train_n}条完成标签，2025年下半年${v.n}条完成标签用于选参。按验证RMSE最小选出discount=${g.discount}、c=${g.covariance_multiplier}，随后用2025年底${m.train_n}条标签初始化。验证RMSE从旧版${fmt(old.validation.rmse,6)}降至${fmt(g.validation.rmse,6)}；与discount=1、c=1的${fmt(near.validation.rmse,6)}仅差${fmt(near.validation.rmse-g.validation.rmse,6)}，不足以断言.995稳定优于1。`;
 const n=data.models.selected.metrics,o=data.models.old.metrics,r=data.models.ridge.metrics;
 el('conclusion').textContent=`相对旧贝叶斯，新参数方向${pct(o.direction_accuracy)}→${pct(n.direction_accuracy)}、Rank IC ${fmt(o.rank_ic,3)}→${fmt(n.rank_ic,3)}、RMSE ${fmt(o.rmse)}→${fmt(n.rmse)}。相对时序Ridge，仅多命中${n.direction_correct-r.direction_correct}次，MAE、RMSE和Rank IC仍不占优。改进更像是降低初期更新幅度，并不是更新越快越好；保留观察，不替换原策略。`;
 el('metrics').innerHTML=keys.map(k=>{const a=data.models[k],x=a.metrics;return `<tr${k==='selected'?' class="selected"':''}><td>${a.label}</td><td class="num">${x.direction_correct}／${x.direction_n}</td><td class="num">${pct(x.direction_accuracy)}</td><td class="num">${pct(x.balanced_accuracy)}</td><td class="num">${fmt(x.rank_ic,3)}</td><td class="num">${fmt(x.mae)}</td><td class="num">${fmt(x.rmse)}</td></tr>`;}).join('');
 const ci=data.paired95.old.direction_accuracy;
 el('ci').textContent=`新减旧同向准确率差的配对区块95%区间为[${fmt(ci.low*100,2)}, ${fmt(ci.high*100,2)}]个百分点，跨0，尚不能确认稳定改善。块长12、2000次；未校正多轮研究选择。`;
 el('grid').innerHTML=Object.entries(data.grid).map(([k,a])=>{const x=a.validation,y=a.metrics;return `<tr${k===m.selected?' class="selected"':''}><td>${a.discount}</td><td class="num">${a.covariance_multiplier}</td><td class="num">${fmt(x.rmse,6)}</td><td class="num">${pct(x.direction_accuracy)}</td><td class="num">${fmt(x.rank_ic,3)}</td><td class="num">${pct(y.direction_accuracy)}</td><td class="num">${fmt(y.rank_ic,3)}</td><td class="num">${fmt(y.mae)}</td><td class="num">${fmt(y.rmse)}</td><td>${k===m.selected?'2025验证选中':k===m.old?'原参数':'敏感性对照'}</td></tr>`;}).join('');
 el('quarters').innerHTML=Object.keys(data.models.ridge.quarters).flatMap(q=>keys.map(k=>{const a=data.models[k],b=a.quarters[q],x=b.metrics;return `<tr><td>${q}</td><td>${a.label}</td><td class="num">${x.n}</td><td>${b.last_complete_signal||'—'}</td><td class="num">${pct(x.direction_accuracy)}</td><td class="num">${fmt(x.rank_ic,3)}</td><td class="num">${fmt(x.mae)}</td><td class="num">${fmt(x.rmse)}</td><td>${x.n<30?'不足30条，不作季度优劣判断':'标签仍有重叠'}</td></tr>`;})).join('');
 chart('forecast',[...keys.map(k=>({label:data.models[k].label,data:data.models[k].prediction})),{label:'实际六日变动',data:data.target,color:'#98a4b3',dash:[4,4]}]);
 trading();
}
const reset=()=>Object.values(charts).forEach(c=>c.resetZoom());
el('rule').onchange=trading;el('reset').onclick=reset;document.getElementById('reset').addEventListener('click',reset);
const jump=document.createElement('a');jump.href='#bayes-parameters';jump.textContent='贝叶斯9组参数对照 ↓';document.querySelector('a[href="#stability"]').after(document.createTextNode('　'),jump);
fetch('bayes_parameter_study.json?v='+Date.now(),{cache:'no-store'}).then(r=>{if(!r.ok)throw Error(r.status);return r.json();}).then(d=>{data=d;render();}).catch(e=>{el('state').textContent='参数研究读取失败：'+e.message;});
})();
