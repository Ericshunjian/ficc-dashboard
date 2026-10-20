(()=>{
'use strict';
const el=id=>document.getElementById('strata-'+id),keys=['ridge_original','ridge_temporal','dlm_temporal'],charts={};let data;
const fmt=(v,n=4)=>v==null||!Number.isFinite(v)?'—':v.toFixed(n),pct=v=>v==null?'—':fmt(v*100,1)+'%';
const direction=d=>d===1?'正预测':'负预测',strength=s=>['','弱','中','强'][s];
const groupName=r=>`${r.direction===1?'正':'负'}·${strength(r.strength)}`;
const interval=(v,scale=1)=>v?.low==null?'区间不足':`[${fmt(v.low*scale,scale===100?1:3)}, ${fmt(v.high*scale,scale===100?1:3)}]`;
function graph(id,labels,series,type='bar',time=false,percent=false){
 if(charts[id])charts[id].destroy();
 charts[id]=new Chart(el(id),{type,data:{labels,datasets:series.map((s,i)=>({label:s.label,data:s.data,borderColor:s.color||['#9eabc4','#16847b'][i],backgroundColor:s.color||(i===0?'#b5c0d7':'#16847bb0'),borderWidth:type==='line'?1.8:1,pointRadius:0,spanGaps:false}))},options:{responsive:true,maintainAspectRatio:false,animation:false,interaction:{mode:'index',intersect:false},plugins:{legend:{position:'bottom',labels:{boxWidth:12,font:{size:10}}},zoom:{pan:{enabled:time,mode:'x'},zoom:{mode:'x',wheel:{enabled:time},pinch:{enabled:time}}}},scales:{x:{grid:{display:false},ticks:time?{maxRotation:0,autoSkip:false,font:{size:10},callback:function(v){const a=Math.ceil(this.min),b=Math.floor(this.max),n=this.width<450?3:5,keep=Array.from({length:n},(_,i)=>Math.round(a+(b-a)*i/(n-1)));return keep.includes(Number(v))?this.getLabelForValue(v):'';}}:{font:{size:10},maxRotation:0}},y:percent?{min:0,max:100}:{}}}});
 el(id).ondblclick=()=>charts[id].resetZoom();
}
function render(){
 if(!data)return;
 const period=el('period').value,key=el('model').value,m=data.models[key],view=m[period],rows=m.test2026.rows,cal=m.calibration2025.rows;
 const label=period==='test2026'?'2026年已冻结／按时递推预测':'2025年内部开发校准预测';
 el('view-note').textContent=`当前表格和时间图：${label}。三张柱图始终并列两期，边界均由2025年120条完整校准预测确定，2026年不重分组。`;
 el('rows').innerHTML=keys.flatMap(k=>{const mm=data.models[k];return mm[period].rows.map(r=>{const cr=mm.calibration2025.rows.find(c=>c.code===r.code);return `<tr${r.small_sample?' style="background:#fff8ed"':''}><td>${mm.label}</td><td>${direction(r.direction)}</td><td>${strength(r.strength)}</td><td class="num">${cr.n}</td><td class="num">${r.forecast_n}／${r.n}</td><td class="num">${r.correct}／${r.direction_n}</td><td class="num">${pct(r.direction_accuracy)}<br><small>${interval(r.interval95.direction_accuracy,100)}${r.interval95.direction_accuracy?.low==null?'':'%'}</small></td><td class="num">${fmt(r.actual_mean)}</td><td class="num">${fmt(r.signed_mean)}<br><small>${interval(r.interval95.signed_mean)}</small></td><td class="num">${fmt(r.mae)}</td><td class="num">${fmt(r.rmse)}</td><td>${r.n===0?'无完整标签':r.small_sample?'少于20条，仅观察':'仍有标签重叠'}</td></tr>`;});}).join('');
 const labels=rows.map(groupName);
 graph('accuracy',labels,[{label:'2025校准',data:cal.map(r=>r.direction_accuracy==null?null:r.direction_accuracy*100)},{label:'2026观察',data:rows.map(r=>r.direction_accuracy==null?null:r.direction_accuracy*100)}],'bar',false,true);
 graph('mean',labels,[{label:'2025校准',data:cal.map(r=>r.signed_mean)},{label:'2026观察',data:rows.map(r=>r.signed_mean)}]);
 graph('counts',labels,[{label:'2025完整标签数',data:cal.map(r=>r.n)},{label:'2026完整标签数',data:rows.map(r=>r.n)}]);
 graph('path',view.dates,[{label:m.label+'预测',data:view.prediction,color:'#16847b'},{label:'实际六日变动',data:view.target,color:'#a6afbd'}],'line',true);
 el('monotonic').innerHTML=[1,-1].map(d=>{const a=view.diagnostics[String(d)],ok=a.accuracy_monotonic;return `<p><b>${direction(d)}：</b>${ok==null?'存在空组，不能比较完整弱中强序列':ok?'同向率点估计随弱→中→强不降':'同向率没有随弱→中→强稳定提高'}；强减弱同向率差${a.strong_minus_weak_accuracy==null?'—':fmt(a.strong_minus_weak_accuracy*100,1)+'个百分点'}，区块95%区间${interval(a.contrast_interval95.direction_accuracy,100)}个百分点。${a.all_groups_at_least_20?'每档均至少20条，但不代表显著有效。':'至少一档少于20条，或无完整标签，结论较弱。'}</p>`;}).join('');
 const q=view.quality;
 el('quality').textContent=`所选模型共${q.forecast_n}条预测、${q.completed_n}条完整标签、${q.pending_n}条待兑现；零预测${q.zero_prediction_n}条，无法分档${q.unsupported_n}条，超出2025年该方向校准幅度范围${q.outside_calibration_range_n}条。训练截止点改变会使预测分布变化，两期各组人数不必相同。`;
}
function init(){
 const m=data.meta;
 el('state').textContent=`分层观察截至${m.asof}；2025年校准模型训练至${m.initial_calibration_train_through}（${m.calibration_train_n}条已兑现标签），取${m.calibration_start}至${m.calibration_end_signal}的${m.calibration_complete_n}条完整预测确定边界。2026年${m.test_prediction_n}条预测、${m.test_complete_n}条标签完成，最后完整信号${m.last_test_complete_signal}。`;
 el('cuts').innerHTML=keys.flatMap(k=>[1,-1].map(d=>{const b=data.models[k].boundaries[String(d)],c=b.cuts;return `<tr><td>${data.models[k].label}</td><td>${direction(d)}</td><td class="num">${b.n}</td><td class="num">${fmt(c?.[0])}</td><td class="num">${fmt(c?.[1])}</td><td class="num">${fmt(b.min_abs)}～${fmt(b.max_abs)}</td></tr>`;})).join('');
 el('latest').innerHTML=keys.map(k=>{const m=data.models[k],v=m.test2026,p=v.prediction.at(-1),c=v.codes.at(-1);return `<tr><td>${m.label}</td><td>${data.meta.asof}</td><td class="num">${fmt(p)}</td><td>${c===0?'零预测':Math.abs(c)===99?'校准不足，无法分档':direction(Math.sign(c))+'·'+strength(Math.abs(c))}</td><td>${v.target.at(-1)==null?'六日结果未完成':'六日结果已完成'}</td></tr>`;}).join('');
 render();
}
el('model').onchange=render;el('period').onchange=render;
el('reset').onclick=()=>Object.values(charts).forEach(c=>c.resetZoom());
document.getElementById('reset').addEventListener('click',()=>Object.values(charts).forEach(c=>c.resetZoom()));
fetch('forecast_signal_diagnostics.json?v='+Date.now(),{cache:'no-store'}).then(r=>{if(!r.ok)throw Error(r.status);return r.json();}).then(d=>{data=d;init();}).catch(e=>{el('state').textContent='分层诊断读取失败：'+e.message;});
})();
