'use strict';
const $=id=>document.getElementById(id),keys=['ridge_original','ridge_temporal','dlm_temporal','xgb_temporal'];
const palette=['#5470c6','#16847b','#b88435','#9065b0'],charts={};let report;
const fmt=(v,n=4)=>v==null||!Number.isFinite(v)?'—':v.toFixed(n);
const pct=(v,n=1)=>v==null?'—':fmt(v*100,n)+'%';
const esc=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const dir=v=>v===1?'做多':v===-1?'做空':'空仓';
function chart(id,labels,series,{percent=false,type='line',stepped=false}={}){
 if(charts[id])charts[id].destroy();
 charts[id]=new Chart($(id),{type,data:{labels,datasets:series.map((s,i)=>({label:s.label,data:s.data,borderColor:s.color||palette[i%4],backgroundColor:type==='bar'?(s.color||palette[i%4]):'rgba(184,132,53,.08)',borderWidth:s.dash?1:1.8,pointRadius:0,spanGaps:false,borderDash:s.dash?[4,4]:[],stepped,fill:s.fill??false}))},options:{responsive:true,maintainAspectRatio:false,animation:false,interaction:{mode:'index',intersect:false},plugins:{legend:{position:'bottom',labels:{boxWidth:12,font:{size:10}}},zoom:{pan:{enabled:true,mode:'x'},zoom:{mode:'x',wheel:{enabled:true},pinch:{enabled:true}}}},scales:{x:{grid:{display:false},ticks:{maxRotation:0,autoSkip:false,font:{size:10},callback:function(v){const a=Math.ceil(this.min),b=Math.floor(this.max),n=this.width<450?3:5,keep=Array.from({length:n},(_,i)=>Math.round(a+(b-a)*i/(n-1)));return keep.includes(Number(v))?this.getLabelForValue(v):'';}}},y:percent?{min:0,max:100}:{}}}});
 $(id).ondblclick=()=>charts[id].resetZoom();
}
function render(){
 if(!report)return;
 const mode=$('mode').value,key=$('model').value,v=report.variants[key],item=v.modes[mode],m=item.sim.metrics,meta=report.meta;
 $('modeNote').textContent=meta[mode];
 $('summary').innerHTML=[['重点模型累计盈亏',fmt(m.total)+' 点'],['重点模型最大回撤',fmt(m.max_drawdown)+' 点'],['重点模型Rank IC',fmt(v.forecast.ic)],['重点模型交易胜率',pct(item.trade_stats.win_rate)]].map(([a,b])=>`<div><small>${a}</small><b>${b}</b></div>`).join('');
 $('comparisonHint').textContent=`${meta.completed_n}条完整预测标签；方向准确率按预测正负计算，交易方向按历史排名，两者口径不同。胜率和平均每笔盈亏只计已平交易，累计盈亏含浮动，单位为点。`;
 $('comparison').innerHTML=keys.map(k=>{const a=report.variants[k],f=a.forecast,r=a.modes[mode],s=r.sim.metrics,t=r.trade_stats;return `<tr><td>${esc(a.label)}</td><td class="num">${a.features_n}</td><td class="num">${fmt(f.ic)}</td><td class="num">${fmt(f.rmse)}</td><td class="num">${pct(f.direction_accuracy)}</td><td class="num">${fmt(s.total)}</td><td class="num">${fmt(s.max_drawdown)}</td><td class="num">${t.closed_n}／${t.open_n}</td><td class="num">${pct(t.win_rate)}</td><td class="num">${fmt(t.mean_pnl)}</td><td class="num">${fmt(t.payoff_ratio,2)}</td></tr>`;}).join('');
 const zero=report.forecast_references;
 $('zeroReference').textContent=`预测基准：始终预测0的RMSE为${fmt(zero.zero.rmse)}；始终预测训练期平均值的RMSE为${fmt(zero.train_mean.rmse)}。误差小于原模型并不等于战胜简单基准，也不等于交易盈利更高。`;
 ['equity','drawdown'].forEach(id=>chart(id,report.dates,keys.map((k,i)=>({label:report.variants[k].label,data:report.variants[k].modes[mode].sim[id],color:palette[i]}))));
 const pred=[{label:'预测均值',data:v.prediction,color:palette[keys.indexOf(key)]},{label:'实际六日变动',data:report.target,color:'#9aa8b9',dash:true}];
 if(v.interval80)pred.push({label:'80%下界',data:v.interval80.low,color:'#d2c4ae',dash:true},{label:'80%上界',data:v.interval80.high,color:'#d2c4ae',dash:true,fill:2});
 chart('prediction',report.dates,pred);
 chart('rank',report.dates,[{label:v.label,data:item.gate.percentile,color:palette[keys.indexOf(key)]},{label:'60%',data:report.dates.map(()=>60),dash:true},{label:'40%',data:report.dates.map(()=>40),dash:true}],{percent:true});
 const months=Object.keys(item.months);
 chart('monthly',months,[{label:'做多盈亏',data:months.map(x=>item.months[x].long_pnl),color:'#16847b'},{label:'做空盈亏',data:months.map(x=>item.months[x].short_pnl),color:'#b88435'}],{type:'bar'});
 chart('position',report.dates,[{label:'收盘观察信号',data:item.gate.signal,color:'#9aa8b9',dash:true},{label:'当日模拟持仓',data:item.sim.position,color:palette[keys.indexOf(key)]}],{stepped:true});
 const controls=[['原28输入',15,report.variants.ridge_original],['原28输入',meta.alpha_temporal,report.controls.original_matched_alpha],['时序92输入',15,report.controls.temporal_original_alpha],['时序92输入',meta.alpha_temporal,report.variants.ridge_temporal]];
 $('controls').innerHTML=controls.map(([name,alpha,a])=>`<tr><td>${name}</td><td class="num">${alpha}</td><td class="num">${fmt(a.forecast.ic)}</td><td class="num">${fmt(a.forecast.rmse)}</td><td class="num">${fmt(a.modes[mode].sim.metrics.total)}</td><td class="num">${fmt(a.modes[mode].sim.metrics.max_drawdown)}</td></tr>`).join('');
 $('months').innerHTML=months.map(month=>`<tr><td>${month}</td>${keys.map(k=>`<td class="num">${fmt(report.variants[k].modes[mode].months[month].pnl)}</td>`).join('')}</tr>`).join('');
 $('latest').innerHTML=keys.map(k=>{const a=report.variants[k],r=a.modes[mode],g=r.gate,s=r.sim;return `<tr><td>${esc(a.label)}</td><td>${meta.asof}</td><td class="num">${fmt(a.prediction.at(-1))}</td><td class="num">${fmt(g.percentile.at(-1),1)}%</td><td>${dir(g.signal.at(-1))}</td><td>${dir(s.position.at(-1))}</td><td>${dir(s.decisions.at(-1).next_position)}（计划）</td></tr>`;}).join('');
 $('sides').innerHTML=[1,-1].map(d=>{const t=item.sides[String(d)];return `<tr><td>${dir(d)}</td><td class="num">${t.closed_n}／${t.open_n}</td><td class="num">${pct(t.win_rate)}</td><td class="num">${fmt(t.average_win)}</td><td class="num">${fmt(t.average_loss)}</td><td class="num">${fmt(t.payoff_ratio,2)}</td><td class="num">${fmt(t.mean_pnl)}</td><td class="num">${fmt(t.closed_pnl)}</td><td class="num">${fmt(t.floating_pnl)}</td></tr>`;}).join('');
 const reasons={max_hold:'满6日',weak:'转中性',reverse:'反向',missing_signal:'信号缺失',open:'未平仓'};
 $('trades').innerHTML=item.sim.trades.slice().reverse().map(t=>`<tr><td>${t.signal_date}</td><td>${t.entry_session}</td><td>${dir(t.direction)}</td><td class="num">${fmt(t.entry_price)}</td><td>${t.exit_session||'未平仓'}</td><td class="num">${fmt(t.exit_price)}</td><td class="num">${t.holding_sessions}</td><td class="num">${fmt(t.pnl)}</td><td>${t.closed?'已平：':'浮动：'}${reasons[t.reason]||esc(t.reason)}</td></tr>`).join('')||'<tr><td colspan="9">无交易</td></tr>';
 $('history').innerHTML=report.dates.map((day,i)=>`<tr><td>${day}</td><td class="num">${fmt(report.level[i])}</td>${keys.map(k=>{const g=report.variants[k].modes[mode].gate;return `<td>${dir(g.signal[i])}／${g.percentile[i]==null?'—':fmt(g.percentile[i],1)+'%'}</td>`;}).join('')}</tr>`).reverse().join('');
}
function audit(){
 const meta=report.meta,val=report.validation,dlm=report.diagnostics.dlm;
 $('finding').textContent=`本轮没有据结果替换每日策略。2025年验证选择时序Ridge alpha=${meta.alpha_temporal}，贝叶斯discount=${meta.discount}${meta.discount===1?'（无额外状态漂移，仍按已完成结果更新后验）':''}，XGBoost仅${meta.xgb_rounds}轮。请同时看预测排序、误差、回撤和收益；更复杂的模型及更多历史输入并未稳定占优。`;
 $('parameters').innerHTML=`<p>内部训练截至${val.initial_train_through}，${val.train_n}条完整标签；最后训练信号${val.last_inner_train_signal}，标签于${val.last_inner_label_exit}兑现。验证信号${val.start}至${val.last_signal}，共${val.n}条，最后标签于${val.last_label_exit}兑现，跨年未完成标签不参与。</p><p>原Ridge固定alpha=15；时序Ridge只比较15／50／150／500。贝叶斯继承选定alpha，比较discount=1／0.995／0.98；每隔6个交易日吸收一条刚到期且互不重叠的六日标签，共${dlm.updates.length}次。初始协方差沿用6倍依赖调整，这不是对全部残差自相关的完整建模。</p><p>XGBoost固定深度2、学习率0.03、min_child_weight=30、L2=10；上限300轮，2025年验证20轮早停，最后保留${meta.xgb_rounds}轮，未因为2026年收益较低而增加轮数。${meta.xgb_rounds===1?'仅1轮意味着这组验证没有支持更复杂的拟合，不代表已证明XGBoost整体无效。':''}</p><p class="muted">同一649条初始训练样本：原28输入缺失${report.quality.train_missing_cells_original}格，92输入缺失${report.quality.train_missing_cells_temporal}格，包括新合约起始窗口及少量日报缺失，均用相应训练中位数填补；2026年92项共同输入缺失${report.quality.test_missing_cells}格。</p>`;
 const rows=[['原Ridge','alpha=15',val.ridge_original,'原参考'],...val.alphas.map(r=>['时序Ridge','alpha='+r.alpha,r,r.alpha===meta.alpha_temporal?'验证选中':'']),...val.discounts.map(r=>['时序贝叶斯','discount='+r.discount,r,r.discount===meta.discount?'验证选中':'']),['时序XGBoost',meta.xgb_rounds+'轮',val.xgb,'验证早停']];
 $('validation').innerHTML=rows.map(([a,b,r,c])=>`<tr><td>${a}</td><td>${b}</td><td class="num">${fmt(r.rmse)}</td><td class="num">${fmt(r.ic)}</td><td class="num">${pct(r.direction_accuracy)}</td><td>${c}</td></tr>`).join('');
 const names={original:'原28输入',spread_history:'价差历史',factor_history:'因子历史'};
 $('features').innerHTML=report.features.map(f=>`<tr><td class="wrap">${esc(f.label)}</td><td>${names[f.block]}</td><td class="num">${fmt(f.latest)}</td><td class="num">${f.train_missing_n}</td><td>${f.clip_enabled?'训练1/99%':'不缩尾'}</td><td class="num">${fmt(f.ridge_coef,6)}</td><td class="num">${fmt(f.dlm_latest_coef,6)}</td><td class="num">${pct(f.xgb_gain_share,2)}</td></tr>`).join('');
}
['model','mode'].forEach(id=>$(id).onchange=render);
$('reset').onclick=()=>Object.values(charts).forEach(c=>c.resetZoom());
fetch('curve_timing_temporal.json?v='+Date.now(),{cache:'no-store'}).then(r=>{if(!r.ok)throw Error(r.status);return r.json();}).then(d=>{
 report=d;$('mode').value=d.meta.default_mode;
 $('state').textContent=`研究截至${d.meta.asof}；统一初始训练至${d.meta.train_through}，${d.meta.train_n}条完整标签（最后训练信号${d.meta.last_train_signal}）。2026年从${d.meta.test_start}起${d.meta.test_n}个信号日，${d.meta.completed_n}条六日标签完成。`;
 audit();render();
}).catch(e=>{$('state').textContent='研究读取失败：'+e.message;});
