(()=>{
'use strict';const U=FiccUI,el=id=>document.getElementById(id),keys=['bayes_temporal_new','ridge_temporal'];let data;
function historyTable(){
 const n=Number(el('count').value),rows=data.records.map((r,i)=>({r,i})).slice(n?-n:0).reverse(),models=data.simulation.models;
 el('signal-history').innerHTML=rows.map(({r,i})=>{const b=r.models.bayes_temporal_new,s=models.bayes_temporal_new.ytd;return `<tr><td>${r.date}</td><td class="num">${U.fmt(b.prediction)}</td><td class="num">${U.pct(b.up_probability)}</td><td>${U.side(b.signal)}</td><td>${U.side(s.position[i])}</td><td>${U.side(s.decisions[i].next_position)}</td><td>${U.side(r.models.ridge_temporal.signal)}</td></tr>`;}).join('');
}
function ledger(){const key=el('ledger-model').value,reasons={max_hold:'满6日',reverse:'预测反向',open:'未平仓'};el('ledger').innerHTML=[...data.simulation.models[key].ytd.trades].reverse().map(t=>`<tr><td>${t.signal_date}</td><td>${t.entry_session}</td><td>${U.side(t.direction)}</td><td>${t.exit_session||'未平'}</td><td>${reasons[t.reason]||t.reason}</td><td class="num">${t.holding_sessions}</td><td class="num">${U.fmt(t.pnl)}</td></tr>`).join('');}
function render(){
 const m=data.meta,s=data.simulation,latest=data.latest,b=latest.models.bayes_temporal_new,models=s.models;
 el('status').classList.toggle('warning',!m.healthy);
 el('status').textContent=`仓库最新可计算信号：${m.asof}；最新源数据：${m.latest_source_date}。${m.healthy?'输入检查通过。':'暂停使用新方向：'+m.issue.problems.join('；')+'。'} 信号日之后的下一交易日执行；中午仅复核，不代表今日行情或账户已成交。`;
 el('signals').innerHTML=keys.map(k=>{const v=latest.models[k],sim=models[k].ytd,plan=sim.decisions.at(-1),prob=k==='bayes_temporal_new';return `<div class="card"><h2>${models[k].label} · ${prob?'主参考':'交叉复核'}</h2><div class="big ${v.signal===1?'up':'down'}">${U.side(v.signal)}</div><p>未来6日预测 <span class="value">${U.fmt(v.prediction,3)}</span> 点</p>${prob?`<div>走扩 <b>${U.pct(b.up_probability)}</b>　／　收窄 <b>${U.pct(b.down_probability)}</b></div><div class="probbar"><span style="width:${b.up_probability*100}%"></span><span style="width:${b.down_probability*100}%"></span></div><p class="muted">80%预测区间 [${U.fmt(b.interval80[0],3)}, ${U.fmt(b.interval80[1],3)}] 点${b.interval80[0]<0&&b.interval80[1]>0?'，仍跨0':''}；概率尚未校准。</p>`:'<p class="muted">Ridge展示预测方向与幅度，方向概率由贝叶斯模型提供。</p>'}<p>当日模拟：${U.side(sim.position.at(-1))}<br>下一交易日计划：${U.side(plan.next_position)}</p><small>基于${m.asof}收盘数据；仅模拟计划。</small></div>`;}).join('');
 el('metrics').innerHTML=keys.map(k=>{const v=models[k].ytd.metrics;return `<tr><td>${models[k].label}</td><td class="num">${U.fmt(v.total)}</td><td class="num">${U.fmt(v.max_drawdown)}</td><td class="num">${v.closed_trades}／${v.open_trades}</td><td class="num">${U.pct(v.win_rate)}</td></tr>`;}).join('');
 U.chart('ytd',s.dates,keys.map(k=>({key:k,label:models[k].label,data:models[k].ytd.equity})));
 U.chart('history',s.history_dates,keys.map(k=>({key:k,label:models[k].label,data:models[k].history.equity,segment:{borderDash:ctx=>ctx.p0DataIndex<s.history_train_n?[5,4]:undefined}})));
 U.chart('position',s.dates,keys.map(k=>({key:k,label:models[k].label,data:models[k].ytd.position})),{position:true});
 U.chart('probability',s.dates,[{key:'bayes_temporal_new',label:'六日走扩概率（%）',data:data.records.map(r=>r.models.bayes_temporal_new.up_probability*100)},{label:'50%',data:s.dates.map(()=>50),color:'#a6afbd',borderDash:[4,4]}],{y:{min:0,max:100}});
 el('boundary').textContent=`两页均使用新参数时序贝叶斯。2023—2025是训练期拟合回放，2026已被反复观察且因子名单含全历史选择；历史不是实盘业绩。新归档流程于${m.deployment_date}接入，目前有${m.forward_record_n}条部署日之后的逐日复核记录；其余均明确为研究回填。未扣成本，不自动下单。`;
 historyTable();ledger();
}
el('count').onchange=historyTable;el('ledger-model').onchange=ledger;el('reset').onclick=U.reset;
U.read('curve_signal_daily.json').then(d=>{data=d;render();}).catch(e=>{el('status').classList.add('warning');el('status').textContent='信号数据读取失败：'+e.message;});
})();
