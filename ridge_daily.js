'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const fmt = (value, n = 3) => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(n);
const rank = value => value == null ? '—' : fmt(value,1) + '%';
const dir = value => value === 1 ? '做多' : value === -1 ? '做空' : '空仓';
const cls = value => value === 1 ? 'long' : value === -1 ? 'short' : 'neutral';
const names = {ridge:'岭回归',tree:'浅层树',xgb:'XGBoost'};
const colors = ['#5470c6','#16847b','#d38b37'];
const reasons = {weak:'信号转中性',reverse:'方向反转',missing_signal:'信号缺失',max_hold:'满6日到期',open:'未平仓'};
const charts = {};
const PAGE_SIZE = 15;
let data, models, research, signalPage = 0, tradePage = 0, comparisonLoaded = false;

async function fetchJson(name) {
  const response = await fetch(name + '?v=' + Date.now(), {cache:'no-store'});
  if (!response.ok) throw Error(name + ' HTTP ' + response.status);
  return response.json();
}

function line(id, labels, series, kind = 'points') {
  if (charts[id]) charts[id].destroy();
  const datasets = series.map((s,i) => ({label:s.label,data:s.values,borderColor:s.color || colors[i % colors.length],
    backgroundColor:s.color || colors[i % colors.length],borderWidth:s.dashed ? 1.2 : 1.8,
    borderDash:s.dashed ? [5,4] : [],pointRadius:0,pointHoverRadius:3,stepped:kind === 'signal',spanGaps:false,fill:false}));
  const y = {title:{display:true,text:kind === 'percentile' ? '预测百分位（%）' : kind === 'signal' ? '方向' : kind === 'drawdown' ? '回撤（点）' : '累计盈亏（点）'},grid:{color:'#edf0f4'}};
  if (kind === 'percentile') { y.min=0; y.max=100; }
  if (kind === 'signal') { y.min=-1; y.max=1; y.ticks={stepSize:1,callback:v => ({'-1':'做空','0':'空仓','1':'做多'}[v])}; }
  charts[id] = new Chart($(id), {type:'line',data:{labels,datasets},options:{responsive:true,maintainAspectRatio:false,animation:false,
    onResize:chart=>{chart.options.scales.x.ticks.maxTicksLimit=chart.width<450?4:6;},
    interaction:{mode:'index',intersect:false},plugins:{legend:{position:'bottom',labels:{boxWidth:12,font:{size:11}}},
      tooltip:{callbacks:{label:ctx => ctx.dataset.label + '：' + (kind === 'signal' ? dir(ctx.parsed.y) : fmt(ctx.parsed.y,kind === 'percentile' ? 1 : 4) + (kind === 'percentile' ? '%' : ' 点'))}}},
    scales:{x:{grid:{display:false},ticks:{maxTicksLimit:innerWidth<600?4:6,maxRotation:0,font:{size:10}}},y}}});
}

function csv(rows, name) {
  if (!rows.length) return;
  const fields=Object.keys(rows[0]), quote=value => '"' + String(value ?? '').replace(/"/g,'""') + '"';
  const content=[fields.map(quote).join(','),...rows.map(r => fields.map(f => quote(r[f])).join(','))].join('\r\n');
  download(new Blob(['\ufeff'+content],{type:'text/csv;charset=utf-8'}),name);
}

function download(blob, name) {
  const url=URL.createObjectURL(blob), a=document.createElement('a'); a.href=url; a.download=name; a.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
}

function paginate(rows, page, prevId, nextId, hintId) {
  const count=Math.max(1,Math.ceil(rows.length/PAGE_SIZE)), current=Math.min(page,count-1);
  $(prevId).disabled=current===0; $(nextId).disabled=current>=count-1;
  $(hintId).textContent=`${rows.length}条 · 最新在前 · ${current+1}/${count}页`;
  return {current,rows:rows.slice(current*PAGE_SIZE,(current+1)*PAGE_SIZE)};
}

function renderHistory() {
  if (!data) return;
  const rows=data.records.map((row,i) => ({row,i})).reverse();
  const page=paginate(rows,signalPage,'signalPrev','signalNext','signalPageHint'); signalPage=page.current;
  $('history').innerHTML=page.rows.map(({row,i}) => {
    const a=row.models.f20,b=row.models.f30,state=r => r.status==='blocked' ? '缺失' : dir(r.signal);
    const source=row.origin==='research-backfill' ? '研究回填' : [a,b].some(r => r.status==='blocked') ? '待补' : '每日归档';
    return `<tr><td>${esc(row.date)}</td><td class="num">${fmt(row.level)}</td><td class="num">${fmt(a.prediction)}</td><td class="num">${rank(a.percentile)}</td><td class="${cls(a.signal)}">${state(a)}</td><td>${dir(data.strategies.f20.position[i])}</td><td class="num">${fmt(b.prediction)}</td><td class="num">${rank(b.percentile)}</td><td class="${cls(b.signal)}">${state(b)}</td><td>${source}</td></tr>`;
  }).join('');
}

function ledger() {
  const choice=$('ledgerSource').value;
  if (choice==='research') {
    if (!research) return null;
    const key=$('family').value+$('size').value,rule=$('rule').value;
    return {sim:research.strategies[rule][key],label:`历史核对：${names[$('family').value]} · ${$('size').value}＋8 · ${rule==='dynamic' ? '动态离场' : '固定6日'}`,
      dates:research.daily.dates,asof:research.meta.asof,key};
  }
  if (!data) return null;
  const key=choice==='daily30' ? 'f30' : 'f20';
  return {sim:data.strategies[key],label:`每日${key.slice(1)}因子＋8 · 动态离场`,dates:data.records.map(r=>r.date),asof:data.meta.signal_date,key};
}

function renderTrades() {
  const source=ledger(); if (!source) { $('ledgerHint').textContent='所选账本尚未加载。'; return; }
  const sim=source.sim,rows=sim.trades.slice().reverse();
  const page=paginate(rows,tradePage,'tradePrev','tradeNext','tradePageHint'); tradePage=page.current;
  const plan=sim.decisions.at(-1);
  $('ledgerHint').textContent=`${source.label} · 截至${source.asof} · 已平${sim.metrics.closed_trades}笔，未平${sim.metrics.open_trades}笔 · 盈亏${fmt(sim.metrics.total)}点，最大回撤${fmt(sim.metrics.max_drawdown)}点`;
  $('trades').innerHTML=page.rows.map(t=>{
    const pending=!t.closed && plan.exit_reason;
    const exitSignal=t.exit_signal_date || (pending ? source.asof+'（计划）' : '—');
    const exitSession=t.exit_session || (pending ? '下一交易日（计划）' : '—');
    const status=t.closed ? (t.exit_execution ? t.exit_execution+' · ' : '')+(reasons[t.reason] || t.reason) : '持仓中'+(pending ? ' · '+(reasons[plan.exit_reason] || plan.exit_reason)+'待执行' : '');
    return `<tr><td>${esc(t.signal_date)}</td><td>${esc(t.entry_session)}</td><td class="${cls(t.direction)}">${dir(t.direction)}</td><td class="num">${rank(t.entry_percentile)}</td><td class="num">${fmt(t.entry_price,4)}</td><td>${esc(exitSignal)}</td><td>${esc(exitSession)}</td><td class="num">${fmt(t.exit_price,4)}</td><td class="num">${t.holding_sessions}</td><td class="num ${t.pnl>=0?'long':'short'}">${fmt(t.pnl,4)}</td><td>${esc(status)}</td></tr>`;
  }).join('');
}

function renderInputs() {
  if (!data || !models) return;
  const key=$('inputModel').value,record=data.latest.models[key];
  const inputs=record.inputs?.length ? record.inputs : models.models[key].features.map(f=>({...f,raw:null,z:null,contribution:null}));
  $('inputs').innerHTML=inputs.map(x=>`<tr><td>${esc(x.label)}</td><td>${fmt(x.raw,4)}</td><td>${esc(x.unit)}</td><td class="num">${fmt(x.z)}</td><td class="num">${fmt(x.contribution,4)}</td><td>${!record.inputs?.length?'未归档':x.missing?'训练中位数':x.clipped?'金额缩尾':x.clip_enabled?'未触及金额界限':'不缩尾'}</td></tr>`).join('');
}

function renderDailyCharts() {
  const dates=data.records.map(r=>r.date),arms=['f20','f30'];
  line('drawdownChart',dates,arms.map((key,i)=>({label:`${key.slice(1)}因子＋8`,values:data.strategies[key].drawdown,color:colors[i]})), 'drawdown');
  $('drawdownHint').textContent=`截至${data.meta.signal_date} · 20因子最大回撤 ${fmt(data.strategies.f20.metrics.max_drawdown)}点；30因子 ${fmt(data.strategies.f30.metrics.max_drawdown)}点。均为动态离场。`;
  line('percentileChart',dates,[...arms.map((key,i)=>({label:`${key.slice(1)}因子预测排名`,values:data.records.map(r=>r.models[key].percentile),color:colors[i]})),
    {label:'40% 做空门槛',values:dates.map(()=>40),dashed:true,color:'#b45f3d'},
    {label:'60% 做多门槛',values:dates.map(()=>60),dashed:true,color:'#8c96a5'}],'percentile');
  $('percentileHint').textContent=`截至${data.meta.signal_date} · 预测值相对历史的排名，不是单个原始因子的分位。40%—60%为空仓区间。`;
  line('signalChart',dates,[{label:'20因子收盘方向信号',values:data.records.map(r=>r.models.f20.signal),color:colors[0]},
    {label:'20因子当日模拟持仓',values:data.strategies.f20.position,color:'#8c96a5',dashed:true},
    {label:'30因子收盘方向信号',values:data.records.map(r=>r.models.f30.signal),color:colors[1]}],'signal');
  $('signalHint').textContent=`截至${data.meta.signal_date} · 收盘信号决定下一交易日方向；模拟持仓为当日计盈亏方向，可能与收盘信号不同。`;
}

async function loadDaily() {
  try {
    [data,models]=await Promise.all(['ridge_daily_signals.json','ridge_daily_models.json'].map(fetchJson));
    if (!data.strategies) throw Error('每日账本尚未更新，请运行每日生成器');
    const m=data.meta,row=data.latest;
    const problems=[...new Set([...(row.models.f20.problems||[]),...(row.models.f30.problems||[]),...(m.unpriced_latest?.problems||[])])];
    $('state').className='notice'+(m.healthy?'':' error');
    $('state').innerHTML=`<b>信号日 ${esc(m.signal_date)}</b> · ${m.healthy?'两组输入检查通过':'数据待补，不是完整新信号'}<br><small>文件更新 ${esc(m.generated.replace('T',' ').replace('+08:00',''))}（北京时间）。${problems.length?esc(problems.join('；')):'显示最近已完成的收盘数据。'}</small>`;
    $('cards').innerHTML=['f20','f30'].map(key=>{
      const a=row.models[key],p=data.positions[key],tr=p.open_trade;
      const action=p.exit_reason==='max_hold' ? '满6日到期，收盘平仓；新交易按次日目标执行' : p.exit_reason ? (reasons[p.exit_reason]||p.exit_reason)+'，计划'+(p.next_position===0?'平仓':'反手') : p.next_position===0?'继续空仓':p.position_during_signal_day===p.next_position?'同向持有':'计划开仓';
      return `<article class="signal-card ${key==='f20'?'main-signal':''}"><h3>${key==='f20'?'主信号 · 20因子＋8基准':'对照 · 30因子＋8基准'}</h3><div class="metrics"><div><div class="value">${fmt(a.prediction)}</div><div class="label">未来6日变动（点）</div></div><div><div class="value">${rank(a.percentile)}</div><div class="label">历史预测百分位</div></div><div><div class="value ${cls(a.signal)}">${a.status==='blocked'?'缺失':dir(a.signal)}</div><div class="label">当日多空信号</div></div></div><p>信号日模拟：<b>${dir(p.position_during_signal_day)}</b> → 次日目标：<b class="${cls(p.next_position)}">${dir(p.next_position)}</b>。${esc(action)}。</p><small>${tr?'当前模拟交易 '+esc(tr.entry_session)+'入场，持有'+tr.holding_sessions+'日。':'暂无未结束的模拟交易。'}次日计划未计为已成交。</small></article>`;
    }).join('');
    $('quality').innerHTML=`<p>机构日报 ${esc(m.source_dates.institution)} · 回购 ${esc(m.source_dates.repo)} · 曲线与期货 ${esc(m.source_dates.curve)}</p><p>7天OMO ${fmt(m.source_dates.omo_rate,2)}%，核验至${esc(m.source_dates.omo_verified)} · <a href="${esc(m.source_dates.omo_source)}" target="_blank" rel="noopener">来源</a></p><p>因子覆盖：20组 ${rank(row.models.f20.coverage==null?null:row.models.f20.coverage*100)}，30组 ${rank(row.models.f30.coverage==null?null:row.models.f30.coverage*100)}；要求≥90%，8基准全部完整。</p><p>模型 ${esc(m.model_version)}；训练截止${esc(models.meta.train_through)}，${models.meta.train_n}条标签；名单截至${esc(models.meta.selection_asof)}，不重新回归或筛选。</p><p>${m.price_revisions.length?'发现'+m.price_revisions.length+'条历史价格修订，已归档价格和预测未改写。':'未发现归档价格修订。'}</p>`;
    renderDailyCharts(); renderHistory(); renderInputs(); renderTrades();
  } catch(error) { $('state').className='notice error'; $('state').textContent='每日信号读取失败：'+error.message; }
}

function researchSim(key) { return research.strategies[$('rule').value][key]; }

function renderResearch() {
  if (!research) return;
  const size=$('size').value,family=$('family').value,key=family+size,rule=$('rule').value;
  const metric=researchSim(key).metrics,forecast=research.forecast[key].test;
  $('researchState').textContent=`研究截至${research.meta.asof} · 训练${research.meta.train_n}条标签 · 测试2025-01-02起${research.meta.test_n}日 · 当前核对：${names[family]} ${size}＋8 / ${rule==='dynamic'?'动态离场':'固定6日'}`;
  $('performance').innerHTML=[['累计盈亏',fmt(metric.total)+' 点','包含期末浮动盈亏'],['最大回撤',fmt(metric.max_drawdown)+' 点','完整历史研究区间'],['预测IC',fmt(forecast.ic,4),forecast.n+'个完成标签'],['已平仓 / 未平仓',metric.closed_trades+' / '+metric.open_trades,'核对模型的历史账本']].map(r=>`<div class="metric"><small>${r[0]}</small><b>${r[1]}</b><small>${r[2]}</small></div>`).join('');
  line('modelEquityChart',research.daily.dates,['ridge','tree','xgb'].map((f,i)=>({label:names[f],values:researchSim(f+size).equity,color:colors[i]})));
  line('factorEquityChart',research.daily.dates,[20,30,40].map((n,i)=>({label:n+'因子＋8',values:researchSim(family+n).equity,color:colors[i]})));
  $('modelChartHint').textContent=`相同${size}因子＋8基准 · ${rule==='dynamic'?'动态离场':'固定6日'} · 累计盈亏（点）`;
  $('factorChartHint').textContent=`相同${names[family]} · ${rule==='dynamic'?'动态离场':'固定6日'} · 累计盈亏（点）`;
  $('researchSummary').textContent='研究截至'+research.meta.asof+' · 以下为动态离场的固定历史对照';
  $('ridgeResults').innerHTML=[20,30,40].map(n=>{const f=research.forecast['ridge'+n].test,s=research.strategies.dynamic['ridge'+n].metrics;return `<tr><td>${n}因子＋8基准</td><td class="num">${fmt(f.ic)}</td><td class="num">${fmt(f.rmse)}</td><td class="num">${fmt(s.total)}</td><td class="num">${fmt(s.max_drawdown)}</td></tr>`;}).join('');
  $('modelResults').innerHTML=research.meta.sizes.flatMap(n=>['ridge','tree','xgb'].map(f=>{
    const k=f+n,forecast=research.forecast[k].test,a=research.strategies.fixed[k].metrics,b=research.strategies.dynamic[k].metrics;
    return `<tr><td>${names[f]}</td><td class="num">${n}＋8</td><td class="num">${fmt(forecast.ic)}</td><td class="num">${fmt(forecast.rmse)}</td><td class="num">${fmt(a.total)}</td><td class="num">${fmt(b.total)}</td><td class="num">${fmt(b.max_drawdown)}</td></tr>`;
  })).join('');
  $('comparisonState').textContent=`历史模型结果截至${research.meta.asof}；每日主模型仍为20因子岭回归、动态离场。`;
  if ($('ledgerSource').value==='research') { tradePage=0; renderTrades(); }
}

async function loadResearch() {
  try { research=await fetchJson('curve_timing_nonlinear.json'); renderResearch(); }
  catch(error) { $('researchState').textContent='历史研究读取失败：'+error.message; }
}

async function loadGateComparison() {
  if (!$('modelDetails').open || comparisonLoaded) return;
  comparisonLoaded=true;
  try {
    const source=await fetchJson('curve_timing_percentile.json');
    $('gateResults').innerHTML=['absolute','signed'].map(key=>{const m=source.strategies[key].dynamic.full.metrics;return `<tr><td>${key==='absolute'?'原绝对强度规则':'历史排名40%／60%'}</td><td class="num">${fmt(m.total)}</td><td class="num">${fmt(m.max_drawdown)}</td><td class="num">${fmt(source.periods[key].dynamic.full['2026'].total)}</td></tr>`;}).join('');
  } catch(error) { comparisonLoaded=false; $('comparisonState').textContent='旧规则对照读取失败：'+error.message; }
}

['size','family','rule'].forEach(id=>$(id).onchange=renderResearch);
$('ledgerSource').onchange=()=>{tradePage=0;renderTrades();};
$('inputModel').onchange=renderInputs;
$('signalPrev').onclick=()=>{signalPage--;renderHistory();}; $('signalNext').onclick=()=>{signalPage++;renderHistory();};
$('tradePrev').onclick=()=>{tradePage--;renderTrades();}; $('tradeNext').onclick=()=>{tradePage++;renderTrades();};
$('download').onclick=()=>{if(data)download(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),'ridge_daily_signals.json');};
$('dailyCsv').onclick=()=>{if(data)csv(data.records.map((r,i)=>({date:r.date,level:r.level,f20_prediction:r.models.f20.prediction,f20_percentile:r.models.f20.percentile,f20_signal:r.models.f20.signal,f20_position:data.strategies.f20.position[i],f30_prediction:r.models.f30.prediction,f30_percentile:r.models.f30.percentile,f30_signal:r.models.f30.signal,f30_position:data.strategies.f30.position[i]})),'3TTL_daily_signals.csv');};
$('tradeCsv').onclick=()=>{const source=ledger();if(source)csv(source.sim.trades,'3TTL_trades_'+source.key+'.csv');};
$('modelDetails').addEventListener('toggle',loadGateComparison);
loadDaily(); loadResearch();
