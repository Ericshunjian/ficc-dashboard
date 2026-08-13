(() => {
  'use strict';
  const dates = D.dates;
  const values = D.vals;
  const normalized = D.norm || values.map(v => v / values[0] * 100);
  const colors = { sub:'#91a5af', grid:'#20313d', blue:'#4dc4ff', red:'#ff6b7a', orange:'#ffb454', green:'#65d68a' };
  let triggerChart;
  let histogram;
  let debounceTimer;

  const byId = id => document.getElementById(id);
  const read = id => Number(byId(id).value);
  const pct = (x, digits=1) => Number.isFinite(x) ? `${(x * 100).toFixed(digits)}%` : '—';
  const num = (x, digits=1) => Number.isFinite(x) ? x.toFixed(digits) : '—';
  const clampInt = (x, lo, hi) => Math.min(hi, Math.max(lo, Math.round(x)));

  function quantile(xs, q) {
    if (!xs.length) return NaN;
    const sorted = [...xs].sort((a,b) => a-b);
    const pos = (sorted.length - 1) * q;
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }

  function wilson(hits, n) {
    if (!n) return [NaN, NaN];
    const z = 1.96, z2 = z*z, p = hits/n, den = 1 + z2/n;
    const mid = (p + z2/(2*n)) / den;
    const half = z * Math.sqrt(p*(1-p)/n + z2/(4*n*n)) / den;
    return [Math.max(0, mid-half), Math.min(1, mid+half)];
  }

  function rollingHighDrawdown(window) {
    const result = new Array(values.length).fill(null);
    const deque = [];
    for (let i=0; i<values.length; i++) {
      while (deque.length && deque[0] < i-window+1) deque.shift();
      while (deque.length && values[deque[deque.length-1]] <= values[i]) deque.pop();
      deque.push(i);
      result[i] = values[i] / values[deque[0]] - 1;
    }
    return result;
  }

  function buildEvents(dd, triggerLevel, horizon, cooldown, touchLevel, putLossLevel) {
    const events = [];
    let lastAccepted = -Infinity;
    for (let t=1; t<values.length; t++) {
      const crossed = dd[t] <= -triggerLevel && dd[t-1] > -triggerLevel;
      if (!crossed || t-lastAccepted < cooldown || t+horizon >= values.length) continue;
      let minValue = Infinity, minIndex = t+1;
      for (let i=t+1; i<=t+horizon; i++) {
        if (values[i] < minValue) { minValue = values[i]; minIndex = i; }
      }
      const worstReturn = minValue/values[t]-1;
      const terminalReturn = values[t+horizon]/values[t]-1;
      events.push({
        index:t, date:dates[t], dd:dd[t], entry:values[t], minValue, minDate:dates[minIndex], daysToMin:minIndex-t,
        worstReturn, terminalReturn, touched:worstReturn <= -touchLevel, terminalHit:terminalReturn <= -touchLevel,
        putLoss:terminalReturn < -putLossLevel
      });
      lastAccepted = t;
    }
    return events;
  }

  function kpi(label, value, detail, cls='blue') {
    return `<div class="kpi"><div class="l">${label}</div><div class="v ${cls}">${value}</div><div class="d">${detail}</div></div>`;
  }

  function renderCharts(events, horizon, extraDrop) {
    const points = events.map(e => ({x:e.date, y:normalized[e.index]}));
    const common = {responsive:true, maintainAspectRatio:false, animation:{duration:180}, plugins:{legend:{labels:{color:colors.sub}},tooltip:{mode:'nearest'}},scales:{x:{ticks:{color:colors.sub,maxTicksLimit:8},grid:{color:colors.grid}},y:{ticks:{color:colors.sub},grid:{color:colors.grid}}}};
    if (!triggerChart) {
      triggerChart = new Chart(byId('triggerChart'), {type:'line',data:{labels:dates,datasets:[
        {label:'沪深300（起点=100）',data:normalized,borderColor:colors.blue,borderWidth:1.2,pointRadius:0},
        {label:'条件事件',data:points,type:'scatter',backgroundColor:colors.red,borderColor:'#ffd2d7',pointRadius:4,pointHoverRadius:6}
      ]},options:common});
    } else { triggerChart.data.datasets[1].data = points; triggerChart.update('none'); }

    const edges = [-Infinity,-.20,-.15,-.10,-.075,-.05,-.025,0,.025,.05,.10,Infinity];
    const labels = ['≤-20%','-20~-15%','-15~-10%','-10~-7.5%','-7.5~-5%','-5~-2.5%','-2.5~0%','0~2.5%','2.5~5%','5~10%','>10%'];
    const counts = new Array(labels.length).fill(0);
    events.forEach(e => { for(let i=0;i<labels.length;i++) if(e.worstReturn>edges[i] && e.worstReturn<=edges[i+1]) { counts[i]++; break; } });
    const bars = labels.map((_,i) => edges[i+1] <= -extraDrop ? colors.red : edges[i] >= 0 ? colors.green : colors.orange);
    const data = {labels,datasets:[{label:`未来 ${horizon} 日内最深额外跌幅（事件数）`,data:counts,backgroundColor:bars,borderRadius:4}]};
    const options = {responsive:true,maintainAspectRatio:false,animation:{duration:180},plugins:{legend:{labels:{color:colors.sub}}},scales:{x:{ticks:{color:colors.sub,maxRotation:45,minRotation:45},grid:{display:false}},y:{beginAtZero:true,ticks:{color:colors.sub,precision:0},grid:{color:colors.grid}}}};
    if (!histogram) histogram = new Chart(byId('histChart'),{type:'bar',data,options});
    else { histogram.data = data; histogram.update('none'); }
  }

  function recompute() {
    const lookback = clampInt(read('lookback'),5,500);
    const triggerLevel = Math.max(.001,read('trigger')/100);
    const cooldown = clampInt(read('cooldown'),0,500);
    const horizon = clampInt(read('horizon'),1,500);
    const extraDrop = Math.max(.001,read('extraDrop')/100);
    const otm = Math.max(0,read('otm')/100);
    const premium = Math.max(0,read('premiumPct')/100);
    const putLossLevel = otm + premium;
    const dd = rollingHighDrawdown(lookback);
    const events = buildEvents(dd,triggerLevel,horizon,cooldown,extraDrop,putLossLevel);
    const n = events.length;
    const touches = events.filter(e=>e.touched).length;
    const terminalHits = events.filter(e=>e.terminalHit).length;
    const putLosses = events.filter(e=>e.putLoss).length;
    const touchCI = wilson(touches,n), termCI=wilson(terminalHits,n), putCI=wilson(putLosses,n);
    const worsts = events.map(e=>e.worstReturn);

    byId('dataRange').textContent = `${D.date0} — ${D.date1} · ${values.length.toLocaleString()}个交易日`;
    byId('formula').innerHTML = `<b>P（未来 ${horizon} 日内较触发价再跌 ≥ ${num(extraDrop*100,1)}%｜当前价较过去 ${lookback} 日高点首次回撤 ≥ ${num(triggerLevel*100,1)}%）</b><br>卖 Put 历史映射：虚值 ${num(otm*100,1)}% + 权利金 ${num(premium*100,1)}% → 到期盈亏平衡跌幅约 ${num(putLossLevel*100,1)}%。`;
    const status = byId('sampleStatus');
    status.textContent = n < 10 ? `有效样本 ${n} 个 · 样本很少` : n < 20 ? `有效样本 ${n} 个 · 区间较宽` : `有效样本 ${n} 个`;
    status.className = `status ${n<10?'bad':n<20?'warn':''}`;
    byId('kpis').innerHTML = [
      kpi(`期内触碰：再跌 ≥ ${num(extraDrop*100,1)}%`,n?pct(touches/n):'—',n?`${touches}/${n}；95%区间 ${pct(touchCI[0])}—${pct(touchCI[1])}`:'没有满足条件的完整事件','red'),
      kpi(`到期：跌幅 ≥ ${num(extraDrop*100,1)}%`,n?pct(terminalHits/n):'—',n?`${terminalHits}/${n}；95%区间 ${pct(termCI[0])}—${pct(termCI[1])}`:'没有满足条件的完整事件','orange'),
      kpi('卖 Put 到期亏损频率',n?pct(putLosses/n):'—',n?`${putLosses}/${n}；95%区间 ${pct(putCI[0])}—${pct(putCI[1])}`:'需要完整的 H 日样本','red'),
      kpi('期内最深再跌中位数',n?pct(quantile(worsts,.5),2):'—',n?`P10 ${pct(quantile(worsts,.1),2)} · 最差 ${pct(Math.min(...worsts),2)}`:'—','blue'),
      kpi('有效历史事件',String(n),`跨越触发阈值且间隔至少 ${cooldown} 个交易日`,'green')
    ].join('');

    byId('eventRows').innerHTML = events.length ? events.slice().reverse().map(e=>`<tr>
      <td>${e.date}</td><td>${pct(e.dd,2)}</td><td>${num(e.entry,2)}</td><td>${num(e.minValue,2)}</td>
      <td class="${e.touched?'hit':''}">${pct(e.worstReturn,2)}</td><td>${e.daysToMin}日<br><span style="color:var(--sub)">${e.minDate}</span></td>
      <td class="${e.terminalReturn<0?'neg':'pos'}">${pct(e.terminalReturn,2)}</td><td class="${e.touched?'hit':'safe'}">${e.touched?'是':'否'}</td>
      <td class="${e.putLoss?'hit':'safe'}">${e.putLoss?'亏损':'盈利/持平'}</td></tr>`).join('') : '<tr><td colspan="9" class="empty">当前参数下没有完整历史事件，请降低触发阈值或缩短观察期。</td></tr>';
    renderCharts(events,horizon,extraDrop);
  }

  document.querySelectorAll('input').forEach(input => input.addEventListener('input',() => { clearTimeout(debounceTimer); debounceTimer=setTimeout(recompute,120); }));
  recompute();
})();
