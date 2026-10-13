'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '—').replace(/[&<>"']/g, char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
const fmt = (value, decimals = 3) => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(decimals);
const rank = value => value == null ? '—' : fmt(value, 1) + '%';
const dir = value => value === 1 ? '做多' : value === -1 ? '做空' : '空仓';
const cls = value => value === 1 ? 'long' : value === -1 ? 'short' : 'neutral';
const reasons = {weak:'中性信号，计划平仓', reverse:'方向反转，计划平仓并反向', missing_signal:'信号缺失，计划平仓', max_hold:'达到6日上限，收盘平仓'};
let data, models, expanded = false, comparisonLoading = false;

async function fetchJson(name) {
  const response = await fetch(name + '?v=' + Date.now(), {cache:'no-store'});
  if (!response.ok) throw Error(name + ' HTTP ' + response.status);
  return response.json();
}

function renderHistory() {
  $('history').innerHTML = (expanded ? data.records : data.records.slice(-10)).slice().reverse().map(row => {
    const a = row.models.f20, b = row.models.f30;
    const state = rec => rec.status === 'blocked' ? '缺失' : dir(rec.signal);
    const source = row.origin === 'research-backfill' ? '研究回填' : [a,b].some(x => x.status === 'blocked') ? '待补数据' : '每日归档';
    return `<tr><td>${esc(row.date)}</td><td>${fmt(row.level)}</td><td>${fmt(a.prediction)}</td><td>${rank(a.percentile)}</td><td class="${cls(a.signal)}">${state(a)}</td><td>${fmt(b.prediction)}</td><td>${rank(b.percentile)}</td><td class="${cls(b.signal)}">${state(b)}</td><td>${source}</td></tr>`;
  }).join('');
}

function renderInputs() {
  const key = $('inputModel').value;
  const record = data.latest.models[key];
  const inputs = record.inputs?.length ? record.inputs : models.models[key].features.map(f => ({...f, raw:null, z:null, contribution:null}));
  $('inputs').innerHTML = inputs.map(x => `<tr><td>${esc(x.label)}</td><td>${fmt(x.raw,4)}</td><td>${esc(x.unit)}</td><td class="num">${fmt(x.z)}</td><td class="num">${fmt(x.contribution,4)}</td><td>${!record.inputs?.length ? '当日输入未归档' : x.missing ? '训练中位数填补' : x.clipped ? '训练界限缩尾' : x.clip_enabled ? '未触及金额界限' : '不缩尾'}</td></tr>`).join('');
}

async function renderDaily() {
  try {
    [data, models] = await Promise.all(['ridge_daily_signals.json','ridge_daily_models.json'].map(fetchJson));
    const m = data.meta, row = data.latest;
    const problems = [...new Set([...(row.models.f20.problems || []), ...(row.models.f30.problems || []), ...(m.unpriced_latest?.problems || [])])];
    $('state').className = 'notice' + (m.healthy ? '' : ' error');
    $('state').innerHTML = `<b>信号日 ${esc(m.signal_date)}</b> · ${m.healthy ? '两组输入检查通过' : '数据待补，不是完整新信号'}<br><small>文件更新 ${esc(m.generated.replace('T',' ').replace('+08:00',''))}（北京时间）。${problems.length ? esc(problems.join('；')) : '当前页显示最近已完成的收盘数据。'}</small>`;
    $('cards').innerHTML = ['f20','f30'].map(key => {
      const a = row.models[key], p = data.positions[key], tr = p.open_trade;
      const action = reasons[p.exit_reason] || (p.next_position === 0 ? '继续空仓' : p.position_during_signal_day === p.next_position ? '同向继续持有' : '计划开仓');
      return `<article class="signal-card ${key === 'f20' ? 'main-signal' : ''}"><h3>${key === 'f20' ? '主信号 · 20因子＋8基准' : '对照 · 30因子＋8基准'}</h3><div class="metrics"><div><div class="value">${fmt(a.prediction)}</div><div class="label">未来6日变动（点）</div></div><div><div class="value">${rank(a.percentile)}</div><div class="label">历史预测排名</div></div><div><div class="value ${cls(a.signal)}">${a.status === 'blocked' ? '缺失' : dir(a.signal)}</div><div class="label">当日方向</div></div></div><p>信号日模拟：<b>${dir(p.position_during_signal_day)}</b> → 下一交易日目标：<b class="${cls(p.next_position)}">${dir(p.next_position)}</b></p><p>${esc(action)}。</p><small>${tr ? '当前模拟交易：' + esc(tr.entry_session) + '入场，已持有' + tr.holding_sessions + '个交易日。' : '暂无未结束的模拟交易。'}最后一天的次日计划未计为已成交。</small></article>`;
    }).join('');
    $('quality').innerHTML = `<p>机构日报 ${esc(m.source_dates.institution)} · 回购 ${esc(m.source_dates.repo)} · 曲线与期货 ${esc(m.source_dates.curve)}</p><p>7天OMO ${fmt(m.source_dates.omo_rate,2)}%，核验至${esc(m.source_dates.omo_verified)} · <a href="${esc(m.source_dates.omo_source)}" target="_blank" rel="noopener">来源</a></p><p>当日因子覆盖：20组${rank(row.models.f20.coverage == null ? null : row.models.f20.coverage * 100)}，30组${rank(row.models.f30.coverage == null ? null : row.models.f30.coverage * 100)}；要求≥90%，8基准全部完整。排名使用此前${row.models.f20.history_n}条有效预测。</p><p>模型 ${esc(m.model_version)}；训练截止${esc(models.meta.train_through)}，${models.meta.train_n}条标签，名单截至${esc(models.meta.selection_asof)}。不重新回归或筛选。</p><p>${m.price_revisions.length ? '发现' + m.price_revisions.length + '条历史价格修订，归档价格与已发预测未改写。' : '未发现历史归档价格修订。'}</p>`;
    renderInputs(); renderHistory();
    $('inputModel').onchange = renderInputs;
    $('all').onclick = () => { expanded = !expanded; renderHistory(); $('all').textContent = expanded ? '只显示最近10日' : '展开全部历史'; };
    $('download').onclick = () => {
      const url = URL.createObjectURL(new Blob([JSON.stringify(data,null,2)], {type:'application/json'}));
      const a = document.createElement('a'); a.href = url; a.download = 'ridge_daily_signals.json'; a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
  } catch (error) {
    $('state').className = 'notice error'; $('state').textContent = '每日信号读取失败：' + error.message;
  }
}

async function renderResearch() {
  try {
    const source = await fetchJson('curve_timing_capacity.json');
    $('researchState').textContent = `历史研究截至${source.meta.asof} · 训练${source.meta.train_n}条完整标签 · 2025-01-02起${source.meta.test_n}个测试预测日，其中420个已完成标签`;
    $('ridgeResults').innerHTML = ['f20','f30','f40'].map(key => {
      const f = source.forecast[key].test, s = source.strategies.dynamic[key].metrics;
      return `<tr><td>${key.slice(1)}因子＋8基准${key === 'f20' ? '（主）' : key === 'f30' ? '（对照）' : '（研究）'}</td><td class="num">${fmt(f.ic)}</td><td class="num">${fmt(f.rmse)}</td><td class="num">${fmt(s.total)}</td><td class="num">${fmt(s.max_drawdown)}</td></tr>`;
    }).join('');
  } catch (error) {
    $('researchState').textContent = '历史研究读取失败：' + error.message;
  }
}

async function renderComparisons() {
  if (!$('modelDetails').open || comparisonLoading) return;
  comparisonLoading = true; $('comparisonState').textContent = '正在读取历史模型与交易规则对照…';
  try {
    const [model, gate] = await Promise.all(['curve_timing_nonlinear.json','curve_timing_percentile.json'].map(fetchJson));
    const names = {ridge:'岭回归', tree:'浅层树', xgb:'XGBoost'};
    $('modelResults').innerHTML = model.meta.sizes.flatMap(n => ['ridge','tree','xgb'].map(family => {
      const key = family + n, f = model.forecast[key].test;
      const fixed = model.strategies.fixed[key].metrics, dynamic = model.strategies.dynamic[key].metrics;
      return `<tr><td>${names[family]}</td><td class="num">${n}＋8</td><td class="num">${fmt(f.ic)}</td><td class="num">${fmt(f.rmse)}</td><td class="num">${fmt(fixed.total)}</td><td class="num">${fmt(dynamic.total)}</td><td class="num">${fmt(dynamic.max_drawdown)}</td></tr>`;
    })).join('');
    $('gateResults').innerHTML = ['absolute','signed'].map(key => {
      const metrics = gate.strategies[key].dynamic.full.metrics;
      return `<tr><td>${key === 'absolute' ? '原绝对强度规则' : '历史排名40%／60%'}</td><td class="num">${fmt(metrics.total)}</td><td class="num">${fmt(metrics.max_drawdown)}</td><td class="num">${fmt(gate.periods[key].dynamic.full['2026'].total)}</td></tr>`;
    }).join('');
    $('comparisonState').textContent = `历史对照截至${model.meta.asof}，每日主信号仍由冻结岭回归生成。`;
    $('modelDetails').removeEventListener('toggle', renderComparisons);
  } catch (error) {
    comparisonLoading = false; $('comparisonState').textContent = '对照读取失败，重新展开可重试：' + error.message;
  }
}

$('modelDetails').addEventListener('toggle', renderComparisons);
renderDaily(); renderResearch();
