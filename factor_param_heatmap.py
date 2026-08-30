# -*- coding: utf-8 -*-
"""因子参数高原热力图生成器

用法：直接运行，产出 factor_param_heatmap.html
改扫描对象：改底部 CONFIG 区的 FACTOR / TARGET / THRESHOLDS / WINDOWS

口径（与 2026-08-28 那轮参数扫描保持一致）：
  信号   cross_up（因子从下方上穿阈值），reset 去重（回落阈值下方才允许再触发）
  方向   收益率下行=赢，收益 = y[建仓] − y[平仓]（bp）
  滞后   lag=0 → T 日建仓（原口径）；lag=1 → T+1 建仓（可执行口径，机构行为 T+1 可得）
  高原   取最优格的 8 邻域（阈值±5、窗口±1）t 均值判定强/弱高原或孤立尖峰
"""
import json
import numpy as np
import pandas as pd

BASE = r'C:\Users\lihaoran\WorkBuddy\2026-06-22-11-07-40'

# ── CONFIG ─────────────────────────────────────────────
FACTOR = '基金·超长债因子'          # factor_data.json 中的因子名
TARGET = '30年国债'                 # yield_curve_data.json 中的曲线名
THRESHOLDS = list(range(0, 100, 5))
WINDOWS = list(range(3, 21))
BONF = 4.14                         # Bonferroni 校正后 |t| 门槛
# ──────────────────────────────────────────────────────


def to_series(o):
    if isinstance(o, dict) and 'dates' in o:
        return pd.Series(o['values'], index=pd.to_datetime(o['dates']))
    raise ValueError(f'unknown series structure: {list(o)[:5]}')


fd = json.load(open(BASE + r'\factor_data.json', encoding='utf-8'))
cd = json.load(open(BASE + r'\yield_curve_data.json', encoding='utf-8'))

f = to_series(fd['series'][FACTOR]).dropna().sort_index()
y = to_series(cd['series'][TARGET]).dropna().sort_index()
df = pd.DataFrame({'f': f, 'y': y}).dropna().sort_index()
fv = df['f'].values
yv = df['y'].values * 100.0
idx = df.index
print(f'因子 {FACTOR}: n={len(f)}')
print(f'标的 {TARGET}: n={len(y)}')
print(f'对齐后: n={len(df)} {idx[0].date()} ~ {idx[-1].date()}')


def scan(thr, win, lag=0):
    n = len(fv)
    above = fv > thr
    rets, dates = [], []
    armed = True
    for i in range(1, n):
        if above[i] and (not above[i - 1]) and armed:
            i0, i1 = i + lag, i + lag + win
            if i1 < n:
                rets.append(yv[i0] - yv[i1])
                dates.append(idx[i])
            armed = False
        elif not above[i]:
            armed = True
    if len(rets) < 5:
        return None
    a = np.array(rets)
    w, l = a[a > 0], a[a < 0]
    mean = float(a.mean())
    std = float(a.std(ddof=1))
    t = mean / (std / np.sqrt(len(a))) if std > 0 else 0.0
    yr = pd.Series(dates).dt.year.value_counts().to_dict()
    full = [yr.get(yy, 0) for yy in (2022, 2023, 2024, 2025)]
    return {'n': len(a), 'mean': mean, 't': float(t),
            'winrate': float(w.size / a.size),
            'pl': float(w.mean() / abs(l.mean())) if w.size and l.size else None,
            'years': {str(k): int(v) for k, v in sorted(yr.items())},
            'pass_sample': bool(min(full) >= 5)}


GRIDS = {}
for lag in (0, 1):
    GRIDS[str(lag)] = {f'{thr}|{win}': r
                       for thr in THRESHOLDS for win in WINDOWS
                       if (r := scan(thr, win, lag))}


def analyse(g):
    items = [(k, v) for k, v in g.items() if v['pass_sample']]
    best_k, best_v = max(items, key=lambda kv: kv[1]['t'])
    bt, bw = [int(x) for x in best_k.split('|')]

    def at(t, w):
        return g.get(f'{t}|{w}', {}).get('t')

    nb = [x for x in (at(bt + dt, bw + dw) for dt in (-5, 0, 5) for dw in (-1, 0, 1))
          if x is not None]
    if best_v['t'] in nb:
        nb.remove(best_v['t'])
    nb_avg = sum(nb) / len(nb) if nb else 0.0
    row = [x for x in (at(bt, w) for w in WINDOWS) if x is not None]
    col = [x for x in (at(t, bw) for t in THRESHOLDS) if x is not None]
    allt = [v['t'] for v in g.values()]

    if nb_avg >= 3.0 and nb_avg >= 0.75 * best_v['t']:
        verdict, vcls = '强高原', 'good'
    elif nb_avg >= 2.0:
        verdict, vcls = '弱高原', 'mid'
    else:
        verdict, vcls = '孤立尖峰', 'bad'

    return {'best_thr': bt, 'best_win': bw, 'best': best_v,
            'nb_avg': round(nb_avg, 2),
            'row_avg': round(sum(row) / len(row), 2),
            'col_avg': round(sum(col) / len(col), 2),
            'global_avg': round(sum(allt) / len(allt), 2),
            'verdict': verdict, 'vcls': vcls,
            'n_sig': len([1 for _, v in items if v['t'] >= 2]),
            'n_pass': len(items)}


ANA = {lag: analyse(g) for lag, g in GRIDS.items()}
a0, a1 = ANA['0'], ANA['1']
b0 = a0['best']
SAME1 = GRIDS['1'][f"{a0['best_thr']}|{a0['best_win']}"]
decay_m = (1 - SAME1['mean'] / b0['mean']) * 100 if b0['mean'] else 0
decay_t = (1 - SAME1['t'] / b0['t']) * 100 if b0['t'] else 0

if a0['verdict'] == '强高原':
    VERDICT_TXT = (f"结论：阈值 {a0['best_thr']} 一带是连片高原 —— 8 邻域 t 均值 "
                   f"{a0['nb_avg']}、该阈值整行均值 {a0['row_avg']}，而全局均值仅 "
                   f"{a0['global_avg']}。参数选择有结构支撑，不是孤立尖峰，过拟合风险可控。")
elif a0['verdict'] == '弱高原':
    VERDICT_TXT = (f"结论：弱高原 —— 邻域 t 均值 {a0['nb_avg']}，"
                   f"行均值 {a0['row_avg']}，全局 {a0['global_avg']}。有一定结构但不够稳。")
else:
    VERDICT_TXT = (f"结论：孤立尖峰 —— 邻域 t 均值仅 {a0['nb_avg']}，"
                   f"最优格 {b0['t']:.2f} 显著高于周边，过拟合嫌疑大，当前参数不可用。")
VERDICT_TXT += (f"　但最优 t={b0['t']:.2f} 仍未达 Bonferroni 门槛 {BONF}；"
                f"T+1 可执行口径下同参数 t 降至 {SAME1['t']:.2f}（−{decay_t:.0f}%）、"
                f"收益缩水 {decay_m:.0f}%。建议样本外跟踪后再考虑上仓位。")

PAYLOAD = json.dumps({'thr': THRESHOLDS, 'win': WINDOWS, 'grids': GRIDS,
                      'ana': ANA, 'bonf': BONF, 'factor': FACTOR,
                      'target': TARGET,
                      'range': [str(idx[0].date()), str(idx[-1].date())]},
                     ensure_ascii=False, separators=(',', ':'))

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>因子参数高原热力图 · {FACTOR} → {TARGET}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;background:#f5f6f8;color:#1f2328;padding:24px;line-height:1.6}}
.wrap{{max-width:1180px;margin:0 auto}}
h1{{font-size:20px;font-weight:500;margin-bottom:4px}}
.sub{{font-size:13px;color:#6b7280;margin-bottom:14px}}
.banner{{background:#fff;border:1px solid #e5e7eb;border-left:3px solid #0f6e56;border-radius:8px;padding:12px 16px;margin-bottom:16px;font-size:13px;color:#374151;line-height:1.75}}
.banner.mid{{border-left-color:#ba7517}}
.banner.bad{{border-left-color:#a32d2d}}
.card{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:18px 20px;margin-bottom:16px}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}}
.kpi{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px 16px}}
.kpi .lbl{{font-size:12px;color:#6b7280;margin-bottom:6px}}
.kpi .val{{font-size:22px;font-weight:500;letter-spacing:-.3px}}
.kpi .note{{font-size:11px;color:#9ca3af;margin-top:4px;line-height:1.5}}
.good{{color:#0f6e56}} .mid{{color:#ba7517}} .bad{{color:#a32d2d}}
.ctrls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:14px}}
.seg{{display:inline-flex;border:1px solid #d1d5db;border-radius:7px;overflow:hidden}}
.seg button{{border:0;background:#fff;padding:6px 14px;font-size:13px;cursor:pointer;color:#374151;font-family:inherit}}
.seg button+button{{border-left:1px solid #d1d5db}}
.seg button.on{{background:#1f2328;color:#fff}}
.lbl2{{font-size:12px;color:#6b7280;margin-right:2px}}
table{{border-collapse:separate;border-spacing:2px;font-size:11px}}
th{{font-weight:500;color:#6b7280;font-size:11px;padding:2px 4px}}
th.rowh{{text-align:right;padding-right:8px;white-space:nowrap}}
td{{width:46px;height:32px;text-align:center;border-radius:4px;cursor:pointer;position:relative;font-variant-numeric:tabular-nums}}
td.null{{background:#f3f4f6;color:#d1d5db;cursor:default}}
td.low{{border:1px dashed #9ca3af}}
td.best{{outline:2px solid #1f2328;outline-offset:1px}}
.legend{{display:flex;gap:14px;align-items:center;font-size:11px;color:#6b7280;margin-top:12px;flex-wrap:wrap}}
.sw{{display:inline-block;width:16px;height:11px;border-radius:2px;vertical-align:-1px;margin-right:4px}}
#detail{{margin-top:14px;padding:12px 14px;background:#fafafa;border:1px solid #e5e7eb;border-radius:8px;font-size:12px;color:#374151;min-height:60px}}
#detail b{{font-weight:500;color:#1f2328}}
.note{{font-size:12px;color:#6b7280;line-height:1.8}}
.note code{{background:#f3f4f6;padding:1px 5px;border-radius:3px;font-size:11px}}
.scroll{{overflow-x:auto}}
</style>
</head>
<body>
<div class="wrap">
<h1>因子参数高原热力图</h1>
<div class="sub">{FACTOR} → {TARGET}　·　样本 {idx[0].date()} ~ {idx[-1].date()}　·　cross_up 上穿 + reset 去重 + 收益率下行=赢　·　生成于 {pd.Timestamp.now():%Y-%m-%d %H:%M}</div>
<div class="banner {a0['vcls']}">{VERDICT_TXT}</div>

<div class="kpis">
  <div class="kpi"><div class="lbl">T+0 最优组合</div>
    <div class="val">阈值{a0['best_thr']} · {a0['best_win']}日</div>
    <div class="note">t={b0['t']:.2f}　n={b0['n']}　胜率{b0['winrate']*100:.1f}%　{b0['mean']:+.2f}bp</div></div>
  <div class="kpi"><div class="lbl">高原判定（T+0）</div>
    <div class="val {a0['vcls']}">{a0['verdict']}</div>
    <div class="note">8邻域 t={a0['nb_avg']}　阈值{a0['best_thr']}行均值 {a0['row_avg']}　全局 {a0['global_avg']}</div></div>
  <div class="kpi"><div class="lbl">显著组合数</div>
    <div class="val">{a0['n_sig']} <span style="font-size:13px;color:#9ca3af">/ {a0['n_pass']}</span></div>
    <div class="note">样本量达标且 |t|≥2　（Bonferroni 门槛 {BONF}）</div></div>
  <div class="kpi"><div class="lbl">T+1 可执行（同参数对照）</div>
    <div class="val">t={SAME1['t']:.2f}</div>
    <div class="note">阈值{a0['best_thr']}·{a0['best_win']}日：t {b0['t']:.2f}→{SAME1['t']:.2f}　收益 {b0['mean']:.2f}→{SAME1['mean']:.2f}bp（−{decay_m:.0f}%）</div></div>
</div>

<div class="card">
  <div class="ctrls">
    <span class="lbl2">建仓时点</span>
    <span class="seg" id="segLag"><button data-v="0" class="on">T+0 原口径</button><button data-v="1">T+1 可执行</button></span>
    <span class="lbl2" style="margin-left:14px">着色指标</span>
    <span class="seg" id="segMetric"><button data-v="t" class="on">t 值</button><button data-v="winrate">胜率</button><button data-v="mean">平均收益</button><button data-v="n">触发次数</button></span>
  </div>
  <div class="scroll"><table id="hm"></table></div>
  <div class="legend" id="legend"></div>
  <div id="detail">点击任意格子查看该组合明细。</div>
</div>

<div class="card">
  <div class="note">
    <b>口径</b>：信号 = 因子从下方上穿阈值（cross_up）；去重 = reset，必须回落到阈值下方才允许下次触发；
    收益 = <code>y[建仓] − y[平仓]</code>（bp），下行记正；t 值为单样本 t 检验（H₀: 均值=0）。<br>
    <b>T+1 口径</b>：机构行为数据 T+1 可得，故建仓日顺延 1 个交易日，用于检验信号是否真的吃得到。<br>
    <b>虚线边框</b> = 样本量不达标（2022–2025 完整年中有某年触发少于 5 次）。黑框 = 该口径下 t 值最优组合。<br>
    <b>高原判读</b>：取最优格的 8 邻域（阈值 ±5、窗口 ±1）t 均值，≥3.0 且 ≥最优值的 75% 记为强高原，
    ≥2.0 记弱高原，否则为孤立尖峰（过拟合嫌疑）。<br>
    <b>多重检验</b>：共 {len(THRESHOLDS)}×{len(WINDOWS)}×2 = {len(THRESHOLDS)*len(WINDOWS)*2} 个组合，
    Bonferroni 校正后显著性门槛约 |t| ≥ {BONF}。
  </div>
</div>
</div>

<script>
const D = {PAYLOAD};
let lag = '0', metric = 't';

function color(v, m) {{
  if (v === null || v === undefined) return ['#f3f4f6', '#d1d5db'];
  if (m === 't') {{
    if (v >= 4.14) return ['#a32d2d', '#fff'];
    if (v >= 3.0) return ['#e24b4a', '#fff'];
    if (v >= 2.0) return ['#f09595', '#501313'];
    if (v >= 1.0) return ['#f7c1c1', '#501313'];
    if (v >= 0)   return ['#fcebeb', '#501313'];
    if (v >= -2)  return ['#e6f1fb', '#042c53'];
    return ['#85b7eb', '#042c53'];
  }}
  if (m === 'winrate') {{
    const p = v * 100;
    if (p >= 70) return ['#a32d2d', '#fff'];
    if (p >= 62) return ['#e24b4a', '#fff'];
    if (p >= 55) return ['#f7c1c1', '#501313'];
    if (p >= 50) return ['#fcebeb', '#501313'];
    return ['#e6f1fb', '#042c53'];
  }}
  if (m === 'mean') {{
    if (v >= 3.0) return ['#a32d2d', '#fff'];
    if (v >= 2.0) return ['#e24b4a', '#fff'];
    if (v >= 1.0) return ['#f7c1c1', '#501313'];
    if (v >= 0)   return ['#fcebeb', '#501313'];
    return ['#e6f1fb', '#042c53'];
  }}
  const p = Math.min(v, 60) / 60;
  return ['rgba(31,35,40,' + (0.06 + 0.5 * p).toFixed(3) + ')', p > 0.55 ? '#fff' : '#374151'];
}}

function fmt(v, m) {{
  if (v === null || v === undefined) return '·';
  if (m === 't') return v.toFixed(2);
  if (m === 'winrate') return (v * 100).toFixed(0);
  if (m === 'mean') return v.toFixed(2);
  return String(v);
}}

function render() {{
  const g = D.grids[lag], ana = D.ana[lag];
  let h = '<tr><th></th>';
  D.win.forEach(w => h += '<th>' + w + '日</th>');
  h += '</tr>';
  D.thr.slice().reverse().forEach(t => {{
    h += '<tr><th class="rowh">' + t + '</th>';
    D.win.forEach(w => {{
      const r = g[t + '|' + w];
      if (!r) {{ h += '<td class="null">·</td>'; return; }}
      const v = r[metric], c = color(v, metric);
      const cls = [r.pass_sample ? '' : 'low',
                   (t === ana.best_thr && w === ana.best_win) ? 'best' : ''].join(' ');
      h += '<td class="' + cls + '" style="background:' + c[0] + ';color:' + c[1] + '" '
         + 'data-t="' + t + '" data-w="' + w + '">' + fmt(v, metric) + '</td>';
    }});
    h += '</tr>';
  }});
  document.getElementById('hm').innerHTML = h;
  document.querySelectorAll('#hm td[data-t]').forEach(
    td => td.onclick = () => show(td.dataset.t, td.dataset.w));
}}

function show(t, w) {{
  const line = (tag, r) => {{
    if (!r) return '<div>' + tag + '：无数据</div>';
    return '<div><b>' + tag + '</b>　n=' + r.n + '　t=' + r.t.toFixed(2)
      + '　胜率=' + (r.winrate * 100).toFixed(1) + '%　均值=' + r.mean.toFixed(2)
      + 'bp　盈亏比=' + (r.pl === null ? '—' : r.pl.toFixed(2))
      + '　样本' + (r.pass_sample ? '达标' : '<span class="bad">不达标</span>') + '</div>'
      + '<div style="color:#9ca3af;font-size:11px;margin:2px 0 8px">分年度触发：'
      + Object.entries(r.years).map(([y, n]) => y + ':' + n).join('　') + '</div>';
  }};
  document.getElementById('detail').innerHTML =
    '<div style="margin-bottom:6px"><b>阈值 ' + t + ' · 观察窗口 ' + w + ' 日</b></div>'
    + line('T+0 原口径', D.grids['0'][t + '|' + w])
    + line('T+1 可执行', D.grids['1'][t + '|' + w]);
}}

function legend() {{
  const el = document.getElementById('legend');
  if (metric === 'n') {{
    el.innerHTML = '<span><i class="sw" style="background:rgba(31,35,40,.06)"></i>少</span>'
      + '<span><i class="sw" style="background:rgba(31,35,40,.56)"></i>多</span>'
      + '<span style="margin-left:10px">虚线框 = 样本量不达标</span>';
    return;
  }}
  const rows = {{
    t: [[4.14, '≥4.14 过 Bonferroni'], [3.0, '≥3.0'], [2.0, '≥2.0 显著'], [1.0, '≥1.0'], [0, '≥0'], [-2, '<0 反向']],
    winrate: [[0.70, '≥70%'], [0.62, '≥62%'], [0.55, '≥55%'], [0.50, '≥50%'], [-1, '<50%']],
    mean: [[3, '≥3bp'], [2, '≥2bp'], [1, '≥1bp'], [0, '≥0bp'], [-99, '<0bp']]
  }}[metric];
  el.innerHTML = rows.map(([v, l]) => '<span><i class="sw" style="background:'
    + color(v, metric)[0] + '"></i>' + l + '</span>').join('')
    + '<span style="margin-left:10px">虚线框 = 样本量不达标（某完整年触发&lt;5次）</span>';
}}

function seg(id, cb) {{
  document.querySelectorAll('#' + id + ' button').forEach(b => b.onclick = () => {{
    document.querySelectorAll('#' + id + ' button').forEach(x => x.classList.remove('on'));
    b.classList.add('on'); cb(b.dataset.v);
  }});
}}
seg('segLag', v => {{ lag = v; render(); }});
seg('segMetric', v => {{ metric = v; render(); legend(); }});
render(); legend();
</script>
</body>
</html>"""

out = BASE + r'\factor_param_heatmap.html'
open(out, 'w', encoding='utf-8').write(HTML)
print(f'\nwritten {out}')
for lag in ('0', '1'):
    a = ANA[lag]
    print(f"lag={lag}: 最优 阈值{a['best_thr']}·{a['best_win']}日 t={a['best']['t']:.2f} "
          f"n={a['best']['n']} 胜率={a['best']['winrate']*100:.1f}% 均值={a['best']['mean']:+.2f}bp "
          f"| 邻域{a['nb_avg']} 行均值{a['row_avg']} 全局{a['global_avg']} → {a['verdict']}")
