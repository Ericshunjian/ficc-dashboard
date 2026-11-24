# -*- coding: utf-8 -*-
"""持续触发信号时一直加多仓：叠加仓位 vs 单仓位不叠加。

同一条平安信号（当日收跌 且 dNet 60 日分位≥80%、T+1 开盘做多、持有 3 日），
只改「持仓期间能否再开新仓」这一件事：

  SINGLE   单仓位
  STACK2   最多 2 组并存
  STACK3   最多 3 组并存
  STACK5   最多 5 组并存
  STACKALL 不限组数，信号触发就加仓

口径与现行一致：每组 5 手 TS、名义 1000 万、盈亏在各自平仓日记账、
净值 = 1 + Σ点数/100（单利累加不复利）、零成本基准，另给成本敏感性。
补充「每日盯市」口径的夏普（叠加版仓位天天在场，脉冲口径会低估平滑度）。

输出：TS_叠加仓位.json
"""
import os
import json
import math
import warnings
import numpy as np

import spider_pingan_signal as sb

warnings.filterwarnings('ignore')
try:
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'TS_叠加仓位.json')
RULES = [
    ('SINGLE', '单仓位不叠加（现行）', 1),
    ('STACK2', '最多 2 组并存', 2),
    ('STACK3', '最多 3 组并存', 3),
    ('STACK5', '最多 5 组并存', 5),
    ('STACKALL', '不限组数·见信号就加', 10 ** 6),
]
COSTS = [0.0, 0.002, 0.005, 0.01, 0.02]
SEED = 20261124
N_JITTER = 500


def run_stack(sig, O, C, H, cap, cost=0.0, a=0, b=None):
    """cap=最大并存组数。返回 (脉冲净值收益, 每日盯市收益, 交易明细)。"""
    n = len(sig)
    if b is None:
        b = n - 1
    pulse = np.zeros(n)
    mark = np.zeros(n)
    occ = np.zeros(n)          # 当日并存组数（收盘时点）
    trades = []
    live = []                   # 尚未平仓的组的到期日
    for t in range(a, b + 1):
        live = [x for x in live if x > t]
        if sig[t] == 0 or len(live) >= cap:
            continue
        e, x = t + 1, t + 1 + H
        if x > b or not np.isfinite(O[e]):
            continue
        if not np.isfinite(C[e:x + 1]).all():
            raise ValueError(f'持仓期间缺少收盘价，索引 {e}–{x}')
        pts = float(sig[t] * (C[x] - O[e]) - cost)
        pulse[x] += pts / sb.PRICE_REF
        sb.mark_trade(mark, O, C, e, x, sig[t], cost)
        occ[e:x + 1] += 1
        live.append(x)
        trades.append({
            'signal_idx': t, 'entry_idx': e, 'exit_idx': x, 'dir': int(sig[t]),
            'entry_px': round(float(O[e]), 4), 'exit_px': round(float(C[x]), 4),
            'pnl_pts': round(pts, 5),
        })
    return pulse, mark, trades, occ


def nav_of(r):
    return [round(float(v), 8) for v in (1.0 + np.cumsum(r))]


def main():
    dates, idx, O, C, dP, series = sb.load()
    n = len(dates)
    split_i = int(np.searchsorted(np.array(dates), sb.SPLIT))
    q = sb.rolling_pct(series[sb.TARGET]['dNet'])
    sig = (make_sig(q, dP))

    out = {
        'meta': {
            'title': '持续加多仓（叠加仓位）vs 单仓位 · 同一条平安信号',
            'generated': __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'start': dates[0], 'end': dates[-1], 'n_days': n, 'split': sb.SPLIT,
            'rule': 'T日盘后：当日收跌 且 dNet 60日分位≥80% → T+1开盘做多5手TS → 持有3交易日 → T+4收盘平仓；'
                    '唯一差别是持仓期间能否再开新仓（cap）',
            'nav': '净值 = 1 + Σ点数/100，固定名义单利累加、不复利；1组=5手=1000万名义',
            'signal_days': int(sig.sum()),
            'costs': COSTS,
        },
        'dates': dates,
        'variants': {},
    }

    for key, label, cap in RULES:
        r, mk, tr, occ = run_stack(sig, O, C, 3, cap, 0.0, 0, n - 1)
        seg = {
            'all': sb.metrics(mk, tr, 0, n - 1),
            'explore': sb.metrics(mk, tr, 0, split_i - 1),
            'verify': sb.metrics(mk, tr, split_i, n - 1),
        }
        # 盯市口径夏普（收益率仍是点数/100，只是记账日不同）
        mk_sharpe = {}
        for nm, a, b in [('all', 0, n - 1), ('explore', 0, split_i - 1),
                         ('verify', split_i, n - 1)]:
            x = mk[a:b + 1]
            sd = float(np.std(x, ddof=1))
            mk_sharpe[nm] = round(float(np.mean(x) / sd * math.sqrt(sb.ANN_DAYS)), 4) if sd > 0 else 0.0
        # 成本（逐笔扣，笔数随 cap 变）
        costs = []
        for c in COSTS:
            rc, _, tc, _ = run_stack(sig, O, C, 3, cap, c, 0, n - 1)
            m = sb.metrics(rc, tc, 0, n - 1)
            costs.append({'成本点': c, '累计点数': m['累计点数'], '交易笔数': m['交易笔数']})
        years = []
        for y in sorted(set(d[:4] for d in dates)):
            a = next((i for i, d in enumerate(dates) if d.startswith(y)), None)
            b = max((i for i, d in enumerate(dates) if d.startswith(y)), default=None)
            if a is not None and b is not None:
                m = sb.metrics(mk, tr, a, b)
                years.append({'年份': y, '累计点数': m['累计点数'],
                              '交易笔数': m['交易笔数'],
                              '最大并存': int(occ[a:b + 1].max()) if b >= a else 0})
        inpos = int((occ > 0).sum())
        out['variants'][key] = {
            'name': label, 'cap': cap,
            'nav': nav_of(mk), 'daily_ret': [round(float(v), 8) for v in r],
            'mark_ret': [round(float(v), 8) for v in mk],
            'trades': [{'信号日': dates[t['signal_idx']], '开仓日': dates[t['entry_idx']],
                        '开仓价': t['entry_px'], '平仓日': dates[t['exit_idx']],
                        '平仓价': t['exit_px'], '盈亏点数': t['pnl_pts']} for t in tr],
            'segments': seg, 'costs': costs, 'yearly': years,
            'occupancy': {
                '在场天数': inpos, '在场比例': round(inpos / n, 4),
                '最大并存组数': int(occ.max()), '平均并存组数': round(float(occ[occ > 0].mean()), 3) if inpos else 0,
                '峰值名义敞口万元': int(occ.max() * 1000),
                '峰值杠杆': round(float(occ.max() * 1000 / 1000), 2),
            },
            'mark_sharpe': mk_sharpe,
        }
        v = out['variants'][key]
        print('%-9s %-22s %6.3f 点 夏普%6.3f 笔%4d 并存峰值%2d 在场%5.1f%% 验证%6.3f' %
              (key, label, seg['all']['累计点数'], seg['all']['夏普比率'],
               seg['all']['交易笔数'], v['occupancy']['最大并存组数'],
               100 * v['occupancy']['在场比例'], seg['verify']['累计点数']), flush=True)

    # 零假设：把信号日在收跌日内抖动 ±3 天，保留信号日数与价格条件
    down = np.where(dP < 0)[0]
    real_idx = np.where(sig == 1)[0]
    cand = sb.jitter_candidates(real_idx, down, 3)
    real_all = out['variants']['STACKALL']['segments']['all']['累计点数']
    real_cap3 = out['variants']['STACK3']['segments']['all']['累计点数']
    rng = np.random.default_rng(SEED)
    hists = {'STACKALL': [], 'STACK3': []}
    for _ in range(N_JITTER):
        pick = sb.sample_jitter(cand, rng)
        sg = np.zeros(n, dtype=int)
        sg[pick] = 1
        for key, cap in [('STACKALL', 10 ** 6), ('STACK3', 3)]:
            rr, _, tt, _ = run_stack(sg, O, C, 3, cap, 0.0, 0, n - 1)
            hists[key].append(sb.metrics(rr, tt, 0, n - 1)['累计点数'])
    out['null'] = {}
    for key, real in [('STACKALL', real_all), ('STACK3', real_cap3)]:
        A = np.array(hists[key])
        out['null'][key] = {
            'real': round(float(real), 4), 'mean': round(float(A.mean()), 4),
            'p95': round(float(np.percentile(A, 95)), 4),
            'p': round(float((A >= real).mean()), 4),
            'hist': [round(float(x), 4) for x in A],
            'n': N_JITTER,
        }
        print('NULL %-9s 随机均值 %.3f p95 %.3f 真实 %.3f p=%.4f' %
              (key, A.mean(), np.percentile(A, 95), real, (A >= real).mean()), flush=True)

    # 与已有口径对照（只读现成结果）
    pure = json.load(open(os.path.join(BASE, 'TS_平安纯多头.json'), encoding='utf-8'))
    seat = json.load(open(os.path.join(BASE, 'TS_席位对比.json'), encoding='utf-8'))
    gd = json.load(open(os.path.join(BASE, 'TS_网格优选_明细.json'), encoding='utf-8'))
    rows = []
    for name, pts, sharpe, note in [
        ('平安·只做多（单仓位）', pure['preset']['all']['累计点数'], pure['preset']['all']['夏普比率'], '现行主口径'),
        ('中信·多空双侧', seat['seats']['中信期货']['full']['all']['累计点数'],
         seat['seats']['中信期货']['full']['all']['夏普比率'], '9 席位取最大，不显著'),
        ('网格·夏普王', gd['variants']['GRID_SHARP']['segments']['all']['累计点数'],
         gd['variants']['GRID_SHARP']['segments']['all']['夏普比率'], '事后挑参'),
        ('网格·点数王', gd['variants']['GRID_PTS']['segments']['all']['累计点数'],
         gd['variants']['GRID_PTS']['segments']['all']['夏普比率'], '事后挑参'),
    ]:
        rows.append({'name': name, '累计点数': pts, '夏普': sharpe, 'note': note})
    for key, label, cap in RULES:
        a = out['variants'][key]['segments']['all']
        rows.append({'name': label, '累计点数': a['累计点数'], '夏普': a['夏普比率'], 'note': key})
    out['compare'] = rows

    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, ensure_ascii=False, separators=(',', ':'))
    print('wrote', OUT, os.path.getsize(OUT), 'bytes')


def make_sig(q, dP):
    ok = (~np.isnan(q)) & (np.nan_to_num(q, nan=-1.0) >= 0.80)
    return (ok & (dP < 0)).astype(int)


if __name__ == '__main__':
    main()
