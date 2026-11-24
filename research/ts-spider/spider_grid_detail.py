# -*- coding: utf-8 -*-
"""补算参数网格里两个「最优格子」的完整净值与分段表现。

原 TS_平安纯多头.json 的 grid.cells 只存了指标、没存净值曲线，
本脚本用完全相同的信号函数与状态机重跑这两格，供汇总页叠加显示。

  夏普王 = dNet / down  / th=0.75 / H=3
  点数王 = dNet / only  / th=0.75 / H=10

down = 当日收跌 且 dNet 60 日分位 >= 0.75；only = 不加价格条件，只看分位。
执行 = T+1 开盘做多 -> T+1+H 收盘平仓，单仓位不叠加，零成本。
输出：TS_网格优选_明细.json
"""
import os
import json
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
OUT = os.path.join(BASE, 'TS_网格优选_明细.json')
SPLIT = sb.SPLIT

VARIANTS = [
    ('GRID_SHARP', '网格·夏普王', 'dNet', 'down', 0.75, 3),
    ('GRID_PTS', '网格·点数王', 'dNet', 'only', 0.75, 10),
]
COSTS = [0.0, 0.002, 0.005, 0.01, 0.02]


def make_long_signal(q, dP, mode, th):
    ok = (~np.isnan(q)) & (np.nan_to_num(q, nan=-1.0) >= th)
    if mode == 'down':
        ok = ok & (dP < 0)
    return ok.astype(int)


def nav_from(r):
    return [round(float(v), 8) for v in (1.0 + np.cumsum(r))]


def norm_trades(tr, dates):
    """统一成汇总页的逐笔字段，与席位对比 / 平安纯多口径一致。"""
    out = []
    for t in tr:
        si, ei, xi = t['signal_idx'], t['entry_idx'], t['exit_idx']
        out.append({
            '信号日': t.get('signal_date') or dates[si],
            '方向': '多' if t.get('dir') == 1 else '空',
            '开仓日': t.get('entry_date') or dates[ei],
            '开仓价': round(float(t['entry_px']), 3),
            '平仓日': t.get('exit_date') or dates[xi],
            '平仓价': round(float(t['exit_px']), 3),
            '持有交易日': int(xi - ei),
            '盈亏点数': round(float(t['pnl_pts']), 3),
        })
    return out


def main():
    dates, idx, O, C, dP, series = sb.load()
    n = len(dates)
    split_i = int(np.searchsorted(np.array(dates), SPLIT))
    q = sb.rolling_pct(series[sb.TARGET]['dNet'])

    out = {
        'meta': {
            'title': '参数网格两个最优格子 · 完整曲线（事后挑参，仅供刻度）',
            'start': dates[0], 'end': dates[-1], 'n_days': n,
            'split': SPLIT, 'costs': COSTS,
            'nav': '净值 = 1 + Σ(Δ价 − 成本)/100，固定名义单利累加、不复利',
            'warning': '这两格是从 120 个组合里按全样本点数/夏普挑出来的最大值，带前视与多重比较，不能当策略',
        },
        'dates': dates,
        'variants': {},
    }

    for key, label, src, mode, th, H in VARIANTS:
        sig = make_long_signal(q, dP, mode, th)
        r, tr = sb.run(sig, O, C, H, 0.0, 0, n - 1)
        seg = {
            'all': sb.metrics(r, tr, 0, n - 1),
            'explore': sb.metrics(r, tr, 0, split_i - 1),
            'verify': sb.metrics(r, tr, split_i, n - 1),
        }
        costs = []
        for c in COSTS:
            rc, tc = sb.run(sig, O, C, H, c, 0, n - 1)
            m = sb.metrics(rc, tc, 0, n - 1)
            costs.append({'成本点': c, '累计点数': m['累计点数'], '夏普比率': m['夏普比率']})
        years = []
        for y in sorted(set(d[:4] for d in dates)):
            a = next((i for i, d in enumerate(dates) if d.startswith(y)), None)
            b = max((i for i, d in enumerate(dates) if d.startswith(y)), default=None)
            if a is not None and b is not None:
                m = sb.metrics(r, tr, a, b)
                years.append({'年份': y, '累计点数': m['累计点数'], '交易笔数': m['交易笔数']})
        out['variants'][key] = {
            'name': label,
            'params': {'source': src, 'mode': mode, 'th': th, 'H': H},
            'rule': ('T 日盘后：dNet 60 日分位≥%.2f%s → T+1 开盘做多 5 手 TS → T+%d 收盘平仓；单仓位不叠加'
                     % (th, ' 且当日收跌' if mode == 'down' else '（不看当日涨跌）', H + 1)),
            'nav': nav_from(r),
            'daily_ret': [round(float(v), 8) for v in r],
            'signal_days': int(sig.sum()),
            'filled': len(tr),
            'skipped': int(sig.sum()) - len(tr),
            'segments': seg,
            'costs': costs,
            'yearly': years,
            'trades': norm_trades(tr, dates),
        }
        v = out['variants'][key]
        print('%s %-12s 信号=%3d 成交=%3d 全样本 %.3f 点 / 夏普 %.3f | 探索 %.3f 验证 %.3f'
              % (key, label, v['signal_days'], v['filled'], seg['all']['累计点数'],
                 seg['all']['夏普比率'], seg['explore']['累计点数'], seg['verify']['累计点数']))

    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, ensure_ascii=False, separators=(',', ':'))
    print('wrote', OUT, os.path.getsize(OUT), 'bytes')


if __name__ == '__main__':
    main()
