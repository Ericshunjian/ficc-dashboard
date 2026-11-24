# -*- coding: utf-8 -*-
"""TS 蜘蛛网 · 每日信号与前瞻记录。

从 5 个已选策略（收益高且夏普高）复现每日信号 / 累计净值 / 当日观察值。
前瞻记录从「锁定日」起算：首次运行锁定为数据末日的下一个交易日，
此后只读不回补，用于干净地验证「看平安席位更准」这一主观直觉。

输出：TS_每日信号.json、TS_前瞻记录.json
"""
import json
import os
import sys
import warnings
from datetime import date, datetime, timedelta, timezone

import numpy as np

import spider_pingan_signal as sb

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(BASE, 'TS_每日信号.json')
STATE_JSON = os.path.join(BASE, 'TS_前瞻记录.json')

# 按「累计点数 + 夏普」选出的 5 个；expected 用于和已有研究产物交叉校验
STRATS = [
    dict(key='GRID_PTS', name='网格·点数王', seat='平安期货', th=0.75, mode='only', H=10,
         tag='事后挑参', expected=4.528, rule='平安 dNet 60日分位≥75%，不看当日涨跌 → T+1开盘做多5手 → T+11收盘平仓'),
    dict(key='GRID_SHARP', name='网格·夏普王', seat='平安期货', th=0.75, mode='down', H=3,
         tag='事后挑参', expected=3.863, rule='当日收跌 且 平安 dNet 60日分位≥75% → T+1开盘做多5手 → T+4收盘平仓'),
    dict(key='PINGAN_LONG', name='平安·只做多', seat='平安期货', th=0.80, mode='down', H=3,
         tag='主口径', expected=3.096, rule='当日收跌 且 平安 dNet 60日分位≥80% → T+1开盘做多5手 → T+4收盘平仓'),
    dict(key='CITIC_BOTH', name='中信·多空双侧', seat='中信期货', th=0.80, mode='contra', H=3,
         tag='9取最大', expected=2.525, rule='收跌+中信dNet分位≥80%做多 / 收涨+分位≤20%做空 → T+1开盘 → T+4收盘'),
    dict(key='TOP20_LONG', name='前20大·只做多', seat='前20大合计', th=0.80, mode='down', H=3,
         tag='', expected=1.995, rule='当日收跌 且 前20大合计 dNet 60日分位≥80% → T+1开盘做多5手 → T+4收盘平仓'),
]

SEATS = sorted(set(s['seat'] for s in STRATS))
NOW = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def next_trading_day(s):
    """锁定起点：数据末日的下一个工作日（不回补当天及以前）。"""
    d = date.fromisoformat(s) + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.isoformat()


def make_sig(q, dP, th, mode):
    q = np.asarray(q, dtype=float)
    valid = ~np.isnan(q)
    hi = valid & (q >= th)
    if mode == 'only':
        return hi.astype(int)
    if mode == 'down':
        return (hi & (dP < 0)).astype(int)
    lo = valid & (q <= 1.0 - th)
    return np.where(hi & (dP < 0), 1, np.where(lo & (dP > 0), -1, 0)).astype(int)


def fmt_trade(t, dates):
    return {
        '信号日': dates[t['signal_idx']],
        '方向': '多' if t['dir'] == 1 else '空',
        '开仓日': dates[t['entry_idx']],
        '平仓日': dates[t['exit_idx']],
        '持有交易日': int(t['exit_idx'] - t['entry_idx']),
        '盈亏点数': round(float(t['pnl_pts']), 3),
    }


def main():
    dates, idx, O, C, dP, series = sb.load()
    n = len(dates)
    last = n - 1

    Q = {seat: sb.rolling_pct(series[seat]['dNet']) for seat in SEATS}
    obs = {}
    for seat in SEATS:
        dn = series[seat]['dNet']
        r = dn[-250:]
        r = r[~np.isnan(r)]
        obs[seat] = {
            'dNet手': None if np.isnan(dn[last]) else round(float(dn[last]), 1),
            '分位': None if np.isnan(Q[seat][last]) else round(float(Q[seat][last]), 3),
            '近250日75分位手': round(float(np.percentile(r, 75)), 1) if len(r) else None,
        }

    # 前瞻状态：起点一旦锁定就不改，records 只追加不回补
    if os.path.exists(STATE_JSON):
        state = json.load(open(STATE_JSON, encoding='utf-8'))
        start = state['start']
        recs = state.get('records', [])
    else:
        start = next_trading_day(dates[last])
        recs = []
        state = {'start': start, 'locked_at': NOW, 'records': [],
                 'note': '锁定日=数据末日的下一交易日；只记锁定日后新发出的信号，不回补历史。'}
    known = set((r['策略key'], r['信号日']) for r in recs)

    out_strats = {}
    today_strats = {}
    for st in STRATS:
        sig = make_sig(Q[st['seat']], dP, st['th'], st['mode'])
        ret, trades = sb.run(sig, O, C, st['H'])
        ret2, _ = sb.run(sig, O, C, st['H'], cost=0.02)
        m = sb.metrics(ret, trades, 0, last)
        y0 = next((i for i, d in enumerate(dates) if d.startswith('2026')), last)
        my = sb.metrics(ret, trades, y0, last)
        m2 = sb.metrics(ret2, [], 0, last)
        if abs(m['累计点数'] - st['expected']) > 0.02:
            print('[WARN] %s 重算 %.3f 点与既有 %.3f 不符，请核对口径' %
                  (st['name'], m['累计点数'], st['expected']))

        holding = [t for t in trades if t['entry_idx'] <= last <= t['exit_idx']]
        sig_today = int(sig[last])
        if holding:
            t = holding[-1]
            position = '%s单持仓中 · 第%d/%d日' % (
                '多' if t['dir'] == 1 else '空', last - t['entry_idx'] + 1, st['H'] + 1)
            hint = '持仓中，不叠加' + ('（今日新信号忽略）' if sig_today else '')
        else:
            position = '空仓'
            hint = ('明日开盘按信号建仓' if sig_today else '观望')

        today_strats[st['key']] = {
            '信号': sig_today, '持仓': position, '提示': hint,
            '分位': obs[st['seat']]['分位'],
            'dNet手': obs[st['seat']]['dNet手'],
            '当日涨跌点': None if np.isnan(dP[last]) else round(float(dP[last]), 3),
        }

        out_strats[st['key']] = {
            'name': st['name'], 'seat': st['seat'], 'tag': st['tag'],
            'rule': st['rule'],
            'params': {'th': st['th'], 'mode': st['mode'], 'H': st['H']},
            'metrics': m, 'ytd': my, 'cost2tick': m2['累计点数'],
            'nav': [round(float(v), 8) for v in (1.0 + np.cumsum(ret))],
            'signal': [int(v) for v in sig],
            'trades': [fmt_trade(t, dates) for t in trades[-20:]],
        }

        for t in trades:
            sd = dates[t['signal_idx']]
            if sd < start or (st['key'], sd) in known:
                continue
            recs.append({
                '策略key': st['key'], '策略': st['name'], '信号日': sd,
                '方向': '多' if t['dir'] == 1 else '空',
                '开仓日': dates[t['entry_idx']], '平仓日': dates[t['exit_idx']],
                '盈亏点数': round(float(t['pnl_pts']), 3), '状态': '已兑现',
            })
            known.add((st['key'], sd))

    recs.sort(key=lambda r: (r['信号日'], r['策略key']))
    cum = {}
    for r in recs:
        cum[r['策略key']] = round(cum.get(r['策略key'], 0.0) + r['盈亏点数'], 3)

    state.update({'asof': dates[last], 'last_check': NOW, 'records': recs,
                  'cumulative': cum})
    with open(STATE_JSON, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(state, f, ensure_ascii=False, indent=1)

    payload = {
        'meta': {
            'generated': NOW, 'asof': dates[last], 'n_days': n,
            'forward_start': start, 'forward_note': state['note'],
            'nav': '净值 = 1 + Σ(方向×Δ价 − 成本)/100，固定5手TS、单倍杠杆、点数累加不复利；零成本回测',
            'warning': '5条曲线均为历史研究回放，其中2条网格线是120格事后挑参；前瞻记录锁定后才算真实验证。',
        },
        'dates': dates,
        'today': {'date': dates[last], 'dP': round(float(dP[last]), 3),
                  'obs': obs, 'strategies': today_strats},
        'strategies': out_strats,
        'forward': {'start': start, 'records': recs, 'cumulative': cum},
    }
    with open(OUT_JSON, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))

    print('asof=%s  前瞻锁定=%s  前瞻笔数=%d' % (dates[last], start, len(recs)))
    for st in STRATS:
        k = st['key']
        print('%-14s %6.3f 点 / 夏普 %.3f / %3d笔  今日信号=%+d  %s' %
              (st['name'], out_strats[k]['metrics']['累计点数'],
               out_strats[k]['metrics']['夏普比率'], out_strats[k]['metrics']['交易笔数'],
               today_strats[k]['信号'], today_strats[k]['持仓']))
    print('JSON: %.0f KB' % (os.path.getsize(OUT_JSON) / 1024))


if __name__ == '__main__':
    main()
