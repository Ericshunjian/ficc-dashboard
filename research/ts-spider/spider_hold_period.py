# -*- coding: utf-8 -*-
"""
持有期专项检验：同一条平安信号，只改持有期 H ∈ {1,3,5,10}（2026-11-24）

信号完全沿用事前锁定的口述规则：当日收跌 且 dNet 60 日分位 ≥ 80%
→ T+1 开盘做多 5 手 TS → T+1+H 收盘平仓；单仓位不叠加、零成本（另给成本档）。

为什么要单独做：主口径 H=3 是事前口述锁定的，不是从 120 格挑的；
但「持有几天」仍是一个自由度，必须看它换档后结论会不会翻。

四项检验：
  ① 全样本 / 探索 / 验证 / 分年 / 成本 / 资金占用，四档并列；
  ② ★ 配对零假设：同一批随机信号日（在收跌日池内抖动 ±3 / ±5 日，保持成簇），
     四个 H 跑完全相同的状态机 ⇒ 得到「H=x 减 H=3」的差值分布，
     排除「长期持有天然吃更多市场漂移」的机械差异；
  ③ 每年用之前数据选 H（只看 H=1/3/5/10），次年应用 —— 持有期能不能稳定选出来；
  ④ 探索段/验证段交叉选择与留一年交叉验证。

输出：TS_持有期专项.json（汇总页只读它，不改任何既有策略产物）
"""
import os, json, math, warnings
from datetime import datetime
import numpy as np

import spider_pingan_signal as sb
import spider_pingan_longonly as pl

warnings.filterwarnings('ignore')
try:
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'TS_持有期专项.json')
HOLDS_TEST = [1, 3, 5, 10]
N_JITTER = 800
SEED = 20261124


def rnd(x, k=4):
    return round(float(x), k)


def slim(m):
    keys = ['累计点数', '夏普比率', '交易笔数', '胜率', '逐笔t值', '每笔平均点数', '最大回撤%']
    return {k: m.get(k) for k in keys if k in m}


def main():
    dates, idx, O, C, dP, series = sb.load()
    n = len(dates)
    split_i = int(np.searchsorted(np.array(dates), sb.SPLIT))
    q = sb.rolling_pct(series[sb.TARGET]['dNet'])
    sig = pl.make_long_signal(q, dP, 'down', 0.80)
    real_idx = np.where(sig == 1)[0]
    years = np.array([int(d[:4]) for d in dates])
    pool_down = np.where((dP < 0) & ~np.isnan(O) & ~np.isnan(C))[0]

    out = {
        'meta': {
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'start': dates[0], 'end': dates[-1], 'n_days': n,
            'split': sb.SPLIT, 'explore': [dates[0], dates[split_i - 1]],
            'verify': [dates[split_i], dates[-1]],
            'signal': '当日收跌 且 平安 dNet 60 日分位 ≥ 80%',
            'holds': HOLDS_TEST, 'n_jitter': N_JITTER, 'seed': SEED,
            'costs': pl.COSTS,
            'rule': 'T 日盘后信号 → T+1 开盘做多 5 手 TS → T+1+H 收盘平仓；单仓位不叠加',
        },
        'dates': dates,
        'variants': {},
    }

    # ---------- ① 四档持有期全口径 ----------
    runs = {}
    for H in HOLDS_TEST:
        r, tr = sb.run(sig, O, C, H, 0.0, 0, n - 1)
        runs[H] = (r, tr)
        v = {
            'H': H,
            'all': slim(sb.metrics(r, tr, 0, n - 1)),
            'explore': slim(sb.metrics(r, tr, 0, split_i - 1)),
            'verify': slim(sb.metrics(r, tr, split_i, n - 1)),
            'nav': sb.nav_from(r),
            'exposure': pl.exposure(r, tr, H, 0, n - 1),
            'costs': [],
            'yearly': [],
            'signal_days': int(sig.sum()),
            'filled': len(tr),
        }
        for c in pl.COSTS:
            rc, tc = sb.run(sig, O, C, H, c, 0, n - 1)
            v['costs'].append({'成本点': c, '累计点数': slim(sb.metrics(rc, tc, 0, n - 1))['累计点数'],
                               '交易笔数': len(tc)})
        for y in range(2021, 2027):
            a = int(np.searchsorted(np.array(dates), f'{y}-01-01'))
            b = int(np.searchsorted(np.array(dates), f'{y+1}-01-01')) - 1
            if b >= n:
                b = n - 1
            if b < a:
                continue
            m = sb.metrics(r, tr, a, b)
            v['yearly'].append({'年份': y, '累计点数': rnd(m.get('累计点数', 0)),
                                '夏普比率': rnd(m.get('夏普比率', 0), 3),
                                '交易笔数': m.get('交易笔数', 0)})
        out['variants'][str(H)] = v

    # ---------- ② 配对零假设：同一批抖动信号日跑四个 H ----------
    # 每个 H 用相同 seed、相同调用顺序 ⇒ picked 日期逐轮一致，可做配对差值检验
    jit = {}
    for W in [3, 5]:
        cand = {i: pool_down[(pool_down >= i - W) & (pool_down <= i + W)] for i in real_idx}
        # 显式独立 rng，保证四个 H 在同一轮拿到完全一样的 pick
        rng = np.random.default_rng(SEED + W)
        acc = {H: [] for H in HOLDS_TEST}
        acc_v = {H: [] for H in HOLDS_TEST}
        diff = {H: [] for H in HOLDS_TEST if H != 3}
        for _ in range(N_JITTER):
            pick = np.unique(np.array([int(rng.choice(cand[i])) for i in real_idx if len(cand[i])], dtype=int))
            sg = np.zeros(n, dtype=int)
            sg[pick] = 1
            vals = {}
            for H in HOLDS_TEST:
                r, tr = sb.run(sg, O, C, H, 0.0, 0, n - 1)
                vals[H] = sb.metrics(r, tr, 0, n - 1)['累计点数']
                acc[H].append(vals[H])
                acc_v[H].append(sb.metrics(r, tr, split_i, n - 1)['累计点数'])
            for H in diff:
                diff[H].append(vals[H] - vals[3])
        block = {}
        for H in HOLDS_TEST:
            A = np.array(acc[H]); AV = np.array(acc_v[H])
            real = out['variants'][str(H)]['all']['累计点数']
            real_v = out['variants'][str(H)]['verify']['累计点数']
            block[str(H)] = {
                'mean': rnd(A.mean()), 'p05': rnd(np.percentile(A, 5)), 'p95': rnd(np.percentile(A, 95)),
                'p': rnd((A >= real).mean()), 'real': rnd(real),
                'excess_vs_null': rnd(real - A.mean()),
                'verify_mean': rnd(AV.mean()), 'verify_p': rnd((AV >= real_v).mean()),
            }
        for H, D in diff.items():
            D = np.array(D)
            obs = out['variants'][str(H)]['all']['累计点数'] - out['variants']['3']['all']['累计点数']
            block[f'{H}_minus_3'] = {
                'obs': rnd(obs), 'mean': rnd(D.mean()),
                'hist': [rnd(x) for x in D],
                'p05': rnd(np.percentile(D, 5)), 'p95': rnd(np.percentile(D, 95)),
                'p_better': rnd((D >= obs).mean()),
                'p_better_abs': rnd((np.abs(D) >= abs(obs)).mean()),
            }
        jit[f'抖动±{W}日·仍收跌'] = block

    out['null'] = jit

    # ---------- ③ 每年用之前数据选 H（限制在四档） ----------
    def select_h(a, b, rule='points'):
        best, bestv = None, -1e9
        for H in HOLDS_TEST:
            r, tr = runs[H]
            m = sb.metrics(r, tr, a, b)
            if m.get('交易笔数', 0) < 12:
                continue
            val = m['累计点数'] if rule == 'points' else m['夏普比率']
            if val > bestv:
                best, bestv = H, val
        return best

    wf_ret = np.zeros(n)
    wf_rows = []
    for y in range(2023, 2027):
        a = int(np.searchsorted(np.array(dates), f'{y}-01-01'))
        b = int(np.searchsorted(np.array(dates), f'{y+1}-01-01')) - 1
        if b >= n:
            b = n - 1
        H = select_h(0, a - 1)
        if H is None:
            continue
        r, tr = runs[H]
        wf_ret[a:b + 1] = r[a:b + 1]
        m = sb.metrics(r, tr, a, b)
        wf_rows.append({'年份': y, '选中H': H, **slim(m)})
    wf_a = int(np.searchsorted(np.array(dates), '2023-01-01'))
    wf_all = {'固定H=%d' % H: slim(sb.metrics(runs[H][0], runs[H][1], wf_a, n - 1))
              for H in HOLDS_TEST}
    wf_tr = [{'signal_idx': i, 'pnl_pts': float(wf_ret[i]) * sb.PRICE_REF}
             for i in range(wf_a, n) if wf_ret[i] != 0]
    wf_all['逐年重选'] = slim(sb.metrics(wf_ret, wf_tr, wf_a, n - 1))
    out['walk_forward'] = {
        'rows': wf_rows, 'window': [dates[wf_a], dates[-1]],
        'all': wf_all,
        'nav': {'逐年重选': sb.nav_from(wf_ret, wf_a, n - 1)},
    }
    for H in HOLDS_TEST:
        out['walk_forward']['nav']['固定H=%d' % H] = sb.nav_from(runs[H][0], wf_a, n - 1)

    # ---------- ④ 探索/验证交叉选择与留一年交叉验证 ----------
    cross = {
        '探索段按点数选H': select_h(0, split_i - 1),
        '探索段按夏普选H': select_h(0, split_i - 1, 'sharpe'),
        '验证段按点数选H': select_h(split_i, n - 1),
        '全样本按点数选H': select_h(0, n - 1),
    }
    loo = []
    for y in range(2021, 2027):
        a = int(np.searchsorted(np.array(dates), f'{y}-01-01'))
        b = int(np.searchsorted(np.array(dates), f'{y+1}-01-01')) - 1
        if b >= n:
            b = n - 1
        if b < a:
            continue
        # 用该年之外所有年份做选择
        sa = 0
        spans = []
        for yy in range(2021, 2027):
            aa = int(np.searchsorted(np.array(dates), f'{yy}-01-01'))
            bb = int(np.searchsorted(np.array(dates), f'{yy+1}-01-01')) - 1
            if bb >= n:
                bb = n - 1
            if bb < aa or aa > b or bb < a:
                if bb >= aa:
                    spans.append((aa, bb))
        # spans 已排除测试年（不相交即保留），逐段选 H 取点数加总最大
        def score(H):
            r, tr = runs[H]
            return sum(sb.metrics(r, tr, p, q)['累计点数'] for p, q in spans)
        H = max(HOLDS_TEST, key=score)
        r, tr = runs[H]
        m = sb.metrics(r, tr, a, b)
        loo.append({'留出年': y, '选中H': H, **slim(m)})
    cross['留一年交叉验证'] = loo
    out['selection'] = cross

    # ---------- 结论数据 ----------
    j3 = jit['抖动±3日·仍收跌']
    hp = out['variants']['3']['all']['累计点数']
    out['summary'] = {
        'best_all': max(HOLDS_TEST, key=lambda H: out['variants'][str(H)]['all']['累计点数']),
        'best_verify': max(HOLDS_TEST, key=lambda H: out['variants'][str(H)]['verify']['累计点数']),
        'best_sharpe': max(HOLDS_TEST, key=lambda H: out['variants'][str(H)]['all']['夏普比率']),
        'preset_H': 3,
        'preset_points': hp,
        'diffs': {f'H{H}-H3': {'全样本': rnd(out['variants'][str(H)]['all']['累计点数'] - hp),
                               '验证段': rnd(out['variants'][str(H)]['verify']['累计点数'] - out['variants']['3']['verify']['累计点数']),
                               '配对p': j3.get(f'{H}_minus_3', {}).get('p_better_abs')}
                  for H in HOLDS_TEST if H != 3},
    }

    json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    print('=== 持有期 1/3/5/10（同一信号，主口径 H=3）===')
    print('%-5s %8s %8s %8s %8s %8s %8s %8s' % ('H', '全样本', '夏普', '探索', '验证', '笔数', '在场%', '扣2t'))
    for H in HOLDS_TEST:
        v = out['variants'][str(H)]
        print('%-5d %8.3f %8.3f %8.3f %8.3f %8d %8.1f %8.3f' %
              (H, v['all']['累计点数'], v['all']['夏普比率'], v['explore']['累计点数'],
               v['verify']['累计点数'], v['all']['交易笔数'], v['exposure']['在场时间占比'] * 100,
               v['costs'][3]['累计点数']))
    print('=== 配对零假设（抖动±3日，仍收跌）===')
    for k, v in j3.items():
        obs = '%.3f' % v['obs'] if 'obs' in v else '-'
        print('%-12s obs=%7s null=%7.3f p=%s' % (k, obs, v['mean'], v.get('p_better_abs')))
    print('=== 逐年重选 H ===')
    for r in wf_rows:
        print(' ', r['年份'], 'H=%-2d' % r['选中H'], r['累计点数'], '点')
    print('窗口合计:', {k: v['累计点数'] for k, v in wf_all.items()})
    print('选择:', {k: v for k, v in cross.items() if k != '留一年交叉验证'})
    print('输出:', OUT, '%.1f KB' % (os.path.getsize(OUT) / 1024))


if __name__ == '__main__':
    main()
