# -*- coding: utf-8 -*-
"""
蜘蛛网策略「报告 vs 本地」差异归因：杠杆假说 + 开平仓规则隔离实验。

不改主回测脚本，只做诊断。输入：
  TS_持仓整理.json / TS.CFE.xlsx / TS_策略回测.json
输出：
  TS_规则杠杆归因.json   控制台表格 + 网页可内嵌的数据

要回答的三个问题：
  1) 报告是不是加了杠杆（1 手 TS 保证金约 2 万，1000 万可开约 500 手）？
  2) 开平仓条件是不是和报告不一致？逐个变体跑，并用报告的胜率 51.97% /
     盈亏比 1.37 反推最可能的规则形态。
  3) 补 2021 数据后，差距缩小了多少？
"""
import os, json, math, collections
import numpy as np
import pandas as pd
import spider_backtest as sb

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'TS_规则杠杆归因.json')


# ---------------------------------------------------------------- 通用模拟器
def simulate(dates, O, C, smap, lo, hi, *, entry_lag=1, exit_lag=2,
             entry_px='open', exit_px='close', mode='independent',
             reverse='close_only', same='ignore', flip=False):
    """返回 (daily_ret, trades)。

    entry_lag / exit_lag：相对信号日 k 的交易日偏移。
    mode='independent' 每个信号独立成笔（允许重叠）；
    mode='state' 单一仓位，反向信号提前平仓，同向信号按 same 处理。
    """
    n = hi - lo + 1
    ret = [0.0] * n
    trades = []
    E = O if entry_px == 'open' else C
    X = O if exit_px == 'open' else C
    hold = max(1, exit_lag - entry_lag)

    def sig_at(k):
        s = smap.get(dates[k])
        if s not in (1, -1):
            return 0
        return -s if flip else s

    if (entry_lag, exit_lag, entry_px, exit_px, mode, reverse, same, flip) == (
        1, 2, 'open', 'close', 'state', 'close_only', 'ignore', False
    ):
        return sb.run_state(dates, O, C, smap, lo, hi)

    if mode == 'independent':
        for k in range(lo, hi + 1):
            s = sig_at(k)
            if s == 0:
                continue
            a, b = k + entry_lag, k + exit_lag
            if a > hi or b > hi or b < a:
                continue
            r = s * (X[b] - E[a]) / sb.PRICE_REF
            if a == b:
                ret[b - lo] += r
            else:
                ret[a - lo] += s * (C[a] - E[a]) / sb.PRICE_REF
                for j in range(a + 1, b):
                    ret[j - lo] += s * (C[j] - C[j - 1]) / sb.PRICE_REF
                ret[b - lo] += s * (X[b] - C[b - 1]) / sb.PRICE_REF
            trades.append({'ret': r, 'sig_day': dates[k],
                           'entry': dates[a], 'exit': dates[b]})
        return ret, trades

    # ---- 状态机 ----
    sched = collections.defaultdict(list)
    pos, entry_price, exit_day = 0, None, None

    def do_close(k, px, how):
        nonlocal pos, entry_price, exit_day
        r = pos * (px - entry_price) / sb.PRICE_REF
        ret[k - lo] += r
        trades.append({'ret': r, 'how': how})
        pos, entry_price, exit_day = 0, None, None

    def do_open(k, px, s, ex):
        nonlocal pos, entry_price, exit_day
        pos, entry_price, exit_day = s, px, ex

    for k in range(lo, hi + 1):
        # 1) 开盘/收盘执行已挂单
        for act in sched[k]:
            if act[0] == 'close' and pos != 0:
                do_close(k, E[k], '反向信号')
                if reverse == 'reverse_same_day' and act[1] != 0 and pos == 0:
                    do_open(k, E[k], act[1], k + hold)
            elif act[0] == 'open' and pos == 0:
                do_open(k, E[k], act[1], k + hold)
        sched.pop(k, None)

        # 2) 到期强平（用平仓价）
        if pos != 0 and k >= exit_day:
            do_close(k, X[k], '到期平仓')

        # 3) 依据 T 日信号安排动作
        s = sig_at(k)
        a = k + entry_lag
        b = k + exit_lag
        if s == 0 or a > hi or b > hi or a < lo:
            continue
        if pos == 0:
            sched[a].append(('open', s))
        elif s == -pos:
            sched[a].append(('close', s if reverse == 'reverse_same_day' else 0))
        elif same == 'extend':
            exit_day = max(exit_day, b)

    if pos != 0:
        do_close(hi, X[hi], '区间末强平')
    return ret, trades


def m(daily_ret, trades):
    return sb.metrics(daily_ret, trades)


def scale_leverage(daily_ret, trades, L):
    """按 L 倍名义敞口线性缩放每日收益（固定名义单利口径，不计追加保证金/强平）。"""
    r = [float(x) * L for x in daily_ret]
    t = [{'ret': float(x['ret']) * L} for x in trades]
    return r, t


def main():
    P, dates, O, C, sig_maps, diag = sb.load_data()
    smap = sig_maps['前20大合计']
    lo, hi = 0, len(dates) - 1
    rp_a = next(i for i, d in enumerate(dates) if d >= '2021-01-04')
    rp_b = max(i for i, d in enumerate(dates) if d <= '2025-04-10')
    y21_a = next(i for i, d in enumerate(dates) if d >= '2021-01-04')
    y21_b = max(i for i, d in enumerate(dates) if d <= '2021-12-31')

    out = {'meta': {
        '说明': '杠杆与开平仓规则隔离实验；同一信号、同一价格，只改规则或杠杆',
        '全样本': [dates[lo], dates[hi]],
        '报告可比段': [dates[rp_a], dates[rp_b]],
        '报告值': sb.REPORT_BASELINE,
        '报告原文': '第3页：初始资金1000W，杠杆倍数为单倍杠杆；'
                    'T+1开盘全仓、T+2收盘平仓；反向信号T+1开盘平仓',
    }, 'variants': [], 'leverage': [], 'periods': {}, 'oracle': {}}

    # ---------------------------------------------------------- 1) 规则变体
    VARIANTS = [
        ('报告字面·状态机', dict(mode='state')),
        ('状态机+反向反手', dict(mode='state', reverse='reverse_same_day')),
        ('状态机+同向续持', dict(mode='state', same='extend')),
        ('逐笔独立（允许重叠）', dict(mode='independent')),
        ('T日收盘开仓→T+2收盘', dict(mode='independent', entry_lag=0, entry_px='close')),
        ('T+1收盘开仓→T+2收盘', dict(mode='independent', entry_px='close')),
        ('T+1开盘→T+1收盘（日内）', dict(mode='independent', exit_lag=1, exit_px='close')),
        ('T+1开盘→T+2开盘', dict(mode='independent', exit_px='open')),
        ('T+1开盘→T+3收盘', dict(mode='independent', exit_lag=3)),
        ('T+1开盘→T+5收盘', dict(mode='independent', exit_lag=5)),
        ('信号延迟1日', dict(mode='independent', entry_lag=2, exit_lag=3)),
        ('信号符号翻转', dict(mode='state', flip=True)),
    ]
    print('=== ① 开平仓规则隔离（全样本 2021-01-04~2026-11-23，零成本，1x）===')
    print(f"{'变体':<26} {'累计%':>8} {'年化%':>7} {'夏普':>6} {'回撤%':>7} {'胜率%':>7} {'盈亏比':>7} {'笔数':>5} | 可比段累计%")
    base_ret = base_tr = None
    for name, kw in VARIANTS:
        r, t = simulate(dates, O, C, smap, lo, hi, **kw)
        mm = m(r, t)
        # 报告可比段单独截
        rc = [0.0] * (rp_b - rp_a + 1)
        for j, v in enumerate(r):
            k = lo + j
            if rp_a <= k <= rp_b:
                rc[k - rp_a] = v
        mc = m(rc, [{'ret': x} for x in rc])
        if name == '报告字面·状态机':
            base_ret, base_tr = r, t
        out['variants'].append({
            '名称': name, '参数': {k: v for k, v in kw.items()},
            '全样本': mm, '报告可比段累计': mc['累计收益率'],
            '报告可比段': mc,
        })
        print(f"{name:<26} {mm['累计收益率']*100:>8.3f} {mm['年化收益率']*100:>7.3f} "
              f"{mm['夏普比率']:>6.2f} {mm['最大回撤']*100:>7.3f} "
              f"{(mm['胜率'] or 0)*100:>7.1f} {mm['盈亏比'] or 0:>7.2f} "
              f"{mm['交易笔数']:>5} | {mc['累计收益率']*100:>7.3f}")

    # ---------------------------------------------------------- 2) 杠杆
    print('\n=== ② 杠杆假说：只缩放「报告字面·状态机」的名义敞口 ===')
    print('（报告原文写明「单倍杠杆」；这里检验若真加杠杆，回撤/夏普是否还能对上）')
    print(f"{'名义敞口L':>10} {'累计%':>9} {'年化%':>8} {'夏普':>6} {'回撤%':>8} | 可比段累计% 可比段回撤%")
    for L in [1, 2, 3, 5, 7.7, 10, 13.8, 20, 100]:
        r, t = scale_leverage(base_ret, base_tr, L)
        mm = m(r, t)
        rc = [r[k - lo] for k in range(rp_a, rp_b + 1)]
        mc = m(rc, [{'ret': x} for x in rc])
        out['leverage'].append({'L': L, '全样本': mm,
                               '报告可比段累计': mc['累计收益率'],
                               '报告可比段回撤': mc['最大回撤']})
        print(f"{L:>10} {mm['累计收益率']*100:>9.3f} {mm['年化收益率']*100:>8.3f} "
              f"{mm['夏普比率']:>6.2f} {mm['最大回撤']*100:>8.3f} | "
              f"{mc['累计收益率']*100:>9.3f} {mc['最大回撤']*100:>9.3f}")

    # 反推隐含杠杆
    need_all = sb.REPORT_BASELINE['strategy']['累计收益率'] / m(base_ret, base_tr)['累计收益率']
    need_rp = sb.REPORT_BASELINE['strategy']['累计收益率'] / out['variants'][0]['报告可比段累计']
    out['implied_leverage'] = {'全样本口径': round(float(need_all), 2),
                               '报告可比段口径': round(float(need_rp), 2)}
    print(f"\n  反推：若信号/价格/规则完全相同，报告 9.52% 需 L≈{need_all:.1f}（全样本）"
          f" 或 L≈{need_rp:.1f}（可比段）")
    print(f"  但 L={need_all:.1f} 时回撤≈{out['leverage'][4]['全样本']['最大回撤']*100:.2f}%，"
          f"报告仅 1.02%；且杠杆不改变夏普（本地 {m(base_ret, base_tr)['夏普比率']:.2f}"
          f" vs 报告 0.83）")

    # ---------------------------------------------------------- 3) 分年 + 2021 贡献
    print('\n=== ③ 分年（报告字面状态机，1x，零成本）===')
    for lab, a, b in [('2021（新补）', y21_a, y21_b),
                      ('2022', None, None), ('2023', None, None),
                      ('2024', None, None), ('2025', None, None),
                      ('2026', None, None),
                      ('报告可比段 2021-2025.04', rp_a, rp_b),
                      ('全样本 2021-2026', lo, hi)]:
        if a is None:
            yr = lab
            idx = [i for i, d in enumerate(dates) if d.startswith(yr)]
            a, b = idx[0], idx[-1]
        rr = [base_ret[k - lo] for k in range(a, b + 1)]
        mm = m(rr, [{'ret': x} for x in rr])
        out['periods'][lab] = mm
        print(f"  {lab:<26} 累计 {mm['累计收益率']*100:>7.3f}% | 年化 {mm['年化收益率']*100:>6.3f}% "
              f"| 夏普 {mm['夏普比率']:>6.2f} | 回撤 {mm['最大回撤']*100:>6.3f}% | 天数 {b-a+1}")

    # ---------------------------------------------------------- 4) Oracle 上限
    print('\n=== ④ 方向 100% 全对的理论上限（逐笔独立，零成本）===')
    for lab, a, b in [('全样本 2021-2026', lo, hi),
                      ('报告可比段 2021-2025.04', rp_a, rp_b)]:
        r, t = simulate(dates, O, C, smap, a, b, mode='independent')
        oracle = sum(abs(x['ret']) for x in t)
        out['oracle'][lab] = {'信号笔数': len(t), '完美方向单利累计': round(float(oracle), 6),
                              '每笔平均|收益|%': round(float(oracle / len(t) * 100), 5)}
        print(f"  {lab:<26} 信号 {len(t):>4} 笔 | 完美方向上限 {oracle*100:>7.3f}% "
              f"| 每笔|收益| {oracle/len(t)*100:>7.4f}% "
              + ("| 研报同段约 6.82%（曲线估读）" if a == rp_a and b == rp_b else ''))

    # ---------------------------------------------------------- 6) 2020 缺口外推
    print('\n=== ⑥ 2020 年（本地无持仓）最多能补多少 ===')
    q = sb.pd.read_excel(sb.PRICE_XLSX, sheet_name=0, header=0)
    q = q[q['日期'].notna()].copy()
    q['date'] = sb.pd.to_datetime(q['日期'])
    q = q.set_index('date').sort_index()
    q20 = q.loc['2020-01-01':'2020-12-31']
    o20 = q20['开盘价(元)'].astype(float).tolist()
    c20 = q20['收盘价(元)'].astype(float).tolist()
    two20 = [abs(c20[k + 2] - o20[k + 1]) / sb.PRICE_REF for k in range(len(o20) - 2)]
    mu20 = float(np.mean(two20))
    n21 = sum(1 for d in dates[y21_a:y21_b + 1] if smap.get(d) in (1, -1))
    oracle20 = mu20 * n21
    capture = m(base_ret, base_tr)['累计收益率'] / out['oracle']['全样本 2021-2026']['完美方向单利累计']
    est20 = oracle20 * capture
    out['gap_2020'] = {
        '2020两日窗口平均绝对收益%': round(mu20 * 100, 5),
        '2020假设信号笔数(按2021密度)': int(n21),
        '2020完美方向上限%': round(oracle20 * 100, 4),
        '本地实际捕获率': round(float(capture), 4),
        '2020最可能贡献%': round(est20 * 100, 4),
        '补2020后报告期估计累计%': round((out['periods']['报告可比段 2021-2025.04']['累计收益率'] + est20) * 100, 4),
        '2021两日窗口平均绝对收益%': round(float(np.mean([
            abs(C[k + 2] - O[k + 1]) / sb.PRICE_REF for k in range(y21_a, y21_b - 1)
        ])) * 100, 5),
    }
    print(f"  2020 两日窗口平均|收益| {mu20*100:.4f}%（2021 为 {out['gap_2020']['2021两日窗口平均绝对收益%']:.4f}%）")
    print(f"  按 2021 信号密度 {n21} 笔 ⇒ 2020 完美方向上限 {oracle20*100:.3f}%")
    print(f"  按本地实际捕获率 {capture*100:.1f}% ⇒ 2020 最可能贡献仅 {est20*100:.3f}%")
    print(f"  补 2020 后报告期估计累计 ≈ {(out['periods']['报告可比段 2021-2025.04']['累计收益率']+est20)*100:.3f}%"
          f"（报告 9.52%）")

    # ---------------------------------------------------------- 5) 报告隐含胜率/盈亏比反推
    print('\n=== ⑤ 用报告的胜率/盈亏比反推每笔收益量级 ===')
    p, pl = 0.5197, 1.37
    coef = (p * pl - (1 - p)) / (p * pl + (1 - p))
    print(f"  报告: 胜率 {p*100:.2f}% 盈亏比 {pl} ⇒ 每笔期望/平均绝对收益 = {coef:.4f}")
    for lab, a, b in [('全样本', lo, hi), ('报告可比段', rp_a, rp_b)]:
        r, t = simulate(dates, O, C, smap, a, b, mode='independent')
        mu = np.mean([abs(x['ret']) for x in t]) * 100
        need = 9.52 / (coef * mu)
        out['oracle'][lab + '_隐含'] = {'每笔平均绝对收益%': round(float(mu), 5),
                                        '达到9.52%所需笔数': int(need)}
        print(f"  {lab:<26} 本地每笔|收益| {mu:.4f}% ⇒ 需 {need:>6.0f} 笔，"
              f"实际仅 {len(t)} 笔（信号密度需 {need/(b-a+1)*100:.0f}%）")

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('\n输出:', OUT, f'({os.path.getsize(OUT)/1024:.1f} KB)')


if __name__ == '__main__':
    main()
