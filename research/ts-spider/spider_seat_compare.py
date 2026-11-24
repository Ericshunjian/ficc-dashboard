# -*- coding: utf-8 -*-
"""
9 个持仓席位 × 2 套规则 的净值对比，以及「中信期货全样本收益」是怎么来的。

背景：第七节（多空双侧规则）把同一套口述参数套到 8 个其他席位做安慰剂，
用同一套规则比较席位表现并分析多空腿。
  ① 把 9 个席位的净值曲线画在一起，两套规则（多空双侧 / 只做多）横向比较；
  ② 把中信那 1.413 点的领先拆成可核对的几块，并给出「9 个里挑最大」的零假设。

口径与第七/八节完全一致（不得事后改特征）：
  dNet = 净持仓日变动；力度 = 过去 60 个交易日滚动分位（含当日、无前视）；
  T 日盘后信号 → T+1 开盘建仓 → 持有 3 个交易日 → T+4 收盘平仓；单仓位不叠加；
  固定 5 手 TS、名义 1000 万、单倍杠杆；净值 = 1 + Σ(方向×Δ价 − 成本)/100，
  点数累加、不复利。零成本为主口径，另附成本敏感性。
"""
import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd

import spider_pingan_signal as sb

BASE = os.path.dirname(os.path.abspath(__file__))
HOLD_JSON = os.path.join(BASE, 'TS_持仓整理.json')
BT_JSON = os.path.join(BASE, 'TS_策略回测.json')
OUT_JSON = os.path.join(BASE, 'TS_席位对比.json')
OUT_XLSX = os.path.join(BASE, 'TS_席位对比_明细.xlsx')
TPL = os.path.join(BASE, 'seat_compare_template.html')
OUT_HTML = os.path.join(BASE, 'TS_席位对比_净值.html')

TH = 0.80
H = 3
COSTS = [0.0, 0.005, 0.01, 0.02]
N_SHIFT = 400          # 单席位循环移位次数
N_MAX = 300            # 「9 席位取最大」随机组数
SEED = 20261124
FOCUS = '中信期货'
TARGET = '平安期货'


# ---------------------------------------------------------------- 工具
def long_signal(q, dP, th=TH):
    """只做多：收跌 + dNet 分位 >= th。"""
    return ((np.nan_to_num(q, nan=-1.0) >= th) & (~np.isnan(q)) & (dP < 0)).astype(int)


def run_block(sig, O, C, H, n, block_short=False, cost=0.0):
    """状态机；block_short=True 时，做空信号仍占用仓位但不开仓。

    用来把「空单自己赚的钱」和「空单挡掉的多单」分离干净。
    """
    return sb.run(sig, O, C, H, cost, b=n - 1, block_short=block_short)


def sum_pts(tr, filt=None):
    return round(float(sum(t['pnl_pts'] for t in tr if filt is None or filt(t))), 4)


def future_pts(O, C, H, n):
    f = np.full(n, np.nan)
    for t in range(n - 1 - H):
        f[t] = C[t + 1 + H] - O[t + 1]
    return f


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else 0.0


def trim_trades(tr, dates, limit=None):
    rows = []
    for t in (tr if limit is None else tr[:limit]):
        rows.append({
            '信号日': dates[t['signal_idx']],
            '方向': '多' if t['dir'] == 1 else '空',
            '开仓日': dates[t['entry_idx']], '开仓价': t['entry_px'],
            '平仓日': dates[t['exit_idx']], '平仓价': t['exit_px'],
            '盈亏点数': t['pnl_pts'],
        })
    return rows


def main():
    dates, idx, O, C, dP, series = sb.load()
    n = len(dates)
    split_i = int(np.searchsorted(np.array(dates), sb.SPLIT))
    Q = {name: sb.rolling_pct(s['dNet']) for name, s in series.items()}
    f = future_pts(O, C, H, n)
    rng = np.random.default_rng(SEED)

    out = {
        'meta': {
            'title': '9 席位净值横向对比与中信归因',
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'start': dates[0], 'end': dates[-1], 'n_days': n,
            'split': sb.SPLIT,
            'explore': [dates[0], dates[split_i - 1]],
            'verify': [dates[split_i], dates[-1]],
            'th': TH, 'H': H, 'win': sb.WIN, 'price_ref': sb.PRICE_REF,
            'costs': COSTS, 'n_shift': N_SHIFT, 'n_max': N_MAX, 'seed': SEED,
            'focus': FOCUS, 'target': TARGET,
            'rule_full': '多空双侧：收跌+dNet分位≥80%→做多；收涨+分位≤20%→做空；H=3',
            'rule_long': '只做多：收跌+dNet分位≥80%→做多；H=3',
            'nav': '净值 = 1 + Σ(方向×Δ价 − 成本)/100，固定名义单利累加、不复利',
            'note': '参数完全沿用第七节口述规则，未重新调参；中信是看过全样本后才被注意到的席位。',
        },
        'dates': dates,
        'price': {'close': [None if np.isnan(x) else round(float(x), 4) for x in C]},
    }

    # ---------- ① 9 席位 × 2 规则 ----------
    seats = {}
    sigs_full, sigs_long = {}, {}
    for name, q in Q.items():
        sg_f = sb.make_signal(q, dP, 'contra', TH)
        sg_l = long_signal(q, dP, TH)
        sigs_full[name], sigs_long[name] = sg_f, sg_l
        rec = {}
        for rule, sg in [('full', sg_f), ('long', sg_l)]:
            r, tr = sb.run(sg, O, C, H, 0.0, 0, n - 1)
            rec[rule] = {
                'all': sb.metrics(r, tr, 0, n - 1),
                'explore': sb.metrics(r, tr, 0, split_i - 1),
                'verify': sb.metrics(r, tr, split_i, n - 1),
                'nav': [round(v, 7) for v in sb.nav_from(r)],
                '多腿': sum_pts(tr, lambda t: t['dir'] == 1),
                '空腿': sum_pts(tr, lambda t: t['dir'] == -1),
                '多腿笔数': sum(1 for t in tr if t['dir'] == 1),
                '空腿笔数': sum(1 for t in tr if t['dir'] == -1),
                'trades': trim_trades(tr, dates),
            }
            # 成本敏感性
            rec[rule]['costs'] = []
            for c in COSTS:
                rr, tt = sb.run(sg, O, C, H, c, 0, n - 1)
                rec[rule]['costs'].append({
                    '成本点': c,
                    '累计点数': sb.metrics(rr, tt, 0, n - 1)['累计点数'],
                    '夏普比率': sb.metrics(rr, tt, 0, n - 1)['夏普比率'],
                })
            # 分年
            yr = []
            for y in range(2021, 2027):
                a = int(np.searchsorted(np.array(dates), f'{y}-01-01'))
                b = int(np.searchsorted(np.array(dates), f'{y+1}-01-01')) - 1
                if b >= n:
                    b = n - 1
                m = sb.metrics(r, tr, a, b)
                tt = [t for t in tr if a <= t['exit_idx'] <= b]
                yr.append({
                    '年份': y, '累计点数': m['累计点数'], '交易笔数': m['交易笔数'],
                    '多腿': sum_pts(tt, lambda t: t['dir'] == 1),
                    '空腿': sum_pts(tt, lambda t: t['dir'] == -1),
                })
            rec[rule]['yearly'] = yr
        rec['q_corr'] = None
        seats[name] = rec
    out['seats'] = seats

    # 席位间 dNet 分位相关性（判断 9 条是不是独立实验）
    names = list(series)
    M = np.array([np.nan_to_num(Q[k], nan=0.5) for k in names])
    corr = np.corrcoef(M)
    out['correlation'] = {
        'names': names,
        'matrix': [[round(float(corr[i, j]), 3) for j in range(len(names))]
                   for i in range(len(names))],
        'offdiag_mean': round(float(np.mean(corr[np.triu_indices(len(names), 1)])), 3),
    }

    # ---------- ② 中信 2.525 的 A/B/C 拆解 ----------
    attribution = {}
    for name in [FOCUS, TARGET, '前20大合计']:
        sg = sigs_full[name]
        sg_l = sigs_long[name]
        _, tA = run_block(sg, O, C, H, n)
        _, tB = run_block(sg_l, O, C, H, n)
        _, tC = run_block(sg, O, C, H, n, block_short=True)
        cA, cB, cC = sum_pts(tA), sum_pts(tB), sum_pts(tC)
        sigB = set(t['signal_idx'] for t in tB)
        sigC = set(t['signal_idx'] for t in tC)
        skipped = [t for t in tB if t['signal_idx'] in (sigB - sigC)]
        # 做空信号：成交 vs 被仓位挡掉
        r_f, tr_f = sb.run(sg, O, C, H, 0.0, 0, n - 1)
        done_sh = set(t['signal_idx'] for t in tr_f if t['dir'] == -1)
        all_sh = set(np.where(sg == -1)[0])
        skip_sh = sorted(all_sh - done_sh)
        v_done = f[list(done_sh)] if done_sh else np.array([])
        v_skip = f[skip_sh] if skip_sh else np.array([])
        attribution[name] = {
            'A_完整双侧': cA, 'B_只做多': cB, 'C_空信号占仓位不开仓': cC,
            '空单自身盈亏': round(cA - cC, 4),
            '挤掉多单的效应': round(cC - cB, 4),
            'A减B': round(cA - cB, 4),
            '被挤掉多单笔数': len(skipped),
            '被挤掉多单在只做多里合计': sum_pts(skipped),
            'A笔数': len(tA), 'B笔数': len(tB), 'C笔数': len(tC),
            '做空信号日数': len(all_sh), '做空成交笔数': len(done_sh),
            '做空信号被跳过笔数': len(skip_sh),
            '成交空信号未来均值点数': round(float(np.nanmean(v_done)), 5) if len(v_done) else None,
            '跳过空信号未来均值点数': round(float(np.nanmean(v_skip)), 5) if len(v_skip) else None,
            '多腿': sum_pts(tr_f, lambda t: t['dir'] == 1),
            '空腿': sum_pts(tr_f, lambda t: t['dir'] == -1),
        }
    out['attribution'] = attribution

    # ---------- ③ 做空信号到底有没有信息 ----------
    sig_short = {}
    for name in [FOCUS, TARGET]:
        sg = sigs_full[name]
        m = (sg == -1) & (~np.isnan(f))
        v = f[m]
        up = (dP > 0) & (~np.isnan(f))
        vu = f[up]
        sig_short[name] = {
            '信号日数': int(m.sum()),
            '未来3日均值点数': round(float(v.mean()), 5),
            '逐笔t值': round(float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v)))), 3),
            '做空方向胜率': round(float((v < 0).mean()), 4),
            '全体收涨日数': int(up.sum()),
            '收涨日未来均值点数': round(float(vu.mean()), 5),
            'Welch_t_vs_收涨日': round(welch(v, vu), 3),
        }
    sig_short['中信减平安'] = round(
        sig_short[FOCUS]['未来3日均值点数'] - sig_short[TARGET]['未来3日均值点数'], 5)
    out['short_info'] = sig_short

    # ---------- ④ 零假设 ----------
    def cum_of(sig):
        _, t = sb.run(sig, O, C, H, 0.0, 0, n - 1)
        return sum(t2['pnl_pts'] for t2 in t)

    real = {name: seats[name]['full']['all']['累计点数'] for name in names}
    shift = {}
    for name in [FOCUS, TARGET]:
        acc = [cum_of(np.roll(sigs_full[name], int(rng.integers(1, n))))
               for _ in range(N_SHIFT)]
        A = np.array(acc)
        shift[name] = {
            'hist': [round(float(x), 4) for x in A],
            'mean': round(float(A.mean()), 4),
            'p95': round(float(np.percentile(A, 95)), 4),
            'real': round(float(real[name]), 4),
            'p': round(float((A >= real[name]).mean()), 4),
            'pct': round(float((A < real[name]).mean()), 4),
        }
    out['null_shift'] = shift

    # 「9 个席位里挑最大」的选择偏差对照：
    # 每组把 9 条信号各自循环移位（保留各自密度与成簇结构），取该组 9 个累计点数的最大值。
    maxs, means = [], []
    for _ in range(N_MAX):
        vals = [cum_of(np.roll(sigs_full[k], int(rng.integers(1, n)))) for k in names]
        maxs.append(max(vals))
        means.append(float(np.mean(vals)))
    MX, ME = np.array(maxs), np.array(means)
    real_max = max(real.values())
    real_mean = float(np.mean(list(real.values())))
    out['null_max'] = {
        'hist': [round(float(x), 4) for x in MX],
        'mean': round(float(MX.mean()), 4),
        'median': round(float(np.median(MX)), 4),
        'p95': round(float(np.percentile(MX, 95)), 4),
        'real_max': round(real_max, 4),
        'p_max': round(float((MX >= real_max).mean()), 4),
        'pct_max': round(float((MX < real_max).mean()), 4),
        'hist_mean': [round(float(x), 4) for x in ME],
        'rand_mean_of_means': round(float(ME.mean()), 4),
        'real_cross_mean': round(real_mean, 4),
        'p_cross_mean': round(float((ME >= real_mean).mean()), 4),
        'n_seats': len(names),
    }

    # 横截面统计
    V = np.array(list(real.values()))
    out['cross_section'] = {
        'values': {k: round(v, 4) for k, v in real.items()},
        'mean': round(float(V.mean()), 4), 'median': round(float(np.median(V)), 4),
        'std': round(float(V.std(ddof=1)), 4),
        'max': round(float(V.max()), 4), 'max_seat': max(real, key=real.get),
        'z_focus': round(float((real[FOCUS] - V.mean()) / V.std(ddof=1)), 3),
        'rank_focus': int(1 + sum(1 for v in V if v > real[FOCUS])),
    }

    # ---------- ⑤ 基准与全家福 ----------
    bt = json.load(open(BT_JSON, encoding='utf-8'))
    bh = np.asarray(bt['benchmark']['ret'], dtype=float)
    spr = np.asarray(bt['series']['前20大合计']['state']['ret'], dtype=float)
    out['benchmark'] = {
        '买入持有': [round(v, 7) for v in (1.0 + np.cumsum(bh))],
        '蜘蛛网原策略_前20': [round(v, 7) for v in (1.0 + np.cumsum(spr))],
        '买入持有累计点数': round(float(bh.sum() * sb.PRICE_REF), 4),
        '蜘蛛网原策略累计点数': round(float(spr.sum() * sb.PRICE_REF), 4),
        '蜘蛛网原策略规则': 'dB>0且dS<0→多 / dB<0且dS>0→空；T+1开盘→T+2收盘（H=1），规则不同仅供刻度',
    }

    def row(lab, seat, rule, extra=''):
        m = seats[seat][rule]['all']
        return {'口径': lab, '席位': seat, '规则': '多空双侧' if rule == 'full' else '只做多',
                '累计点数': m['累计点数'], '夏普比率': m['夏普比率'],
                '交易笔数': m['交易笔数'], '胜率': m['胜率'],
                '扣1tick点数': seats[seat][rule]['costs'][1]['累计点数'],
                '扣2tick点数': seats[seat][rule]['costs'][2]['累计点数'],
                '探索段': seats[seat][rule]['explore']['累计点数'],
                '验证段': seats[seat][rule]['verify']['累计点数'], '备注': extra}
    out['overview'] = [
        row('中信期货 多空双侧', FOCUS, 'full', '第七节席位安慰剂'),
        row('平安期货 多空双侧（第七节主口径）', TARGET, 'full', '口述规则原始版本'),
        row('中信期货 只做多', FOCUS, 'long', '关掉做空腿后反而不如平安'),
        row('平安期货 只做多（第八节主口径）', TARGET, 'long', '两项研究里最抗成本'),
        row('前20大合计 多空双侧', '前20大合计', 'full', ''),
        row('前20大合计 只做多', '前20大合计', 'long', ''),
    ]

    # ---------- ⑥ 结论 ----------
    a, p = attribution[FOCUS], attribution[TARGET]
    cs, nm = out['cross_section'], out['null_max']
    yr_txt = ', '.join('{}:{:+.2f}'.format(r['年份'], r['累计点数'])
                       for r in seats[FOCUS]['full']['yearly'][:5])
    y26 = seats[FOCUS]['full']['yearly'][5]['累计点数']
    out['conclusion'] = {
        'headline': '中信多空拆解：分别查看空单自身和单仓位占用的影响，并用九席位对照处理筛选偏差。',
        'points': [
            f"中信当前结果使用第七节**多空双侧**规则（收跌+大幅加多→多、收涨+大幅减多→空、H=3）套到中信的结果；"
            f"同一规则平安 {real[TARGET]:.3f} 点，差 {real[FOCUS]-real[TARGET]:.3f} 点。"
            f"换成**只做多**，中信只剩 {seats[FOCUS]['long']['all']['累计点数']:.3f} 点（验证段还是负的），"
            f"平安却是 {seats[TARGET]['long']['all']['累计点数']:.3f} 点 ⇒ 中信的领先只存在于这个特定口径。",
            f"差额几乎全在做空腿：中信多腿 {a['多腿']:+.3f} / 空腿 {a['空腿']:+.3f}，"
            f"平安多腿 {p['多腿']:+.3f} / 空腿 {p['空腿']:+.3f}。"
            f"两边做多腿几乎一样（中信还略低 {a['多腿']-p['多腿']:+.3f}），空腿差 {a['空腿']-p['空腿']:+.3f} 点。",
            f"更细的 A/B/C 隔离：中信从只做多 {a['B_只做多']:.3f} → 完整 {a['A_完整双侧']:.3f}，"
            f"其中空单自身赚 {a['空单自身盈亏']:+.3f} 点、空单占仓位挡掉 {a['被挤掉多单笔数']} 笔多单带来 "
            f"{a['挤掉多单的效应']:+.3f} 点（那批单在只做多口径里合计 {a['被挤掉多单在只做多里合计']:+.3f} 点）。"
            f"平安恰好反过来：空单亏 {p['空单自身盈亏']:+.3f}、还挡掉 {p['被挤掉多单笔数']} 笔本可赚 "
            f"{p['被挤掉多单在只做多里合计']:+.3f} 点的多单。",
            f"但中信的空头信号本身没有信息：{sig_short[FOCUS]['信号日数']} 个做空信号日未来 3 日均值 "
            f"{sig_short[FOCUS]['未来3日均值点数']:+.5f} 点、逐笔 t={sig_short[FOCUS]['逐笔t值']:+.2f}，"
            f"相对全体收涨日 Welch t={sig_short[FOCUS]['Welch_t_vs_收涨日']:+.2f}，均不显著。"
            f"真正赚的是执行结构：{a['做空信号日数']} 个空信号只成交 {a['做空成交笔数']} 笔，"
            f"成交的未来 3 日均值 {a['成交空信号未来均值点数']:+.5f} 点（跌），被挡掉的 "
            f"{a['做空信号被跳过笔数']} 个反而 {a['跳过空信号未来均值点数']:+.5f} 点（涨）——"
            "「成簇时只进第一笔」替它把亏损的追单过滤掉了。",
            f"选择偏差：中信单看自己的循环移位 p={shift[FOCUS]['p']:.4f} 确实显著，"
            f"但它是 {nm['n_seats']} 个席位里最大的一个。按「9 席位取最大」的随机分布，"
            f"随机最大值均值 {nm['mean']:.3f}、p95 {nm['p95']:.3f}，真实 {nm['real_max']:.3f} 只排 "
            f"{nm['pct_max']*100:.1f} 分位、p={nm['p_max']:.4f} ⇒ **多重比较后不显著**。"
            f"横截面 z={cs['z_focus']:.2f}，也不算离群。",
            f"反倒说明规则本身普遍有效：随机移位后 9 席位平均只有 {nm['rand_mean_of_means']:+.3f} 点，"
            f"真实 9 席位均值 {nm['real_cross_mean']:.3f} 点（p={nm['p_cross_mean']:.4f}）⇒ "
            "赚的是「这段 TS 慢牛 + 收跌做多/收涨做空 + 不叠加」的共同结构，不是某个席位独占的信息。",
            f"成本：中信双侧每笔来回扣 0.005 点 {real[FOCUS]:.3f} → {seats[FOCUS]['full']['costs'][1]['累计点数']:.3f} 点，"
            f"扣 0.010 点 → {seats[FOCUS]['full']['costs'][2]['累计点数']:.3f} 点，"
            f"扣 0.020 点 → {seats[FOCUS]['full']['costs'][3]['累计点数']:.3f} 点。"
            f"平安纯多扣 0.005 点后有 {seats[TARGET]['long']['costs'][1]['累计点数']:.3f} 点，最抗成本。",
            f"分年：中信双侧 2021–2025 全正（{yr_txt}）、2026 {y26:+.3f}；但空腿分年忽正忽负，"
            "「连续 5 年正」主要来自做多腿 + 执行结构，不代表空头判断稳定。",
            "定位：**刻度 + 证伪，不能上仓**。中信是看过全样本净值之后才被注意到的席位，"
            "样本外切分防不了「挑席位」；网页里的 9 条净值是横向刻度，不是 9 条可交易策略。",
        ],
    }

    # ---------- ⑦ 输出 ----------
    json.dump(out, open(OUT_JSON, 'w', encoding='utf-8'),
              ensure_ascii=False, separators=(',', ':'))
    print('JSON:', os.path.getsize(OUT_JSON) // 1024, 'KB')

    with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as w:
        pd.DataFrame(out['overview']).to_excel(w, sheet_name='全家福', index=False)
        prows = []
        for name, rec in seats.items():
            for rule in ['full', 'long']:
                m = rec[rule]['all']
                prows.append({
                    '席位': name, '规则': '多空双侧' if rule == 'full' else '只做多',
                    '累计点数': m['累计点数'], '夏普比率': m['夏普比率'],
                    '最大回撤': m['最大回撤'], '交易笔数': m['交易笔数'],
                    '胜率': m['胜率'], '逐笔t值': m['逐笔t值'],
                    '多腿点数': rec[rule]['多腿'], '空腿点数': rec[rule]['空腿'],
                    '探索段': rec[rule]['explore']['累计点数'],
                    '验证段': rec[rule]['verify']['累计点数'],
                    '扣1tick': rec[rule]['costs'][1]['累计点数'],
                    '扣2tick': rec[rule]['costs'][2]['累计点数'],
                })
        pd.DataFrame(prows).to_excel(w, sheet_name='席位绩效', index=False)
        pd.DataFrame([{'席位': k, **v} for k, v in attribution.items()]) \
            .to_excel(w, sheet_name='中信归因', index=False)
        pd.DataFrame([{'指标': k, **v} for k, v in sig_short.items()
                      if isinstance(v, dict)]).to_excel(w, sheet_name='做空信息', index=False)
        yrows = []
        for name, rec in seats.items():
            for r0 in rec['full']['yearly']:
                yrows.append({'席位': name, **r0})
        pd.DataFrame(yrows).to_excel(w, sheet_name='分年', index=False)
        pd.DataFrame(seats[FOCUS]['full']['trades']).to_excel(w, sheet_name='中信逐笔', index=False)
        pd.DataFrame(seats[TARGET]['full']['trades']).to_excel(w, sheet_name='平安逐笔', index=False)
    print('Excel:', os.path.getsize(OUT_XLSX) // 1024, 'KB')

    tpl = open(TPL, encoding='utf-8').read()
    assert tpl.count('/*__DATA__*/') == 1, '模板占位符数量不对'
    js = json.dumps(out, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    open(OUT_HTML, 'w', encoding='utf-8', newline='\n').write(tpl.replace('/*__DATA__*/', js))
    print('HTML:', os.path.getsize(OUT_HTML) // 1024, 'KB')

    # ---------- ⑧ 控制台摘要 ----------
    print('\n=== 9 席位 × 2 规则（全样本累计点数）===')
    for name, rec in seats.items():
        print(f"  {name:<14} 双侧 {rec['full']['all']['累计点数']:>7.3f} "
              f"(多{rec['full']['多腿']:+.3f} 空{rec['full']['空腿']:+.3f})  "
              f"纯多 {rec['long']['all']['累计点数']:>7.3f}")
    print('\n=== 中信 A/B/C ===')
    for k, v in attribution[FOCUS].items():
        print(' ', k, v)
    print('\n=== 零假设 ===')
    for k, v in shift.items():
        print(' ', k, 'p', v['p'], '随机均值', v['mean'])
    print('  9席位取最大 p', nm['p_max'], '随机最大均值', nm['mean'],
          '真实最大', nm['real_max'], '横截面均值 p', nm['p_cross_mean'])
    print('\n=== 结论 ===')
    for s in out['conclusion']['points']:
        print('  •', s[:110])


if __name__ == '__main__':
    main()
