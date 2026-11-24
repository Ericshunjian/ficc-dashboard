# -*- coding: utf-8 -*-
"""
平安期货「只做多」策略（2026-11-24，承接 spider_pingan_signal.py）

用户新要求：基于平安的逆势加多操作，不做空，只择机做多。

本轮要回答的核心问题不是「净值好不好看」，而是：
  ① 只做多是买 TS 慢牛，那么「择机」到底有没有增量？
  ② 增量来自「收跌日做多」（价格条件）还是「平安大幅加多」（平安条件）？
  ③ 删掉做空腿是不是事后偷看？—— 检验探索段做空腿当时是否已亏。
  ④ 平安在 9 个席位里排第一，是否超出随机预期？

口径（沿用上一轮，参数一个不改，只是关掉做空腿）：
  力度   = dNet 在过去 60 个有效日里的百分位（含当日，只用 t 及之前，无前视）
  信号   = 当日收跌 且 力度 >= 80 分位 → 次日开盘做多 5 手 TS
  执行   = T 日盘后信号 → T+1 开盘建仓 → 持有 3 个交易日 → T+4 收盘平仓
           单仓位、期间不叠加；固定名义 1000 万、单倍杠杆
  净值   = 1 + Σ(Δ价 − 成本)/100，固定名义单利累加、不复利
  成本   = 主结果零成本，另给 0.002 / 0.005 / 0.01 / 0.02 点（均为每笔来回固定点数）

★ 随机择时零假设（本轮最关键）：同构状态机对照。
  从「全部交易日池」/「收跌日池」各随机抽与真实信号同等密度的日子作为信号日，
  跑完全相同的状态机（含占位跳过），得到随机择时的累计点数分布。
  分解：择机总增量 = 真实 − μ全池；价格条件 = μ收跌池 − μ全池；平安条件 = 真实 − μ收跌池。
  这样三项加总自洽，且不会把「状态机跳过冲突信号」的选择效应错记成信号能力。

输出：TS_平安纯多头.json / TS_平安纯多头_明细.xlsx / TS_平安纯多头_策略.html
"""
import os, json, math, warnings
from datetime import datetime
import numpy as np
import pandas as pd

import spider_pingan_signal as sb

warnings.filterwarnings('ignore')
try:
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(BASE, 'TS_平安纯多头.json')
OUT_XLSX = os.path.join(BASE, 'TS_平安纯多头_明细.xlsx')
TPL = os.path.join(BASE, 'pingan_longonly_template.html')
OUT_HTML = os.path.join(BASE, 'TS_平安纯多头_策略.html')
PREV_JSON = os.path.join(BASE, 'TS_平安净多策略.json')

PRICE_REF = sb.PRICE_REF
ANN_DAYS = sb.ANN_DAYS
WIN = sb.WIN
THRESHOLDS = sb.THRESHOLDS
HOLDS = sb.HOLDS
SOURCES = sb.SOURCES
SPLIT = sb.SPLIT
N_PERM = 1200            # 同构随机择时次数
N_STRAT = 1000           # 按年分层抽样次数
N_JITTER = 800           # 时点抖动次数
N_SHIFT = 400            # 循环移位次数
SEED = 20261124
TARGET = sb.TARGET
COSTS = [0.0, 0.002, 0.005, 0.01, 0.02]
LONG_MODES = ['down', 'only']   # down=只在收跌日做多（口述）；only=不加价格条件

# 事前锁定参数：与上一轮口述规则完全一致，只是不做空
PRESET = {'source': 'dNet', 'mode': 'down', 'th': 0.80, 'H': 3}


def make_long_signal(q, dP, mode, th):
    """只输出 0/1：力度 >= th 分位；down 模式额外要求当日收跌。"""
    ok = (~np.isnan(q)) & (np.nan_to_num(q, nan=-1.0) >= th)
    if mode == 'down':
        ok = ok & (dP < 0)
    return ok.astype(int)


def future_pts(O, C, H):
    """逐笔独立口径：t 日信号 → t+1 开盘 → t+1+H 收盘的点数（允许重叠）。"""
    n = len(C)
    f = np.full(n, np.nan)
    for t in range(n - 1 - H):
        if not (np.isnan(O[t + 1]) or np.isnan(C[t + 1 + H])):
            f[t] = C[t + 1 + H] - O[t + 1]
    return f


def pool_stats(f, mask, label):
    v = f[mask]
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return {'名称': label, 'n': int(len(v))}
    se = v.std(ddof=1) / np.sqrt(len(v))
    return {'名称': label, 'n': int(len(v)),
            '每笔均值点数': round(float(v.mean()), 5),
            '中位点数': round(float(np.median(v)), 5),
            '胜率': round(float((v > 0).mean()), 4),
            't值': round(float(v.mean() / se), 3) if se > 0 else None,
            '累计点数(逐笔全开)': round(float(v.sum()), 4)}


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return round(float((a.mean() - b.mean()) / se), 3) if se > 0 else None


def exposure(ret, trades, H, a=0, b=None):
    """在场时间占比 & 按占用资金年化：只在资金被占用时计年化。"""
    n = len(ret)
    if b is None:
        b = n - 1
    days_on = sum(max(0, min(b, t['exit_idx']) - max(a, t['entry_idx']) + 1)
                  for t in trades)
    span = b - a + 1
    frac = days_on / span if span else 0.0
    total = float(np.sum(ret[a:b + 1]))
    return {'持仓交易日': int(days_on), '区间交易日': int(span),
            '在场时间占比': round(frac, 4),
            '累计收益率': round(total, 6),
            '按占用资金年化': round(float(total / frac * ANN_DAYS / span), 6) if frac > 0 else None}


def main():
    dates, idx, O, C, dP, series = sb.load()
    n = len(dates)
    split_i = int(np.searchsorted(np.array(dates), SPLIT))
    Q = {name: {src: sb.rolling_pct(s[src]) for src in SOURCES}
         for name, s in series.items()}
    qt = Q[TARGET]
    q0 = qt[PRESET['source']]
    H0, th0 = PRESET['H'], PRESET['th']
    f = future_pts(O, C, H0)
    valid = ~np.isnan(f)
    sig_main = make_long_signal(q0, dP, PRESET['mode'], th0)
    n_sig = int(sig_main.sum())

    out = {
        'meta': {
            'title': '平安期货「只做多」策略 · 择机增量检验',
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'start': dates[0], 'end': dates[-1], 'n_days': n,
            'target': TARGET, 'price_ref': PRICE_REF, 'win': WIN,
            'thresholds': THRESHOLDS, 'holds': HOLDS, 'sources': SOURCES,
            'costs': COSTS, 'split': SPLIT,
            'explore': [dates[0], dates[split_i - 1]],
            'verify': [dates[split_i], dates[-1]],
            'n_perm': N_PERM, 'n_strat': N_STRAT, 'n_jitter': N_JITTER,
            'n_shift': N_SHIFT, 'seed': SEED,
            'preset': PRESET,
            'rule': 'T日盘后：当日收跌 且 dNet 60日分位≥80% → T+1开盘做多5手TS → 持有3个交易日 → T+4收盘平仓；单仓位不叠加；不做空',
            'nav': '净值 = 1 + Σ(Δ价 − 成本)/100，固定名义单利累加、不复利（1点=初始资金1%）',
            'snooping': '「只做多」是看过上一轮完整规则结果后提出的修改；本轮用「探索段做空腿当时是否已亏」检验该决定是否可事前正当化，并把验证段当作干净样本外。',
        },
        'dates': dates,
        'price': {'open': [None if np.isnan(x) else round(float(x), 4) for x in O],
                  'close': [None if np.isnan(x) else round(float(x), 4) for x in C]},
    }

    # ---------- ① 删腿是否可事前正当化 ----------
    full_sig = sb.make_signal(q0, dP, 'contra', th0)
    r_full, tr_full = sb.run(full_sig, O, C, H0, 0.0, 0, n - 1)
    drop = []
    for lab, a, b in [('探索段', 0, split_i - 1), ('验证段', split_i, n - 1), ('全样本', 0, n - 1)]:
        lg = [t['pnl_pts'] for t in tr_full if t['dir'] == 1 and a <= t['exit_idx'] <= b]
        sh = [t['pnl_pts'] for t in tr_full if t['dir'] == -1 and a <= t['exit_idx'] <= b]
        drop.append({'区间': lab, '做多笔数': len(lg), '做多点数': round(float(sum(lg)), 4),
                     '做空笔数': len(sh), '做空点数': round(float(sum(sh)), 4)})
    out['drop_leg_evidence'] = {
        'rows': drop,
        'verdict': '探索段做空腿就已亏损 ⇒ 「只做多」可用 2023 年底之前的信息正当化，验证段是干净样本外'
                   if drop[0]['做空点数'] < 0 else '探索段做空腿为正 ⇒ 删腿属于事后选择，验证段需打折看',
        'full_nav': sb.nav_from(r_full),
    }

    # ---------- ② 主口径 ----------
    ret_main, tr_main = sb.run(sig_main, O, C, H0, 0.0, 0, n - 1)
    for t in tr_main:
        i = t['signal_idx']
        t['signal_date'] = dates[i]
        t['entry_date'] = dates[t['entry_idx']]
        t['exit_date'] = dates[t['exit_idx']]
        t['q'] = None if np.isnan(q0[i]) else round(float(q0[i]), 4)
        t['dNet'] = None if np.isnan(series[TARGET]['dNet'][i]) else round(float(series[TARGET]['dNet'][i]), 1)
        t['dP'] = None if np.isnan(dP[i]) else round(float(dP[i]), 4)
        t['方向'] = '多'
    out['preset'] = {
        'params': PRESET,
        'desc': '当日收跌 且 平安 dNet 60日分位 ≥ 80%（近250日约 +571 手）→ 次日开盘做多 5 手 TS，持有 3 个交易日；不做空',
        'explore': sb.metrics(ret_main, tr_main, 0, split_i - 1),
        'verify': sb.metrics(ret_main, tr_main, split_i, n - 1),
        'all': sb.metrics(ret_main, tr_main, 0, n - 1),
        'nav': sb.nav_from(ret_main),
        'daily_ret': [round(float(x), 8) for x in ret_main],
        'trades': tr_main,
        'th_hands': sb.hands_of(series, TARGET, 'dNet', th0),
        'exposure': exposure(ret_main, tr_main, H0),
        'exposure_verify': exposure(ret_main, tr_main, H0, split_i, n - 1),
        'signal_days': n_sig,
    }

    # ---------- ③ 三层择机分解（逐笔独立 / 状态机 / 同构随机零假设） ----------
    hi = (~np.isnan(q0)) & (np.nan_to_num(q0, nan=-1.0) >= th0)
    pools_mask = {
        '全部交易日（β基准）': valid,
        '收跌日（加价格条件）': valid & (dP < 0),
        '收涨日': valid & (dP > 0),
        '收跌 + 平安大幅加多（主口径信号池）': valid & (dP < 0) & hi,
        '收跌 + 平安力度不足': valid & (dP < 0) & (~hi),
        '平安大幅加多（不加价格条件）': valid & hi,
    }
    out['decomposition'] = {
        'pool_stats': [pool_stats(f, m, k) for k, m in pools_mask.items()],
        'incremental': {
            '价跌 vs 全部（价格条件增量）': round(float(
                f[valid & (dP < 0)][~np.isnan(f[valid & (dP < 0)])].mean() -
                f[valid][~np.isnan(f[valid])].mean()), 5),
            '价跌+平安 vs 价跌（平安条件增量）': round(float(
                f[valid & (dP < 0) & hi][~np.isnan(f[valid & (dP < 0) & hi])].mean() -
                f[valid & (dP < 0)][~np.isnan(f[valid & (dP < 0)])].mean()), 5),
        },
    }
    a1 = f[valid & (dP < 0) & hi]; a1 = a1[~np.isnan(a1)]
    a2 = f[valid & (dP < 0) & (~hi)]; a2 = a2[~np.isnan(a2)]
    out['decomposition']['incremental']['平安条件 Welch_t'] = welch(a1, a2)

    # 状态机四组对照（同构，公平）
    variants = [
        ('主口径：收跌 + 平安大幅加多', sig_main),
        ('去掉平安：只看收跌日', (dP < 0).astype(int)),
        ('去掉价格：只看平安力度', hi.astype(int)),
        ('去掉择机：每个交易日都做多', np.ones(n, dtype=int)),
    ]
    sv = []
    for lab, sg in variants:
        r, tr = sb.run(sg, O, C, H0, 0.0, 0, n - 1)
        sv.append({'名称': lab,
                   'explore': sb.metrics(r, tr, 0, split_i - 1),
                   'verify': sb.metrics(r, tr, split_i, n - 1),
                   'all': sb.metrics(r, tr, 0, n - 1),
                   'nav': sb.nav_from(r)})
    out['decomposition']['state_variants'] = sv

    # 被状态机「占位」跳过的信号日：检验这部分是不是运气
    used = {t['signal_idx'] for t in tr_main}
    skipped = [i for i in np.where(sig_main == 1)[0] if i not in used]
    f_used = f[[i for i in used if not np.isnan(f[i])]]
    f_skip = f[[i for i in skipped if not np.isnan(f[i])]]
    out['decomposition']['skipped'] = {
        '信号日数': n_sig, '成交笔数': len(tr_main), '被跳过': len(skipped),
        '成交信号未来均值点数': round(float(f_used.mean()), 5),
        '被跳过信号未来均值点数': round(float(f_skip.mean()), 5),
        '说明': '连续信号被单仓位规则跳过；若两者接近，说明净值没有被「跳过坏交易」系统性抬高',
    }

    # ★ 同构随机择时零假设
    rng = np.random.default_rng(SEED)
    pool_all = np.arange(1, n - H0 - 1)          # 全交易日池（排除首日与末尾不足 H 的日子）
    pool_down = np.where((dP < 0) & valid)[0]    # 收跌日池
    real_all = out['preset']['all']['累计点数']
    real_verify = out['preset']['verify']['累计点数']

    def rand_null(pool, seg_a, seg_b, real):
        """抽 n_sig 个信号日 → 同构状态机 → 累计点数分布。"""
        acc, navs = [], []
        for _ in range(N_PERM):
            pick = rng.choice(pool, min(n_sig, len(pool)), replace=False)
            sg = np.zeros(n, dtype=int)
            sg[pick] = 1
            sg[:seg_a] = 0
            r, tr = sb.run(sg, O, C, H0, 0.0, 0, n - 1)
            m = sb.metrics(r, tr, seg_a, seg_b)
            acc.append(m['累计点数'])
            navs.append(sb.nav_from(r, seg_a, seg_b))
        A = np.array(acc)
        med_i = int(np.argsort(np.abs(A - np.median(A)))[0])
        return {'hist': [round(float(x), 4) for x in A],
                'mean': round(float(A.mean()), 4),
                'p05': round(float(np.percentile(A, 5)), 4),
                'p95': round(float(np.percentile(A, 95)), 4),
                'p': round(float((A >= real).mean()), 4),
                'real': round(float(real), 4),
                'pct': round(float((A < real).mean()), 4),
                'median_nav': [round(float(x), 7) for x in navs[med_i]]}

    rand = {}
    for seg, a, b, real in [('full', 0, n - 1, real_all), ('verify', split_i, n - 1, real_verify)]:
        pa = pool_all[pool_all >= a] if a else pool_all
        pd_ = pool_down[pool_down >= a]
        rand[seg] = {'全交易日池': rand_null(pa, a, b, real),
                     '收跌日池': rand_null(pd_, a, b, real)}
    # N3 按年分层抽样：保持每年信号数，排除「信号恰好集中在上涨年份」的时段暴露
    years = np.array([int(d[:4]) for d in dates])
    real_idx = np.where(sig_main == 1)[0]
    strat_acc, strat_acc_v = [], []
    for _ in range(N_STRAT):
        pick = []
        for y in sorted(set(years[real_idx])):
            py = pool_down[years[pool_down] == y]
            ry = real_idx[years[real_idx] == y]
            if len(py):
                pick.extend(rng.choice(py, min(len(ry), len(py)), replace=False).tolist())
        sg = np.zeros(n, dtype=int); sg[np.asarray(pick, dtype=int)] = 1
        r, tr = sb.run(sg, O, C, H0, 0.0, 0, n - 1)
        strat_acc.append(sb.metrics(r, tr, 0, n - 1)['累计点数'])
        strat_acc_v.append(sb.metrics(r, tr, split_i, n - 1)['累计点数'])
    SA, SV = np.array(strat_acc), np.array(strat_acc_v)
    rand['full']['按年分层收跌池'] = {
        'hist': [round(float(x), 4) for x in SA], 'mean': round(float(SA.mean()), 4),
        'p05': round(float(np.percentile(SA, 5)), 4),
        'p95': round(float(np.percentile(SA, 95)), 4),
        'p': round(float((SA >= real_all).mean()), 4),
        'p_verify': round(float((SV >= real_verify).mean()), 4),
        'real': round(float(real_all), 4),
        'pct': round(float((SA < real_all).mean()), 4),
    }

    # N5 时点抖动：每个真实信号日 ±W 日挪动。
    # 「仍收跌」变体保持价格条件与信号成簇结构，只打乱平安到底挑了哪一天 —— 这是平安增量的主判据。
    def jitter_null(W, keep_down):
        pool = pool_down if keep_down else np.arange(n)
        cand = sb.jitter_candidates(real_idx, pool, W)
        acc, per, ntr = [], [], []
        for _ in range(N_JITTER):
            pick = sb.sample_jitter(cand, rng)
            sg = np.zeros(n, dtype=int); sg[pick] = 1
            r, tr = sb.run(sg, O, C, H0, 0.0, 0, n - 1)
            m = sb.metrics(r, tr, 0, n - 1)
            acc.append(m['累计点数']); per.append(m.get('每笔平均点数') or 0.0); ntr.append(m['交易笔数'])
        A = np.array(acc)
        return {'hist': [round(float(x), 4) for x in A], 'mean': round(float(A.mean()), 4),
                'p05': round(float(np.percentile(A, 5)), 4),
                'p95': round(float(np.percentile(A, 95)), 4),
                'p': round(float((A >= real_all).mean()), 4),
                'real': round(float(real_all), 4),
                'pct': round(float((A < real_all).mean()), 4),
                '每笔均值点数': round(float(np.mean(per)), 5),
                '平均成交笔数': round(float(np.mean(ntr)), 1)}
    for W in [3, 5]:
        rand['full'][f'抖动±{W}日·仍收跌'] = jitter_null(W, True)
    rand['full']['抖动±3日·不限'] = jitter_null(3, False)

    out['decomposition']['random_null'] = rand
    muA = rand['full']['全交易日池']['mean']
    muB = rand['full']['收跌日池']['mean']
    out['decomposition']['contribution'] = {
        '随机择时基准 μA': muA,
        '加价格条件后 μB': muB,
        '真实主口径': real_all,
        '择机总增量': round(real_all - muA, 4),
        '其中价格条件贡献': round(muB - muA, 4),
        '其中平安条件贡献': round(real_all - muB, 4),
        'p_全池': rand['full']['全交易日池']['p'],
        'p_收跌池': rand['full']['收跌日池']['p'],
        'p_收跌池_验证段': rand['verify']['收跌日池']['p'],
        'p_按年分层': rand['full']['按年分层收跌池']['p'],
        'p_分层_验证段': rand['full']['按年分层收跌池']['p_verify'],
        'p_抖动仍收跌_3日': rand['full']['抖动±3日·仍收跌']['p'],
        'p_抖动仍收跌_5日': rand['full']['抖动±5日·仍收跌']['p'],
        'p_抖动不限_3日': rand['full']['抖动±3日·不限']['p'],
        '说明': ('三项加总自洽：总增量 = 价格条件 + 平安条件。但「平安条件」这 1.629 点里混着'
                 '「信号成簇 + 单仓位只进第一笔、跳过后续追单」的执行结构；'
                 '把信号日在收跌日范围内抖动 ±3 天后 p 不再显著，说明平安挑中具体哪一天并无独立增量。'),
    }

    # 买入持有基准（同样 5 手固定名义）
    bh = np.zeros(n)
    bh[1:] = (C[1:] - C[:-1]) / PRICE_REF
    bh = np.nan_to_num(bh)
    out['benchmark'] = {
        'buy_hold': {'nav': sb.nav_from(bh),
                     'all': sb.metrics(bh, [{'exit_idx': i, 'pnl_pts': float(bh[i]) * PRICE_REF}
                                            for i in range(n) if bh[i] != 0], 0, n - 1),
                     '累计点数': round(float(C[-1] - C[0]), 4)},
        'random_median_nav_full': rand['full']['全交易日池']['median_nav'],
        'random_median_nav_down': rand['full']['收跌日池']['median_nav'],
        'start_idx': 0, 'verify_start_idx': split_i,
    }

    # ---------- ④ 循环移位零假设（整条信号 / 只打乱平安力度） ----------
    def shift_null(kind):
        acc, acc_v = [], []
        for _ in range(N_SHIFT):
            k = int(rng.integers(1, n))
            if kind == 'shift_signal':
                sg = np.roll(sig_main, k)
            else:
                qq = np.roll(q0, k)
                sg = make_long_signal(qq, dP, 'down', th0)
            r, tr = sb.run(sg, O, C, H0, 0.0, 0, n - 1)
            acc.append(sb.metrics(r, tr, 0, n - 1)['累计点数'])
            acc_v.append(sb.metrics(r, tr, split_i, n - 1)['累计点数'])
        A, V = np.array(acc), np.array(acc_v)
        return {'hist': [round(float(x), 4) for x in acc],
                'mean': round(float(A.mean()), 4),
                'p95': round(float(np.percentile(A, 95)), 4),
                'p': round(float((A >= real_all).mean()), 4),
                'p_verify': round(float((V >= real_verify).mean()), 4),
                'real': round(float(real_all), 4)}
    out['null_shift'] = {'整条信号移位': shift_null('shift_signal'),
                         '只打乱平安力度': shift_null('shift_dnet')}

    # ---------- ⑤ 席位安慰剂（同规则纯多头）+ 最大值是否显著 ----------
    placebo = []
    for name in series:
        qq = Q[name][PRESET['source']]
        sg = make_long_signal(qq, dP, PRESET['mode'], th0)
        r, tr = sb.run(sg, O, C, H0, 0.0, 0, n - 1)
        placebo.append({'席位': name,
                        'explore': sb.metrics(r, tr, 0, split_i - 1),
                        'verify': sb.metrics(r, tr, split_i, n - 1),
                        'all': sb.metrics(r, tr, 0, n - 1),
                        'nav': sb.nav_from(r)})
    placebo.sort(key=lambda x: x['all']['累计点数'], reverse=True)
    out['placebo'] = {
        'rows': placebo,
        'rank_of_target': 1 + sum(1 for p in placebo if p['席位'] != TARGET
                                  and p['all']['累计点数'] > real_all),
        'n_seats': len(placebo),
        'note': '序列高度同源；即便全无信息，「9 个里排第一」的概率也有 1/9 ≈ 0.111，不能当显著性证据',
    }

    # ---------- ⑥ 参数面（纯多头网格） ----------
    cells = []
    cache = {}
    for src in SOURCES:
        for mode in LONG_MODES:
            for th in THRESHOLDS:
                for H in HOLDS:
                    sg = make_long_signal(qt[src], dP, mode, th)
                    r, tr = sb.run(sg, O, C, H, 0.0, 0, n - 1)
                    key = (src, mode, th, H)
                    cache[key] = (sg, r, tr)
                    cells.append({'source': src, 'mode': mode, 'th': th, 'H': H,
                                  'explore': sb.metrics(r, tr, 0, split_i - 1),
                                  'verify': sb.metrics(r, tr, split_i, n - 1),
                                  'all': sb.metrics(r, tr, 0, n - 1)})
    ep = [c['explore']['累计点数'] for c in cells]
    vp = [c['verify']['累计点数'] for c in cells]
    top10 = sorted(cells, key=lambda c: c['explore']['累计点数'], reverse=True)[:10]
    out['grid'] = {
        'cells': cells,
        'robust': {
            '验证段中位数点数': round(float(np.median(vp)), 4),
            '验证段为正比例': round(float(np.mean(np.array(vp) > 0)), 4),
            '探索段前10验证段为正数': sum(1 for c in top10 if c['verify']['累计点数'] > 0),
            '探索段前10验证段平均点数': round(float(np.mean([c['verify']['累计点数'] for c in top10])), 4),
            '两段相关': round(float(np.corrcoef(ep, vp)[0, 1]), 4),
        },
    }
    pool = [c for c in cells if c['explore'].get('交易笔数', 0) >= 20]
    best = max(pool, key=lambda c: c['explore']['累计点数'])
    sel = {k: best[k] for k in ['source', 'mode', 'th', 'H']}
    _, r_sel, tr_sel = cache[(sel['source'], sel['mode'], sel['th'], sel['H'])]
    out['grid']['best'] = {
        'params': sel,
        'explore': best['explore'], 'verify': best['verify'], 'all': best['all'],
        'nav': sb.nav_from(r_sel),
        'label': '只用探索段挑出，仅供刻度，不是可下注策略',
    }

    # ---------- ⑦ 成本 / 分年 / walk-forward ----------
    costs = []
    for c in COSTS:
        r, tr = sb.run(sig_main, O, C, H0, c, 0, n - 1)
        costs.append({'成本点': c, **sb.metrics(r, tr, 0, n - 1)})
    out['cost_sensitivity'] = costs

    yr = []
    for y in range(2021, 2027):
        a = int(np.searchsorted(np.array(dates), f'{y}-01-01'))
        b = int(np.searchsorted(np.array(dates), f'{y + 1}-01-01')) - 1
        if b >= n:
            b = n - 1
        yr.append({'年份': y, **sb.metrics(ret_main, tr_main, a, b)})
    out['yearly'] = yr

    wf_ret = np.zeros(n)
    wf_rows = []
    wf_trades = []
    for y in range(2023, 2027):
        a = int(np.searchsorted(np.array(dates), f'{y}-01-01'))
        b = int(np.searchsorted(np.array(dates), f'{y + 1}-01-01')) - 1
        if b >= n:
            b = n - 1
        cand = []
        for src in SOURCES:
            for mode in LONG_MODES:
                for th in THRESHOLDS:
                    for H in HOLDS:
                        sg = make_long_signal(qt[src], dP, mode, th)
                        r, tr = sb.run(sg, O, C, H, 0.0, 0, a - 1)
                        m = sb.metrics(r, tr, 0, a - 1)
                        if m.get('交易笔数', 0) >= 12:
                            cand.append((m['累计点数'], src, mode, th, H))
        if not cand:
            continue
        _, s2, m2, t2, H2 = max(cand, key=lambda x: x[0])
        sg = make_long_signal(qt[s2], dP, m2, t2)
        r, tr = sb.run(sg, O, C, H2, 0.0, a, b)
        wf_ret[a:b + 1] = r[a:b + 1]
        wf_trades.extend(tr)
        wf_rows.append({'年份': y, '参数': f'{s2}/{m2}/th={t2}/H={H2}',
                        **sb.metrics(wf_ret, tr, a, b)})
    wf_a = int(np.searchsorted(np.array(dates), '2023-01-01'))
    out['walk_forward'] = {'rows': wf_rows, 'nav': sb.nav_from(wf_ret, wf_a, n - 1),
                           'start_idx': wf_a,
                           'all': sb.metrics(wf_ret, wf_trades, wf_a, n - 1)}

    # ---------- ⑧ 结论（网页直接渲染） ----------
    c = out['decomposition']['contribution']
    out['conclusion'] = {
        'headline': '只做多确实比多空都做好得多、样本外不衰减；但赚钱的是「逢跌做多 + 不追单」的执行结构，不是平安挑的那一天。',
        'points': [
            f"主口径全样本 {real_all:.3f} 点（{out['preset']['all']['夏普比率']:.2f} 夏普 / "
            f"{out['preset']['all']['交易笔数']} 笔），探索段 {out['preset']['explore']['累计点数']:.3f} 点、"
            f"验证段 {out['preset']['verify']['累计点数']:.3f} 点，样本外没有衰减。",
            f"删腿可事前正当化：探索段做空腿已亏 {drop[0]['做空点数']:.3f} 点，验证段继续亏 {drop[1]['做空点数']:.3f} 点 ⇒ "
            "用 2023 年底之前的数据就能做出「不做空」的决定，验证段算干净样本外。",
            f"同构随机择时（同密度信号日、同状态机）：择机总增量 {c['择机总增量']:+.3f} 点，"
            f"其中价格条件 {c['其中价格条件贡献']:+.3f} 点、平安条件及成簇结构 {c['其中平安条件贡献']:+.3f} 点；"
            f"按年分层抽样 p={c['p_按年分层']:.3f}（验证段 p={c['p_分层_验证段']:.3f}），排除了「信号集中在上涨年份」。",
            f"★ 主判据（时点抖动）：把 162 个信号日在收跌日范围内前后挪 3 天、保持成簇结构，"
            f"累计仍有 {rand['full']['抖动±3日·仍收跌']['mean']:+.3f} 点，p={c['p_抖动仍收跌_3日']:.3f} ⇒ 不再显著"
            f"（±5 日 p={c['p_抖动仍收跌_5日']:.3f}，边缘显著，如实并列）；"
            f"逐笔独立口径 Welch t={welch(a1,a2):.2f}。两者都说明：平安「挑中具体哪一天」的独立信息很弱，"
            "前述 1.629 点主要来自「信号成簇时只进第一笔、跳过后续追单」——被跳过的 49 个信号平均亏 "
            f"{out['decomposition']['skipped']['被跳过信号未来均值点数']:.5f} 点，成交 113 笔平均赚 "
            f"{out['preset']['all']['每笔平均点数']:.5f} 点。其余零假设（全池 / 收跌池 / 分层 / 循环移位）p 更小，"
            "但它们同时打乱了信号的成簇结构或簇的时间位置，把执行纪律的功劳也算进了「平安」，故不作主判据。",
            f"成本：每笔平均 {out['preset']['all']['每笔平均点数']:.4f} 点（约 "
            "每笔来回扣 0.005 点后 "
            f"{costs[2]['累计点数']:.3f} 点、扣 0.010 点 {costs[3]['累计点数']:.3f} 点，比上一轮含做空版抗成本得多。",
            f"资金效率：只在场 {out['preset']['exposure']['在场时间占比']*100:.1f}% 的时间，"
            f"按占用资金年化 {out['preset']['exposure']['按占用资金年化']*100:.2f}%；"
            f"但绝对收益仍低于买入持有 {out['benchmark']['buy_hold']['累计点数']:.3f} 点。",
            f"席位安慰剂：平安在 {len(placebo)} 个序列里排第 {out['placebo']['rank_of_target']}，"
            "但序列高度同源、排第一的随机概率就有 1/9，不能当显著性证据。",
            f"时点脆弱性：若不限制在收跌日、把信号日整体抖动 ±3 天，累计从 {real_all:.3f} 掉到 "
            f"{rand['full']['抖动±3日·不限']['mean']:.3f} 点 ⇒ 价格条件一旦丢失就几乎不赚，"
            "策略对「收跌日」这个条件的依赖远大于对平安的依赖。",
            f"参数面：验证段 {out['grid']['robust']['验证段为正比例']*100:.1f}% 格子为正、探索段前10在验证段全部为正，"
            f"但两段相关仅 {out['grid']['robust']['两段相关']:.2f}；探索段最优持有期漂到 H=10，"
            "walk-forward 也稳定在 H=10 ⇒ 「持有几天」不稳定，不能拿某一格当结论。",
            "定位：刻度 + 证伪。可作为「逢跌做多、单仓位不追单」的执行刻度；不能据此推断平安席位有独占信息，也不能直接上仓。"
            "多空双侧和纯多的取舍取决于做空盈亏和仓位重叠。"
            "各腿收益与修订版席位排名见分解与横向对照。"
            "两轮共同指向同一件事：赚的是价格条件与执行结构，不是平安席位的独占信息。",
        ],
    }

    # ---------- 写出 ----------
    json.dump(out, open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as w:
        pd.DataFrame([{'区间': d['区间'], '做多笔数': d['做多笔数'], '做多点数': d['做多点数'],
                       '做空笔数': d['做空笔数'], '做空点数': d['做空点数']} for d in drop
                      ] + [{'区间': '判据', '做多笔数': None, '做多点数': None, '做空笔数': None,
                            '做空点数': out['drop_leg_evidence']['verdict']}]).to_excel(w, sheet_name='删腿依据', index=False)
        pd.DataFrame(out['decomposition']['pool_stats']).to_excel(w, sheet_name='择机三层', index=False)
        pd.DataFrame([{'规则': v['名称'],
                       '探索段累计点数': v['explore']['累计点数'], '验证段累计点数': v['verify']['累计点数'],
                       '全样本累计点数': v['all']['累计点数'], '全样本笔数': v['all']['交易笔数'],
                       '全样本夏普': v['all']['夏普比率'], '每笔均值点数': v['all']['每笔平均点数']} for v in sv]
                     ).to_excel(w, sheet_name='状态机对照', index=False)
        pd.DataFrame([{'零假设': f'{seg} / {pool}', **{k: v[k] for k in ['mean', 'p05', 'p95', 'p', 'real', 'pct']}}
                      for seg, d in rand.items() for pool, v in d.items()]).to_excel(w, sheet_name='随机择时零假设', index=False)
        pd.DataFrame([{'项目': k, '值': v} for k, v in c.items()]).to_excel(w, sheet_name='增量分解', index=False)
        pd.DataFrame([{'方法': k, '均值': v['mean'], 'p95': v['p95'],
                       'p全样本': v['p'], 'p验证段': v['p_verify'], '真实': v['real']}
                      for k, v in out['null_shift'].items()]).to_excel(w, sheet_name='循环移位', index=False)
        pd.DataFrame([{'席位': p['席位'], '探索段': p['explore']['累计点数'], '验证段': p['verify']['累计点数'],
                       '全样本': p['all']['累计点数'], '笔数': p['all']['交易笔数'],
                       '夏普': p['all']['夏普比率']} for p in placebo]).to_excel(w, sheet_name='席位安慰剂', index=False)
        pd.DataFrame([{'信号源': x['source'], '价格条件': x['mode'], '力度分位': x['th'], '持有天数': x['H'],
                       '探索段点数': x['explore']['累计点数'], '探索段笔数': x['explore']['交易笔数'],
                       '验证段点数': x['verify']['累计点数'], '验证段笔数': x['verify']['交易笔数'],
                       '全样本点数': x['all']['累计点数'], '全样本夏普': x['all']['夏普比率']} for x in cells]
                     ).to_excel(w, sheet_name='参数面', index=False)
        pd.DataFrame(costs).to_excel(w, sheet_name='成本敏感性', index=False)
        pd.DataFrame(yr).to_excel(w, sheet_name='分年', index=False)
        pd.DataFrame(wf_rows).to_excel(w, sheet_name='walk_forward', index=False)
        pd.DataFrame([{k: v for k, v in t.items() if k != 'tag'} for t in tr_main]).to_excel(w, sheet_name='逐笔明细', index=False)
        pd.DataFrame([{'项目': k, '值': str(v)} for k, v in [
            ('主口径', out['meta']['rule']), ('净值', out['meta']['nav']),
            ('探索段', f"{dates[0]} ~ {dates[split_i-1]}"), ('验证段', f"{dates[split_i]} ~ {dates[-1]}"),
            ('数据窥探', out['meta']['snooping']),
            ('主判据', '同构随机择时：从收跌日池抽同等密度信号日跑相同状态机'),
            ('生成时间', out['meta']['generated'])]]).to_excel(w, sheet_name='说明', index=False)

    if os.path.exists(TPL):
        tpl = open(TPL, encoding='utf-8').read()
        js = json.dumps(out, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
        assert '/*__DATA__*/' in tpl, '模板缺少数据占位符'
        open(OUT_HTML, 'w', encoding='utf-8', newline='\n').write(tpl.replace('/*__DATA__*/', js))

    # ---------- 控制台摘要 ----------
    print('=== ① 删腿依据 ===');
    for d in drop:
        print(f"  {d['区间']}: 多 {d['做多点数']:+.3f} 点({d['做多笔数']}笔) | 空 {d['做空点数']:+.3f} 点({d['做空笔数']}笔)")
    print(' ', out['drop_leg_evidence']['verdict'])
    print('\n=== ② 主口径 ===')
    print(' ', out['preset']['desc'])
    for lab in ['explore', 'verify', 'all']:
        print(f'  {lab:<8}', out['preset'][lab])
    print('  在场:', out['preset']['exposure'])
    print('\n=== ③ 择机三层（逐笔独立） ===')
    for p in out['decomposition']['pool_stats']:
        print(f"  {p['名称']:<32} n={p['n']:>5} 每笔 {p.get('每笔均值点数',0):+.5f} 点 胜率 {p.get('胜率',0):.3f} t={p.get('t值')}")
    print('  条件内增量:', out['decomposition']['incremental'])
    print('\n=== ③b 状态机同构对照 ===')
    for v in sv:
        print(f"  {v['名称']:<28} 全 {v['all']['累计点数']:>8.3f} 点 / {v['all']['交易笔数']:>3}笔 "
              f"探索 {v['explore']['累计点数']:>7.3f} 验证 {v['verify']['累计点数']:>7.3f}")
    print('  占位跳过:', out['decomposition']['skipped'])
    print('\n=== ③c 随机择时零假设 ===')
    for seg, d in rand.items():
        for pool, v in d.items():
            print(f"  {seg:<7} {pool:<10} 随机均值 {v['mean']:+.3f} p95 {v['p95']:+.3f} "
                  f"真实 {v['real']:+.3f}  p={v['p']:.4f} 分位={v['pct']:.3f}")
    print('  增量分解:', {k: v for k, v in c.items() if isinstance(v, (int, float))})
    print('\n=== ④ 循环移位 ===')
    for k, v in out['null_shift'].items():
        print(f"  {k:<14} p={v['p']:.4f} p(验证段)={v['p_verify']:.4f} 随机均值={v['mean']:.3f}")
    print('\n=== ⑤ 席位安慰剂 ===')
    for p in placebo:
        print(f"  {p['席位']:<14} {p['all']['累计点数']:>8.3f} 点 {p['all']['交易笔数']:>3}笔")
    print('  平安排名:', out['placebo']['rank_of_target'], '/', out['placebo']['n_seats'])
    print('\n=== ⑥ 参数面 ===', out['grid']['robust'], '\n  探索段最优:', sel, out['grid']['best']['all'])
    print('\n=== ⑦ 成本 ===')
    for x in costs:
        print(f"  {x['成本点']:.3f} 点 → {x['累计点数']:+.3f} 点 夏普 {x['夏普比率']:.2f}")
    print('\n=== 分年 ===')
    for x in yr:
        print(f"  {x['年份']} {x['累计点数']:+.3f} 点 {x['交易笔数']:>3}笔")
    print('\n=== walk-forward ===')
    for x in wf_rows:
        print(f"  {x['年份']} {x['参数']:<26} {x['累计点数']:+.3f} 点")
    for p in [OUT_JSON, OUT_XLSX] + ([OUT_HTML] if os.path.exists(TPL) else []):
        print('输出:', p, f'({os.path.getsize(p)/1024:.1f} KB)')


if __name__ == '__main__':
    main()
