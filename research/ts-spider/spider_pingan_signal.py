# -*- coding: utf-8 -*-
"""
平安期货「逆势加多」主观感受 → 可检验策略（2026-11-24）

用户的观察：
  ① 价格下跌时平安净多头增加，随后几个交易日价格上涨；
  ② 价格还在涨时平安净多头下降，随后几个交易日价格下跌。
把它拆成两个可证伪的部件：
  部件 A：平安净多变动 dNet 与当日价格变动反向（逆势行为是否真实）；
  部件 B：在「价跌」条件下，平安加多是否对未来收益有增量（还是只是价格均值回复）。

口径（事前锁定，不按结果调）：
  力度   = dNet 在过去 60 个有效日里的百分位（含当日；只用 t 及之前数据，无前视）
           分位 >= th 视为「大幅加多」，<= 1-th 视为「大幅减多」
  逆势   = contra 模式要求 sign(dP) = -sign(dNet)；only 模式不加价格条件
  执行   = T 日盘后拿到持仓 → T+1 开盘建仓 → 持有 H 个交易日 → T+1+H 收盘平仓
           期间不再开新仓（单一仓位、固定 5 手 TS、名义 1000 万、单倍杠杆）
  净值   = 1 + Σ(方向×Δ价 − 成本)/100，固定名义单利累加、不复利（与 spider_backtest.py 一致）
  成本   = 主结果零成本，另给 0.002 / 0.005 / 0.01 点（TS 一个最小变动价位 = 0.005 点）

防数据窥探：
  探索段 2021-01-04~2023-12-31 选参数，验证段 2024-01-01~2026-11-23 只报告不调；
  另给出逐年 walk-forward（每年只用之前的数据选参数）。
  ★ 但「平安期货」本身是看过全样本净值后挑出来的序列，此切分只能防参数过拟合，
    防不了序列选择的信息泄漏。故同时做：其他 8 席位安慰剂 + 循环移位零假设 +
    「只留价格反转、去掉平安」的归因对照。

输出：TS_平安净多策略.json / TS_平安净多策略_明细.xlsx / TS_平安净多_策略.html
"""
import os, json, math, warnings
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
try:
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
HOLD_JSON = os.path.join(BASE, 'TS_持仓整理.json')
PRICE_XLSX = os.path.join(BASE, 'TS.CFE.xlsx')
OUT_JSON = os.path.join(BASE, 'TS_平安净多策略.json')
OUT_XLSX = os.path.join(BASE, 'TS_平安净多策略_明细.xlsx')
TPL = os.path.join(BASE, 'pingan_signal_template.html')
OUT_HTML = os.path.join(BASE, 'TS_平安净多_策略.html')

PRICE_REF = 100.0          # TS 报价面值：1 点 = 初始资金 1000 万的 1%
INIT_CAPITAL = 10_000_000.0
ANN_DAYS = 243
WIN = 60                   # 力度分位回看窗口（交易日）
MIN_OBS = 30
THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]
HOLDS = [1, 2, 3, 5, 8, 10]
SOURCES = ['dNet', 'dB']   # 净持仓变动 / 多单变动（用户口述「净多头」用 dNet）
MODES = ['contra', 'only']
COSTS = [0.0, 0.002, 0.005, 0.01]
SPLIT = '2024-01-01'       # 探索 / 验证分界
N_PERM = 400
SEED = 20261124
TARGET = '平安期货'


# ---------------------------------------------------------------- 数据
def load():
    P = json.load(open(HOLD_JSON, encoding='utf-8'))
    dates = P['dates']
    idx = pd.to_datetime(dates)
    q = pd.read_excel(PRICE_XLSX, sheet_name=0, header=0)
    q = q[q['日期'].notna()].copy()
    q['date'] = pd.to_datetime(q['日期'])
    q = q.set_index('date').sort_index()
    q = q[~q.index.duplicated(keep='last')]
    O = q['开盘价(元)'].astype(float).reindex(idx).to_numpy()
    C = q['收盘价(元)'].astype(float).reindex(idx).to_numpy()
    dP = np.full(len(C), np.nan)
    dP[1:] = C[1:] - C[:-1]
    series = {}
    for name, s in P['series'].items():
        net = pd.Series(s['net'], index=idx, dtype=float).to_numpy()
        lng = pd.Series(s['long'], index=idx, dtype=float).to_numpy()
        dn = np.full(len(net), np.nan)
        dn[1:] = net[1:] - net[:-1]
        db = np.full(len(lng), np.nan)
        db[1:] = lng[1:] - lng[:-1]
        series[name] = {'dNet': dn, 'dB': db, 'net': net}
    return dates, idx, O, C, dP, series


def rolling_pct(x, win=WIN, min_obs=MIN_OBS):
    """当日值在过去 win 个有效日里的百分位（含当日，只用 t 及之前，无前视）。"""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        if np.isnan(x[i]):
            continue
        w = x[max(0, i - win + 1): i + 1]
        w = w[~np.isnan(w)]
        if len(w) < min_obs:
            continue
        out[i] = float(np.mean(w <= x[i]))
    return out


# ---------------------------------------------------------------- 信号与回测
def make_signal(q, dP, mode, th, flip=False):
    lo = 1.0 - th
    lng = np.nan_to_num(q, nan=-1.0) >= th
    sht = (np.nan_to_num(q, nan=2.0) <= lo) & (~np.isnan(q))
    lng = lng & (~np.isnan(q))
    if mode == 'contra':
        lng = lng & (dP < 0)
        sht = sht & (dP > 0)
    sig = np.where(lng, 1, np.where(sht, -1, 0)).astype(int)
    return -sig if flip else sig


def run(sig, O, C, H, cost=0.0, a=0, b=None, tag=''):
    """单一仓位状态机：信号日 t → t+1 开盘建仓 → t+1+H 收盘平仓，期间不叠加。"""
    n = len(sig)
    if b is None:
        b = n - 1
    ret = np.zeros(n)
    trades = []
    free = a
    for t in range(a, b + 1):
        if sig[t] == 0 or t < free:
            continue
        e, x = t + 1, t + 1 + H
        if x > b or np.isnan(O[e]) or np.isnan(C[x]):
            continue
        pts = float(sig[t] * (C[x] - O[e]) - cost)
        ret[x] += pts / PRICE_REF
        trades.append({
            'signal_idx': t, 'signal_date': '', 'entry_idx': e, 'exit_idx': x,
            'dir': int(sig[t]), 'q': None, 'dNet': None, 'dP': None,
            'entry_px': round(float(O[e]), 4), 'exit_px': round(float(C[x]), 4),
            'pnl_pts': round(pts, 5), 'ret': round(pts / PRICE_REF, 8),
            'tag': tag,
        })
        free = x + 1
    return ret, trades


def metrics(ret, trades, a=0, b=None):
    n = len(ret)
    if b is None:
        b = n - 1
    r = ret[a:b + 1]
    if len(r) == 0:
        return {}
    cum = 1.0 + np.cumsum(r)
    total = float(cum[-1] - 1.0)
    ann = float(total * ANN_DAYS / len(r))
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    sharpe = float(np.mean(r) / sd * math.sqrt(ANN_DAYS)) if sd > 0 else 0.0
    mdd = float((1.0 - cum / np.maximum.accumulate(cum)).max())
    tr = [t for t in trades if a <= t['signal_idx'] <= b]
    pts = [t['pnl_pts'] for t in tr]
    win = [x for x in pts if x > 0]
    loss = [x for x in pts if x < 0]
    tstat = float(np.mean(pts) / (np.std(pts, ddof=1) / np.sqrt(len(pts)))) \
        if len(pts) > 2 and np.std(pts, ddof=1) > 0 else 0.0
    return {
        '累计点数': round(sum(pts), 4), '累计收益率': round(total, 6),
        '年化收益率': round(ann, 6), '最大回撤': round(mdd, 6),
        '夏普比率': round(sharpe, 4),
        '胜率': round(len(win) / len(pts), 4) if pts else None,
        '盈亏比': round(float(np.mean(win) / abs(np.mean(loss))), 4) if win and loss else None,
        '交易笔数': len(pts), '逐笔t值': round(tstat, 3),
        '每笔平均点数': round(float(np.mean(pts)), 5) if pts else None,
    }


def nav_from(ret, a=0, b=None):
    if b is None:
        b = len(ret) - 1
    return (1.0 + np.cumsum(ret[a:b + 1])).tolist()


def side_break(tr):
    """多空两条腿各自的贡献：判断收益是不是只来自做多 beta。"""
    res = {}
    for lab, cond in [('做多', lambda t: t['dir'] == 1),
                      ('做空', lambda t: t['dir'] == -1)]:
        v = [t['pnl_pts'] for t in tr if cond(t)]
        res[lab] = {'笔数': len(v), '累计点数': round(float(sum(v)), 4),
                    '每笔均值点数': round(float(np.mean(v)), 5) if v else None,
                    '胜率': round(float(np.mean(np.array(v) > 0)), 4) if v else None}
    return res


def hands_of(series, name, src, th):
    """分位阈值换算成最近 250 日实际手数，方便主观执行。"""
    arr = series[name][src][-250:]
    return {'上阈值手数': round(float(np.nanpercentile(arr, th * 100)), 1),
            '下阈值手数': round(float(np.nanpercentile(arr, (1 - th) * 100)), 1),
            '窗口': '最近 250 个交易日'}


# ---------------------------------------------------------------- 主流程
def main():
    dates, idx, O, C, dP, series = load()
    n = len(dates)
    split_i = int(np.searchsorted(np.array(dates), SPLIT))
    Q = {name: {src: rolling_pct(s[src]) for src in SOURCES}
         for name, s in series.items()}
    q_t = Q[TARGET]
    src0 = 'dNet'

    out = {
        'meta': {
            'title': '平安期货「逆势加多」策略 · 主观感受检验',
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'start': dates[0], 'end': dates[-1], 'n_days': n,
            'target': TARGET, 'price_ref': PRICE_REF, 'init_capital': INIT_CAPITAL,
            'win': WIN, 'thresholds': THRESHOLDS, 'holds': HOLDS,
            'sources': SOURCES, 'modes': MODES, 'costs': COSTS,
            'split': SPLIT, 'explore': [dates[0], dates[split_i - 1]],
            'verify': [dates[split_i], dates[-1]],
            'n_perm': N_PERM, 'seed': SEED,
            'rule': 'T日盘后信号 → T+1开盘建仓 → 持有H个交易日 → T+1+H收盘平仓；期间不叠加；固定5手TS/名义1000万/单倍杠杆',
            'nav': '净值 = 1 + Σ(方向×Δ价 − 成本)/100，固定名义单利累加、不复利',
        },
        'dates': dates, 'price': {'open': [None if np.isnan(x) else round(float(x), 4) for x in O],
                                  'close': [None if np.isnan(x) else round(float(x), 4) for x in C]},
    }

    # ---------- ① 描述统计：用户的感受在数据里是什么样 ----------
    dn = series[TARGET][src0]
    fut = {}
    for H in HOLDS:
        f = np.full(n, np.nan)
        for t in range(n - 1 - H):
            f[t] = C[t + 1 + H] - O[t + 1]
        fut[H] = f
    def sp(a, b):
        m = (~np.isnan(a)) & (~np.isnan(b))
        if m.sum() < 10:
            return None
        return round(float(pd.Series(a[m]).corr(pd.Series(b[m]), method='spearman')), 4)
    quad = {}
    H3 = fut[3]
    for lab, m in [
        ('价跌+净多增（用户①）', (dP < 0) & (dn > 0)),
        ('价涨+净多减（用户②）', (dP > 0) & (dn < 0)),
        ('价跌+净多减', (dP < 0) & (dn < 0)),
        ('价涨+净多增', (dP > 0) & (dn > 0)),
    ]:
        v = H3[m]
        v = v[~np.isnan(v)]
        quad[lab] = {'n': int(len(v)), '未来3日均值点数': round(float(v.mean()), 5),
                     '胜率': round(float((v > 0).mean()), 4),
                     't': round(float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))), 3) if len(v) > 2 else None}
    yearly_ic = {}
    for y in range(2021, 2027):
        m = np.asarray(idx.year == y)
        yearly_ic[str(y)] = {'IC_dNet_未来3日': sp(dn[m], H3[m]),
                             'IC_dNet_当日涨跌': sp(dn[m], dP[m])}
    out['descriptive'] = {
        'corr_dNet_dP': sp(dn, dP),
        'IC_dNet_future': {str(H): sp(dn, fut[H]) for H in HOLDS},
        'quad': quad, 'yearly_ic': yearly_ic,
        'dNet_pct_recent': {str(int(p * 100)): round(float(np.nanpercentile(dn[-250:], p * 100)), 1)
                            for p in [0.1, 0.2, 0.5, 0.8, 0.9]},
    }

    # ---------- ② 参数面（全网格，探索段选参数，验证段只报告） ----------
    cells = []
    cache = {}
    for src in SOURCES:
        for mode in MODES:
            for th in THRESHOLDS:
                for H in HOLDS:
                    sig = make_signal(q_t[src], dP, mode, th)
                    ret, tr = run(sig, O, C, H, 0.0, 0, n - 1)
                    key = (src, mode, th, H)
                    cache[key] = (sig, ret, tr)
                    cells.append({
                        'source': src, 'mode': mode, 'th': th, 'H': H,
                        'explore': metrics(ret, tr, 0, split_i - 1),
                        'verify': metrics(ret, tr, split_i, n - 1),
                        'all': metrics(ret, tr, 0, n - 1),
                    })
    out['grid'] = {'cells': cells}

    # 用户口述的字面规则（事前锁定，不参与任何调参；这才是真正要检验的对象）
    preset = {'source': 'dNet', 'mode': 'contra', 'th': 0.80, 'H': 3,
              'desc': '价跌且净多增力度≥80分位 → 次日开盘做多；价涨且净多减≤20分位 → 次日开盘做空；持有3个交易日'}
    sig_pre = make_signal(q_t['dNet'], dP, 'contra', 0.80)
    ret_pre, tr_pre = run(sig_pre, O, C, 3, 0.0, 0, n - 1)
    for t in tr_pre:
        t['signal_date'] = dates[t['signal_idx']]
        t['entry_date'] = dates[t['entry_idx']]
        t['exit_date'] = dates[t['exit_idx']]
        t['q'] = None if np.isnan(q_t['dNet'][t['signal_idx']]) else round(float(q_t['dNet'][t['signal_idx']]), 4)
        t['dNet'] = None if np.isnan(dn[t['signal_idx']]) else round(float(dn[t['signal_idx']]), 1)
        t['dP'] = None if np.isnan(dP[t['signal_idx']]) else round(float(dP[t['signal_idx']]), 4)
        t['方向'] = '多' if t['dir'] == 1 else '空'
    out['preset'] = {
        'params': preset,
        'explore': metrics(ret_pre, tr_pre, 0, split_i - 1),
        'verify': metrics(ret_pre, tr_pre, split_i, n - 1),
        'all': metrics(ret_pre, tr_pre, 0, n - 1),
        'nav': nav_from(ret_pre), 'daily_ret': [round(float(x), 8) for x in ret_pre],
        'trades': tr_pre, 'sides': side_break(tr_pre),
        'th_hands': hands_of(series, TARGET, 'dNet', 0.80),
    }
    # 选参：探索段累计点数最高、且探索段笔数 >= 20（避免只做 3 笔的假高收益）
    pool = [c for c in cells if c['explore'].get('交易笔数', 0) >= 20]
    best = max(pool, key=lambda c: c['explore']['累计点数'])
    sel = {'source': best['source'], 'mode': best['mode'],
           'th': best['th'], 'H': best['H']}
    key = (sel['source'], sel['mode'], sel['th'], sel['H'])
    sig0, ret0, tr0 = cache[key]
    # 补上逐笔的辅助字段
    for t in tr0:
        t['signal_date'] = dates[t['signal_idx']]
        t['entry_date'] = dates[t['entry_idx']]
        t['exit_date'] = dates[t['exit_idx']]
        t['q'] = None if np.isnan(q_t[sel['source']][t['signal_idx']]) else round(float(q_t[sel['source']][t['signal_idx']]), 4)
        t['dNet'] = None if np.isnan(dn[t['signal_idx']]) else round(float(dn[t['signal_idx']]), 1)
        t['dP'] = None if np.isnan(dP[t['signal_idx']]) else round(float(dP[t['signal_idx']]), 4)
        t['方向'] = '多' if t['dir'] == 1 else '空'
    out['selected'] = {
        'params': sel,
        'explore': metrics(ret0, tr0, 0, split_i - 1),
        'verify': metrics(ret0, tr0, split_i, n - 1),
        'all': metrics(ret0, tr0, 0, n - 1),
        'nav': nav_from(ret0), 'daily_ret': [round(float(x), 8) for x in ret0],
        'trades': tr0,
        'params_label': '探索段网格调参所得（仅供刻度，不等于可下注策略）',
        'sides': side_break(tr0),
        'th_hands': hands_of(series, TARGET, sel['source'], sel['th']),
    }
    # 参数面稳健性：验证段所有格子是否整体为正（成片 vs 孤点）
    vp = [c['verify']['累计点数'] for c in cells]
    ep = [c['explore']['累计点数'] for c in cells]
    top10 = sorted(cells, key=lambda c: c['explore']['累计点数'], reverse=True)[:10]
    out['grid']['robust'] = {
        '验证段中位数点数': round(float(np.median(vp)), 4),
        '验证段为正比例': round(float(np.mean(np.array(vp) > 0)), 4),
        '探索段前10在验证段为正数': sum(1 for c in top10 if c['verify']['累计点数'] > 0),
        '探索段前10验证段平均点数': round(float(np.mean([c['verify']['累计点数'] for c in top10])), 4),
        '两段相关': round(float(np.corrcoef(ep, vp)[0, 1]), 4),
    }

    # ---------- ③ 归因对照：平安到底有没有增量 ----------
    # 归因必须在同一套「口述参数」下比较，否则 A/B/C 的差异会被参数差异污染
    H = preset['H']; th = preset['th']; src = preset['source']
    qs = q_t[src]
    long_only = (np.nan_to_num(qs, nan=-1.0) >= th) & (dP < 0)
    short_only = (np.nan_to_num(qs, nan=2.0) <= 1 - th) & (~np.isnan(qs)) & (dP > 0)
    sig_only = np.where((np.nan_to_num(qs, nan=-1.0) >= th) & (~np.isnan(qs)), 1,
                        np.where((np.nan_to_num(qs, nan=2.0) <= 1 - th) & (~np.isnan(qs)), -1, 0))
    sig_price = np.where(dP < 0, 1, np.where(dP > 0, -1, 0))   # 纯价格反转
    variants = [
        ('A 完整规则（价跌+大幅加多→多；价涨+大幅减多→空）', make_signal(qs, dP, 'contra', th)),
        ('B 只看平安力度（去掉价格条件）', sig_only),
        ('C 只看价格反转（去掉平安）', sig_price),
        ('D 完整规则反向做', -make_signal(qs, dP, 'contra', th)),
        ('E 只做多：价跌+大幅加多', np.where(long_only, 1, 0)),
        ('F 只做空：价涨+大幅减多', np.where(short_only, -1, 0)),
        ('G 价跌+平安未大幅加多（安慰剂）', np.where((dP < 0) & (~long_only) & (np.nan_to_num(qs, nan=-1.0) < th), 1, 0)),
    ]
    attr = []
    for lab, sg in variants:
        r, tr = run(sg, O, C, H, 0.0, 0, n - 1)
        attr.append({'规则': lab,
                     'explore': metrics(r, tr, 0, split_i - 1),
                     'verify': metrics(r, tr, split_i, n - 1),
                     'all': metrics(r, tr, 0, n - 1),
                     'nav': nav_from(r)})
    out['attribution'] = attr

    # 条件内增量：价跌日子里，大幅加多 vs 其余，未来 H 日收益差
    fH = fut[H]
    m_base = (dP < 0) & (~np.isnan(fH))
    m_hi = m_base & (np.nan_to_num(qs, nan=-1.0) >= th)
    m_lo = m_base & (np.nan_to_num(qs, nan=-1.0) < th)
    a1, a2 = fH[m_hi], fH[m_lo]
    a1, a2 = a1[~np.isnan(a1)], a2[~np.isnan(a2)]
    se = math.sqrt(a1.var(ddof=1) / len(a1) + a2.var(ddof=1) / len(a2))
    out['incremental'] = {
        'H': H, 'th': th,
        '价跌且大幅加多': {'n': int(len(a1)), '均值点数': round(float(a1.mean()), 5)},
        '价跌但力度不足': {'n': int(len(a2)), '均值点数': round(float(a2.mean()), 5)},
        '差值点数': round(float(a1.mean() - a2.mean()), 5),
        'Welch_t': round(float((a1.mean() - a2.mean()) / se), 3) if se > 0 else None,
        '说明': '在「价跌」这个条件内部比较平安力度的增量；若 t 不显著，平安只是价跌反弹的伴随现象',
    }

    # ---------- ④ 横截面安慰剂：同一规则套到其他 8 个席位 ----------
    placebo = []
    for name in series:
        if name == TARGET:
            continue
        qq = Q[name][src]
        sg = make_signal(qq, dP, preset['mode'], th)
        r, tr = run(sg, O, C, H, 0.0, 0, n - 1)
        placebo.append({'席位': name,
                        'explore': metrics(r, tr, 0, split_i - 1),
                        'verify': metrics(r, tr, split_i, n - 1),
                        'all': metrics(r, tr, 0, n - 1)})
    out['placebo'] = placebo

    # ---------- ⑤ 零假设：两种移位 ----------
    rng = np.random.default_rng(SEED)
    real_all = out['preset']['all']['累计点数']
    real_shp = out['preset']['all']['夏普比率']
    real_verify = out['preset']['verify']['累计点数']

    def perm_stats(kind):
        acc, shp, acc_v = [], [], []
        base_sig = sig_pre
        for _ in range(N_PERM):
            k = int(rng.integers(1, n))
            if kind == 'shift_signal':
                sg = np.roll(base_sig, k)
            else:
                qq = np.roll(qs, k)          # 只打乱平安力度，保留价格路径
                sg = make_signal(qq, dP, preset['mode'], th)
            r, tr = run(sg, O, C, H, 0.0, 0, n - 1)
            m = metrics(r, tr, 0, n - 1)
            mv = metrics(r, tr, split_i, n - 1)
            acc.append(m['累计点数']); shp.append(m['夏普比率'])
            acc_v.append(mv['累计点数'])
        A, V = np.array(acc), np.array(acc_v)
        return {'hist_acc': [round(float(x), 4) for x in acc],
                'hist_sharpe': [round(float(x), 4) for x in shp],
                'rand_mean_acc': round(float(np.mean(A)), 4),
                'rand_p95_acc': round(float(np.percentile(A, 95)), 4),
                'p_acc': round(float((A >= real_all).mean()), 4),
                'p_sharpe': round(float((np.array(shp) >= real_shp).mean()), 4),
                'pct_acc': round(float((A < real_all).mean()), 4),
                'p_acc_verify': round(float((V >= real_verify).mean()), 4),
                'rand_mean_verify': round(float(np.mean(V)), 4)}
    out['null_test'] = {
        'shift_signal': perm_stats('shift_signal'),
        'shift_dnet': perm_stats('shift_dnet'),
    }

    # ---------- ⑥ 成本敏感性 ----------
    costs = []
    for c in COSTS:
        r, tr = run(sig_pre, O, C, preset['H'], c, 0, n - 1)
        row = {'成本点': c, **metrics(r, tr, 0, n - 1)}
        r2, tr2 = run(sig0, O, C, sel['H'], c, 0, n - 1)
        row['网格最优累计点数'] = metrics(r2, tr2, 0, n - 1)['累计点数']
        costs.append(row)
    out['cost_sensitivity'] = costs

    # ---------- ⑦ 逐年 walk-forward（每年只用之前的数据选参数） ----------
    wf_ret = np.zeros(n)
    wf_rows = []
    for y in range(2023, 2027):
        a = int(np.searchsorted(np.array(dates), f'{y}-01-01'))
        b = int(np.searchsorted(np.array(dates), f'{y + 1}-01-01')) - 1
        if b >= n:
            b = n - 1
        cand = []
        for src2 in SOURCES:
            for mode2 in MODES:
                for th2 in THRESHOLDS:
                    for H2 in HOLDS:
                        sg = make_signal(q_t[src2], dP, mode2, th2)
                        r, tr = run(sg, O, C, H2, 0.0, 0, a - 1)
                        m = metrics(r, tr, 0, a - 1)
                        if m.get('交易笔数', 0) >= 12:
                            cand.append((m['累计点数'], src2, mode2, th2, H2))
        if not cand:
            continue
        _, s2, m2, t2, H2 = max(cand, key=lambda x: x[0])
        sg = make_signal(q_t[s2], dP, m2, t2)
        r, tr = run(sg, O, C, H2, 0.0, a, b)
        wf_ret[a:b + 1] = r[a:b + 1]
        wf_rows.append({'年份': y, '参数': f'{s2}/{m2}/th={t2}/H={H2}',
                        **metrics(wf_ret, [t for t in tr], a, b)})
    wf_a = int(np.searchsorted(np.array(dates), '2023-01-01'))
    wf_dummy = [{'signal_idx': i, 'pnl_pts': float(wf_ret[i]) * PRICE_REF}
                for i in range(wf_a, n) if wf_ret[i] != 0]
    out['walk_forward'] = {'rows': wf_rows, 'nav': nav_from(wf_ret, wf_a, n - 1),
                           'start_idx': int(wf_a),
                           'all': metrics(wf_ret, wf_dummy, wf_a, n - 1)}

    # ---------- ⑧ 分年绩效 ----------
    yr_rows = []
    for y in range(2021, 2027):
        ia = int(np.searchsorted(np.array(dates), f'{y}-01-01'))
        ib = int(np.searchsorted(np.array(dates), f'{y + 1}-01-01')) - 1
        if ib >= n:
            ib = n - 1
        yr_rows.append({'年份': y, **metrics(ret0, tr0, ia, ib)})
    out['yearly'] = yr_rows

    # ---------- 写出 ----------
    json.dump(out, open(OUT_JSON, 'w', encoding='utf-8'),
              ensure_ascii=False, separators=(',', ':'))
    with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as w:
        pd.DataFrame([{
            '信号源': c['source'], '模式': c['mode'], '力度分位': c['th'], '持有天数': c['H'],
            '探索段累计点数': c['explore'].get('累计点数'), '探索段笔数': c['explore'].get('交易笔数'),
            '验证段累计点数': c['verify'].get('累计点数'), '验证段笔数': c['verify'].get('交易笔数'),
            '全样本累计点数': c['all'].get('累计点数'), '全样本夏普': c['all'].get('夏普比率'),
        } for c in cells]).to_excel(w, sheet_name='参数面', index=False)
        pd.DataFrame([{'规则': a['规则'],
                       '探索段累计点数': a['explore'].get('累计点数'),
                       '验证段累计点数': a['verify'].get('累计点数'),
                       '全样本累计点数': a['all'].get('累计点数'),
                       '全样本笔数': a['all'].get('交易笔数'),
                       '全样本夏普': a['all'].get('夏普比率'),
                       '全样本胜率': a['all'].get('胜率')} for a in attr]).to_excel(w, sheet_name='归因对照', index=False)
        pd.DataFrame([{'席位': p['席位'],
                       '探索段累计点数': p['explore'].get('累计点数'),
                       '验证段累计点数': p['verify'].get('累计点数'),
                       '全样本累计点数': p['all'].get('累计点数'),
                       '全样本笔数': p['all'].get('交易笔数')} for p in placebo]).to_excel(w, sheet_name='席位安慰剂', index=False)
        pd.DataFrame([{'成本点': c['成本点'], **{k: v for k, v in c.items() if k != '成本点'}}
                      for c in costs]).to_excel(w, sheet_name='成本敏感性', index=False)
        pd.DataFrame(yr_rows).to_excel(w, sheet_name='分年', index=False)
        pd.DataFrame(out['selected']['trades']).to_excel(w, sheet_name='选定规则逐笔', index=False)
        pd.DataFrame([{'年份': y, **v} for y, v in yearly_ic.items()]).to_excel(w, sheet_name='分年IC', index=False)
        pd.DataFrame([{'项目': k, '值': str(v)} for k, v in [
            ('选定参数', str(sel)), ('探索段', f"{dates[0]} ~ {dates[split_i-1]}"),
            ('验证段', f"{dates[split_i]} ~ {dates[-1]}"),
            ('力度定义', f'dNet 过去 {WIN} 个有效日百分位（含当日，无前视）'),
            ('执行', 'T+1 开盘 → T+1+H 收盘；期间不叠加；固定 5 手 TS'),
            ('净值口径', '净值 = 1 + Σ(方向×Δ价 − 成本)/100，单利累加不复利；1 点 = 初始资金 1%'),
            ('主判据', '零假设 shift_dnet：只打乱平安力度、保留价格路径'),
            ('数据窥探警告', '平安是看过全样本净值后挑出的序列，样本外切分只能防参数过拟合'),
            ('生成时间', out['meta']['generated'])]]).to_excel(w, sheet_name='说明', index=False)

    # 网页
    if os.path.exists(TPL):
        tpl = open(TPL, encoding='utf-8').read()
        js = json.dumps(out, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
        assert '/*__DATA__*/' in tpl, '模板缺少数据占位符'
        open(OUT_HTML, 'w', encoding='utf-8').write(tpl.replace('/*__DATA__*/', js))

    # ---------- 控制台摘要 ----------
    print('=== ① 描述统计 ===')
    print('  dNet 与当日涨跌 Spearman:', out['descriptive']['corr_dNet_dP'])
    print('  IC(dNet, 未来H日):', out['descriptive']['IC_dNet_future'])
    for k, v in quad.items():
        print(f"  {k:<22} n={v['n']:>4} 未来3日 {v['未来3日均值点数']:+.4f} 点 t={v['t']:+.2f}")
    print('\n=== ②a 口述规则（事前锁定，主口径） ===')
    print(' ', preset['desc'])
    print('  力度阈值(手):', out['preset']['th_hands'])
    for lab in ['explore', 'verify', 'all']:
        print(f"  {lab:<8}", out['preset'][lab])
    print('  多空分解:', out['preset']['sides'])
    print('\n=== ②b 网格最优（只用探索段选参，仅供刻度） ===')
    print(' ', sel, ' 力度对应手数(近250日):', out['selected']['th_hands'])
    print('  多空分解:', out['selected']['sides'])
    for lab in ['explore', 'verify', 'all']:
        print(f"  {lab:<8}", out['selected'][lab])
    print('  参数面稳健性:', out['grid']['robust'])
    print('\n=== ③ 归因对照（全样本累计点数） ===')
    for a in attr:
        print(f"  {a['规则']:<46} 累计 {a['all']['累计点数']:>8.3f} 点 笔数 {a['all']['交易笔数']:>4} "
              f"验证段 {a['verify']['累计点数']:>8.3f}")
    print('  条件内增量:', out['incremental'])
    print('\n=== ④ 席位安慰剂（同规则） ===')
    for p in placebo:
        print(f"  {p['席位']:<12} 全样本 {p['all']['累计点数']:>8.3f} 点 验证段 {p['verify']['累计点数']:>8.3f}")
    print('\n=== ⑤ 零假设 ===')
    for k, v in out['null_test'].items():
        print(f"  {k:<14} p(累计)={v['p_acc']:.4f} p(夏普)={v['p_sharpe']:.4f} "
              f"p(验证段累计)={v['p_acc_verify']:.4f} "
              f"随机均值={v['rand_mean_acc']:.3f} 95分位={v['rand_p95_acc']:.3f}")
    print('\n=== ⑥ 成本 ===')
    for c in costs:
        print(f"  {c['成本点']:.3f} 点 → 口述规则 {c['累计点数']:.3f} 点 / {c['累计收益率']*100:.3f}% "
              f"| 网格最优 {c['网格最优累计点数']:.3f} 点")
    print('\n=== 分年（口述规则） ===')
    for r in yr_rows:
        print(f"  {r['年份']} 累计 {r['累计点数']:>7.3f} 点 笔数 {r['交易笔数']:>3} 夏普 {r['夏普比率']:>6.2f}")
    print('\n=== ⑦ walk-forward ===')
    for r in wf_rows:
        print(f"  {r['年份']} {r['参数']:<28} 累计 {r['累计点数']:>7.3f} 点 笔数 {r['交易笔数']:>3}")
    for p in [OUT_JSON, OUT_XLSX] + ([OUT_HTML] if os.path.exists(TPL) else []):
        print('输出:', p, f'({os.path.getsize(p)/1024:.1f} KB)')


if __name__ == '__main__':
    main()
