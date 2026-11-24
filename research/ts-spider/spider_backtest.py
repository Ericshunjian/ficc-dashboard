# -*- coding: utf-8 -*-
"""
TS「蜘蛛网」策略 · 静态版回测（复现国泰海通《债市量化系列之五》）

报告口径（PDF 第 85 行原文）：
  T 日信号 dB>0 且 dS<0 -> 做多；dB<0 且 dS>0 -> 做空；其余无信号
  T 日仓位 0     -> T+1 开盘价全仓建仓（单倍杠杆），T+2 收盘价平仓
  T 日仓位与信号反向 -> T+1 开盘价平仓（不反手）
  初始资金 1000 万，单倍杠杆，报告回测区间 2020-01-01 ~ 2025-04-10

净值口径（2026-11-24 改）：固定手数、固定名义敞口，盈亏按点数线性累加、不复利。
  TS 合约面值 200 万、报价按 100 元面值 => 1 点 = 2 万元/手；
  单倍杠杆 1000 万 = 5 手 => 1 点 = 10 万元 = 初始资金的 1%，
  故 r = 方向 × Δ价(点) / 100，净值 = 1 + Σr（= 1 + 累计点数/100）。

数据：
  TS_持仓整理.json  前一脚本生成的持仓/信号（避免重复清洗）
  TS.CFE.xlsx       Wind 行情（开盘价/收盘价，2018-08-17 ~ 2026-11-23）

输出：
  TS_策略回测.json             网页数据源
  TS_策略回测_交易明细.xlsx    各序列逐笔交易 + 绩效汇总
  TS_蜘蛛网_策略回测.html      单文件网页（模板 spider_backtest_template.html）
"""
import sys, os, json, math, warnings
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
HOLD_JSON = os.path.join(BASE, 'TS_持仓整理.json')
PRICE_XLSX = os.path.join(BASE, 'TS.CFE.xlsx')
OUT_JSON = os.path.join(BASE, 'TS_策略回测.json')
OUT_XLSX = os.path.join(BASE, 'TS_策略回测_交易明细.xlsx')
TPL = os.path.join(BASE, 'spider_backtest_template.html')
OUT_HTML = os.path.join(BASE, 'TS_蜘蛛网_策略回测.html')

INIT_CAPITAL = 10_000_000.0
PRICE_REF = 100.0       # 报价面值；1 点 = 初始资金的 1%（固定名义，不用浮动开仓价做分母）
ANN_DAYS = 243          # 实测每年 242~245 个交易日
COST_POINTS = [0.0, 0.002, 0.005]   # 每笔来回成本（价格点）；固定点数

# 报告公布的 TS 静态策略绩效（回测区间 2020-01-01~2025-04-10），仅用于对照
REPORT_BASELINE = {
    'period': '2020-01-01 ~ 2025-04-10',
    'strategy': {'累计收益率': 0.0952, '年化收益率': 0.0184, '最大回撤': 0.0102,
                 '夏普比率': 0.83, '胜率': 0.5197, '盈亏比': 1.37},
    'benchmark': {'累计收益率': 0.0195, '年化收益率': 0.0039, '最大回撤': 0.0289,
                  '夏普比率': -0.63},
}


def load_data():
    P = json.load(open(HOLD_JSON, encoding='utf-8'))
    hold_dates = P['dates']
    q = pd.read_excel(PRICE_XLSX, sheet_name=0, header=0)
    q = q[q['日期'].notna()].copy()
    q['date'] = pd.to_datetime(q['日期'])
    q = q.set_index('date').sort_index()
    q = q[~q.index.duplicated(keep='last')]

    # 回测日历 = 行情交易日 ∩ 持仓覆盖区间
    d0, d1 = pd.Timestamp(hold_dates[0]), pd.Timestamp(hold_dates[-1])
    q = q.loc[(q.index >= d0) & (q.index <= d1)]
    dates = [d.date().isoformat() for d in q.index]
    opens = q['开盘价(元)'].astype(float).tolist()
    closes = q['收盘价(元)'].astype(float).tolist()

    sig_by_series = {}
    for name, s in P['series'].items():
        sig_by_series[name] = {d: v for d, v in zip(hold_dates, s['signal']) if v is not None}

    diag = {
        '持仓交易日': len(hold_dates),
        '行情交易日(区间内)': len(dates),
        '持仓日不在行情日历': sorted(set(hold_dates) - set(dates))[:10],
        '行情日不在持仓日历': sorted(set(dates) - set(hold_dates))[:10],
        '行情起止': [dates[0], dates[-1]],
        '行情OHLC空值': int(q[['开盘价(元)', '收盘价(元)']].isna().sum().sum()),
    }
    return P, dates, opens, closes, sig_by_series, diag


def metrics(daily_ret, trades, nav=None):
    r = np.asarray(daily_ret, dtype=float)
    n = len(r)
    if n == 0:
        return {}
    cum = 1.0 + np.cumsum(r)          # 固定名义敞口：逐笔盈亏线性累加，不复利
    total = float(cum[-1] - 1.0)
    ann = float(total * ANN_DAYS / n)  # 单利年化，与净值口径一致
    sd = float(np.std(r, ddof=1)) if n > 1 else 0.0
    sharpe = float(np.mean(r) / sd * math.sqrt(ANN_DAYS)) if sd > 0 else 0.0
    mdd = float((1.0 - cum / np.maximum.accumulate(np.r_[1.0, cum])[1:]).max())
    tr = [t['ret'] for t in trades]
    wins = [x for x in tr if x > 0]
    loss = [x for x in tr if x < 0]
    return {
        '累计收益率': round(total, 6), '累计点数': round(total * PRICE_REF, 4),
        '年化收益率': round(float(ann), 6),
        '最大回撤': round(mdd, 6), '夏普比率': round(sharpe, 4),
        '胜率': round(len(wins) / len(tr), 4) if tr else None,
        '盈亏比': round(float(np.mean(wins) / abs(np.mean(loss))), 4)
                 if wins and loss else None,
        '交易笔数': len(tr), '日均收益bp': round(float(np.mean(r)) * 10000, 4),
    }


def nav_series(daily_ret):
    cum = 1.0 + np.cumsum(np.asarray(daily_ret, dtype=float))
    dd = 1.0 - cum / np.maximum.accumulate(np.r_[1.0, cum])[1:]
    return cum.tolist(), dd.tolist()


def run_state(dates, opens, closes, sig_map, lo, hi, cost_pt=0.0, *, live=False):
    """报告口径：仓位状态机。反向信号只平仓不反手；开仓后 T+2 收盘强平。"""
    n = hi - lo + 1
    ret = [0.0] * n
    trades = []
    pos, entry_price, entry_k, exit_k = 0, None, None, None
    pend_open, pend_close = 0, False

    def close_at(k, px, how):
        nonlocal pos, entry_price, entry_k, exit_k
        r = pos * (px - entry_price) / PRICE_REF - cost_pt / PRICE_REF
        j = k - lo
        ref = entry_price if k == entry_k else closes[k - 1]
        ret[j] += (pos * (px - ref) - cost_pt) / PRICE_REF
        trades.append({
            '信号日': dates[entry_k - 1] if entry_k is not None and entry_k > 0 else '',
            '开仓日': dates[entry_k] if entry_k is not None else '',
            '开仓价': round(entry_price, 4), '方向': '多' if pos == 1 else '空',
            '平仓日': dates[k], '平仓价': round(px, 4), '平仓方式': how,
            'ret': round(float(r), 8), '盈亏元': round(float(r) * INIT_CAPITAL, 2),
        })
        pos, entry_price, entry_k, exit_k = 0, None, None, None

    for k in range(lo, hi + 1):
        # 1) 开盘执行上一交易日挂出的单
        if pend_close and pos != 0:
            close_at(k, opens[k], '反向信号·次日开盘')
        pend_close = False
        if pend_open and pos == 0:
            pos, entry_price, entry_k, exit_k = pend_open, opens[k], k, k + 1
        pend_open = 0
        # 2) 收盘执行到期的强制平仓（T+2）
        if pos != 0:
            if exit_k == k:
                close_at(k, closes[k], 'T+2收盘')
            else:
                ref = entry_price if k == entry_k else closes[k - 1]
                ret[k - lo] += pos * (closes[k] - ref) / PRICE_REF
        # 3) 依据 T 日信号，安排下一交易日动作
        sig = sig_map.get(dates[k])
        if sig in (1, -1):
            if pos == 0 and (live or k + 2 <= hi):
                pend_open = sig
            elif sig == -pos and (live or k + 1 <= hi):
                pend_close = True
    if live:
        return ret, trades, {'position': pos, 'entry': entry_k, 'exit': exit_k,
                             'next_open': pend_open, 'next_close': pend_close}
    return ret, trades


def run_independent(dates, opens, closes, sig_map, lo, hi, cost_pt=0.0):
    """对照口径：每个信号独立成一笔 T+1开盘 -> T+2收盘，允许仓位叠加。"""
    n = hi - lo + 1
    ret = [0.0] * n
    trades = []
    for k in range(lo, hi - 1):
        sig = sig_map.get(dates[k])
        if sig not in (1, -1) or k + 2 > hi:
            continue
        e, x = opens[k + 1], closes[k + 2]
        r = sig * (x - e) / PRICE_REF - cost_pt / PRICE_REF
        ret[k + 1 - lo] += sig * (closes[k + 1] - e) / PRICE_REF
        ret[k + 2 - lo] += (sig * (x - closes[k + 1]) - cost_pt) / PRICE_REF
        trades.append({
            '信号日': dates[k], '开仓日': dates[k + 1], '开仓价': round(e, 4),
            '方向': '多' if sig == 1 else '空', '平仓日': dates[k + 2],
            '平仓价': round(x, 4), '平仓方式': '逐笔独立·T+2收盘',
            'ret': round(float(r), 8), '盈亏元': round(float(r) * INIT_CAPITAL, 2),
        })
    return ret, trades


def benchmark_ret(opens, closes, lo, hi):
    r = [0.0] * (hi - lo + 1)
    for k in range(lo + 1, hi + 1):
        r[k - lo] = (closes[k] - closes[k - 1]) / PRICE_REF
    return r


def permutation_test(dates, opens, closes, sig_map, lo, hi, n_perm=200, seed=20261123):
    """零假设：把信号序列循环移位（保持信号密度与聚集结构），重跑同一流水线。"""
    n = hi - lo + 1
    arr = np.array([sig_map.get(dates[lo + i], 0) or 0 for i in range(n)], dtype=int)
    real_r, real_tr = run_state(dates, opens, closes, sig_map, lo, hi, 0.0)
    real = metrics(real_r, real_tr)
    rng = np.random.default_rng(seed)
    acc, shp = [], []
    for _ in range(n_perm):
        k = int(rng.integers(1, n))
        sh = np.roll(arr, k)
        smap = {dates[lo + i]: int(sh[i]) for i in range(n)}
        r, tr = run_state(dates, opens, closes, smap, lo, hi, 0.0)
        m = metrics(r, tr)
        acc.append(m['累计收益率'])
        shp.append(m['夏普比率'])
    acc, shp = np.array(acc), np.array(shp)
    flip = {d: (-v if v else 0) for d, v in sig_map.items()}
    fr, ftr = run_state(dates, opens, closes, flip, lo, hi, 0.0)
    return {
        'n_perm': n_perm, 'seed': seed,
        'real': {k: real[k] for k in ['累计收益率', '夏普比率', '交易笔数']},
        'rand_mean_sharpe': round(float(shp.mean()), 4),
        'rand_median_sharpe': round(float(np.median(shp)), 4),
        'rand_p95_sharpe': round(float(np.percentile(shp, 95)), 4),
        'p_sharpe': round(float((shp >= real['夏普比率']).mean()), 4),
        'p_acc': round(float((acc >= real['累计收益率']).mean()), 4),
        'pct_sharpe': round(float((shp < real['夏普比率']).mean()), 4),
        'flip': metrics(fr, ftr),
        'hist_sharpe': [round(float(x), 4) for x in shp],
        'hist_acc': [round(float(x), 6) for x in acc],
    }


def regime_diag(price_xlsx):
    """报告区间 2020-01-01~2025-04-10；本地持仓 2021 起，量化缺 2020 的样本差异。"""
    q = pd.read_excel(price_xlsx, sheet_name=0, header=0)
    q = q[q['日期'].notna()].copy()
    q['date'] = pd.to_datetime(q['日期'])
    q = q.set_index('date').sort_index()['收盘价(元)'].astype(float)
    r = q.pct_change().dropna()
    seg = {}
    for lab, a, b in [('2020（报告独有，本地无持仓）', '2020-01-01', '2020-12-31'),
                      ('2021（本次新补）', '2021-01-04', '2021-12-31'),
                      ('2022-2026（原本地样本）', '2022-01-04', '2026-11-23'),
                      ('报告可比段 2021-2025.04', '2021-01-04', '2025-04-10')]:
        x = r.loc[a:b]
        seg[lab] = {'交易日': len(x), '日波动std': round(float(x.std() * 100), 4),
                    '区间涨跌': round(float((q.loc[a:b].iloc[-1] / q.loc[a:b].iloc[0] - 1) * 100), 3),
                    '日均绝对涨跌': round(float(x.abs().mean() * 100), 4)}
    return seg


def period_bounds(dates, p):
    if p == 'all':
        return 0, len(dates) - 1
    if p == 'report':
        a = next((i for i, d in enumerate(dates) if d >= '2021-01-04'), 0)
        b = max((i for i, d in enumerate(dates) if d <= '2025-04-10'), default=a)
        return a, b
    idx = [i for i, d in enumerate(dates) if d.startswith(p)]
    return (idx[0], idx[-1]) if idx else (0, -1)


def main():
    P, dates, opens, closes, sig_maps, diag = load_data()
    periods = ['all', 'report', '2021', '2022', '2023', '2024', '2025', '2026']
    bounds = {p: period_bounds(dates, p) for p in periods}
    bounds = {p: b for p, b in bounds.items() if b[1] > b[0]}

    lo, hi = bounds['all']
    bench = benchmark_ret(opens, closes, lo, hi)

    out = {
        'meta': {
            'title': 'TS 蜘蛛网策略 · 静态版回测',
            'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'start': dates[lo], 'end': dates[hi], 'n_days': len(dates),
            'init_capital': INIT_CAPITAL, 'ann_days': ANN_DAYS,
            'rule': '报告第85行：T信号 / T+1开盘 / T+2收盘 / 反向信号次日开盘平仓 / 单倍杠杆',
            'price_source': 'TS.CFE.xlsx（Wind，开盘价、收盘价）',
            'price_ref': PRICE_REF,
            'nav_method': '固定名义单利：r=方向×Δ价/100（1点=初始资金1%），净值=1+Σr 直接累加，不复利',
            'periods': {p: {'start': dates[a], 'end': dates[b], 'n': b - a + 1}
                        for p, (a, b) in bounds.items()},
            'report_period': '2020-01-01 ~ 2025-04-10',
            'local_report_period': '2021-01-04 ~ 2025-04-10（本地缺 2020 全年）',
            'holdings_since': '2021-01-04',
        },
        'dates': dates, 'open': opens, 'close': closes,
        'report_baseline': REPORT_BASELINE,
        'diagnostics': diag,
        'benchmark': {'ret': bench,
                      'metrics': metrics(bench, [{'ret': x} for x in bench])},
        'series': {}, 'periods': {}, 'null_test': {},
        'regime': regime_diag(PRICE_XLSX),
    }

    for name, smap in sig_maps.items():
        rec = {}
        for rule, fn in [('state', run_state), ('independent', run_independent)]:
            r, tr = fn(dates, opens, closes, smap, lo, hi, 0.0)
            nav, _ = nav_series(r)
            m = metrics(r, tr)
            sens = []
            for c in COST_POINTS:
                rr, tt = fn(dates, opens, closes, smap, lo, hi, c)
                sens.append({'成本点': c, **metrics(rr, tt)})
            rec[rule] = {
                'ret': r, 'metrics': m, 'trades': tr,
                'cost_sensitivity': sens,
                '资金终值元': round(INIT_CAPITAL * nav[-1], 2),
            }
        out['series'][name] = rec

    for name in ['前20大合计', '前10大合计', '平安期货', '国泰君安']:
        out['null_test'][name] = permutation_test(
            dates, opens, closes, sig_maps[name], lo, hi)

    for p, (a, b) in bounds.items():
        b_r = benchmark_ret(opens, closes, a, b)
        b_m = metrics(b_r, [{'ret': x} for x in b_r])
        row = {'benchmark': b_m, 'series': {}}
        for name, smap in sig_maps.items():
            r, tr = run_state(dates, opens, closes, smap, a, b, 0.0)
            row['series'][name] = metrics(r, tr)
        out['periods'][p] = row

    json.dump(out, open(OUT_JSON, 'w', encoding='utf-8'),
              ensure_ascii=False, separators=(',', ':'))

    # Excel：绩效汇总 + 各序列逐笔交易
    with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as w:
        rows = []
        for p, row in out['periods'].items():
            for name, m in row['series'].items():
                rows.append({'区间': p, '序列': name, **m})
        pd.DataFrame(rows).to_excel(w, sheet_name='绩效汇总', index=False)
        pd.DataFrame([{
            '序列': n, '真实累计': t['real']['累计收益率'], '真实夏普': t['real']['夏普比率'],
            '随机夏普均值': t['rand_mean_sharpe'], '随机夏普中位': t['rand_median_sharpe'],
            '随机夏普95分位': t['rand_p95_sharpe'], '真实夏普分位': t['pct_sharpe'],
            'p(随机>=真实夏普)': t['p_sharpe'], 'p(随机>=真实累计)': t['p_acc'],
            '符号翻转夏普': t['flip']['夏普比率'], '符号翻转累计': t['flip']['累计收益率'],
            '循环移位次数': t['n_perm'],
        } for n, t in out['null_test'].items()]).to_excel(w, sheet_name='零假设对照', index=False)
        for name in ['前20大合计', '前10大合计', '前5大合计', '中信期货', '平安期货', '国泰君安']:
            pd.DataFrame(out['series'][name]['state']['trades']).to_excel(
                w, sheet_name=('逐笔_' + name)[:31], index=False)
        pd.DataFrame([{
            '项目': k, '值': str(v)} for k, v in
            [('回测区间', out['meta']['start'] + ' ~ ' + out['meta']['end']),
             ('交易日数', out['meta']['n_days']), ('初始资金', INIT_CAPITAL),
             ('年化因子', ANN_DAYS), ('报告原文区间', REPORT_BASELINE['period']),
             ('主口径', '仓位状态机'), ('对照口径', '逐笔独立'),
             ('净值口径', '固定名义单利：净值 = 1 + Σr，r = 方向×Δ价/100；1点=初始资金1%，不复利'),
             ('成本', '主结果零成本，另给 0.002 / 0.005 点敏感性'),
             ('价格口径提示', '持仓表结算价与 TS.CFE 收盘价长期差约 2.9 点（不同合约），日变化相关 0.84、差值std 0.027 点；水平不可比、变动可比'),
             ('基准', '买入持有 TS.CFE，按收盘价日收益，单倍杠杆'),
             ('生成时间', out['meta']['generated'])]]).to_excel(
            w, sheet_name='说明', index=False)

    # 网页
    tpl = open(TPL, encoding='utf-8').read()
    js = json.dumps(out, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    assert '/*__DATA__*/' in tpl, '模板缺少数据占位符'
    html = tpl.replace('/*__DATA__*/', js)
    diag_path = os.path.join(BASE, 'TS_规则杠杆归因.json')
    if os.path.exists(diag_path):
        dj = open(diag_path, encoding='utf-8').read().replace('</', '<\\/')
        if '/*__DIAG__*/' in html:
            html = html.replace('/*__DIAG__*/', dj)
    open(OUT_HTML, 'w', encoding='utf-8').write(html)

    print('=== 回测完成 ===')
    print(f"区间 {out['meta']['start']} ~ {out['meta']['end']}，{len(dates)} 交易日")
    print('日历诊断:', {k: v for k, v in diag.items() if k in
                    ('持仓交易日', '行情交易日(区间内)', '持仓日不在行情日历', '行情日不在持仓日历')})
    print('\n=== 全样本 · 报告口径（状态机）===')
    bm = out['benchmark']['metrics']
    print(f"  {'基准(买入持有)':<18} 累计 {bm['累计收益率']*100:>7.2f}% 年化 {bm['年化收益率']*100:>6.2f}% "
          f"夏普 {bm['夏普比率']:>6.2f} 回撤 {bm['最大回撤']*100:>5.2f}%")
    for name in out['series']:
        m = out['series'][name]['state']['metrics']
        print(f"  {name:<18} 累计 {m['累计收益率']*100:>7.2f}% 年化 {m['年化收益率']*100:>6.2f}% "
              f"夏普 {m['夏普比率']:>6.2f} 回撤 {m['最大回撤']*100:>5.2f}% "
              f"胜率 {(m['胜率'] or 0)*100:>5.1f}% 笔数 {m['交易笔数']:>4}")
    print('\n=== 报告可比区间（2021-01-04 ~ 2025-04-10，本地缺 2020）前20大合计 ===')
    m = out['periods']['report']['series']['前20大合计']
    b = out['periods']['report']['benchmark']
    print('  策略:', m)
    print('  基准:', b)
    print('  报告公布:', REPORT_BASELINE['strategy'], '/ 基准', REPORT_BASELINE['benchmark'])
    print('\n=== 零假设对照（200 次循环移位，报告口径）===')
    for n, t in out['null_test'].items():
        print(f"  {n:<12} 真实夏普 {t['real']['夏普比率']:>6.2f} | 随机中位 {t['rand_median_sharpe']:>6.2f} "
              f"95分位 {t['rand_p95_sharpe']:>6.2f} | 分位 {t['pct_sharpe']*100:>5.1f}% "
              f"p(夏普) {t['p_sharpe']:.3f} p(累计) {t['p_acc']:.3f} | 翻转 {t['flip']['夏普比率']:>6.2f}")
    print('\n=== 行情分段波动 ===')
    for k, v in out['regime'].items():
        print(' ', k, v)
    for p in [OUT_JSON, OUT_XLSX, OUT_HTML]:
        print('输出:', p, f'({os.path.getsize(p)/1024:.1f} KB)')


if __name__ == '__main__':
    main()
