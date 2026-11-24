# -*- coding: utf-8 -*-
"""
TS 机构持仓整理 · 蜘蛛网策略复现（第一步：数据整理）

输入：本目录下 2021年 ~ 2026年（截至11月23日）/ 会员：XXX 品种：TS 2年期国债期货.xlsx
输出：
  - TS_持仓整理_2021-2026.xlsx   多 sheet 数据表
  - TS_持仓整理.json             网页数据源
  - TS_蜘蛛网_持仓整理.html      单文件网页（内嵌数据，离线双击可开）

可复现：改完源 Excel 后直接重跑本脚本即可。

口径对齐（国泰海通《债市量化系列之五》）：
  dB = 多单量日变动；dS = 空单量日变动（本报告口径：空头增加为正）
  T 日 dB>0 且 dS<0 -> 做多(+1)；dB<0 且 dS>0 -> 做空(-1)；其余 0
  Wind 导出的「空单量」带负号，故先取绝对值再算 dS。
"""
import sys, os, re, glob, json, warnings
from datetime import datetime
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
XLSX_OUT = os.path.join(BASE, 'TS_持仓整理_2021-2026.xlsx')
JSON_OUT = os.path.join(BASE, 'TS_持仓整理.json')
BT_JSON = os.path.join(BASE, 'TS_策略回测.json')
HTML_TPL = os.path.join(BASE, 'spider_web_template.html')
HTML_OUT = os.path.join(BASE, 'TS_蜘蛛网_持仓整理.html')

# 显示名 -> Wind 会员名（None 表示由其它序列合成）
GROUPS = {
    '合计类': [
        ('前5大合计', '前五名合计'),
        ('前10大合计', '前十名合计'),
        ('前20大合计', '前二十名合计'),
    ],
    '席位类': [
        ('中信期货', '中信期货'),
        ('平安期货', '平安期货'),
        ('华泰期货', '华泰期货'),
        ('国泰君安', '国泰君安'),
        ('海通期货', '海通期货'),
        ('国泰海通（合成）', None),
    ],
}
ALL_NAMES = [n for g in GROUPS.values() for n, _ in g]
COLS = ['long', 'short', 'net', 'dB', 'dS', 'dNet', 'signal']
CN = {
    'long': '多单量', 'short': '空单量', 'net': '净持仓',
    'dB': 'dB', 'dS': 'dS', 'dNet': '净持仓变动', 'signal': '信号',
}


# ---------------------------------------------------------------- 读取与清洗
def read_source_files():
    files = sorted(glob.glob(os.path.join(BASE, '**', '*.xlsx'), recursive=True))
    files = [f for f in files if '会员：' in os.path.basename(f)
             and not os.path.basename(f).startswith('~$')
             and 'TS_持仓整理' not in os.path.basename(f)]
    raw, quality = {}, []
    for p in files:
        fn = os.path.basename(p)
        m_mem = re.search(r'会员：(.+?)\s+品种：', fn)
        m_yr = re.search(r'(\d{4})年', os.path.basename(os.path.dirname(p)))
        if not (m_mem and m_yr):
            continue
        mem, yr = m_mem.group(1), int(m_yr.group(1))
        d = pd.read_excel(p, sheet_name=0, header=0)
        d['date'] = pd.to_datetime(d.iloc[:, 0], errors='coerce')
        d = d[d['date'].notna()].copy()
        d = d[~d['date'].duplicated(keep='last')].set_index('date').sort_index()

        lng = pd.to_numeric(d['多单量'], errors='coerce')
        net = pd.to_numeric(d['净持仓'], errors='coerce')
        sraw = pd.to_numeric(d['空单量'], errors='coerce')
        sht = -sraw                       # Wind 空单量为负 -> 取正
        settle = pd.to_numeric(d.get('结算价', pd.Series(index=d.index, dtype=float)),
                               errors='coerce')

        # 占位行：持仓三列全空（只有结算价），视为无效
        placeholder = lng.isna() & sht.isna() & net.isna()
        # 会员当日未进空单前20榜 -> 空单量为 NaN，按 0 处理（此时净持仓==多单量）
        impute = sht.isna() & lng.notna() & net.notna()
        sht = sht.where(~impute, 0.0)
        pos_short = int((sraw > 0).sum())   # 空单量异常为正的行数

        df = pd.DataFrame({'long': lng, 'short': sht, 'net': net, 'settle': settle})
        df = df.where(~placeholder, np.nan)
        # 恒等式检验（仅在三列齐全的行）
        ok = df[['long', 'short', 'net']].notna().all(axis=1)
        eq = float((df.loc[ok, 'net'] - (df.loc[ok, 'long'] - df.loc[ok, 'short'])).abs().max()) \
            if ok.any() else float('nan')

        raw[(yr, mem)] = df
        quality.append({
            '年份': yr, '会员': mem, '文件': os.path.relpath(p, BASE),
            '有效行数': int((~placeholder).sum()),
            '起始日': d.index.min().date().isoformat() if len(d) else '',
            '结束日': d.index.max().date().isoformat() if len(d) else '',
            '空单NaN填0行数': int(impute.sum()),
            '持仓全空占位行数': int(placeholder.sum()),
            '空单量为正异常行数': pos_short,
            '净持仓恒等式最大偏差': round(eq, 4) if pd.notna(eq) else None,
        })
    if not raw:
        raise RuntimeError('未找到任何源 Excel（需形如「会员：XXX 品种：TS ...xlsx」）')
    return raw, quality


def concat_years(raw, mem):
    parts = [df for (yr, m), df in sorted(raw.items()) if m == mem]
    if not parts:
        return None
    out = pd.concat(parts).sort_index()
    if out.index.duplicated().any():
        dup = out.index[out.index.duplicated()].date.tolist()[:5]
        raise RuntimeError(f'{mem} 跨年日期重复：{dup}')
    return out


def build_series(raw):
    """返回 {显示名: DataFrame}，索引为全局日期轴。"""
    date_sets = [set(df.index) for df in raw.values()]
    all_dates = pd.DatetimeIndex(sorted(set().union(*date_sets)))
    series = {}
    for disp, mem in [x for g in GROUPS.values() for x in g]:
        if mem is None:
            continue
        d = concat_years(raw, mem)
        series[disp] = d.reindex(all_dates) if d is not None \
            else pd.DataFrame(np.nan, index=all_dates,
                              columns=['long', 'short', 'net', 'settle'], dtype=float)

    # 国泰海通（合成）= 国泰君安 + 海通期货 逐日相加；缺任一腿则该日留空
    g, h = series['国泰君安'], series['海通期货']
    both = g[['long', 'short', 'net']].notna().all(axis=1) & \
           h[['long', 'short', 'net']].notna().all(axis=1)
    syn = pd.DataFrame(index=all_dates, columns=['long', 'short', 'net', 'settle'], dtype=float)
    for c in ['long', 'short', 'net']:
        syn[c] = (g[c] + h[c]).where(both)
    syn['settle'] = g['settle'].fillna(h['settle'])
    series['国泰海通（合成）'] = syn

    # 派生 dB / dS / dNet / 信号（严格相邻交易日，不跨越缺失日）
    for name, d in series.items():
        pl, ps, pn = d['long'].shift(1), d['short'].shift(1), d['net'].shift(1)
        valid = d['long'].notna() & d['short'].notna() & d['net'].notna() & \
                pl.notna() & ps.notna() & pn.notna()
        d['dB'] = (d['long'] - pl).where(valid)
        d['dS'] = (d['short'] - ps).where(valid)
        d['dNet'] = (d['net'] - pn).where(valid)
        sig = pd.Series(np.nan, index=d.index)
        sig[valid] = 0
        sig[(d['dB'] > 0) & (d['dS'] < 0)] = 1     # 多头增、空头减 -> 做多
        sig[(d['dB'] < 0) & (d['dS'] > 0)] = -1    # 多头减、空头增 -> 做空
        d['signal'] = sig
    return series, all_dates, both


# ---------------------------------------------------------------- 数值与统计
def as_num(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    f = float(v)
    return int(round(f)) if abs(f - round(f)) < 1e-9 else round(f, 4)


def arr(d, col):
    return [as_num(x) for x in d[col].tolist()]


def settle_base(series, all_dates):
    """全局结算价：优先前20大合计，逐日用其它序列补缺。"""
    s = pd.Series(np.nan, index=all_dates, dtype=float)
    for n in ['前20大合计', '前10大合计', '前5大合计', '中信期货', '国泰君安']:
        s = s.fillna(series[n]['settle'])
    return [as_num(x) for x in s.tolist()]


def signal_stats(d, settle):
    """信号统计 + 结算价口径的示意收益（非报告口径，仅用于直觉检查）。"""
    sig = d['signal']
    n1, nm1, n0 = int((sig == 1).sum()), int((sig == -1).sum()), int((sig == 0).sum())
    pnl = []
    s = pd.Series(settle, dtype=float)
    for i in np.where(sig.notna())[0]:
        v = sig.iloc[i]
        if v == 0 or i + 2 >= len(s):
            continue
        a, b = s.iloc[i + 1], s.iloc[i + 2]
        if pd.notna(a) and pd.notna(b):
            pnl.append(float(v) * (b - a))
    wins = sum(1 for x in pnl if x > 0)
    stat = {
        'n_long': n1, 'n_short': nm1, 'n_none': n0,
        'n_valid': n1 + nm1 + n0,
        'last_signal': as_num(sig.dropna().iloc[-1]) if sig.notna().any() else None,
        'last_signal_date': sig.dropna().index[-1].date().isoformat() if sig.notna().any() else None,
        '示意笔数': len(pnl),
        '示意胜率': round(wins / len(pnl), 4) if pnl else None,
        '示意均值_价格点': round(float(np.mean(pnl)), 5) if pnl else None,
        '示意合计_价格点': round(float(np.sum(pnl)), 4) if pnl else None,
    }
    return stat


# ---------------------------------------------------------------- 写出产物
def write_excel(series, all_dates, quality, meta):
    sheets = {}
    for gname, members in GROUPS.items():
        frame = pd.DataFrame({'日期': [x.date().isoformat() for x in all_dates]})
        for disp, _ in members:
            d = series[disp]
            for c in COLS:
                frame[f'{disp}_{CN[c]}'] = [as_num(x) for x in d[c].tolist()]
        sheets[gname] = frame

    sigmat = pd.DataFrame({'日期': [x.date().isoformat() for x in all_dates]})
    for n in ALL_NAMES:
        sigmat[n] = [as_num(x) for x in series[n]['signal'].tolist()]
    sheets['信号矩阵'] = sigmat

    q = pd.DataFrame(quality)
    sheets['数据质检'] = q

    notes = [
        ['项目', '说明'],
        ['品种', 'TS 2年期国债期货（中信所会员持仓，Wind 导出）'],
        ['数据区间', f"{meta['start']} ~ {meta['end']}，共 {meta['n_days']} 个交易日"],
        ['源数据', f"{meta['n_files']} 个 Excel（2021–2026）；前5/前10/前20 为会员合计，其余为单会员席位"],
        ['空单量符号', 'Wind 导出为负数，本表统一取绝对值（正值=空头持仓量）'],
        ['净持仓', '= 多单量 − 空单量；与 Wind 净持仓列逐日恒等，偏差见数据质检'],
        ['空单量为 NaN', '会员当日未进入空单前20榜，Wind 留空；本表按 0 处理（此时 Wind 净持仓 = 多单量，自洽）'],
        ['占位行', '2026-11-23 前5/前10 合计行只有结算价、持仓三列全空，判为当日未刷新，留空不参与计算'],
        ['跨年拼接', '按全局交易日序列连续 diff，不在每年首日断开'],
        ['dB / dS', '日变动；dS>0 表示空头增加（报告口径）。相邻交易日任一端缺失则当日留空，不跨越'],
        ['信号规则', 'T 日 dB>0 且 dS<0 → +1 做多；dB<0 且 dS>0 → −1 做空；其余 0'],
        ['执行口径', '报告为 T+1 开盘建仓 / T+2 收盘平仓；净值回测见 TS_蜘蛛网_策略回测.html（用 TS.CFE 开/收盘价）'],
        ['国泰海通（合成）', '国泰君安 + 海通期货 逐日相加；缺任一腿则该日留空。2026 缺海通源表，故该年合成序列为空；2021–2025 为两席位逐日相加'],
        ['2021 补充', '2021 年 7 个源表已补齐（243 天，恒等式零误差、跨年衔接连续）；报告回测起点 2020-01-01，本地仍缺 2020 全年'],
        ['动态版复现缺口', '报告动态调参需 n=1~20 逐档合计，现有仅 5/10/20 三档；且需 T+1 开盘价、T+2 收盘价'],
        ['生成时间', meta['generated']],
    ]
    sheets['说明'] = pd.DataFrame(notes[1:], columns=notes[0])

    with pd.ExcelWriter(XLSX_OUT, engine='openpyxl') as w:
        for name in ['说明', '合计类', '席位类', '信号矩阵', '数据质检']:
            sheets[name].to_excel(w, sheet_name=name, index=False)
            ws = w.sheets[name]
            ws.freeze_panes = 'B2'
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col[:60] if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 34)
    return XLSX_OUT


def build_payload(series, all_dates, quality, settle, meta, both):
    syn_missing = [d.date().isoformat() for d, ok in zip(all_dates, both) if not ok]
    payload = {
        'meta': meta,
        'dates': [d.date().isoformat() for d in all_dates],
        'settle': settle,
        'colNames': CN,
        'groups': {g: [n for n, _ in members] for g, members in GROUPS.items()},
        'series': {},
        'quality': {
            'files': quality,
            'synthetic_missing': syn_missing,
            'synthetic_missing_2026': sum(1 for x in syn_missing if x.startswith('2026')),
            'notes': meta['alerts'],
        },
    }
    for n in ALL_NAMES:
        d = series[n]
        payload['series'][n] = {
            'group': '合计类' if n in payload['groups']['合计类'] else '席位类',
            **{c: arr(d, c) for c in COLS},
            'stats': signal_stats(d, settle),
            'validDays': int(d['long'].notna().sum()),
            'note': ('2026 缺海通期货源表，合成序列该年留空' if n == '国泰海通（合成）' else
                     '2026-11-23 持仓未刷新，该日留空' if n in ('前5大合计', '前10大合计') else ''),
        }
    return payload


def build_navdata(dates_iso):
    """从 TS_策略回测.json 抽取各序列日收益与多空分解净值，内嵌进整理网页。

    净值口径与回测脚本完全一致：T 日信号 → T+1 开盘建仓 → T+2 收盘平仓、
    单倍杠杆、初始 1000 万、零成本。整理网页只负责展示，不重新计算收益。
    """
    if not os.path.exists(BT_JSON):
        print('警告：未找到 TS_策略回测.json，网页中的信号净值图将不可用')
        return {'available': False, 'ret': {}, 'bench': [], 'metrics': {},
                'decomp': {}, 'names': [], 'note': '请先运行 spider_backtest.py'}
    bt = json.load(open(BT_JSON, encoding='utf-8'))
    bd = list(bt['dates'])
    if bd != list(dates_iso):
        raise RuntimeError('回测日期轴与整理日期轴不一致，请先重跑 spider_backtest.py')
    n_day = len(bd)
    pos = {d: i for i, d in enumerate(bd)}
    ret, metrics, decomp = {}, {}, {}
    for name in ALL_NAMES:
        if name not in bt['series']:
            continue
        sr = bt['series'][name]
        ret[name] = {'state': sr['state']['ret'], 'independent': sr['independent']['ret']}
        metrics[name] = sr['state']['metrics']
        # 按平仓日把逐笔独立交易拆成「只做多 / 只做空 / 合计」三条净值
        lg = [0.0] * n_day
        sh = [0.0] * n_day
        al = [0.0] * n_day
        for t in sr['independent']['trades']:
            j = pos.get(t.get('平仓日'))
            if j is None:
                continue
            r = float(t.get('ret', 0.0))
            al[j] += r
            if t.get('方向') == '多':
                lg[j] += r
            else:
                sh[j] += r

        def _nav(arr):
            v, out = 1.0, []
            for x in arr:
                v += x                  # 固定名义单利：盈亏直接累加，不复利
                out.append(round(v, 7))
            return out

        decomp[name] = {'long': _nav(lg), 'short': _nav(sh), 'all': _nav(al)}

    # 自洽校验：三条分解中的「合计」应等于逐笔独立日收益的累计净值
    max_err = 0.0
    for name, arr in decomp.items():
        v, ref = 1.0, []
        for x in ret[name]['independent']:
            v += x
            ref.append(v)
        max_err = max(max_err, max(abs(a - b) for a, b in zip(arr['all'], ref)))
    print(f'多空分解自洽校验：最大偏差 {max_err:.2e}')
    if max_err > 1e-4:
        print('  警告：偏差偏大，分解图可能与主净值口径不一致，请核对 trades 与 ret')

    return {
        'available': True,
        'ret': ret,
        'bench': bt['benchmark']['ret'],
        'metrics': metrics,
        'decomp': decomp,
        'names': [n for n in ALL_NAMES if n in ret],
        'note': ('净值口径：T 日信号 → T+1 开盘建仓 → T+2 收盘平仓；固定 5 手、名义 1000 万、单倍杠杆、零成本；'
                 'TS 1 点 = 初始资金 1%，净值 = 1 + 累计盈亏点数/100，逐笔直接累加、不复利。'
                 '数据源 TS.CFE.xlsx（Wind 开/收盘价）。多空分解按逐笔独立口径、以平仓日计入，'
                 '仅用于观察多空两侧贡献，与状态机净值略有差异。'),
        'generated': bt['meta']['generated'],
    }


def write_html(payload, navdata):
    if not os.path.exists(HTML_TPL):
        raise RuntimeError(f'缺少网页模板：{HTML_TPL}')
    tpl = open(HTML_TPL, encoding='utf-8').read()
    js = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    nj = json.dumps(navdata, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    if '/*__DATA__*/' not in tpl or '/*__NAVDATA__*/' not in tpl:
        raise RuntimeError('模板缺少 /*__DATA__*/ 或 /*__NAVDATA__*/ 占位符')
    out = tpl.replace('/*__DATA__*/', js).replace('/*__NAVDATA__*/', nj)
    open(HTML_OUT, 'w', encoding='utf-8').write(out)
    return HTML_OUT


def main():
    raw, quality = read_source_files()
    series, all_dates, both = build_series(raw)
    settle = settle_base(series, all_dates)

    n_impute = sum(q['空单NaN填0行数'] for q in quality)
    n_placeholder = sum(q['持仓全空占位行数'] for q in quality)
    bad_eq = [q for q in quality if q['净持仓恒等式最大偏差'] not in (None, 0, 0.0)]
    alerts = [
        f'2026-11-23：前5/前10 合计行持仓未刷新（仅结算价），该日留空；前20合计、中信期货有当日值；单会员席位整体只到 2026-11-20。',
        f'海通期货 2026 源表缺失 → 国泰海通（合成）2026 全段留空（{int((~both & (all_dates.year == 2026)).sum())} 天）；2021–2025 为两席位逐日相加。',
        f'2021 年 7 个源表已补入（源文件共 {len(quality)} 个）；报告回测起点 2020-01-01，本地仍缺 2020 全年，报告可比段为 2021-01-04 ~ 2025-04-10。',
        f'空单量为 NaN 共 {n_impute} 行（会员当日未进空单前20榜），按 0 处理；此时 Wind 净持仓恰等于多单量，口径自洽。',
        '源表本身缺 3 个交易日：海通期货 2025-11-27、2025-12-05，平安期货 2024-04-29（会员当日未进榜，Wind 不生成行）。这些日子及次日不生成信号，不跨越缺失做 diff。',
        '开平仓规则按报告原文逐条实现，净值已用 TS.CFE 开/收盘价复现，详见 TS_蜘蛛网_策略回测.html。',
        '报告动态调参需 n=1~20 逐档合计，现有仅 5/10/20 三档 → 目前只能复现静态版。',
    ]
    meta = {
        'title': 'TS 机构持仓整理 · 蜘蛛网策略复现',
        'variety': 'TS 2年期国债期货',
        'start': all_dates.min().date().isoformat(),
        'end': all_dates.max().date().isoformat(),
        'n_days': len(all_dates),
        'n_files': len(quality),
        'n_series': len(ALL_NAMES),
        'aligned_end': max(d.date().isoformat() for d, ok in zip(all_dates, both) if ok),
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'alerts': alerts,
        'quality_ok': len(bad_eq) == 0,
    }

    write_excel(series, all_dates, quality, meta)
    payload = build_payload(series, all_dates, quality, settle, meta, both)
    open(JSON_OUT, 'w', encoding='utf-8').write(
        json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    dates_iso = [d.date().isoformat() for d in all_dates]
    navdata = build_navdata(dates_iso)
    write_html(payload, navdata)

    print('=== 完成 ===')
    print('源文件数:', len(quality), '| 交易日:', len(all_dates),
          f"| {meta['start']} ~ {meta['end']}", '| 序列:', len(ALL_NAMES))
    print('恒等式异常文件数:', len(bad_eq), '| 空单NaN填0:', n_impute, '| 占位行:', n_placeholder)
    print('对齐末日(合成序列可用):', meta['aligned_end'])
    print('\n=== 各序列信号统计 ===')
    for n in ALL_NAMES:
        s = payload['series'][n]['stats']
        print(f"  {n:<16} 有效日 {payload['series'][n]['validDays']:>4} | "
              f"多 {s['n_long']:>3} 空 {s['n_short']:>3} 无 {s['n_none']:>4} | "
              f"末次信号 {s['last_signal_date']} {s['last_signal']} | "
              f"示意胜率 {s['示意胜率']}")
    for p in (XLSX_OUT, JSON_OUT, HTML_OUT):
        print('输出:', p, f'({os.path.getsize(p)/1024:.1f} KB)')


if __name__ == '__main__':
    main()
