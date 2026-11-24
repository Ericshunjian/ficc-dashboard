# -*- coding: utf-8 -*-
"""把前期 5 个研究网页的结果汇成一个单文件网页：TS_研究汇总.html

数据源（均为本目录已有产物，不重跑回测）：
  TS_席位对比.json / TS_平安纯多头.json / TS_平安净多策略.json
  TS_策略回测.json / TS_持仓整理.json / TS_规则杠杆归因.json
  TS_网格优选_明细.json（由 spider_grid_detail.py 补算两个最优格子的净值）
输出：TS_研究汇总.html（数据内嵌、离线可开）
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))


def J(name):
    with open(os.path.join(BASE, name), encoding='utf-8') as f:
        return json.load(f)


def rnd(v, k=6):
    return round(float(v), k)


def nav_of(x):
    return [rnd(v) for v in x]


def num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def main():
    seat = J('TS_席位对比.json')
    pure = J('TS_平安纯多头.json')
    pa7 = J('TS_平安净多策略.json')
    bt = J('TS_策略回测.json')
    hold = J('TS_持仓整理.json')
    diag = J('TS_规则杠杆归因.json')
    gd = J('TS_网格优选_明细.json')

    dates = seat['dates']
    N = len(dates)
    S = seat['seats']
    M = seat['meta']

    # ---------- ① 净值序列（6 条） ----------
    series = {}

    def add(key, name, raw, color, width, tag, m=None):
        series[key] = {
            'name': name, 'nav': nav_of(raw), 'color': color,
            'width': width, 'tag': tag, 'metrics': m or {},
        }

    add('PA_LONG', '平安 · 只做多（第八节）', pure['preset']['nav'], '#d64545', 2.6,
        '主口径 A', pure['preset']['all'])
    add('ZX_FULL', '中信 · 多空双侧（第九节）', S['中信期货']['full']['nav'], '#2c5aa0', 2.6,
        '主口径 B', S['中信期货']['full']['all'])
    add('PA_FULL', '平安 · 多空双侧（第七节）', S['平安期货']['full']['nav'], '#e2a03f', 1.5,
        '对照：同一席位多空都做', S['平安期货']['full']['all'])
    add('ZX_LONG', '中信 · 只做多', S['中信期货']['long']['nav'], '#3f8fa8', 1.5,
        '对照：同一席位不做空', S['中信期货']['long']['all'])
    add('P20_LONG', '前20大合计 · 只做多', S['前20大合计']['long']['nav'], '#7b61b8', 1.4,
        '对照：换个席位', S['前20大合计']['long']['all'])
    add('BH', '买入持有 TS（基准）', seat['benchmark']['买入持有'], '#9aa5b4', 1.4,
        '全程在场 · 不做任何择时', bt['benchmark']['metrics'])
    add('SPIDER', '蜘蛛网原策略（前20）', seat['benchmark']['蜘蛛网原策略_前20'], '#b0b8c4', 1.3,
        '报告口径 · dB/dS 规则', bt['periods']['all']['series']['前20大合计'])
    gv = gd['variants']
    add('GRID_SHARP', '网格·夏普王（事后挑参）', gv['GRID_SHARP']['nav'], '#b7791f', 1.7,
        '120 格按夏普挑出 · 不能上仓', gv['GRID_SHARP']['segments']['all'])
    add('GRID_PTS', '网格·点数王（事后挑参）', gv['GRID_PTS']['nav'], '#8f5b34', 1.7,
        '120 格按点数挑出 · 不能上仓', gv['GRID_PTS']['segments']['all'])

    # ---------- ② 指标对照表 ----------
    key_of = {
        '中信期货 多空双侧 ← 用户问的 2.525': ('中信期货', 'full'),
        '平安期货 多空双侧（第七节主口径）': ('平安期货', 'full'),
        '中信期货 只做多': ('中信期货', 'long'),
        '平安期货 只做多（第八节主口径）': ('平安期货', 'long'),
        '前20大合计 多空双侧': ('前20大合计', 'full'),
        '前20大合计 只做多': ('前20大合计', 'long'),
    }
    short_name = {
        '中信期货 多空双侧': '中信 · 多空双侧', '平安期货 多空双侧': '平安 · 多空双侧',
        '中信期货 只做多': '中信 · 只做多', '平安期货 只做多': '平安 · 只做多',
        '前20大合计 多空双侧': '前20 · 多空双侧', '前20大合计 只做多': '前20 · 只做多',
    }
    marks = {
        '平安期货 只做多': '★ 主口径 A', '中信期货 多空双侧': '★ 主口径 B',
        '平安期货 多空双侧': '第七节原始', '中信期货 只做多': '对照',
        '前20大合计 只做多': '对照', '前20大合计 多空双侧': '对照',
    }
    rows = []
    for o in seat['overview']:
        s, m = key_of[o['口径']]
        a = S[s][m]['all']
        name = o['口径'].replace(' ← 用户问的 2.525', '').replace('（第七节主口径）', '')
        name = name.replace('（第八节主口径）', '')
        rows.append({
            '口径': short_name.get(name, name),
            '标记': marks.get(name, ''), '累计点数': o['累计点数'], '夏普比率': o['夏普比率'],
            '交易笔数': o['交易笔数'], '胜率': o['胜率'],
            '逐笔t值': a['逐笔t值'], '每笔平均点数': a['每笔平均点数'],
            '最大回撤%': rnd(a['最大回撤'] * 100, 3),
            '探索段': o['探索段'], '验证段': o['验证段'],
            '扣1tick': o['扣1tick点数'], '扣2tick': o['扣2tick点数'],
            '备注': o['备注'],
        })
    for key, name in [('GRID_SHARP', '网格·夏普王'), ('GRID_PTS', '网格·点数王')]:
        v = gd['variants'][key]
        a = v['segments']['all']
        rows.append({
            '口径': name, '标记': '事后挑参 · 不可上仓',
            '累计点数': a['累计点数'], '夏普比率': a['夏普比率'],
            '交易笔数': a['交易笔数'], '胜率': a['胜率'],
            '逐笔t值': a['逐笔t值'], '每笔平均点数': a['每笔平均点数'],
            '最大回撤%': rnd(a['最大回撤'] * 100, 3),
            '探索段': v['segments']['explore']['累计点数'],
            '验证段': v['segments']['verify']['累计点数'],
            '扣1tick': v['costs'][2]['累计点数'], '扣2tick': v['costs'][3]['累计点数'],
            '备注': v['rule'] + '；120 格里按指标挑出，带前视',
        })
    # 按两个主口径优先排序
    order = ['平安 · 只做多', '中信 · 多空双侧', '网格·夏普王', '网格·点数王',
             '平安 · 多空双侧', '中信 · 只做多', '前20 · 只做多', '前20 · 多空双侧']
    rows.sort(key=lambda x: order.index(x['口径']) if x['口径'] in order else 99)
    rows.append({
        '口径': '买入持有 TS', '标记': '基准', '累计点数': bt['benchmark']['metrics']['累计点数'],
        '夏普比率': bt['benchmark']['metrics']['夏普比率'], '交易笔数': N,
        '胜率': bt['benchmark']['metrics']['胜率'], '逐笔t值': None,
        '每笔平均点数': None, '最大回撤%': rnd(bt['benchmark']['metrics']['最大回撤'] * 100, 3),
        '探索段': None, '验证段': None, '扣1tick': None, '扣2tick': None,
        '备注': '全程在场；平安纯多只在场 31.7%，按占用资金年化 1.66% vs 买入持有 0.61%',
    })
    p20 = bt['periods']['all']['series']['前20大合计']
    rows.append({
        '口径': '蜘蛛网原策略（前20）', '标记': '基准 · dB/dS', '累计点数': p20['累计点数'],
        '夏普比率': p20['夏普比率'], '交易笔数': p20['交易笔数'], '胜率': p20['胜率'],
        '逐笔t值': None, '每笔平均点数': None,
        '最大回撤%': rnd(p20['最大回撤'] * 100, 3),
        '探索段': None, '验证段': None, '扣1tick': None, '扣2tick': None,
        '备注': '报告静态版口径，信号规则与本次两套不同，仅作刻度',
    })

    # ---------- ③ 分年 / 成本 ----------
    yearly = {}
    for k, s, m in [('PA_LONG', '平安期货', 'long'), ('ZX_FULL', '中信期货', 'full'),
                    ('PA_FULL', '平安期货', 'full'), ('ZX_LONG', '中信期货', 'long')]:
        yearly[k] = [{'年份': str(y['年份']), '点数': num(y['累计点数']),
                      '笔数': int(y['交易笔数'])} for y in S[s][m]['yearly']]
    for k in ('GRID_SHARP', 'GRID_PTS'):
        yearly[k] = [{'年份': str(y['年份']), '点数': num(y['累计点数']),
                      '笔数': int(y['交易笔数'])} for y in gd['variants'][k]['yearly']]

    costs = {'labels': ['0', '1 tick', '2 tick', '4 tick'],
             'series': []}
    for name, key, color in [('平安·只做多', 'PA_LONG', '#d64545'),
                             ('中信·多空双侧', 'ZX_FULL', '#2c5aa0')]:
        s, m = ('平安期货', 'long') if key == 'PA_LONG' else ('中信期货', 'full')
        vals = [num(c['累计点数']) for c in S[s][m]['costs']]
        costs['series'].append({'name': name, 'values': vals, 'color': color})
    for name, key, color in [('网格·夏普王', 'GRID_SHARP', '#b7791f'),
                             ('网格·点数王', 'GRID_PTS', '#8f5b34')]:
        cs = gd['variants'][key]['costs']
        costs['series'].append({'name': name,
                                'values': [num(cs[i]['累计点数']) for i in (0, 2, 3, 4)],
                                'color': color})

    # ---------- ④ A/B/C 归因与多空腿 ----------
    abc = []
    for name in ['中信期货', '平安期货']:
        a = seat['attribution'][name]
        abc.append({
            'label': '中信' if name == '中信期货' else '平安',
            'values': [a['B_只做多'], a['C_空信号占仓位不开仓'], a['A_完整双侧']],
            'extra': {
                '空单自身盈亏': a['空单自身盈亏'], '挤掉多单的效应': a['挤掉多单的效应'],
                'A减B': a['A减B'], '被挤掉多单笔数': a['被挤掉多单笔数'],
                '被挤掉多单在只做多里合计': a['被挤掉多单在只做多里合计'],
                '多腿': a['多腿'], '空腿': a['空腿'],
            },
        })

    # ---------- ⑤ 证伪 / 零假设 ----------
    jitter = pure['decomposition']['random_null']['full']['抖动±3日·仍收跌']
    jitter5 = pure['decomposition']['random_null']['full']['抖动±5日·仍收跌']
    maxnull = seat['null_max']
    falsify = [
        {'对象': '平安·只做多', '做法': '信号日在收跌日内前后抖动 ±3 天（保持成簇 / 不叠加）',
         '真实': 3.096, '随机均值': jitter['mean'], 'p': jitter['p'],
         '判读': '主判据：不显著 ⇒ 平安「挑中哪天」没有独立增量'},
        {'对象': '平安·只做多', '做法': '同样抖动 ±5 天（并列如实报告）',
         '真实': 3.096, '随机均值': jitter5['mean'], 'p': jitter5['p'],
         '判读': '边缘显著，但不改变主判据结论'},
        {'对象': '中信·多空双侧', '做法': '9 个席位各自循环移位，每组取 9 个里的最大值（300 组）',
         '真实': maxnull['real_max'], '随机均值': maxnull['mean'], 'p': maxnull['p_max'],
         '判读': '多重比较后不显著 ⇒ 2.525 是「9 取最大」的选择偏差'},
        {'对象': '中信·多空双侧', '做法': '中信自己一条信号循环移位（400 次）',
         '真实': seat['null_shift']['中信期货']['real'],
         '随机均值': seat['null_shift']['中信期货']['mean'],
         'p': seat['null_shift']['中信期货']['p'],
         '判读': '单看自己显著，但它正是被挑出来的最大值，不能单独作数'},
        {'对象': '平安·多空双侧', '做法': '整条信号 / 只打乱平安力度循环移位（400 次）',
         '真实': 1.112, '随机均值': pa7['null_test']['shift_signal']['rand_mean_acc'],
         'p': pa7['null_test']['shift_signal']['p_acc'],
         '判读': 'p=0.213（只打乱力度 0.338）⇒ 第七节规则本身不显著'},
        {'对象': '蜘蛛网原策略', '做法': '前20大合计信号循环移位（200 次）',
         '真实': 1.307, '随机均值': bt['null_test']['前20大合计']['rand_mean_sharpe'],
         'p': bt['null_test']['前20大合计']['p_acc'],
         '判读': '随机均值列为夏普均值（口径不同，仅作刻度）'},
    ]

    # ---------- ⑥ 前期研究精简卡片 ----------
    quad = pa7['descriptive']['quad']
    diagvars = [v['报告可比段累计'] * 100 for v in diag['variants']]
    cards = [
        {
            'no': '①', 'title': '持仓整理', 'href': 'TS_蜘蛛网_持仓整理.html',
            'tag': '数据层',
            'points': [
                'Wind 导出 <b>45 张表</b> → <b>%d 个交易日</b>（2021-01-04 ~ 2026-11-23）、<b>9 个持仓序列</b>（前5/10/20 合计 + 6 个单会员席位）。' % hold['meta']['n_days'],
                '空单量带负号先取绝对值；<code>dB</code>=多单日变动、<code>dS</code>=空单日变动、<code>dNet</code>=净持仓日变动；跨年按连续交易日 diff。',
                '空单 NaN = 当日未进空单前 20 榜，按 0 处理（共 118 行，与 Wind 净持仓自洽）；缺 3 个交易日不跨越做 diff。',
                '缺口：2020 全年缺（报告回测起点）；2026 海通源表缺 ⇒ 国泰海通合成 2026 全段留空；2026-11-23 部分合计行未刷新。',
            ],
        },
        {
            'no': '②', 'title': '蜘蛛网策略回测 · 与报告差异', 'href': 'TS_蜘蛛网_策略回测.html',
            'tag': '报告复现',
            'points': [
                '静态版（dB>0 且 dS<0 → 多，T+1 开盘 / T+2 收盘）：前20大合计全样本 <b>1.307 点 / 夏普 0.67 / 183 笔</b>；报告公布 <b>9.52% / 0.83</b>，报告可比段本地只有 0.727 点。',
                '差异<b>不是</b>成本、杠杆或开平仓规则：报告明写单倍杠杆；12 种开平仓变体可比段只落在 <b>%.2f~%.2f 点</b>；反推需 7.3~13.1 倍杠杆，但回撤会同步放大、夏普不变。' % (min(diagvars), max(diagvars)),
                'Oracle 上限（方向 100% 全对、零成本）报告区间仅 <b>7.997%</b>，比报告值还低 19% ⇒ 报告净值应先当待证伪对象。',
                '零假设：前20 循环移位 p(累计)=0.080 / p(夏普)=0.065 未过显著性；每笔来回扣 1 个 tick（0.005 点）收益大幅缩水。',
            ],
        },
        {
            'no': '③', 'title': '平安「逆势加多」主观感受检验', 'href': 'TS_平安净多_策略.html',
            'tag': '第七节',
            'points': [
                '口述规则：收跌 + <code>dNet</code> 60 日分位 ≥80%（约 +571 手）→ 做多；收涨 + ≤20%（约 −466 手）→ 做空；持有 3 日。全样本 <b>1.112 点 / 164 笔 / 夏普 0.45</b>。',
                '「平安确实逆势」成立（dNet 与当日涨跌相关 <b>−0.158</b>），但「价涨+净多减→跌」不成立：未来 3 日 −0.003 点、t=−0.71；反例象限也都涨。',
                '去掉平安只看价跌反弹反而更赚（<b>1.639 点 &gt; 1.112 点</b>）；价跌内部「大幅加多」比「力度不足」只多 0.0055 点、Welch t=0.68 ⇒ 平安没有增量。',
                '同规则套到 8 个其他席位全部为正，中信 <b>2.525 点</b>最高；循环移位 p=0.213（只打乱力度 0.338）⇒ 赚的是 TS 慢牛里「逢跌做多」的钱。',
                '每笔平均仅 0.0068 点（约 1.4 个 tick），扣 1 tick 后 1.112 → 0.292 点（−74%），2 tick 转负。',
            ],
        },
        {
            'no': '④', 'title': '平安「只做多」策略', 'href': 'TS_平安纯多头_策略.html',
            'tag': '第八节 · 本次对比 A',
            'points': [
                '只关掉做空腿、参数一个没调：<b>3.096 点 / 3.096%</b>、<b>113 笔</b>、夏普 <b>1.43</b>、胜率 66.4%、逐笔 t=3.62、最大回撤 0.29%；探索段 1.431 / 验证段 <b>1.665 点</b>，6 个年份全正。',
                '删腿不是纯事后偷看：做空腿在<b>探索段就亏 −0.243 点</b>、验证段继续亏 −0.369 点 ⇒ 用 2023 年底之前的数据就能正当化「不做空」。',
                '择机三层（每笔未来 3 日均值）：全部交易日 0.00790 → 收跌日 0.01342 → 收跌+平安大幅加多 0.01754；平安条件增量 0.0055 点、<b>Welch t=0.682</b> 不显著。',
                '★ 主判据：把 162 个信号日在收跌日内抖动 ±3 天、保持成簇与不叠加，仍赚 <b>2.350 点、p=0.1037</b> ⇒ 赚的是「成簇时只进第一笔」的执行纪律，不是平安挑的那天。被跳过的 49 个追单平均亏 −0.00518 点。',
                '成本 / 资金效率：扣 1 tick 仍有 <b>2.531 点</b>（最抗成本）；只在场 31.7%，按占用资金年化 1.66%；但绝对收益仍低于买入持有 3.595 点。参数面两段相关仅 0.30，「持有几天」不稳定。',
            ],
        },
        {
            'no': '⑤', 'title': '9 席位横向对比 · 中信 2.525 归因', 'href': 'TS_席位对比_净值.html',
            'tag': '第九节 · 本次对比 B',
            'points': [
                '同一套参数只换席位，9 席位多空双侧均值 1.219 / 中位 1.112 / 标准差 0.746；中信 2.525 排 1/9、z=1.75。<b>换成只做多中信只剩 1.016 点（验证段 −0.863），平安却是 3.096 点</b>。',
                '差额几乎全在做空腿：中信多腿 +1.614 / 空腿 <b>+0.911</b>，平安多腿 +1.724 / 空腿 <b>−0.612</b>；两边做多腿几乎一样。',
                'A/B/C 隔离：中信从只做多 1.016 → 完整 2.525，其中空单自身赚 0.911、空单占仓位挡掉 29 笔本来亏 −0.769 点的多单（+0.598）；平安恰好反过来。',
                '但中信空头信号本身没信息：113 个信号日未来 3 日均值 −0.00509 点、<b>t=−0.59</b>；起作用的是「成簇时只进第一笔」把 45 个会涨的追单挡掉了。',
                '★ 「9 个里挑最大」零假设：随机最大值均值 1.572、p95 2.749，真实 2.525 只排 92.7 分位、<b>p=0.0733 不显著</b>；扣 4 tick 后 2.525 → 0.005 点（归零）。',
            ],
        },
        {
            'no': '⑥', 'title': '参数网格里的两个「最优格子」', 'href': 'TS_平安纯多头_策略.html',
            'tag': '事后挑参 · 只作刻度',
            'points': [
                '规则与前面完全同源，只改两个旋钮：<b>夏普王</b> = dNet 分位≥0.75 <b>且当日收跌</b>、持有 3 日（T+1 开盘 → T+4 收盘）；<b>点数王</b> = 分位≥0.75 <b>不看涨跌</b>、持有 10 日（T+1 开盘 → T+11 收盘）。都是单仓位、不做空。',
                '全样本 <b>3.863 点 / 夏普 1.617 / 128 笔</b>（验证段 1.756、6 年全正、扣 2 tick 还有 2.583 点）；点数王 <b>4.528 点 / 夏普 1.346 / 92 笔</b>（验证段 1.146、扣 2 tick 3.608 点，但 <b>2025 年亏 0.591 点</b>）。两条已叠加在顶部净值图里，勾选可见。',
                '<b>为什么不能当策略</b>：这两格是从 2 个信号源 × 2 种价格条件 × 5 个阈值 × 6 个持有期 = <b>120 个组合</b>里按全样本点数/夏普挑出来的最大值，天然带前视与多重比较。',
                '证据：探索段与验证段点数相关只有 <b>0.30</b>；探索段挑出的最优持有期是 H=10，到验证段只剩 0.683 点；按「每年用之前数据选参」的 walk-forward 跑下来 4 年合计 <b>1.753 点 / 夏普 0.876</b>，2025 年 −0.546 点。',
                '所以图中这两条更高的线只说明「规则面还有空间」这个刻度，<b>不能推断真能拿到 4.5 点</b>。要真用，必须事前锁参数再走一次干净的样本外。',
            ],
        },
    ]

    # ---------- ⑦ 其它数字 ----------
    bmk = seat['benchmark']
    cross = seat['cross_section']
    extra = {
        'buy_hold': bmk['买入持有累计点数'],
        'spider': bmk['蜘蛛网原策略累计点数'],
        'exposure': pure['preset']['exposure']['在场时间占比'],
        'exposure_ann': pure['preset']['exposure']['按占用资金年化'],
        'bh_ann': bt['benchmark']['metrics']['年化收益率'],
        'split': M['split'],
        'cross_mean': cross['mean'], 'cross_median': cross['median'], 'cross_std': cross['std'],
        'rand_max_mean': maxnull['mean'], 'rand_max_p95': maxnull['p95'],
        'rand_mean_of_means': maxnull['rand_mean_of_means'],
        'real_cross_mean': maxnull['real_cross_mean'],
        'short_t': seat['short_info']['中信期货']['逐笔t值'],
        'short_welch': seat['short_info']['中信期货']['Welch_t_vs_收涨日'],
        'skipped': pure['decomposition']['skipped'],
        'grid_wf_pts': pure['walk_forward']['all']['累计点数'],
        'grid_wf_sharpe': pure['walk_forward']['all']['夏普比率'],
        'grid_corr': pure['grid']['robust']['两段相关'],
        'grid_n': 120,
        'quad': quad,
        'generated': M['generated'],
        'start': M['start'], 'end': M['end'], 'n_days': N,
    }

    payload = {
        'meta': {
            'title': 'TS 蜘蛛网研究 · 一页汇总',
            'start': M['start'], 'end': M['end'], 'n_days': N,
            'split': M['split'], 'generated': M['generated'],
            'nav': M['nav'],
        },
        'dates': dates,
        'series': series,
        'rows': rows,
        'yearly': yearly,
        'costs': costs,
        'abc': abc,
        'falsify': falsify,
        'hist_jitter': {'hist': [rnd(v, 4) for v in jitter['hist']],
                        'real': jitter['real'], 'mean': jitter['mean'],
                        'p95': jitter['p95'], 'p': jitter['p']},
        'hist_max': {'hist': [rnd(v, 4) for v in maxnull['hist']],
                     'real': maxnull['real_max'], 'mean': maxnull['mean'],
                     'p95': maxnull['p95'], 'p': maxnull['p_max']},
        'cards': cards,
        'extra': extra,
    }

    with open(os.path.join(BASE, 'spider_summary_template.html'), encoding='utf-8') as f:
        tpl = f.read()
    js = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    out = tpl.replace('/*__DATA__*/', js)
    path = os.path.join(BASE, 'TS_研究汇总.html')
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(out)
    print('wrote', path, len(out), 'chars; payload', len(js), 'chars')


if __name__ == '__main__':
    main()
