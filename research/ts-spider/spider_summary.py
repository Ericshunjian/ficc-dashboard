"""Build the five-strategy dashboard and the machine-readable daily signal report."""
import argparse
from datetime import date
from html import escape
import json
from pathlib import Path

import numpy as np

import spider_backtest as bt
import spider_daily_data as exchange
import spider_expiry_benchmark as expiry
import spider_pingan_signal as sb

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / 'spider_summary_template.html'
HTML = BASE / 'index.html'
SUMMARY_ALIAS = BASE / 'TS_研究汇总.html'
SIGNALS = BASE / 'TS_每日信号.json'
FORWARD = BASE / 'TS_前瞻记录.json'
FORWARD_START = '2026-11-24'
SELECTION_END = '2026-11-23'
COST = 0.005  # Fixed round-trip points for comparable research metrics.
COLORS = ('#2e65c2', '#0d836a', '#ac542f', '#7951b5', '#9a690b')
SEATS = ('前5大合计', '前10大合计', '前20大合计', '中信期货', '平安期货',
         '国泰君安')
RULES = {
    'static': ('持仓双变动', '多增空减做多，反向做空，T+2 收盘平仓，可提前平仓'),
    'long': ('逢跌纯多', '收跌且净持仓变化达到 60 日 80% 分位，T+{exit} 收盘平仓'),
    'both': ('逆势多空', '收跌且净增达到 80% 分位做多，收涨且净变动低于或等于 20% 分位做空，T+{exit} 收盘平仓'),
    'gridpts': ('网格·点数王', '平安 dNet 60日分位≥75%，不看当日涨跌 → T+1开盘做多、T+{exit}收盘平仓'),
    'gridsharp': ('网格·夏普王', '当日收跌 且 平安 dNet 60日分位≥75% → T+1开盘做多、T+{exit}收盘平仓'),
}
GRID = (('gridpts', 10), ('gridsharp', 3))  # 用户指定保留：120格事后挑参，仅作刻度


def append_sessions(dates, opens, closes, positions, snapshots):
    anchor = dates[-1]
    previous = snapshots.get(anchor)
    for day, session in sorted(snapshots.items()):
        if day <= anchor:
            continue
        if previous is None:
            raise ValueError(f'缺少价格衔接日 {anchor} 的中金所行情')
        contract = max(previous['prices'], key=lambda key: (previous['prices'][key][2], key))
        if contract not in session['prices']:
            raise ValueError(f'{day} 缺少前一交易日主力合约 {contract}')
        c0 = previous['prices'][contract][1]
        op, cl, _ = session['prices'][contract]
        opens.append(round(closes[-1] + op - c0, 6))
        closes.append(round(closes[-1] + cl - c0, 6))
        dates.append(day)
        for seat in SEATS:
            positions[seat].append(session['positions'].get(seat, (None, None)))
        previous = session
    return dates, np.array(opens), np.array(closes), positions


def load_data(snapshots, target=None):
    holdings, dates, opens, closes, _, _ = bt.load_data()
    indices = {day: i for i, day in enumerate(holdings['dates'])}
    positions = {}
    for seat in SEATS:
        series = holdings['series'][seat]
        pairs = [(series['long'][indices[day]], series['short'][indices[day]]) for day in dates]
        # Fill an absent Wind snapshot only when the exchange published a complete row.
        positions[seat] = [
            snapshots.get(day, {}).get('positions', {}).get(seat, pair)
            if None in pair else pair for day, pair in zip(dates, pairs)
        ]
    if target:
        snapshots = {day: value for day, value in snapshots.items() if day <= target.isoformat()}
    return append_sessions(dates, opens, closes, positions, snapshots)


def signal_inputs(closes, positions, rule):
    values = np.asarray(positions, dtype=float)
    buy, sell = np.diff(values, axis=0, prepend=np.full((1, 2), np.nan)).T
    net = buy - sell
    change = np.diff(closes, prepend=np.nan)
    if rule == 'static':
        raw = ((buy > 0) & (sell < 0)).astype(int) - ((buy < 0) & (sell > 0)).astype(int)
        return raw, np.isfinite(net), {'duo': buy[-1], 'kong': sell[-1]}
    q = sb.rolling_pct(net)
    if rule == 'gridpts':
        raw = np.maximum(sb.make_signal(q, change, 'none', 0.75), 0)
    elif rule == 'gridsharp':
        raw = np.maximum(sb.make_signal(q, change, 'contra', 0.75), 0)
    else:
        raw = sb.make_signal(q, change, 'contra', 0.8)
        if rule == 'long':
            raw = np.maximum(raw, 0)
    return raw, np.isfinite(q) & np.isfinite(change), {'dnet': net[-1], 'pct': q[-1], 'price_change': change[-1]}


def run_rule(dates, opens, closes, signals, rule, start, end, *, hold=3, live=False, cost=None):
    c = COST if cost is None else cost
    if rule == 'static':
        mapping = {day: int(signal) for day, signal in zip(dates, signals) if signal}
        result = bt.run_state(dates, opens, closes, mapping, start, end, c, live=live)
        returns, trades = result[:2]
        state = dict(result[2], trades=trades) if live else None
        return np.asarray(returns), bt.metrics(returns, trades), state
    returns, trades = sb.run(signals, opens, closes, hold, c, start, end, mark_open=live)
    state = None
    if live:
        state = {'position': 0, 'entry': None, 'exit': None, 'next_open': 0,
                 'next_close': False, 'trades': trades}
        for signal, entry, exit_day in sb.orders(signals, opens, hold, start, end):
            if entry == end + 1:
                state['next_open'] = int(signals[signal])
            elif entry <= end < exit_day:
                state.update(position=int(signals[signal]), entry=entry, exit=exit_day)
    return returns[start:end + 1], sb.metrics(returns, trades, start, end), state


def rank_candidates(dates, opens, closes, positions):
    cutoff = int(np.searchsorted(dates, SELECTION_END, side='right')) - 1
    split = int(np.searchsorted(dates, sb.SPLIT))
    candidates = []
    for seat in SEATS:
        for rule in RULES:
            if rule in dict(GRID):
                continue
            raw, valid, drivers = signal_inputs(closes, positions[seat], rule)
            if valid[max(0, cutoff - 59):cutoff + 1].mean() < 0.9:
                continue
            pingan_long = (seat, rule) == ('平安期货', 'long')
            holds = (3,)  # 用户 2026-11-24 指定：平安纯多维持 H=3
            c = 0.0 if pingan_long else COST  # 用户指定：平安纯多不扣成本
            for hold in holds:
                _, full, _ = run_rule(dates, opens, closes, raw, rule, 0, cutoff, hold=hold, cost=c)
                _, recent, _ = run_rule(dates, opens, closes, raw, rule, split, cutoff, hold=hold, cost=c)
                if full['累计收益率'] <= 0 or recent['累计收益率'] <= 0 or full['交易笔数'] < 40:
                    continue
                candidates.append({'seat': seat, 'rule': rule, 'hold': hold, 'raw': raw, 'valid': valid,
                                   'drivers': drivers, 'full': full, 'recent': recent, 'rank': 0, 'cost': c})
    # 网格两条由用户指定保留，不参与排名，直接占两席；事后挑参，只作刻度。
    forced = []
    for rule, hold in GRID:
        raw, valid, drivers = signal_inputs(closes, positions['平安期货'], rule)
        _, full, _ = run_rule(dates, opens, closes, raw, rule, 0, cutoff, hold=hold)
        _, recent, _ = run_rule(dates, opens, closes, raw, rule, split, cutoff, hold=hold)
        forced.append({'seat': '平安期货', 'rule': rule, 'hold': hold, 'raw': raw, 'valid': valid,
                       'drivers': drivers, 'full': full, 'recent': recent, 'rank': 0, 'forced': True,
                       'cost': COST})
    for metric in ('累计收益率', '夏普比率'):
        for rank, candidate in enumerate(sorted(candidates, key=lambda c: -c['full'][metric])):
            candidate['rank'] += rank
    by_rule = {}
    for candidate in sorted(candidates, key=lambda c: (c['rank'], -c['recent']['夏普比率'])):
        by_rule.setdefault((candidate['seat'], candidate['rule']), candidate)
    selected = forced + list(by_rule.values())[:5 - len(forced)]
    if len(selected) != 5:
        raise ValueError(f'历史筛选仅有 {len(selected)} 个合格策略')
    return selected


def display_state(raw, valid, state, dates):
    side = {1: '多', -1: '空', 0: '空仓'}
    if state['next_close']:
        action = '下个交易日开盘平仓'
    elif state['next_open']:
        action = '下个交易日开' + side[state['next_open']]
    elif state['position'] and state['exit'] == len(dates):
        action = '下个交易日收盘平仓'
    elif state['position']:
        action = '持' + side[state['position']]
    elif valid:
        action = '维持空仓'
    else:
        action = '待数据'
    return {'raw': side[int(raw)] if raw and valid else '无触发' if valid else '数据缺失',
            'position': side[state['position']], 'action': action,
            'entry': dates[state['entry']] if state['entry'] is not None else None}


def update_forward(dates, opens, closes, selected, payload, *, record_today):
    """Keep first-published signals and append settlements as those positions close."""
    state = (json.loads(FORWARD.read_text(encoding='utf-8')) if FORWARD.exists()
             else {'start': FORWARD_START, 'records': [], 'cumulative': {}})
    asof = payload['asof']
    daily = state.setdefault('sessions', {})
    if (record_today and asof >= state['start'] and payload['source'] == '中金所盘后数据'
            and asof > max(daily, default='')):
        daily[asof] = {s['name']: s['raw'] for s in payload['strategies']}
        state['asof'] = asof

    if daily and asof >= max(daily):
        first = int(np.searchsorted(dates, state['start']))
        known = {(t['策略key'], t['信号日']) for t in state['records']}
        for candidate, item in zip(selected, payload['strategies']):
            name, rule = item['name'], candidate['rule']
            directions = {'多': 1, '空': -1}
            raw = np.array([directions.get(daily.get(day, {}).get(name), 0) for day in dates])
            _, _, result = run_rule(dates, opens, closes, raw, rule, first, len(dates) - 1,
                                    hold=candidate['hold'], live=True, cost=candidate['cost'])
            for trade in result['trades']:
                if rule == 'static':
                    signal, entry, exit_day = (trade[key] for key in ('信号日', '开仓日', '平仓日'))
                    direction, points = trade['方向'], round(trade['ret'] * 100, 5)
                else:
                    signal, entry, exit_day = (dates[trade[key]] for key in ('signal_idx', 'entry_idx', 'exit_idx'))
                    direction, points = ('多' if trade['dir'] > 0 else '空'), trade['pnl_pts']
                if (name, signal) not in known:
                    state['records'].append({'策略key': name, '策略': name, '信号日': signal,
                                             '方向': direction, '开仓日': entry, '平仓日': exit_day,
                                             '盈亏点数': points, '状态': '已兑现'})
                    known.add((name, signal))
        state['records'].sort(key=lambda t: (t['信号日'], t['策略key']))
        cumulative = {}
        for record in state['records']:
            key = record['策略key']
            cumulative[key] = round(cumulative.get(key, 0.0) + record['盈亏点数'], 5)
        state['cumulative'] = cumulative
        write_if_changed(FORWARD, json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    return {'start': state['start'], 'sessions': sum(day <= asof for day in daily),
            'closed': sum(t['平仓日'] <= asof for t in state['records'])}


def digest(payload, target):
    current = payload['asof'] == target.isoformat()
    status = '已更新' if current else '目标日暂无完整结算数据'
    print(f"{status} | 目标日 {target} | 最新 {payload['asof']} | {payload['source']}")
    for s in payload['strategies']:
        print(f"{s['name']} | 信号 {s['raw']} | 盘后 {s['position']} | {s['action']}")


def write_if_changed(path, data):
    if path.exists() and path.read_text(encoding='utf-8') == data:
        return
    temporary = path.with_suffix('.tmp')
    temporary.write_text(data, encoding='utf-8', newline='\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', help='从中金所获取目标日数据')
    parser.add_argument('--date', type=date.fromisoformat, default=exchange.today())
    args = parser.parse_args()
    _, historical, _, _, _, _ = bt.load_data()
    anchor = date.fromisoformat(historical[-1])
    if args.date < anchor:
        parser.error(f'目标日期早于当前历史快照 {anchor}')
    snapshots = (exchange.refresh_inputs(anchor, args.date) if args.refresh else exchange.read_inputs())
    dates, opens, closes, positions = load_data(snapshots, args.date)
    asof = dates[-1]
    payload = {
        'asof': asof, 'selection_end': SELECTION_END, 'start': dates[0],
        'selection_note': '用户 2026-11-24 指定：剔除华泰系；保留网格点数王/夏普王（120格事后挑参，仅作刻度）；平安纯多零成本 + H=3，其余策略成本 0.005 点',
        'source': '中金所盘后数据' if asof > anchor.isoformat() else 'Wind 历史快照',
        'close_change': round(float(closes[-1] - closes[-2]), 4),
        'strategies': [],
    }
    nav = {}
    selected = rank_candidates(dates, opens, closes, positions)
    for candidate, color in zip(selected, COLORS):
        seat, rule, raw, valid = (candidate[k] for k in ('seat', 'rule', 'raw', 'valid'))
        hold = candidate['hold']
        r, m, state = run_rule(dates, opens, closes, raw, rule, 0, len(dates) - 1, hold=hold, live=True,
                               cost=candidate['cost'])
        label, description = RULES[rule]
        name = label if candidate.get('forced') else f'{seat} · {label}' + (f' H={hold}' if rule != 'static' else '')
        payload['strategies'].append({
            'name': name, 'rule': description.format(exit=hold + 1), 'color': color,
            'cost': candidate['cost'],
            'total': m['累计收益率'], 'sharpe': m['夏普比率'], 'drawdown': m['最大回撤'],
            'trades': m['交易笔数'], 'validation': candidate['recent']['累计收益率'],
            **display_state(raw[-1], valid[-1], state, dates),
            'drivers': {k: round(float(v), 4) if np.isfinite(v) else None
                        for k, v in candidate['drivers'].items()},
        })
        nav[name] = np.round(1 + np.cumsum(r), 7).tolist()
    nav[expiry.NAME] = expiry.net_values(dates, expiry.read_quotes(snapshots), COST, bt.PRICE_REF)
    payload['forward'] = update_forward(
        dates, opens, closes, selected, payload,
        record_today=args.date == exchange.today() and asof == args.date.isoformat(),
    )
    write_if_changed(SIGNALS, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    page_data = json.dumps({**payload, 'dates': dates, 'nav': nav,
                            'benchmark': {'name': expiry.NAME, 'color': '#46596f'}}, ensure_ascii=False,
                           separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    page = TEMPLATE.read_text(encoding='utf-8').replace('__PAYLOAD__', page_data)
    page = page.replace('__ASOF__', escape(asof))
    write_if_changed(HTML, page)
    write_if_changed(SUMMARY_ALIAS, page)
    write_if_changed(BASE / 'TS_每日信号.html',
                     (BASE / 'daily_signal_template.html').read_text(encoding='utf-8'))
    digest(payload, args.date)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
