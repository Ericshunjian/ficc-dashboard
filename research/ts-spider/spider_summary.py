"""Publish the TS archive strategies and the daily signal report."""
import argparse
from datetime import date
from html import escape
import json
from pathlib import Path

import numpy as np

import spider_archive_rules as archive
import spider_backtest as bt
import spider_daily_data as exchange
import spider_expiry_benchmark as expiry
import spider_pingan_signal as pingan

BASE = Path(__file__).resolve().parent
TEMPLATE = BASE / 'spider_summary_template.html'
HTML = BASE / 'index.html'
SUMMARY_ALIAS = BASE / 'TS_研究汇总.html'
SIGNALS = BASE / 'TS_每日信号.json'
FORWARD = BASE / 'TS_前瞻记录.json'
COLORS = ('#2e65c2', '#0d836a', '#7951b5', '#ac542f', '#8a6408')
SEATS = tuple(dict.fromkeys(strategy.seat for strategy in archive.ACTIVE_RULES))
ORDER_ACTIONS = {'下个交易日开盘做多', '下个交易日开盘平仓', '下个交易日收盘平仓'}


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
        for seat in positions:
            positions[seat].append(session['positions'].get(seat, (None, None)))
        previous = session
    return dates, np.array(opens), np.array(closes), positions


def load_data(snapshots, target=None, *, seats=SEATS):
    holdings, dates, opens, closes, _, _ = bt.load_data()
    indices = {day: i for i, day in enumerate(holdings['dates'])}
    positions = {}
    for seat in seats:
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


def year_activity(key, asof, historical, simulated, open_trades, records, sessions):
    year = asof[:4]
    closed = {trade['信号日']: trade for trade in simulated}
    for saved in records:
        if saved['策略key'] != key or saved['平仓日'] > asof:
            continue
        replayed = closed.get(saved['信号日'], {})
        same_trade = all(replayed.get(field) == saved.get(field)
                         for field in ('合约', '开仓日', '平仓日', '平仓原因'))
        closed[saved['信号日']] = {**replayed, **saved} if same_trade else saved

    active = [trade for trade in open_trades if trade['信号日'] not in closed]
    trades = []
    for source, rows in (('历史回放', historical), ('前瞻模拟', [*closed.values(), *active])):
        for trade in rows:
            if any((trade.get(field) or '').startswith(year) for field in ('信号日', '开仓日', '平仓日')):
                trades.append({**trade, '来源': source})
    trades.sort(key=lambda t: (t['平仓日'] or t['开仓日'], t['信号日']), reverse=True)

    orders = []
    for day in sorted(sessions, reverse=True):
        if day <= asof and day.startswith(year):
            signal = sessions[day].get(key, {})
            if signal.get('action') in ORDER_ACTIONS:
                orders.append({'日期': day, '动作': signal['action'], '合约': signal.get('contract')})
    return {'trades': trades, 'orders': orders}


def build_dashboard(dates, closes, positions, snapshots, *, publish):
    asof = dates[-1]
    history_end = int(np.searchsorted(dates, archive.ASOF, side='right'))
    split = int(np.searchsorted(dates, archive.SPLIT))
    quotes = expiry.read_quotes(snapshots)
    contracts = archive.contract_ranks(dates, snapshots)
    changes = np.diff(closes, prepend=np.nan)
    state = json.loads(FORWARD.read_text(encoding='utf-8')) if FORWARD.exists() else {}
    tracking = state.setdefault('archive_v1', {'after': archive.ASOF, 'sessions': {}, 'records': []})
    sessions = tracking['sessions']
    records = tracking['records']
    known = {(trade['策略key'], trade['信号日']) for trade in records}
    publish_today = publish and asof > archive.ASOF
    current = sessions.setdefault(asof, {}) if publish_today else sessions.get(asof, {})

    plans = {hold: archive.contract_plans(dates[:history_end], contracts, quotes, hold)
             for hold in {rule.hold for rule in archive.ACTIVE_RULES}}
    results, books = {}, {}
    for strategy in archive.ACTIVE_RULES:
        entries, ranks = archive.signal_inputs(strategy, closes, positions)
        holding = archive.holding_changes(positions[strategy.seat]) if strategy.flow_entry else None
        exits = holding[:, 0] < holding[:, 1] if strategy.negative_exit else None
        historical = []
        full = archive.simulate(
            strategy, dates, quotes, plans[strategy.hold], entries, ranks, 0, history_end - 1,
            exit_signals=exits, trade_log=historical,
        )
        recent = archive.simulate(
            strategy, dates, quotes, plans[strategy.hold], entries, ranks, split, history_end - 1,
            exit_signals=exits,
        )

        live_signals = {}
        for i in range(history_end, len(dates)):
            day = dates[i]
            issued = sessions.get(day, {}).get(strategy.key)
            if issued:
                live_signals[day] = issued
            else:
                live_signals[day] = archive.forward_signal(
                    strategy, day, contracts[day], entries[i], ranks[i], changes[i],
                    holding_change=holding[i] if holding is not None else None,
                )
        open_trades = []
        live_pnl, trades, live = archive.simulate_forward(
            strategy, dates, quotes, live_signals, history_end, open_trades=open_trades,
        )
        today = live_signals.get(asof) or archive.forward_signal(
            strategy, asof, contracts[asof], entries[-1], ranks[-1], changes[-1],
            holding_change=holding[-1] if holding is not None else None,
        )
        if holding is not None:
            today = {**today, **{key: int(value) if np.isfinite(value) else None
                                 for key, value in zip(('long_change', 'short_change'), holding[-1])}}
        if publish_today:
            current.setdefault(strategy.key, {**today, **live})
            for trade in trades:
                key = trade['策略key'], trade['信号日']
                if key not in known:
                    records.append(trade)
                    known.add(key)

        books[strategy.key] = year_activity(strategy.key, asof, historical, trades, open_trades, records, sessions)
        report = current.get(strategy.key, {**today, **live})
        item = {
            'key': strategy.key, 'name': strategy.name,
            'seat': strategy.seat, 'require_down': strategy.require_down,
            'trend_window': strategy.trend_window, 'flow_entry': strategy.flow_entry,
            'negative_exit': strategy.negative_exit,
            'hold': strategy.hold, 'entry_rank': strategy.entry_rank, 'exit_rank': strategy.exit_rank,
            'entry_on_close_day': not strategy.cooldown,
            **{key: report[key] for key in ('raw', 'position', 'contract', 'action', 'entry')},
            'drivers': {key: report.get(key) for key in ('pct', 'long_change', 'short_change')},
        }
        results[strategy.key] = (item, full, recent, live_pnl)

    items, nav = [], {}
    for strategy, color in zip(archive.FEATURED, COLORS, strict=True):
        item, full, recent, live_pnl = (combine_portfolio(strategy, results)
                                        if isinstance(strategy, archive.Portfolio) else results[strategy.key])
        history = archive.metrics(*full)
        validation = archive.metrics(*recent)
        items.append({
            **item, 'color': color, 'cost': 0,
            'total': history['total_pct'] / 100, 'sharpe': history['sharpe'],
            'drawdown': history['drawdown_pct'] / 100, 'trades': history['trades'],
            'validation': validation['total_pct'] / 100, 'validation_sharpe': validation['sharpe'],
        })
        nav[strategy.name] = np.round(1 + np.cumsum(np.r_[full[0], live_pnl]) / bt.PRICE_REF, 7).tolist()
    nav[expiry.NAME] = expiry.net_values(dates, quotes, 0, bt.PRICE_REF)
    if publish_today:
        records.sort(key=lambda trade: (trade['信号日'], trade['策略key']))
        write_if_changed(FORWARD, json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    forward = {'after': archive.ASOF, 'sessions': sum(day <= asof for day in sessions),
               'closed': sum(trade['平仓日'] <= asof for trade in records)}
    return items, nav, forward, {'year': asof[:4], 'books': books}


def combine_portfolio(strategy, results):
    (first, first_full, first_recent, first_live), (second, second_full, second_recent, second_live) = (
        results[leg.key] for leg in strategy.legs
    )
    sides = [first, second]
    triggers = {leg['raw'] for leg in sides}
    exposure = sum(leg['position'] == '多' for leg in sides) * 50
    actions = '、'.join(f"{leg['name']} {leg['action']}" + (f" {leg['contract']}" if leg['contract'] else '')
                       for leg in sides)
    item = {
        'key': strategy.key, 'name': strategy.name, 'legs': sides,
        'raw': '多' if '多' in triggers else '数据缺失' if '数据缺失' in triggers else '无触发',
        'position': f'多 {exposure}%' if exposure else '空仓',
        'action': actions, 'contract': None, 'entry': None,
    }
    full = ((first_full[0] + second_full[0]) / 2, first_full[1] + second_full[1], 0)
    recent = ((first_recent[0] + second_recent[0]) / 2, first_recent[1] + second_recent[1], 0)
    return item, full, recent, (first_live + second_live) / 2


def digest(payload, target):
    current = payload['asof'] == target.isoformat()
    status = '已更新' if current else '目标日暂无完整结算数据'
    print(f"{status} | 目标日 {target} | 最新 {payload['asof']} | {payload['source']}")
    for s in payload['strategies']:
        contract = f" {s['contract']}" if s['contract'] else ''
        print(f"{s['name']} | 信号 {s['raw']} | 盘后 {s['position']} | {s['action']}{contract}")


def write_if_changed(path, data):
    if path.exists() and path.read_text(encoding='utf-8') == data:
        return
    temporary = path.with_suffix('.tmp')
    temporary.write_text(data, encoding='utf-8', newline='\n')
    temporary.replace(path)


def pingan_quantiles(positions):
    holdings = np.asarray(positions['平安期货'], dtype=float)
    daily_net = np.diff(holdings[:, 0] - holdings[:, 1], prepend=np.nan)
    return {
        str(pct): [int(value) if np.isfinite(value) else None for value in series]
        for pct, series in zip((20, 75), pingan.rolling_quantile_levels(daily_net, (20, 75)), strict=True)
    }


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
    snapshots = {day: row for day, row in snapshots.items() if day <= args.date.isoformat()}
    dates, _, closes, positions = load_data(snapshots)
    asof = dates[-1]
    publish = args.date == exchange.today() and asof == args.date.isoformat() and asof > anchor.isoformat()
    strategies, nav, forward, activity = build_dashboard(dates, closes, positions, snapshots, publish=publish)
    payload = {
        'asof': asof, 'selection_end': min(archive.ASOF, asof), 'start': dates[0],
        'source': '中金所盘后数据' if asof > anchor.isoformat() else 'Wind 历史快照',
        'close_change': round(float(closes[-1] - closes[-2]), 4),
        'strategies': strategies, 'forward': forward, 'activity': activity,
    }
    write_if_changed(SIGNALS, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    page_data = json.dumps({**payload, 'dates': dates, 'nav': nav, 'pingan_quantiles': pingan_quantiles(positions),
                            'benchmark': {'name': expiry.NAME, 'color': '#46596f'}}, ensure_ascii=False,
                           separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    page = TEMPLATE.read_text(encoding='utf-8').replace('__PAYLOAD__', page_data)
    page = page.replace('__ASOF__', escape(asof))
    write_if_changed(HTML, page)
    alias = (BASE / 'daily_signal_template.html').read_text(encoding='utf-8')
    write_if_changed(SUMMARY_ALIAS, alias)
    write_if_changed(BASE / 'TS_每日信号.html', alias)
    digest(payload, args.date)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
