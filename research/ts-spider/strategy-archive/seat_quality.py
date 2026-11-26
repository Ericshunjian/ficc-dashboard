"""Compare per-trade TS signal outcomes across seats on common data dates."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spider_archive_rules as archive
import spider_daily_data as exchange
import spider_expiry_benchmark as expiry
import spider_summary as summary

SEATS = ('平安期货', '中信期货', '华泰期货', '国泰君安', '海通期货')
WINDOWS = (
    ('五席共同区间', '2023-01-01', '2025-12-31', SEATS),
    ('四席近期', '2024-01-01', archive.ASOF, SEATS[:4]),
    ('四席 2026', '2026-01-01', archive.ASOF, SEATS[:4]),
)


def fixed_outcome(dates, quotes, plan):
    entry, exit_day, contract = plan
    return round(quotes[dates[exit_day]][contract][1] - quotes[dates[entry]][contract][0], 5)


def describe(label, outcomes):
    points = np.array(list(outcomes.values()), dtype=float)
    wins, losses = points[points > 0], -points[points < 0]
    if not len(points):
        return f'{label} | 0 笔'
    win_mean = wins.mean() if len(wins) else np.nan
    loss_mean = losses.mean() if len(losses) else np.nan
    return (f'{label} | {len(points)} 笔 | 胜率 {len(wins)/len(points):.2%} | '
            f'盈亏比 {win_mean/loss_mean:.3f} | 平均赢 {win_mean:.4f} | 平均亏 {loss_mean:.4f} | '
            f'每笔 {points.mean():+.4f} | 平 {np.count_nonzero(points==0)} 笔')


def paired_interval(days, first, second):
    paths = [np.array([pnl.get(day, np.nan) for day in days]) for pnl in (first, second)]
    block = 20
    draws = 3000
    rng = np.random.default_rng(20261126)
    starts = rng.integers(0, len(days) - block + 1, (draws, (len(days) + block - 1) // block))
    sampled = (starts[:, :, None] + np.arange(block)).reshape(draws, -1)[:, :len(days)]
    means = [np.nanmean(p[sampled], axis=1) for p in paths]
    return np.quantile(means[0] - means[1], (.025, .975))


def report_raw(dates, quotes, plans, positions):
    changes = {seat: archive.holding_changes(positions[seat]) for seat in SEATS}
    print('原始持仓双变动 | 每次触发独立成交 | T+1 开盘进、T+2 收盘出')
    for label, first, last, seats in WINDOWS:
        lo = int(np.searchsorted(dates, first))
        hi = int(np.searchsorted(dates, last, side='right')) - 1
        common = np.logical_and.reduce([np.isfinite(changes[seat]).all(axis=1) for seat in seats])
        eligible = [t for t in range(lo, hi + 1) if common[t] and t in plans and plans[t][1] <= hi]
        print(f'\n{label} {dates[lo]} 至 {dates[hi]} | 共同有效 {common[lo:hi+1].sum()} / {hi-lo+1} 日')
        for seat in seats:
            long, short = changes[seat].T
            longs = {dates[t]: fixed_outcome(dates, quotes, plans[t])
                     for t in eligible if long[t] > 0 and short[t] < 0}
            shorts = {dates[t]: -fixed_outcome(dates, quotes, plans[t])
                      for t in eligible if long[t] < 0 and short[t] > 0}
            print(describe(seat + ' 多空合计', {**longs, **shorts}))
            print(describe(seat + ' 做多', longs))
            print(describe(seat + ' 做空', shorts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', action='store_true', help='仅比较原始多空双变动与固定退出')
    args = parser.parse_args()
    snapshots = {day: row for day, row in exchange.read_inputs().items() if day <= archive.ASOF}
    dates, _, closes, positions = summary.load_data(snapshots, seats=SEATS)
    if dates[-1] != archive.ASOF:
        raise ValueError(f'归案行情需要覆盖到 {archive.ASOF}')
    quotes = expiry.read_quotes(snapshots)
    contracts = archive.contract_ranks(dates, snapshots)
    if args.raw:
        report_raw(dates, quotes, archive.contract_plans(dates, contracts, quotes, 1), positions)
        return
    plans = {hold: archive.contract_plans(dates, contracts, quotes, hold) for hold in (3, 5)}
    calendar = {day: i for i, day in enumerate(dates)}

    for base in (archive.STRATEGIES[1], archive.PA_FLOW3):
        prepared = {}
        for seat in SEATS:
            rule = replace(base, name=seat, seat=seat)
            entries, ranks = archive.signal_inputs(rule, closes, positions)
            valid = (np.isfinite(archive.holding_changes(positions[seat])).all(axis=1)
                     if rule.flow_entry else np.isfinite(ranks) & np.isfinite(np.diff(closes, prepend=np.nan)))
            prepared[seat] = (rule, entries, ranks, valid)

        print(f'\n{base.name}')
        for label, first, last, seats in WINDOWS:
            lo = int(np.searchsorted(dates, first))
            hi = int(np.searchsorted(dates, last, side='right')) - 1
            common = np.logical_and.reduce([prepared[seat][3] for seat in seats])
            print(f'\n{label} {dates[lo]} 至 {dates[hi]} | 共同有效 {common[lo:hi+1].sum()} / {hi-lo+1} 日')
            actual = {}
            for seat in seats:
                rule, entries, ranks, _ = prepared[seat]
                entries = entries & common
                exits = (ranks < rule.exit_rank) & common if rule.exit_rank is not None else None
                ledger = []
                archive.simulate(rule, dates, quotes, plans[rule.hold], entries, ranks, lo, hi,
                                 exit_signals=exits, trade_log=ledger)
                outcomes = {t['信号日']: t['盈亏点数'] for t in ledger}
                actual[seat] = outcomes
                print(describe(seat + ' 策略逐笔', outcomes))
                if label == '四席 2026':
                    continue
                events = {dates[t]: fixed_outcome(dates, quotes, plans[3][t])
                          for t in np.flatnonzero(entries[lo:hi+1]) + lo
                          if t in plans[3] and plans[3][t][1] <= hi}
                print(describe(seat + ' 每次触发固定 T+4', events))
                if rule.exit_rank is not None:
                    fixed = {day: fixed_outcome(dates, quotes, plans[rule.hold][calendar[day]]) for day in outcomes}
                    print(describe(seat + ' 同入场固定 T+6', fixed))
            if label == '五席共同区间':
                ci = paired_interval(dates[lo:hi+1], actual[SEATS[0]], actual[SEATS[4]])
                print(f'平安减海通每笔均值差的配对区块重采样 95% 区间 | {ci[0]:+.4f} 至 {ci[1]:+.4f} 点')


if __name__ == '__main__':
    main()
