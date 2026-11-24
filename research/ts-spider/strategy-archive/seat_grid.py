"""Compare entry rules for TS holdings aggregates on the frozen sample."""
import argparse
from datetime import date
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spider_archive_rules as archive
import spider_daily_data as exchange
import spider_expiry_benchmark as expiry
import spider_summary as summary

SEATS = ('前5大合计', '前10大合计')
HOLDS = (1, 3, 5, 10)
RANKS = (.60, .65, .70, .75, .80, .85, .90, .95)


def report(seat, candidates):
    fit = {rule: result for rule, result in candidates.items()
           if result['前段']['trades'] >= 20 and result['前段']['total_pct'] > 0}
    ranks = {
        field: {rule: i for i, rule in enumerate(sorted(fit, key=lambda r: -fit[r]['前段'][field]))}
        for field in ('total_pct', 'sharpe')
    }
    fit_first = min(fit, key=lambda rule: (
        ranks['total_pct'][rule] + ranks['sharpe'][rule], ranks['sharpe'][rule],
    ))
    paired = [rule for rule, result in fit.items()
              if result['后段']['trades'] >= 20 and result['后段']['total_pct'] > 0]
    balanced = max(paired, key=lambda rule: min(fit[rule]['前段']['sharpe'], fit[rule]['后段']['sharpe']))
    late_first = max(paired, key=lambda rule: fit[rule]['后段']['sharpe'])

    print(f'\n{seat} | {len(candidates)} 组')
    for label, rule in [('前段收益与夏普等权排名', fit_first), ('两段较低夏普最高', balanced),
                        ('后段夏普最高', late_first)]:
        print(f'{label} | H={rule.hold} | 开仓分位 {rule.entry_rank:.0%} | '
              f'收跌过滤 {rule.require_down} | 趋势均线 {rule.trend_window} | 提前退出分位 {rule.exit_rank}')
        for period, metrics in candidates[rule].items():
            print(f"  {period} | {metrics['total_pct']:.3f}% | 夏普 {metrics['sharpe']:.3f}"
                  f" | 最大回撤 {metrics['drawdown_pct']:.3f}% | {metrics['trades']} 笔")


def report_directions(seat, close, positions, backtest):
    holdings = np.asarray(positions[seat], dtype=float)
    long_change, short_change = np.diff(holdings, axis=0, prepend=np.full((1, 2), np.nan)).T
    both = (long_change > 0) & (short_change < 0)
    net = long_change > short_change
    dip = np.diff(close, prepend=np.nan) < 0
    yesterday = pd.Series(close).shift(1)
    trend = (yesterday > yesterday.rolling(60).mean()).to_numpy()
    no_rank = np.full(len(close), np.nan)

    print(f'\n{seat} | 累计收益 / 夏普 / 最大回撤 / 笔数')
    print('规则 | H | 触发日数 | 全样本 | 前段 | 后段 | 2024 | 2025 | 2026')
    rules = (
        ('多增空减', both, HOLDS),
        ('净多增加', net, HOLDS),
        ('多增空减附加价格条件', both & dip & trend, (5,)),
        ('净多增加附加价格条件', net & dip & trend, (5,)),
    )
    for name, entries, holds in rules:
        for hold in holds:
            rule = archive.Strategy('direction', name, seat, hold, 0, 0)
            results = backtest(rule, entries, no_rank)
            cells = [f"{m['total_pct']:.3f}% / {m['sharpe']:.3f} / "
                     f"{m['drawdown_pct']:.3f}% / {m['trades']}" for m in results.values()]
            print(' | '.join([name, str(hold), str(np.count_nonzero(entries)), *cells]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direction', action='store_true', help='比较前五、前十、前二十的不设分位入场规则')
    args = parser.parse_args()
    seats = (*SEATS, '前20大合计') if args.direction else SEATS
    cutoff = date.fromisoformat(archive.ASOF)
    snapshots = {day: row for day, row in exchange.read_inputs().items() if day <= archive.ASOF}
    dates, _, close, positions = summary.load_data(snapshots, cutoff, seats=seats)
    end = int(np.searchsorted(dates, archive.ASOF, side='right')) - 1
    dates = dates[:end + 1]
    if not dates or dates[-1] != archive.ASOF:
        raise ValueError(f'归案行情需要覆盖到 {archive.ASOF}')
    close = close[:end + 1]
    positions = {seat: rows[:end + 1] for seat, rows in positions.items()}
    split = int(np.searchsorted(dates, archive.SPLIT))
    periods = [('全样本', 0, end), ('前段', 0, split - 1), ('后段', split, end)]
    for year in (2024, 2025, 2026):
        lo = int(np.searchsorted(dates, f'{year}-01-01'))
        hi = int(np.searchsorted(dates, f'{year + 1}-01-01')) - 1
        periods.append((str(year), lo, hi))
    quotes = expiry.read_quotes(snapshots)
    contracts = archive.contract_ranks(dates, snapshots)
    plans = {hold: archive.contract_plans(dates, contracts, quotes, hold) for hold in HOLDS}

    def backtest(rule, entries, ranks):
        return {
            label: archive.metrics(*archive.simulate(rule, dates, quotes, plans[rule.hold], entries, ranks, lo, hi))
            for label, lo, hi in periods
        }

    for seat in seats:
        if args.direction:
            report_directions(seat, close, positions, backtest)
            continue
        candidates = {}
        for hold, rank, trend, exit_rank, down in product(HOLDS, RANKS, (0, 60), (None, .20), (True, False)):
            rule = archive.Strategy('grid', seat, seat, hold, rank, 0, exit_rank, trend, down)
            entries, ranks = archive.signal_inputs(rule, close, positions)
            candidates[rule] = backtest(rule, entries, ranks)
        report(seat, candidates)


if __name__ == '__main__':
    main()
