"""Compare entry rules for TS holdings aggregates on the frozen sample."""
import argparse
from dataclasses import replace
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
FLOW_SEATS = (*SEATS, '前20大合计')
FLOW_FLOORS = (0, 50, 100, 250, 500)


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


def holding_changes(seat, positions):
    return archive.holding_changes(positions[seat]).T


def print_periods(identifier, results):
    cells = [f"{m['total_pct']:.3f}% / {m['sharpe']:.3f} / "
             f"{m['drawdown_pct']:.3f}% / {m['trades']}" for m in results.values()]
    print(' | '.join([*identifier, *cells]))


def report_directions(seat, close, positions, backtest):
    long_change, short_change = holding_changes(seat, positions)
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
            print_periods([name, str(hold), str(np.count_nonzero(entries))], backtest(rule, entries, no_rank))


def report_exit_study(seat, positions, backtest):
    long_change, short_change = holding_changes(seat, positions)
    both = (long_change > 0) & (short_change < 0)
    exits = {
        '固定': None,
        '反向双变动': (long_change < 0) & (short_change > 0),
        '净多转负': long_change < short_change,
    }
    results = {}
    for hold, floor, exit_name in product((0, *HOLDS), FLOW_FLOORS, exits):
        if hold == 0 and exit_name != '固定':
            continue
        entries = both & (long_change >= floor) & (-short_change >= floor)
        rule = archive.Strategy('flow-exit', seat, seat, hold, 0, 0)
        results[hold, floor, exit_name] = backtest(rule, entries, None, exit_signals=exits[exit_name])

    eligible = [key for key, periods in results.items()
                if all(periods[phase]['trades'] >= 20 and periods[phase]['total_pct'] > 0
                       for phase in ('前段', '后段'))]
    ordered = sorted(eligible, key=lambda key: min(results[key]['前段']['sharpe'],
                                                   results[key]['后段']['sharpe']), reverse=True)
    print(f'\n{seat} | {len(results)} 组 | 收益 / 夏普 / 回撤 / 笔数')
    print('H | 两侧最低手数 | 退出 | 全样本 | 前段 | 后段 | 2024 | 2025 | 2026')
    controls = [(0, 0, '固定'), (1, 0, '固定'), (3, 0, '固定'), (3, 0, '净多转负')]
    for key in dict.fromkeys([*ordered[:3], *controls]):
        print_periods([str(part) for part in key], results[key])


def portfolio_uncertainty(first, second):
    """Resample matched 20-session blocks of both sleeves together."""
    size, block, draws = len(first), 20, 3000
    rng = np.random.default_rng(20261125)
    starts = rng.integers(0, size - block + 1, size=(draws, (size + block - 1) // block))
    indices = (starts[:, :, None] + np.arange(block)).reshape(draws, -1)[:, :size]
    benchmark = first[indices]
    combined = (benchmark + second[indices]) / 2

    def sharpe(pnl):
        return pnl.mean(axis=1) / pnl.std(axis=1, ddof=1) * np.sqrt(243)

    return (np.percentile((combined - benchmark).sum(axis=1), (2.5, 97.5)),
            np.percentile(sharpe(combined) - sharpe(benchmark), (2.5, 97.5)))


def report_portfolio(close, positions, run, periods):
    five_day = archive.STRATEGIES[1]
    flow = archive.PA_FLOW3
    entries, ranks = archive.signal_inputs(five_day, close, positions)
    flow_entries, _ = archive.signal_inputs(flow, close, positions)
    print('\n平安等权组合 | 各占一半固定名义仓位，日盈亏相关性及收益 / 夏普 / 回撤')
    print('区间 | 相关性 | 五日转弱 | 多增空减三日 | 等权')
    for name, lo, hi in periods:
        first = run(five_day, entries, ranks, lo, hi)[0]
        second = run(flow, flow_entries, None, lo, hi)[0]
        cells = []
        for pnl in (first, second, (first + second) / 2):
            m = archive.metrics(pnl, 0, 0)
            cells.append(f"{m['total_pct']:.3f}% / {m['sharpe']:.3f} / {m['drawdown_pct']:.3f}%")
        correlation = np.corrcoef(first, second)[0, 1]
        print(' | '.join([name, f'{correlation:.3f}', *cells]))
        if name in ('全样本', '后段'):
            gain, sharpe = portfolio_uncertainty(first, second)
            print(f'  相对五日转弱的 95% 区间 | 收益差 {gain[0]:+.3f} 到 {gain[1]:+.3f} 个百分点'
                  f' | 夏普差 {sharpe[0]:+.3f} 到 {sharpe[1]:+.3f}')


def report_neighborhood(close, positions, run, periods):
    windows = periods[:3]
    five = archive.STRATEGIES[1]
    flow = archive.PA_FLOW3
    front = archive.TOP20_FLOW3
    five_entries, five_ranks = archive.signal_inputs(five, close, positions)
    flow_entries, _ = archive.signal_inputs(flow, close, positions)
    front_entries, _ = archive.signal_inputs(front, close, positions)
    long_change, short_change = holding_changes(front.seat, positions)
    negative = long_change < short_change
    twice = negative & np.r_[False, negative[:-1]]
    fixed_leg = {label: run(five, five_entries, five_ranks, lo, hi) for label, lo, hi in windows}

    print('\n邻近参数 | 全样本、前段、后段的累计收益 / 夏普 / 最大回撤 / 笔数')
    for hold in (2, 3, 4):
        varied = replace(flow, hold=hold)
        paths = {label: run(varied, flow_entries, None, lo, hi) for label, lo, hi in windows}
        for weight in (.4, .5, .6):
            metrics = {
                label: archive.metrics(weight * fixed_leg[label][0] + (1 - weight) * pnl,
                                       fixed_leg[label][1] + count, 0)
                for label, (pnl, count, _) in paths.items()
            }
            print_periods(['平安组合', f'多增空减 H={hold}', f'转弱版 {weight:.0%}'], metrics)
        varied = replace(front, hold=hold)
        for label, mask in [('单日转负', negative), ('连续两日为负', twice)]:
            metrics = {window: archive.metrics(*run(varied, front_entries, None, lo, hi, exit_signals=mask))
                       for window, lo, hi in windows}
            print_periods(['前二十', f'H={hold}', label], metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--direction', action='store_true', help='比较前五、前十、前二十的不设分位入场规则')
    mode.add_argument('--exit-study', action='store_true', help='比较多增空减的手数门槛、退出规则与等权组合')
    args = parser.parse_args()
    seats = (*FLOW_SEATS, '平安期货') if args.exit_study else FLOW_SEATS if args.direction else SEATS
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
    holds = (0, 1, 2, 3, 4, 5, 10) if args.exit_study else HOLDS
    plans = {hold: archive.contract_plans(dates, contracts, quotes, hold) for hold in holds}

    def run(rule, entries, ranks, lo, hi, *, exit_signals=None):
        return archive.simulate(rule, dates, quotes, plans[rule.hold], entries, ranks, lo, hi,
                                exit_signals=exit_signals)

    def backtest(rule, entries, ranks, *, exit_signals=None):
        return {
            label: archive.metrics(*run(rule, entries, ranks, lo, hi, exit_signals=exit_signals))
            for label, lo, hi in periods
        }

    if args.exit_study:
        for seat in seats:
            report_exit_study(seat, positions, backtest)
        report_portfolio(close, positions, run, periods)
        report_neighborhood(close, positions, run, periods)
        return

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
