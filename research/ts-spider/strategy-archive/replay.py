"""Replay the frozen, zero-cost TS research rules and their matched controls."""
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import spider_daily_data as exchange
import spider_expiry_benchmark as expiry
import spider_pingan_signal as signal
import spider_summary as summary

ASOF = '2026-11-24'
SPLIT = '2024-01-01'


@dataclass(frozen=True)
class Strategy:
    key: str
    seat: str
    hold: int
    entry_rank: float
    cooldown: int
    exit_rank: float | None = None
    trend_window: int = 0

    def control(self):
        return replace(self, key=f'{self.key}-control', exit_rank=None, trend_window=0)


STRATEGIES = (
    Strategy('PA-DIP3-EXIT20-v1', '平安期货', 3, .80, 1, exit_rank=.20),
    Strategy('PA-DIP5-EXIT20-v1', '平安期货', 5, .75, 0, exit_rank=.20),
    Strategy('TOP20-TREND60-DIP5-v1', '前20大合计', 5, .80, 0, trend_window=60),
)


def load_data():
    snapshots = {day: row for day, row in exchange.read_inputs().items() if day <= ASOF}
    dates, _, close, positions = summary.load_data(snapshots, date.fromisoformat(ASOF))
    end = int(np.searchsorted(dates, ASOF, side='right'))
    dates = dates[:end]
    if not dates or dates[-1] != ASOF:
        raise ValueError(f'归案行情需要覆盖到 {ASOF}')
    ranks = pd.read_csv(HERE / 'ts-contracts-by-volume.tsv', sep='\t', dtype=str)
    contracts = dict(zip(ranks['date'], ranks['contracts'].str.split()))
    quotes = expiry.read_quotes(snapshots)
    missing = (set(dates) - contracts.keys()) | (set(dates) - quotes.keys())
    if missing:
        raise ValueError(f'缺少 {min(missing)} 的合约排序或报价')
    return dates, close[:end], {seat: rows[:end] for seat, rows in positions.items()}, contracts, quotes


def contract_plans(dates, contracts, quotes, hold):
    """Choose from signal-day volume ranks and lock the contract through exit."""
    traded_contracts = {contract for day in dates for contract in contracts[day]}
    expiries = {contract: expiry.scheduled_expiry(contract).isoformat() for contract in traded_contracts}
    plans = {}
    for t, day in enumerate(dates[:-(hold + 1)]):
        entry, planned_exit = t + 1, t + 1 + hold
        contract = next((c for c in contracts[day] if expiries[c] > dates[planned_exit]), None)
        if contract is None:
            continue
        quote = quotes[dates[entry]].get(contract)
        if quote is None or quote[0] is None:
            continue
        plans[t] = (entry, planned_exit, contract)
    return plans


def signal_inputs(strategy, close, positions):
    values = np.asarray(positions[strategy.seat], dtype=float)
    daily_net = np.diff(values[:, 0] - values[:, 1], prepend=np.nan)
    ranks = signal.rolling_pct(daily_net, win=60, min_obs=30)
    entries = (np.diff(close, prepend=np.nan) < 0) & (ranks >= strategy.entry_rank)
    if strategy.trend_window:
        yesterday = pd.Series(close).shift(1)
        average = yesterday.rolling(strategy.trend_window, min_periods=strategy.trend_window).mean()
        entries &= (yesterday > average).to_numpy()
    return entries, ranks


def simulate(strategy, dates, quotes, plans, entries, ranks, start, end):
    """Return daily fixed-notional point P&L, filled trades and exposure days."""
    pnl = np.zeros(end - start + 1)
    trades, exposure = 0, 0
    free = start
    for t in np.flatnonzero(entries[start:end + 1]) + start:
        if t < free or t not in plans:
            continue
        entry, exit_day, contract = plans[t]
        if exit_day > end:
            continue
        entry_price = quotes[dates[entry]][contract][0]
        exit_price = quotes[dates[exit_day]][contract][1]
        if strategy.exit_rank is not None:
            for j in range(entry, exit_day):
                if ranks[j] < strategy.exit_rank:
                    next_open = quotes[dates[j + 1]][contract][0]
                    if next_open is not None:
                        exit_day, exit_price = j + 1, next_open
                        break
        marks = [quotes[dates[j]][contract][1] for j in range(entry, exit_day)]
        pnl[entry - start:exit_day - start + 1] += np.diff([entry_price, *marks, exit_price])
        free = exit_day + strategy.cooldown
        trades += 1
        exposure += exit_day - entry + 1
    return pnl, trades, exposure


def metrics(pnl, trades, exposure):
    nav = 1 + np.cumsum(pnl) / 100
    dd = 1 - nav / np.maximum.accumulate(np.r_[1, nav])[1:]
    stdev = pnl.std(ddof=1) if len(pnl) > 1 else 0
    return {
        'total_pct': pnl.sum(),
        'sharpe': pnl.mean() / stdev * np.sqrt(243) if stdev else 0,
        'drawdown_pct': 100 * dd.max(),
        'trades': trades,
        'exposure_pct': 100 * exposure / len(pnl),
    }


def main():
    dates, close, positions, contracts, quotes = load_data()
    split = int(np.searchsorted(dates, SPLIT))
    end = len(dates) - 1
    ranges = (('全样本', 0, end), ('2021–2023', 0, split - 1), ('2024 年起', split, end))
    for year in (2024, 2025, 2026):
        lo = int(np.searchsorted(dates, f'{year}-01-01'))
        hi = int(np.searchsorted(dates, f'{year + 1}-01-01')) - 1
        ranges += ((str(year), lo, hi),)
    plans = {hold: contract_plans(dates, contracts, quotes, hold)
             for hold in {strategy.hold for strategy in STRATEGIES}}
    print('编号 | 区间 | 累计收益 | 夏普 | 最大回撤 | 笔数 | 在场')
    for strategy in STRATEGIES:
        for rule in (strategy, strategy.control()):
            entries, ranks = signal_inputs(rule, close, positions)
            for label, lo, hi in ranges:
                result = metrics(*simulate(rule, dates, quotes, plans[rule.hold], entries, ranks, lo, hi))
                print(f'{rule.key} | {label} | {result["total_pct"]:.3f}% | {result["sharpe"]:.3f}'
                      f' | {result["drawdown_pct"]:.3f}% | {result["trades"]} | {result["exposure_pct"]:.1f}%')


if __name__ == '__main__':
    main()
