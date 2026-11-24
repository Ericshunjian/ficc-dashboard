"""Replay the frozen, zero-cost TS research rules and their matched controls."""
from datetime import date
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spider_archive_rules as archive
import spider_daily_data as exchange
import spider_expiry_benchmark as expiry
import spider_summary as summary


def main():
    snapshots = {day: row for day, row in exchange.read_inputs().items() if day <= archive.ASOF}
    dates, _, close, positions = summary.load_data(snapshots, date.fromisoformat(archive.ASOF))
    end = int(np.searchsorted(dates, archive.ASOF, side='right')) - 1
    dates = dates[:end + 1]
    if not dates or dates[-1] != archive.ASOF:
        raise ValueError(f'归案行情需要覆盖到 {archive.ASOF}')
    contracts = archive.contract_ranks(dates, snapshots)
    quotes = expiry.read_quotes(snapshots)
    split = int(np.searchsorted(dates, archive.SPLIT))
    ranges = [('全样本', 0, end), ('2021–2023', 0, split - 1), ('2024 年起', split, end)]
    for year in (2024, 2025, 2026):
        lo = int(np.searchsorted(dates, f'{year}-01-01'))
        hi = int(np.searchsorted(dates, f'{year + 1}-01-01')) - 1
        ranges.append((str(year), lo, hi))
    plans = {hold: archive.contract_plans(dates, contracts, quotes, hold)
             for hold in {strategy.hold for strategy in archive.STRATEGIES}}
    print('编号 | 区间 | 累计收益 | 夏普 | 最大回撤 | 笔数 | 在场')
    for strategy in archive.STRATEGIES:
        for rule in (strategy, strategy.control()):
            entries, ranks = archive.signal_inputs(rule, close, positions)
            for label, lo, hi in ranges:
                result = archive.metrics(*archive.simulate(
                    rule, dates, quotes, plans[rule.hold], entries, ranks, lo, hi,
                ))
                print(f'{rule.key} | {label} | {result["total_pct"]:.3f}% | {result["sharpe"]:.3f}'
                      f' | {result["drawdown_pct"]:.3f}% | {result["trades"]} | {result["exposure_pct"]:.1f}%')


if __name__ == '__main__':
    main()
