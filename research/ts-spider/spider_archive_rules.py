"""Zero-cost TS archive rules, marked on actual contract prices."""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import spider_expiry_benchmark as expiry
import spider_pingan_signal as signal

RANKS = Path(__file__).resolve().with_name('strategy-archive') / 'ts-contracts-by-volume.tsv'
ASOF = '2026-11-24'
SPLIT = '2024-01-01'
# 国办发明电〔2026〕10号。周末按周末休市，周末调休不补交易日。
HOLIDAYS_2027 = np.concatenate([
    np.arange(start, end, dtype='datetime64[D]')
    for start, end in (
        ('2027-01-01', '2027-01-04'), ('2027-02-05', '2027-02-14'),
        ('2027-04-03', '2027-04-06'), ('2027-05-01', '2027-05-06'),
        ('2027-06-09', '2027-06-10'), ('2027-09-15', '2027-09-16'),
        ('2027-10-01', '2027-10-08'),
    )
])


@dataclass(frozen=True)
class Strategy:
    key: str
    name: str
    seat: str
    hold: int
    entry_rank: float
    cooldown: int
    exit_rank: float | None = None
    trend_window: int = 0
    require_down: bool = True
    flow_entry: bool = False
    negative_exit: bool = False

    def control(self):
        return replace(self, key=f'{self.key}-control', exit_rank=None, trend_window=0, negative_exit=False)


@dataclass(frozen=True)
class Portfolio:
    key: str
    name: str
    legs: tuple[Strategy, Strategy]


STRATEGIES = (
    Strategy('PA-DIP3-EXIT20-v1', '平安回踩三日·转弱退出', '平安期货', 3, .80, 1, exit_rank=.20),
    Strategy('PA-DIP5-EXIT20-v1', '平安回踩五日·转弱退出', '平安期货', 5, .75, 0, exit_rank=.20),
    Strategy('TOP20-TREND60-DIP5-v1', '前二十顺势回踩五日', '前20大合计', 5, .80, 0, trend_window=60),
)
PA_FLOW3 = Strategy('PA-FLOW3-v1', '平安多增空减·三日', '平安期货', 3, 0, 0,
                    require_down=False, flow_entry=True)
TOP20_FLOW3 = Strategy('TOP20-FLOW3-NEG-v1', '前二十多增空减·净多转负退出', '前20大合计', 3, 0, 0,
                       require_down=False, flow_entry=True, negative_exit=True)
FEATURED = (
    Portfolio('PA-COMB50-v1', '平安等权组合', (STRATEGIES[1], PA_FLOW3)),
    STRATEGIES[1],
    Strategy('TOP5-TREND60-DIP5-Q70-v1', '前五顺势回踩五日', '前5大合计', 5, .70, 0, trend_window=60),
    STRATEGIES[2],
    TOP20_FLOW3,
)
ACTIVE_RULES = tuple(dict.fromkeys(rule for item in FEATURED
                                   for rule in (item.legs if isinstance(item, Portfolio) else (item,))))


def contract_ranks(dates, snapshots):
    ranks = pd.read_csv(RANKS, sep='\t', dtype=str)
    contracts = dict(zip(ranks['date'], ranks['contracts'].str.split()))
    for day, session in snapshots.items():
        if day > ASOF:
            prices = session['prices']
            contracts[day] = sorted(prices, key=lambda contract: (-prices[contract][2], contract))
    missing = set(dates) - contracts.keys()
    if missing:
        raise ValueError(f'缺少 {min(missing)} 的 TS 合约成交量排序')
    return contracts


def choose_contract(contracts, planned_exit):
    return next((contract for contract in contracts
                 if expiry.scheduled_expiry(contract).isoformat() > planned_exit), None)


def contract_plans(dates, contracts, quotes, hold):
    """Choose from signal-day volume ranks and lock the contract through exit."""
    plans = {}
    for t, day in enumerate(dates[:-(hold + 1)]):
        entry, planned_exit = t + 1, t + 1 + hold
        contract = choose_contract(contracts[day], dates[planned_exit])
        if contract is None:
            continue
        quote = quotes[dates[entry]].get(contract)
        if quote is None or quote[0] is None:
            continue
        plans[t] = (entry, planned_exit, contract)
    return plans


def holding_changes(rows):
    return np.diff(np.asarray(rows, dtype=float), axis=0, prepend=np.full((1, 2), np.nan))


def signal_inputs(strategy, close, positions):
    values = np.asarray(positions[strategy.seat], dtype=float)
    if strategy.flow_entry:
        long, short = holding_changes(values).T
        ranks = np.full(len(values), np.nan)
        entries = (long > 0) & (short < 0)
    else:
        daily_net = np.diff(values[:, 0] - values[:, 1], prepend=np.nan)
        ranks = signal.rolling_pct(daily_net, win=60, min_obs=30)
        entries = ranks >= strategy.entry_rank
    if strategy.require_down:
        entries &= np.diff(close, prepend=np.nan) < 0
    if strategy.trend_window:
        yesterday = pd.Series(close).shift(1)
        average = yesterday.rolling(strategy.trend_window, min_periods=strategy.trend_window).mean()
        entries &= (yesterday > average).to_numpy()
    return entries, ranks


def simulate(strategy, dates, quotes, plans, entries, ranks, start, end, *, exit_signals=None):
    """Return daily fixed-notional point P&L, filled trades and exposure days."""
    if exit_signals is None and strategy.exit_rank is not None:
        exit_signals = ranks < strategy.exit_rank
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
        if exit_signals is not None:
            for j in range(entry, exit_day):
                if exit_signals[j]:
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


def forward_signal(strategy, day, contracts, entry, rank, price_change, *, holding_change=None):
    contract = None
    if entry:
        cutoff = np.busday_offset(day, strategy.hold + 1, roll='forward', holidays=HOLIDAYS_2027)
        if cutoff >= np.datetime64('2028-01-01'):
            # Allow for a holiday closure until the next yearly calendar is available.
            cutoff += np.timedelta64(9, 'D')
        contract = choose_contract(contracts, str(cutoff))
    valid = np.isfinite(rank) and (not strategy.require_down or np.isfinite(price_change))
    weak = strategy.exit_rank is not None and rank < strategy.exit_rank
    if strategy.flow_entry:
        valid = bool(np.isfinite(holding_change).all())
    if strategy.negative_exit:
        long, short = holding_change
        weak = long < short
    return {
        'raw': '多' if entry else '无触发' if valid else '数据缺失',
        'buy_contract': contract, 'weak': bool(weak),
        'pct': round(float(rank), 6) if np.isfinite(rank) else None,
    }


def simulate_forward(strategy, dates, quotes, signals, start):
    """Run the first-published inputs from flat, in open/close order."""
    pnl = np.zeros(len(dates) - start)
    trades = []
    active = None
    buy_contract = None
    sell_at_open = False
    state = {'position': '空仓', 'entry': None, 'contract': None,
             'action': '下一交易日起观察'}
    for i in range(start, len(dates)):
        day = dates[i]
        opening = quotes[day].get(buy_contract, (None,))[0] if buy_contract else None
        if opening is not None:
            active = {'contract': buy_contract, 'signal': dates[i - 1], 'entry': day,
                      'exit_idx': i + strategy.hold, 'opening': opening, 'mark': opening}

        closed = False
        if active is not None:
            opening, close, settlement = quotes[day][active['contract']]
            expiry_day = expiry.scheduled_expiry(active['contract']).isoformat()
            exit_price = None
            if sell_at_open and opening is not None:
                exit_price = opening
                exit_reason = '净多日变动转负开盘' if strategy.negative_exit else '转弱开盘'
            elif day >= expiry_day:
                exit_price, exit_reason = settlement, '到期结算'
            elif i == active['exit_idx']:
                exit_price, exit_reason = close, '计划日收盘'
            pnl[i - start] = (close if exit_price is None else exit_price) - active['mark']
            if exit_price is None:
                active['mark'] = close
            else:
                trades.append({'策略key': strategy.key, '策略': strategy.name, '合约': active['contract'],
                               '信号日': active['signal'], '开仓日': active['entry'], '平仓日': day,
                               '方向': '多', '盈亏点数': round(exit_price - active['opening'], 5),
                               '平仓原因': exit_reason, '状态': '已兑现'})
                active, closed = None, True

        signal_today = signals[day]
        buy_contract = None
        sell_at_open = False
        if active is not None:
            sell_at_open = signal_today['weak']
            action = ('下个交易日开盘平仓' if sell_at_open else
                      '下个交易日收盘平仓' if active['exit_idx'] == i + 1 else '继续持多')
        elif closed and strategy.cooldown:
            action = '平仓日跳过新信号'
        else:
            buy_contract = signal_today['buy_contract']
            if buy_contract:
                action = '下个交易日开盘做多'
            elif signal_today['raw'] == '多':
                action = '等待合约'
            else:
                action = '待数据' if signal_today['raw'] == '数据缺失' else '维持空仓'
        state = {'position': '多' if active else '空仓',
                 'entry': active['entry'] if active else None,
                 'contract': active['contract'] if active else buy_contract, 'action': action}
    return pnl, trades, state
