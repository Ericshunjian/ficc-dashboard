"""Build a TS buy-and-roll-at-expiry benchmark from exchange contract prices."""
import argparse
import csv
import io
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from zipfile import ZipFile

import spider_daily_data as exchange

ARCHIVE = Path(__file__).resolve().with_name('TS_到期持有行情.csv')
COLUMNS = ('date', 'contract', 'open', 'mark', 'settlement')
NAME = '持有到期基准'


def scheduled_expiry(contract):
    month_start = date(2000 + int(contract[2:4]), int(contract[4:]), 1)
    return month_start + timedelta(days=(4 - month_start.weekday()) % 7 + 7)


def read_quotes(snapshots):
    quotes = {}
    with ARCHIVE.open(encoding='utf-8', newline='') as source:
        for row in csv.DictReader(source):
            opening = float(row['open']) if row['open'] else None
            quotes.setdefault(row['date'], {})[row['contract']] = (
                opening, float(row['mark']), float(row['settlement']),
            )
    for day, session in snapshots.items():
        quotes.setdefault(day, {}).update(session.get('expiry_quotes', {}))
    return quotes


def net_values(dates, quotes, cost, price_ref):
    """Open the nearest expiry at the open and value expiry at exchange settlement."""
    nav = []
    current = None
    previous_close = 0.0
    points = 0.0
    for iso_day in dates:
        day = date.fromisoformat(iso_day)
        prices = quotes.get(iso_day)
        if not prices:
            raise ValueError(f'{iso_day} 缺少持有到期基准的中金所合约行情')
        if current is None:
            contract = min(prices)  # TS contract codes are chronological YYMM.
            opening = prices[contract][0]
            if opening is not None:
                current = contract
                expiry = scheduled_expiry(current)
                previous_close = opening
        if current is not None:
            if current not in prices:
                raise ValueError(f'{iso_day} 缺少在仓合约 {current} 的中金所行情')
            _, close, settlement = prices[current]
            if day >= expiry:
                points += settlement - previous_close - cost
                current = None
            else:
                points += close - previous_close
                previous_close = close
        nav.append(round(1.0 + points / price_ref, 7))
    return nav


def download_month(month):
    body = exchange.fetch(f'/sj/historysj/{month}/zip/{month}.zip')
    if body is None:
        raise ValueError(f'中金所缺少 {month} 月度行情归档')
    rows = []
    with ZipFile(io.BytesIO(body)) as archive:
        for filename in archive.namelist():
            match = re.fullmatch(r'(\d{4})(\d{2})(\d{2})_1\.csv', Path(filename).name)
            if not match:
                continue
            day = '-'.join(match.groups())
            for contract, prices in exchange.parse_expiry_quotes(archive.read(filename)).items():
                rows.append((day, contract, *prices))
    return rows


def download_history(start, end):
    months = sorted({(start + timedelta(days=i)).strftime('%Y%m') for i in range((end - start).days + 1)})
    month_rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for rows in pool.map(download_month, months):
            month_rows.extend(rows)
    selected = sorted(row for row in month_rows if start.isoformat() <= row[0] <= end.isoformat())
    if not selected:
        raise ValueError('历史区间没有 TS 合约行情')
    temp = ARCHIVE.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8', newline='') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(COLUMNS)
        writer.writerows(selected)
    temp.replace(ARCHIVE)
    print(f'中金所 TS 月度行情：{len({r[0] for r in selected})} 个交易日，{len(selected)} 条合约报价')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=date.fromisoformat, required=True)
    parser.add_argument('--end', type=date.fromisoformat, required=True)
    args = parser.parse_args()
    if args.start > args.end:
        parser.error('开始日期应早于或等于结束日期')
    download_history(args.start, args.end)


if __name__ == '__main__':
    main()
