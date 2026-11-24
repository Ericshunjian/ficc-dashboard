"""Fetch and validate public, end-of-day TS data from CFFEX."""
import csv
import io
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

BASE = Path(__file__).resolve().parent
INPUTS = BASE / 'TS_交易所日更.json'
EXCHANGE = 'http://www.cffex.com.cn'
CHINA = timezone(timedelta(hours=8))
SEATS = ('平安期货', '中信期货', '华泰期货', '国泰君安', '海通期货')


def read_inputs():
    return json.loads(INPUTS.read_text(encoding='utf-8')) if INPUTS.exists() else {}


def fetch(path):
    try:
        with urlopen(EXCHANGE + path, timeout=12) as response:
            if '/error_page/' in response.geturl():
                return None
            return response.read()
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def parse_ranks(body, day):
    root = ET.fromstring(body)
    if root.tag != 'positionRank':
        raise ValueError('持仓榜格式异常')
    sides = {1: defaultdict(int), 2: defaultdict(int)}
    parties = {}
    for item in root:
        row = {field.tag.lower(): field.text for field in item}
        side = row.get('datatypeid')
        if side not in ('1', '2') or not (row.get('rank') or '').isdigit():
            continue
        if not 1 <= int(row['rank']) <= 20:
            continue
        if row.get('tradingday') != day or not re.fullmatch(r'TS\d{4}', row['instrumentid']):
            raise ValueError(f'持仓榜日期或品种异常：{day}')
        party = row['partyid']
        if not party:
            raise ValueError('持仓榜缺少会员编号')
        sides[int(side)][party] += int(row['volume'])
        parties[party] = row['shortname'].split('(')[0].strip()
    if min(map(len, sides.values())) < 20:
        raise ValueError(f'持仓榜缺少前 20 名：{day}')
    top = {}
    for n in (5, 10, 20):
        top[f'前{n}大合计'] = [sum(sorted(vals.values(), reverse=True)[:n]) for vals in sides.values()]
    for name in SEATS:
        matches = [party for party, label in parties.items() if label == name]
        if matches:
            top[name] = [sum(vals.get(party, 0) for party in matches) for vals in sides.values()]
    return top


def parse_prices(body):
    reader = csv.DictReader(io.StringIO(body.decode('gb18030')))
    expected = {'合约代码', '今开盘', '今收盘', '成交量'}
    if not expected.issubset(reader.fieldnames or ()):
        raise ValueError('日行情列名异常')
    prices = {}
    for row in reader:
        contract = row['合约代码'].strip()
        if not re.fullmatch(r'TS\d{4}', contract):
            continue
        if (row['今开盘'] or '').strip().lower() in ('', 'null'):
            continue
        numbers = [float(row['今开盘']), float(row['今收盘']), int(row['成交量'])]
        if not all(math.isfinite(v) and v > 0 for v in numbers):
            raise ValueError(f'TS 日行情数值异常：{contract}')
        prices[contract] = numbers
    if not prices:
        raise ValueError('日行情缺少 TS 合约')
    return prices


def fetch_session(day):
    stamp = day.strftime('%Y%m%d')
    directory = f'{stamp[:6]}/{stamp[6:]}'
    prices = fetch(f'/sj/hqsj/rtj/{directory}/{stamp}_1.csv')
    ranks = fetch(f'/sj/ccpm/{directory}/TS.xml')
    if prices is None and ranks is None:
        return None
    if prices is None or ranks is None:
        raise ValueError(f'{day} 的行情或持仓榜尚未齐全')
    return {'positions': parse_ranks(ranks, stamp), 'prices': parse_prices(prices)}


def refresh_inputs(anchor, target):
    existing = read_inputs()
    updated = dict(existing)
    # Always reload the target, so an exchange correction can replace the saved close.
    start = max(anchor, date.fromisoformat(max(existing, default=anchor.isoformat())))
    day = min(start, target)
    while day <= target:
        if day.weekday() < 5:
            session = fetch_session(day)
            if session is not None:
                updated[day.isoformat()] = session
        day += timedelta(days=1)
    if updated != existing:
        temp = INPUTS.with_suffix('.tmp')
        temp.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temp.replace(INPUTS)
    return updated


def today():
    return datetime.now(CHINA).date()
