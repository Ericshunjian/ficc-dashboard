# -*- coding: utf-8 -*-
"""
股债相关性数据源：沪深300 + 债券（国债ETF/国债指数）+ 国债期限利差

数据源：
  - 股票/债券日线：东方财富 push2his（前复权，含分红再投，近似全收益）
  - 期限利差：本地 yield_curve_data.json（Wind 导出的国债收益率曲线，日频）

输出：stock_bond_data.json（前端按所选窗口实时计算滚动相关系数）

注意：
  - 盘中不采未收盘数据（15:30 前丢弃当日 bar）
  - 写入走 jsonio.write_json_skip_unchanged（幂等，无变化不重写）
"""
import os
import json
import datetime
import logging

import requests

BASE = os.path.dirname(os.path.abspath(__file__))
CURVE_JSON = os.path.join(BASE, 'yield_curve_data.json')
OUT_JSON = os.path.join(BASE, 'stock_bond_data.json')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('stock_bond')

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
EM_KLINE = ('https://push2his.eastmoney.com/api/qt/stock/kline/get'
            '?secid={secid}&fields1=f1,f2,f3&fields2=f51,f52,f53,f54,f55,f56,f57'
            '&klt=101&fqt=1&beg=19900101&end=20500101&lmt=10000')

# 标的池：股票 + 债券（顺序即前端展示顺序）；东财 secid + 腾讯代码（兜底源）
INSTRUMENTS = [
    ('沪深300', '1.000300', 'sh000300'),
    ('十年国债ETF(511260)', '1.511260', 'sh511260'),
    ('中债30年国债财富ETF(511090)', '1.511090', 'sh511090'),
    ('五年国债ETF(511010)', '1.511010', 'sh511010'),
    ('国债ETF平安(511020)', '1.511020', 'sh511020'),
    ('上证国债指数(000012)', '1.000012', 'sh000012'),
]

# 期限利差定义（基于本地国债收益率曲线，单位 bp）
SPREAD_DEFS = [
    ('30Y-10Y', '30年国债', '10年国债'),
    ('10Y-1Y', '10年国债', '1年国债'),
    ('30Y-1Y', '30年国债', '1年国债'),
    ('5Y-1Y', '5年国债', '1年国债'),
]

MARKET_CLOSE_MIN = 15 * 60 + 30  # 15:30 前认为当日未收盘，丢弃当日 bar


def _settled_cutoff_date():
    """返回可用的最后一个已收盘交易日（含）；盘中则丢弃今天。"""
    now = datetime.datetime.now()
    today = now.date()
    if now.hour * 60 + now.minute < MARKET_CLOSE_MIN:
        return today - datetime.timedelta(days=1)
    return today


def _http_get_json(url, timeout=30):
    """requests 优先，失败换 urllib（沙箱代理对部分域名时通时断）"""
    try:
        return requests.get(url, timeout=timeout, headers=HEADERS).json()
    except Exception:
        import urllib.request
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))


def fetch_kline(secid, name, retries=4):
    """东方财富日线（前复权）→ (dates, close)；失败重试（带退避），全失败抛异常"""
    last_err = None
    for i in range(retries):
        try:
            data = _http_get_json(EM_KLINE.format(secid=secid)).get('data') or {}
            klines = data.get('klines') or []
            dates, closes = [], []
            cutoff = _settled_cutoff_date()
            for line in klines:
                p = line.split(',')
                d = datetime.date(int(p[0][:4]), int(p[0][5:7]), int(p[0][8:10]))
                if d > cutoff:
                    continue
                try:
                    c = float(p[2])
                except ValueError:
                    continue
                if c <= 0:
                    continue
                dates.append(p[0])
                closes.append(round(c, 6))
            if not dates:
                raise RuntimeError('返回空数据')
            logger.info('  %s: %d 条, %s ~ %s%s', name, len(dates), dates[0], dates[-1],
                        f'（第{i + 1}次尝试）' if i else '')
            return dates, closes
        except Exception as e:
            last_err = e
            logger.warning('  %s 第%d次抓取失败: %s', name, i + 1, str(e)[:120])
            import time
            time.sleep(0.8 * (i + 1))
    raise RuntimeError(f'{name} 连续{retries}次抓取失败: {last_err}')


def fetch_tencent(code, name):
    """腾讯兜底源：fqkline 前复权日线（上限约1000根）"""
    url = f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,1000,qfq'
    r = requests.get(url, timeout=30, headers=HEADERS)
    d = r.json().get('data', {}).get(code, {})
    k = d.get('qfqday') or d.get('day') or []
    if not k:
        raise RuntimeError('返回空数据')
    dates, closes = [], []
    cutoff = _settled_cutoff_date()
    for row in k:
        ds = row[0]
        d = datetime.date(int(ds[:4]), int(ds[5:7]), int(ds[8:10]))
        if d > cutoff:
            continue
        c = float(row[2])
        if c <= 0:
            continue
        dates.append(ds)
        closes.append(round(c, 6))
    if not dates:
        raise RuntimeError('全部被过滤')
    logger.info('  %s[腾讯兜底]: %d 条, %s ~ %s', name, len(dates), dates[0], dates[-1])
    return dates, closes


def fetch_instrument(name, secid, tcode, retries=3):
    """东财为主，腾讯兜底"""
    try:
        return fetch_kline(secid, name, retries), 'eastmoney'
    except Exception as e:
        logger.warning('  %s 东财失败，尝试腾讯兜底: %s', name, str(e)[:120])
        try:
            return fetch_tencent(tcode, name), 'tencent'
        except Exception as e2:
            raise RuntimeError(f'东财与腾讯均失败: {e2}')


def _merge_with_existing(prices):
    """合并保护：抓取失败或明显变短的序列保留存量，防止部分失败覆盖好数据"""
    old = None
    if os.path.exists(OUT_JSON):
        try:
            with open(OUT_JSON, encoding='utf-8') as f:
                old = json.load(f)
        except Exception:
            old = None
    if not old:
        return prices
    oldp = old.get('prices', {})
    for name in list(prices.keys()):
        new_len = len(prices[name]['dates'])
        old_len = len(oldp.get(name, {}).get('dates', []))
        # 容忍 5 天以内差异（正常增量），明显变短视为抓取不完整
        if old_len and new_len < old_len - 5:
            logger.warning('  %s 新抓取(%d条)明显短于存量(%d条)，保留存量', name, new_len, old_len)
            prices[name] = oldp[name]
    for name, p in oldp.items():
        if name not in prices:
            logger.warning('  %s 本次抓取失败，保留存量(%d条)', name, len(p['dates']))
            prices[name] = p
    return prices


def load_spreads():
    """从本地收益率曲线读取各期限利差（%）→ bp"""
    with open(CURVE_JSON, encoding='utf-8') as f:
        curve = json.load(f)
    series = curve.get('series', {})
    out = {}
    for label, long_name, short_name in SPREAD_DEFS:
        if long_name not in series or short_name not in series:
            logger.warning('  曲线缺少 %s / %s，跳过利差 %s', long_name, short_name, label)
            continue
        L, S = series[long_name], series[short_name]
        s_map = dict(zip(S['dates'], S['values']))
        dates, vals = [], []
        for d, v in zip(L['dates'], L['values']):
            if v is None or d not in s_map or s_map[d] is None:
                continue
            dates.append(d)
            vals.append(round((v - s_map[d]) * 100, 4))  # % → bp
        out[label] = {'dates': dates, 'values': vals}
        logger.info('  利差 %s: %d 条, %s ~ %s', label, len(dates), dates[0], dates[-1])
    return out


def main():
    logger.info('股债数据源更新开始')
    prices = {}
    sources = {}
    for name, secid, tcode in INSTRUMENTS:
        try:
            (d, c), src = fetch_instrument(name, secid, tcode)
            if d:
                prices[name] = {'dates': d, 'close': c}
                sources[name] = src
        except Exception as e:
            logger.warning('  %s: %s', name, e)
        import time
        time.sleep(0.5)

    prices = _merge_with_existing(prices)
    # 存量保留的序列，来源标记沿用
    for name in prices:
        sources.setdefault(name, 'kept-existing')

    spreads = {}
    if os.path.exists(CURVE_JSON):
        try:
            spreads = load_spreads()
        except Exception as e:
            logger.warning('  利差读取失败: %s', e)
    else:
        logger.warning('  未找到 %s，跳过利差', CURVE_JSON)

    all_dates = []
    for p in prices.values():
        all_dates.extend(p['dates'])
    for s in spreads.values():
        all_dates.extend(s['dates'])

    payload = {
        'meta': {
            'last_updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'date_range': [min(all_dates), max(all_dates)] if all_dates else None,
            'instruments': list(prices.keys()),
            'spreads': list(spreads.keys()),
            'source': 'eastmoney push2his (前复权,主) / 腾讯 fqkline (兜底) + 本地国债收益率曲线',
            'price_sources': sources,
            'note': '债券端用国债ETF/国债指数的前复权价格近似全收益；前端按所选窗口实时计算滚动相关系数',
        },
        'prices': prices,
        'spreads': spreads,
    }

    try:
        import jsonio
        changed = jsonio.write_json_skip_unchanged(OUT_JSON, payload)
    except Exception:
        changed = True
        with open(OUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, separators=(',', ':'))

    size_mb = os.path.getsize(OUT_JSON) / 1024 / 1024
    logger.info('已写出 %s（%.2f MB, 区间 %s）%s',
                os.path.basename(OUT_JSON), size_mb, payload['meta']['date_range'],
                '' if changed else '[无变化，跳过重写]')
    return changed


if __name__ == '__main__':
    main()
