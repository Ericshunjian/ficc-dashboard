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
from urllib.parse import urlencode

BASE = os.path.dirname(os.path.abspath(__file__))
CURVE_JSON = os.path.join(BASE, 'yield_curve_data.json')
OUT_JSON = os.path.join(BASE, 'stock_bond_data.json')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('stock_bond')

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
EM_KLINE = ('https://push2his.eastmoney.com/api/qt/stock/kline/get'
            '?secid={secid}&fields1=f1,f2,f3&fields2=f51,f52,f53,f54,f55,f56,f57'
            '&klt=101&fqt=1&beg=19900101&end=20500101&lmt=10000')

# (名称, 东财secid, 腾讯代码, 新浪symbol)
STOCK = ('沪深300', '1.000300', 'sh000300', 'sh000300')

# 债券端默认口径：中债-国债总财富(总值)指数（中债估值中心官方，财富指数含利息再投）
BOND_MAIN = '中债国债总财富指数'
CHINABOND_TREE_URL = 'https://yield.chinabond.com.cn/cbweb-mn/indices/queryTree?locale=zh_CN'
CHINABOND_QUERY_URL = 'https://yield.chinabond.com.cn/cbweb-mn/indices/singleIndexQueryResult'
CHINABOND_INDEX_ID = '2c9081e50e8767dc010e879acb220021'  # 中债-国债总指数（树节点 id）
CHINABOND_SERIES_KEY = 'CFZS_00'  # CFZS=财富指数, 00=总值

# 中债接口不可用时的兜底：国债 ETF 前复权（近似全收益）
BOND_FALLBACKS = [('五年国债ETF(511010)', '1.511010', 'sh511010', 'sh511010')]

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


SINA_KLINE = ('https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/'
              'CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen=6000')


def fetch_sina_index(symbol, name, retries=3):
    """新浪日线（最长 6000 根，指数无需复权）→ (dates, close)

    沙箱代理对东财 push2his 时通时断，新浪作为主源更稳定。
    """
    h = dict(HEADERS, **{'Referer': 'https://finance.sina.com.cn'})
    last_err = None
    for i in range(retries):
        try:
            import urllib.request
            req = urllib.request.Request(SINA_KLINE.format(symbol=symbol), headers=h)
            with urllib.request.urlopen(req, timeout=40) as resp:
                raw = resp.read().decode('utf-8', 'ignore')
            data = json.loads(raw)
            dates, closes = [], []
            cutoff = _settled_cutoff_date()
            for it in data:
                ds = str(it.get('day', ''))[:10]
                if len(ds) != 10:
                    continue
                d = datetime.date(int(ds[:4]), int(ds[5:7]), int(ds[8:10]))
                if d > cutoff:
                    continue
                try:
                    c = float(it['close'])
                except (KeyError, TypeError, ValueError):
                    continue
                if c <= 0:
                    continue
                dates.append(ds)
                closes.append(round(c, 6))
            if len(dates) < 500:
                raise RuntimeError(f'返回数据过少({len(dates)}条)')
            logger.info('  %s[新浪]: %d 条, %s ~ %s%s', name, len(dates), dates[0], dates[-1],
                        f'（第{i + 1}次尝试）' if i else '')
            return dates, closes
        except Exception as e:
            last_err = e
            logger.warning('  %s 新浪第%d次抓取失败: %s', name, i + 1, str(e)[:120])
            import time
            time.sleep(1.0 * (i + 1))
    raise RuntimeError(f'{name} 新浪连续{retries}次失败: {last_err}')


def fetch_chinabond_index(retries=3):
    """中债估值中心：中债-国债总财富(总值)指数 → (dates, close)

    参数经 queryTree 拿到的指数 id + 固定组合（qxlxt=00 总值、zslxt=CFZS 财富）拼在 URL 上，
    必须以 POST 方式请求，返回 {CFZS_00: {毫秒时间戳: 点位}}。
    """
    params = {
        'indexid': CHINABOND_INDEX_ID,
        'qxlxt': '00',      # 期限类型：00=总值
        'ltcslx': '00',     # 流通场所：00=全市场
        'zslxt': 'CFZS',    # 指数类型：财富指数
        'zslxt1': '',
        'lx': '1',
        'locale': 'zh_CN',
    }
    url = CHINABOND_QUERY_URL + '?' + urlencode(params)
    last_err = None
    for i in range(retries):
        try:
            import urllib.request
            req = urllib.request.Request(url, data=b'', headers=dict(HEADERS, **{
                'X-Requested-With': 'XMLHttpRequest'}))
            with urllib.request.urlopen(req, timeout=40) as resp:
                raw = json.loads(resp.read().decode('utf-8'))
            series = raw.get(CHINABOND_SERIES_KEY) or {}
            items = []
            cutoff = _settled_cutoff_date()
            for ms, v in series.items():
                if v is None:
                    continue
                d = datetime.datetime.fromtimestamp(int(ms) / 1000, datetime.timezone.utc).date()
                if d > cutoff:
                    continue
                items.append((d.strftime('%Y-%m-%d'), round(float(v), 6)))
            items.sort()
            dates = [x[0] for x in items]
            closes = [x[1] for x in items]
            if len(dates) < 200:
                raise RuntimeError(f'返回数据过少({len(dates)}条)')
            logger.info('  %s: %d 条, %s ~ %s%s', BOND_MAIN, len(dates), dates[0], dates[-1],
                        f'（第{i + 1}次尝试）' if i else '')
            return dates, closes
        except Exception as e:
            last_err = e
            logger.warning('  %s 第%d次抓取失败: %s', BOND_MAIN, i + 1, str(e)[:120])
            import time
            time.sleep(1.0 * (i + 1))
    raise RuntimeError(f'{BOND_MAIN} 连续{retries}次抓取失败: {last_err}')


def fetch_instrument(name, secid, tcode, sina_symbol=None, retries=3):
    """新浪(长历史) → 东财 → 腾讯 依次兜底"""
    if sina_symbol:
        try:
            return fetch_sina_index(sina_symbol, name), 'sina'
        except Exception as e:
            logger.warning('  %s 新浪失败，尝试东财: %s', name, str(e)[:120])
    try:
        return fetch_kline(secid, name, retries), 'eastmoney'
    except Exception as e:
        logger.warning('  %s 东财失败，尝试腾讯兜底: %s', name, str(e)[:120])
        try:
            return fetch_tencent(tcode, name), 'tencent'
        except Exception as e2:
            raise RuntimeError(f'新浪/东财/腾讯均失败: {e2}')


def _merge_with_existing(prices, keep_names):
    """合并保护：抓取失败或明显变短的序列保留存量，防止部分失败覆盖好数据

    keep_names 之外的旧序列（如已废弃的 ETF 标的）不再保留，避免残留。
    """
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
    for name in keep_names:
        if name not in prices and name in oldp:
            logger.warning('  %s 本次抓取失败，保留存量(%d条)', name, len(oldp[name]['dates']))
            prices[name] = oldp[name]
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
    import time
    prices = {}
    sources = {}

    # 1) 股票：沪深300
    name, secid, tcode, ssym = STOCK
    try:
        (d, c), src = fetch_instrument(name, secid, tcode, ssym)
        if d:
            prices[name] = {'dates': d, 'close': c}
            sources[name] = src
    except Exception as e:
        logger.warning('  %s: %s', name, e)
    time.sleep(0.5)

    # 2) 债券：中债-国债总财富(总值)指数（官方主源），失败降级到国债 ETF 前复权
    default_bond = BOND_MAIN
    try:
        d, c = fetch_chinabond_index()
        prices[BOND_MAIN] = {'dates': d, 'close': c}
        sources[BOND_MAIN] = 'chinabond (中债-国债总财富/总值/CFZS_00)'
    except Exception as e:
        logger.warning('  %s 中债源失败: %s，尝试国债ETF兜底', BOND_MAIN, str(e)[:120])
        for fb_name, fb_secid, fb_tcode, fb_ssym in BOND_FALLBACKS:
            try:
                (d, c), src = fetch_instrument(fb_name, fb_secid, fb_tcode, fb_ssym)
                if d:
                    prices[fb_name] = {'dates': d, 'close': c}
                    sources[fb_name] = src + ' (中债接口不可用时的代理)'
                    default_bond = fb_name
                    break
            except Exception as e2:
                logger.warning('  %s 兜底也失败: %s', fb_name, str(e2)[:120])
            time.sleep(0.5)

    keep_names = [STOCK[0], BOND_MAIN] + [x[0] for x in BOND_FALLBACKS]
    prices = _merge_with_existing(prices, keep_names)
    # 存量保留的序列，来源标记沿用
    for name in prices:
        sources.setdefault(name, 'kept-existing')
    # 中债序列若本次抓取失败但存量有，则默认口径仍用中债
    if default_bond != BOND_MAIN and BOND_MAIN in prices:
        default_bond = BOND_MAIN

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
            'default_bond': default_bond,
            'spreads': list(spreads.keys()),
            'source': ('中债估值中心 中债-国债总财富(总值)指数(主) / '
                       'eastmoney push2his + 腾讯 fqkline(兜底) + 本地国债收益率曲线'),
            'price_sources': sources,
            'note': ('债券端默认中债-国债总财富(总值)指数（含利息再投，官方口径）；'
                     '中债接口不可用时降级为国债ETF前复权。前端按所选窗口实时计算滚动相关系数'),
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
