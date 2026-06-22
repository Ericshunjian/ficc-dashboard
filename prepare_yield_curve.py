"""
收益率曲线数据预处理
从 FICC原始数据（衍生品、收益率曲线）.xlsx 读取 4 个 sheet，输出 yield_curve_data.json

Sheet 结构：
  - 基准 / 期货 / 债券：row 0=Wind, row 1=指标名称, row 2+=数据
  - IRS：row 0=日期 + 指标, row 1+=数据
自动使用简短名称（如 10年国债、FR007、IRS-1Y、2年期货IRR、10年国开 等）
"""
import pandas as pd
import json
import os
import sys
import re
import logging
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = r"D:\工作1\研究课题\收益率曲线\FICC原始数据（衍生品、收益率曲线）.xlsx"
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "yield_curve_data.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def simplify_name(raw):
    """将 Wind 指标名称简化为短名"""
    s = str(raw).strip()

    # 国债到期收益率: 中债国债到期收益率:10年 → 10年国债
    m = re.match(r'^中债国债到期收益率:(\d+)年$', s)
    if m: return f'{m.group(1)}年国债'

    # 国开债: 中债国开债到期收益率:10年 → 10年国开
    m = re.match(r'^中债国开债到期收益率:(\d+)年$', s)
    if m: return f'{m.group(1)}年国开'

    # 地方债: 财政部-中国地方政府债券收益率曲线:10年 → 10年地方债
    m = re.match(r'^财政部-中国地方政府债券收益率曲线:(\d+)年$', s)
    if m: return f'{m.group(1)}年地方债'

    # 国债期货隐含利率: 中国:10年期国债期货隐含利率:主连 → 10年期货隐含利率
    m = re.match(r'^中国:(\d+)年期国债期货隐含利率:主连$', s)
    if m: return f'{m.group(1)}年期货隐含利率'

    # 国债期货IRR: 中国:10年期国债期货IRR:主连 → 10年期货IRR
    m = re.match(r'^中国:(\d+)年期国债期货IRR:主连$', s)
    if m: return f'{m.group(1)}年期货IRR'

    # FR007 系列
    if s == '中国:回购定盘利率:7天(FR007)': return 'FR007'
    m = re.match(r'^中国:回购定盘利率:7天\(FR007\):(\d+)日移动平均:算术平均$', s)
    if m: return f'FR007-MA{m.group(1)}'

    # DR001 系列
    if s == '中国:存款类机构质押式回购加权利率:1天': return 'DR001'
    m = re.match(r'^中国:存款类机构质押式回购加权利率:1天:(\d+)日移动平均:算术平均$', s)
    if m: return f'DR001-MA{m.group(1)}'

    # IRS REPO
    m = re.match(r'^(\d+)YREPO$', s)
    if m: return f'IRS-{m.group(1)}Y'

    return s  # fallback


def parse_sheet(df, sheet_name):
    """解析单个 sheet，返回 {short_name: {dates, values}}"""
    # 判断表头结构
    first_cell = str(df.iloc[0, 0]).strip() if pd.notna(df.iloc[0, 0]) else ''
    if first_cell == 'Wind':
        # 基准/期货/债券: row 1=指标, row 2+=数据
        header_row = 1
        data_start = 2
    elif first_cell == '日期' or '日期' in first_cell:
        # IRS: row 0=日期+指标, row 1+=数据
        header_row = 0
        data_start = 1
    else:
        # 尝试自动检测
        header_row = 1 if pd.notna(df.iloc[1, 0]) and df.iloc[1, 0] == '指标名称' else 0
        data_start = header_row + 1

    # 构建 col → short_name 映射
    col_names = {}
    for c in range(1, df.shape[1]):
        raw = df.iloc[header_row, c]
        if pd.notna(raw):
            short = simplify_name(raw)
            col_names[c] = short

    # 读数据
    data = df.iloc[data_start:].copy()
    data.columns = ['date'] + [f'c{c}' for c in range(1, df.shape[1])]
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data[data['date'].notna()].copy()

    # 过滤未来日期
    today = pd.Timestamp.now().normalize()
    data = data[data['date'] <= today].copy()

    # 构建每条序列（过滤 0/NaN）
    series = {}
    for c, short in col_names.items():
        col_name = f'c{c}'
        sub = data[['date', col_name]].copy()
        sub = sub[sub[col_name].notna() & (sub[col_name] != 0)]
        if len(sub) == 0:
            continue
        dates = sub['date'].dt.strftime('%Y-%m-%d').tolist()
        vals = [round(float(v), 4) for v in sub[col_name].tolist()]
        series[short] = {'dates': dates, 'values': vals}

    return series, data['date'].min(), data['date'].max()


def main():
    if not os.path.exists(EXCEL_PATH):
        log.error(f"文件不存在: {EXCEL_PATH}")
        return False

    xl = pd.ExcelFile(EXCEL_PATH)
    log.info(f"Sheets: {xl.sheet_names}")

    all_series = {}      # short_name → series
    categories = {}      # sheet_name → [short_names]
    date_min = None
    date_max = None

    for i, sn in enumerate(xl.sheet_names):
        df = pd.read_excel(EXCEL_PATH, sheet_name=i, header=None)
        series, dmin, dmax = parse_sheet(df, sn)
        categories[sn] = list(series.keys())
        for name, s in series.items():
            all_series[name] = s
        if date_min is None or dmin < date_min:
            date_min = dmin
        if date_max is None or dmax > date_max:
            date_max = dmax
        log.info(f"  Sheet [{sn}]: {len(series)} 条曲线, {dmin.date()} ~ {dmax.date()}")

    output = {
        'meta': {
            'data_source': 'FICC原始数据（衍生品、收益率曲线）.xlsx',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'date_range': [
                date_min.strftime('%Y-%m-%d'),
                date_max.strftime('%Y-%m-%d'),
            ],
            'categories': categories,
            'series_count': len(all_series),
        },
        'series': all_series,
    }

    tmp = OUTPUT_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUTPUT_PATH)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    log.info(f"输出: {OUTPUT_PATH} ({size_kb:.1f} KB)")
    log.info(f"总曲线数: {len(all_series)}")
    for cat, names in categories.items():
        log.info(f"  {cat}: {names}")
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
