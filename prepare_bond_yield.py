"""
FICC 现券收益率数据预处理
从 FICC原始数据（现券）.xlsx 读取，输出 bond_yield_data.json

Excel 结构（单 sheet "现券"）:
  row 0: 分类（30年国债 / 7-10年国债 / 2-7年国债，合并单元格）
  row 1: 期限（实际剩余期限，单位年，数值）
  row 2: 代码（债券代码，如 2600002.IB）
  row 3+: 数据，col 0 = 日期
"""
import pandas as pd
import json
import os
import sys
import logging
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = r"D:\工作1\研究课题\收益率曲线\FICC原始数据（现券）.xlsx"
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "bond_yield_data.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def main():
    if not os.path.exists(EXCEL_PATH):
        log.error(f"文件不存在: {EXCEL_PATH}")
        return False

    df = pd.read_excel(EXCEL_PATH, sheet_name=0, header=None)
    log.info(f"原始 shape: {df.shape}")

    # ── 解析表头 ──
    # row 0: 分类（向前填充合并单元格）
    # row 1: 期限（数值）
    # row 2: 代码
    category_row = df.iloc[0].values
    duration_row = df.iloc[1].values
    code_row = df.iloc[2].values

    cur_cat = None
    col_category = {}
    for c in range(1, df.shape[1]):
        v = category_row[c]
        if pd.notna(v) and v != '分类':
            cur_cat = str(v).strip()
        col_category[c] = cur_cat

    # 构建期限分组 → 债券代码列表；同时建立 col → (code, duration, category)
    terms = {}
    col_info = {}  # col -> {code, duration, category}
    for c in range(1, df.shape[1]):
        code = code_row[c]
        if pd.notna(code):
            code_str = str(code).strip()
            category = col_category[c]
            duration = float(duration_row[c]) if pd.notna(duration_row[c]) else None
            terms.setdefault(category, []).append(code_str)
            col_info[c] = {'code': code_str, 'duration': duration, 'category': category}

    # ── 读取数据，过滤未来日期 ──
    data = df.iloc[3:].copy()
    data.columns = ['date'] + [f'c{c}' for c in range(1, df.shape[1])]
    data['date'] = pd.to_datetime(data['date'])
    today = pd.Timestamp.now().normalize()
    data = data[data['date'] <= today].copy()
    log.info(f"过滤到今天 ({today.date()}) 后行数: {len(data)}")

    # ── 构建每只债券的时间序列 ──
    series = {}
    for c, info in col_info.items():
        col_name = f'c{c}'
        sub = data[['date', col_name]].copy()
        sub = sub[sub[col_name].notna() & (sub[col_name] != 0)]
        if len(sub) == 0:
            continue
        dates = sub['date'].dt.strftime('%Y-%m-%d').tolist()
        yields = [round(float(v), 4) for v in sub[col_name].tolist()]
        series[info['code']] = {
            'category': info['category'],
            'duration': round(info['duration'], 4) if info['duration'] is not None else None,
            'dates': dates,
            'yields': yields,
        }

    # 期限分组：按 Excel 中出现的顺序保留所有分类（不再硬编码）
    terms_sorted = {}
    for c in range(1, df.shape[1]):
        cat = col_category.get(c)
        if cat and cat not in terms_sorted:
            terms_sorted[cat] = terms.get(cat, [])

    output = {
        'meta': {
            'data_source': 'FICC原始数据（现券）.xlsx',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'date_range': [
                data['date'].min().strftime('%Y-%m-%d'),
                data['date'].max().strftime('%Y-%m-%d'),
            ],
            'terms': terms_sorted,
            'bond_count': len(series),
            'total_records': sum(len(s['dates']) for s in series.values()),
        },
        'series': series,
    }

    tmp = OUTPUT_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUTPUT_PATH)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    log.info(f"输出: {OUTPUT_PATH} ({size_kb:.1f} KB)")
    log.info(f"债券数: {len(series)}, 总记录: {output['meta']['total_records']}")
    for t, codes in terms_sorted.items():
        log.info(f"  {t}: {len(codes)} 只券")
    # 打印前几只券的剩余期限示例
    for code, s in list(series.items())[:5]:
        log.info(f"  示例: {code} 分类={s['category']} 剩余期限={s['duration']}年")
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
