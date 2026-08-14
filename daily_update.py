"""
每日定时更新债券交易数据 + FICC 现券收益率数据 + 收益率曲线数据 + 因子数据
- bond_data.xlsx → bond_trading_data.json（机构行为）
- FICC原始数据（现券）.xlsx → bond_yield_data.json（个券收益率）
- FICC原始数据（衍生品、收益率曲线）.xlsx → yield_curve_data.json（收益率曲线走势）
- 增量更新合并数据 → bond_trading_data_merged.json
- 因子计算 → factor_data.json

执行顺序：
  [0] 先运行用户的 现券数据处理_2026.py，从原始日报生成 bond_data.xlsx
  [1] 机构行为数据
  [2] FICC 现券收益率
  [3] 收益率曲线
  [4] 增量更新合并数据
  [5] 因子计算
"""
import pandas as pd
import json
import os
import sys
import re
import subprocess
import logging
from datetime import datetime, date
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BOND_DATA_EXCEL = r"C:\Users\lihaoran\Documents\工作\现券交易\bond_data.xlsx"
OLD_BOND_DATA_EXCEL = r"C:\Users\lihaoran\Documents\工作\现券交易\bond_data_备份截至2025年.xlsx"
BOND_DATA_OUTPUT = os.path.join(SCRIPT_DIR, "bond_trading_data.json")
FICC_EXCEL = r"D:\工作1\研究课题\收益率曲线\FICC原始数据（现券）.xlsx"
FICC_OUTPUT = os.path.join(SCRIPT_DIR, "bond_yield_data.json")
CURVE_EXCEL = r"D:\工作1\研究课题\收益率曲线\FICC原始数据（衍生品、收益率曲线）.xlsx"
CURVE_OUTPUT = os.path.join(SCRIPT_DIR, "yield_curve_data.json")
FACTOR_OUTPUT = os.path.join(SCRIPT_DIR, "factor_data.json")
FACTOR_MERGED_DATA = os.path.join(SCRIPT_DIR, "bond_trading_data_merged.json")
LOG_PATH = os.path.join(SCRIPT_DIR, "daily_update.log")

# 用户预处理脚本：从原始日报生成 bond_data.xlsx
USER_PREPROCESS_SCRIPT = r"C:\Users\lihaoran\Documents\工作\现券交易\2026年交易\现券数据处理_2026.py"
USER_PREPROCESS_CWD = r"C:\Users\lihaoran\Documents\工作\现券交易\2026年交易"
PYTHON_EXE = r"C:\Users\lihaoran\AppData\Local\Programs\Python\Python313\python.exe"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── 机构行为数据相关 ──
INSTITUTIONS = [
    "大型银行", "中小型银行", "证券公司", "保险公司",
    "基金公司及产品", "货币市场基金", "理财子公司及理财类产品", "其他",
]
INSTITUTIONS_SUMMARY = [
    "大型银行", "中小型银行", "证券公司", "保险公司",
    "基金公司及产品", "理财子公司及理财类产品", "货币市场基金", "其他",
]
MATURITIES = ["≤1年", "1-3年", "3-5年", "5-7年", "7-10年",
              "10-15年", "15-20年", "20-30年", ">30年"]
BOND_TYPES_DETAIL = ["国债", "政金债", "地方债", "同业存单", "信用债"]
BOND_TYPES_SUMMARY = ["国债", "政金债", "地方债", "同业存单", "信用债", "利率债"]

# ── 旧文件（备份截至2025年）机构映射 ──
# 旧文件明细sheet(11)有13个机构，汇总sheet(0-5)有12个机构（无境外机构）
# 映射到新口径8个机构，其他产品类/境外机构忽略
OLD_INSTITUTIONS_DETAIL = [
    "大型商业银行/政策性银行",  # → 大型银行
    "股份制商业银行",            # → 中小型银行
    "城市商业银行",              # → 中小型银行
    "外资银行",                  # → 中小型银行
    "农村金融机构",              # → 中小型银行
    "证券公司",                  # → 证券公司
    "保险公司",                  # → 保险公司
    "基金公司及产品",            # → 基金公司及产品
    "理财子公司及理财类产品",    # → 理财子公司及理财类产品
    "其他产品类",                # → 忽略
    "境外机构",                  # → 忽略
    "货币市场基金",              # → 货币市场基金
    "其他",                      # → 其他
]
OLD_TO_NEW_MAP = {
    "大型商业银行/政策性银行": "大型银行",
    "股份制商业银行": "中小型银行",
    "城市商业银行": "中小型银行",
    "外资银行": "中小型银行",
    "农村金融机构": "中小型银行",
    "证券公司": "证券公司",
    "保险公司": "保险公司",
    "基金公司及产品": "基金公司及产品",
    "理财子公司及理财类产品": "理财子公司及理财类产品",
    "货币市场基金": "货币市场基金",
    "其他": "其他",
    "其他产品类": None,  # 忽略
    "境外机构": None,    # 忽略
}
# 旧文件汇总sheet(0-5)机构列顺序（12个，无境外机构）
OLD_INSTITUTIONS_SUMMARY = [
    "大型商业银行/政策性银行",  # col 1 → 大型银行
    "股份制商业银行",            # col 2 → 中小型银行
    "城市商业银行",              # col 3 → 中小型银行
    "外资银行",                  # col 4 → 中小型银行
    "农村金融机构",              # col 5 → 中小型银行
    "证券公司",                  # col 6 → 证券公司
    "保险公司",                  # col 7 → 保险公司
    "基金公司及产品",            # col 8 → 基金公司及产品
    "理财子公司及理财类产品",    # col 9 → 理财子公司及理财类产品
    "其他产品类",                # col 10 → 忽略
    "货币市场基金",              # col 11 → 货币市场基金
    "其他",                      # col 12 → 其他
]


def is_file_today(file_path):
    if not os.path.exists(file_path):
        return False
    mtime = os.path.getmtime(file_path)
    mdate = datetime.fromtimestamp(mtime).date()
    today = date.today()
    log.info(f"  {os.path.basename(file_path)} 修改时间: {mdate}, 今天: {today}")
    return mdate == today


def format_date(val):
    s = str(int(val))
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


# ── 机构行为数据处理 ──
def build_detail():
    df = pd.read_excel(BOND_DATA_EXCEL, sheet_name=11)
    date_counts = df.iloc[:, 0].value_counts().sort_index()
    records = []
    for day_offset, date_val in enumerate(date_counts.index):
        start_idx = day_offset * 5
        for bond_offset in range(5):
            row_idx = start_idx + bond_offset
            if row_idx >= len(df):
                break
            row = df.iloc[row_idx]
            date_str = format_date(date_val)
            bond_type = BOND_TYPES_DETAIL[bond_offset]
            for inst_idx, inst_name in enumerate(INSTITUTIONS):
                col_start = inst_idx * 10 + 1
                for mat_idx, mat_name in enumerate(MATURITIES):
                    val = row.iloc[col_start + mat_idx]
                    if pd.notna(val) and float(val) != 0:
                        records.append({
                            "date": date_str, "bond_type": bond_type,
                            "institution": inst_name, "maturity": mat_name,
                            "value": round(float(val), 4),
                        })
                total_val = row.iloc[col_start + 9]
                if pd.notna(total_val):
                    records.append({
                        "date": date_str, "bond_type": bond_type,
                        "institution": inst_name, "maturity": "合计",
                        "value": round(float(total_val), 4),
                    })
    return pd.DataFrame(records)


def build_summary():
    records = []
    for sheet_idx, bond_type in enumerate(BOND_TYPES_SUMMARY):
        df = pd.read_excel(BOND_DATA_EXCEL, sheet_name=sheet_idx, header=None)
        for _, row in df.iloc[1:].iterrows():
            date_str = format_date(row.iloc[0])
            for inst_idx, inst_name in enumerate(INSTITUTIONS_SUMMARY):
                val = row.iloc[inst_idx + 1]
                if pd.notna(val):
                    records.append({
                        "date": date_str, "bond_type": bond_type,
                        "institution": inst_name, "maturity": "合计",
                        "value": round(float(val), 4),
                    })
    return pd.DataFrame(records)


def build_detail_old():
    """解析旧文件 sheet 11（13 机构 × 10 列，映射到新口径 8 机构）"""
    df = pd.read_excel(OLD_BOND_DATA_EXCEL, sheet_name=11)
    date_counts = df.iloc[:, 0].value_counts().sort_index()
    records = []
    for day_offset, date_val in enumerate(date_counts.index):
        start_idx = day_offset * 5
        for bond_offset in range(5):
            row_idx = start_idx + bond_offset
            if row_idx >= len(df):
                break
            row = df.iloc[row_idx]
            date_str = format_date(date_val)
            bond_type = BOND_TYPES_DETAIL[bond_offset]
            # 按新口径机构聚合（加总）
            new_inst_values = {inst: [0.0] * 10 for inst in INSTITUTIONS}
            for old_idx, old_inst in enumerate(OLD_INSTITUTIONS_DETAIL):
                new_inst = OLD_TO_NEW_MAP.get(old_inst)
                if new_inst is None:
                    continue
                col_start = old_idx * 10 + 1
                for mat_idx in range(10):  # 9 期限 + 合计
                    val = row.iloc[col_start + mat_idx]
                    if pd.notna(val):
                        new_inst_values[new_inst][mat_idx] += float(val)
            for inst_name in INSTITUTIONS:
                vals = new_inst_values[inst_name]
                for mat_idx, mat_name in enumerate(MATURITIES):
                    v = vals[mat_idx]
                    if v != 0:
                        records.append({
                            "date": date_str, "bond_type": bond_type,
                            "institution": inst_name, "maturity": mat_name,
                            "value": round(v, 4),
                        })
                total_v = vals[9]
                if total_v != 0:
                    records.append({
                        "date": date_str, "bond_type": bond_type,
                        "institution": inst_name, "maturity": "合计",
                        "value": round(total_v, 4),
                    })
    return pd.DataFrame(records)


def build_summary_old():
    """解析旧文件 sheets 0-5（12 机构，映射到新口径 8 机构）"""
    records = []
    for sheet_idx, bond_type in enumerate(BOND_TYPES_SUMMARY):
        df = pd.read_excel(OLD_BOND_DATA_EXCEL, sheet_name=sheet_idx, header=None)
        for _, row in df.iloc[1:].iterrows():
            date_str = format_date(row.iloc[0])
            # 按新口径机构聚合（加总）
            new_inst_values = {inst: 0.0 for inst in INSTITUTIONS_SUMMARY}
            for col_offset, old_inst in enumerate(OLD_INSTITUTIONS_SUMMARY):
                new_inst = OLD_TO_NEW_MAP.get(old_inst)
                if new_inst is None:
                    continue
                val = row.iloc[col_offset + 1]
                if pd.notna(val):
                    new_inst_values[new_inst] += float(val)
            for inst_name in INSTITUTIONS_SUMMARY:
                v = new_inst_values[inst_name]
                if v != 0:
                    records.append({
                        "date": date_str, "bond_type": bond_type,
                        "institution": inst_name, "maturity": "合计",
                        "value": round(v, 4),
                    })
    return pd.DataFrame(records)


def update_bond_trading_data():
    """更新机构行为数据（合并旧文件历史数据 + 新文件当日数据）"""
    if not os.path.exists(BOND_DATA_EXCEL):
        log.warning(f"机构行为数据文件不存在: {BOND_DATA_EXCEL}")
        return False
    if not is_file_today(BOND_DATA_EXCEL):
        log.warning("  bond_data.xlsx 不是今天下载的，跳过")
        return True

    log.info("  开始处理机构行为数据...")
    # 新文件（当日数据）
    df_detail_new = build_detail()
    df_summary_new = build_summary()
    log.info(f"  新文件明细: {len(df_detail_new)} 条, 汇总: {len(df_summary_new)} 条")

    # 旧文件（历史数据 2021-06-03 ~ 2025-12-30，机构映射 13→8）
    df_detail_old = pd.DataFrame()
    df_summary_old = pd.DataFrame()
    if os.path.exists(OLD_BOND_DATA_EXCEL):
        log.info(f"  合并旧文件历史数据: {os.path.basename(OLD_BOND_DATA_EXCEL)}")
        df_detail_old = build_detail_old()
        df_summary_old = build_summary_old()
        log.info(f"  旧文件明细: {len(df_detail_old)} 条, 汇总: {len(df_summary_old)} 条")
        if len(df_detail_old) > 0:
            log.info(f"  旧文件日期范围: {df_detail_old['date'].min()} ~ {df_detail_old['date'].max()}")
    else:
        log.warning(f"  旧文件不存在: {OLD_BOND_DATA_EXCEL}")

    # 合并（旧 + 新），去重（如有重叠保留新文件数据）
    df_detail = pd.concat([df_detail_old, df_detail_new], ignore_index=True)
    if len(df_detail) > 0:
        df_detail = df_detail.drop_duplicates(
            subset=['date', 'bond_type', 'institution', 'maturity'], keep='last')
        df_detail = df_detail.sort_values(
            ['date', 'bond_type', 'institution', 'maturity']).reset_index(drop=True)

    df_summary = pd.concat([df_summary_old, df_summary_new], ignore_index=True)
    if len(df_summary) > 0:
        df_summary = df_summary.drop_duplicates(
            subset=['date', 'bond_type', 'institution', 'maturity'], keep='last')
        df_summary = df_summary.sort_values(
            ['date', 'bond_type', 'institution', 'maturity']).reset_index(drop=True)

    log.info(f"  合并后明细: {len(df_detail)} 条, 汇总: {len(df_summary)} 条")
    if len(df_detail) > 0:
        log.info(f"  日期范围: {df_detail['date'].min()} ~ {df_detail['date'].max()}")

    # 索引化压缩输出（54MB → ~6MB）
    dates_list = sorted(df_detail['date'].unique().tolist()) if len(df_detail) > 0 else []
    bond_types_list = sorted(df_detail['bond_type'].unique().tolist()) if len(df_detail) > 0 else []
    institutions_list = INSTITUTIONS
    maturities_list = MATURITIES + ["合计"]
    date_idx = {d: i for i, d in enumerate(dates_list)}
    bt_idx = {b: i for i, b in enumerate(bond_types_list)}
    inst_idx = {i: idx for idx, i in enumerate(institutions_list)}
    mat_idx = {m: i for i, m in enumerate(maturities_list)}

    compact_detail = []
    for _, r in df_detail.iterrows():
        compact_detail.append([date_idx[r['date']], bt_idx[r['bond_type']],
                               inst_idx[r['institution']], mat_idx[r['maturity']],
                               round(float(r['value']), 2)])
    compact_summary = []
    for _, r in df_summary.iterrows():
        compact_summary.append([date_idx.get(r['date'], -1), bt_idx.get(r['bond_type'], -1),
                                inst_idx.get(r['institution'], -1), mat_idx.get(r['maturity'], -1),
                                round(float(r['value']), 2)])

    output = {
        "meta": {
            "data_source": "bond_data.xlsx + bond_data_备份截至2025年.xlsx (merged)",
            "date_range": [df_detail['date'].min(), df_detail['date'].max()] if len(df_detail) > 0 else ["", ""],
            "bond_types": bond_types_list,
            "institutions": institutions_list,
            "maturities": maturities_list,
            "total_records": len(df_detail),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note": "合并数据：旧文件(2021-2025)机构映射13→8，新文件(2026-)原样保留",
            "format": "compact_v1",  # 标记压缩格式版本
        },
        "idx": {
            "dates": dates_list,
            "bond_types": bond_types_list,
            "institutions": institutions_list,
            "maturities": maturities_list,
        },
        "detail": compact_detail,
        "summary": compact_summary,
    }
    tmp = BOND_DATA_OUTPUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(',', ':'))
    os.replace(tmp, BOND_DATA_OUTPUT)
    log.info(f"  输出: {BOND_DATA_OUTPUT} ({os.path.getsize(BOND_DATA_OUTPUT)/1024/1024:.1f} MB)")
    return True


# ── FICC 现券收益率处理 ──
def update_ficc_yield_data():
    """更新 FICC 现券收益率数据"""
    if not os.path.exists(FICC_EXCEL):
        log.warning(f"FICC 数据文件不存在: {FICC_EXCEL}")
        return False
    if not is_file_today(FICC_EXCEL):
        log.warning("  FICC原始数据（现券）.xlsx 不是今天下载的，跳过")
        return True

    log.info("  开始处理 FICC 现券收益率数据...")
    df = pd.read_excel(FICC_EXCEL, sheet_name=0, header=None)

    # Excel 结构：row 0=分类(期限分组), row 1=期限(数值), row 2=代码, row 3+=数据
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

    terms = {}
    col_info = {}
    seen_codes = set()  # 同一 code 跨多个分组时，保留首次出现的分类（按 Excel 列顺序）
    for c in range(1, df.shape[1]):
        code = code_row[c]
        if pd.notna(code):
            code_str = str(code).strip()
            if code_str in seen_codes:
                log.warning(f"  跳过重复券 {code_str} (col{c})，已归入更早的分类 [{col_category[c]}]")
                continue
            seen_codes.add(code_str)
            category = col_category[c]
            duration = float(duration_row[c]) if pd.notna(duration_row[c]) else None
            terms.setdefault(category, []).append(code_str)
            col_info[c] = {'code': code_str, 'duration': duration, 'category': category}

    # 数据从 row 3 开始，过滤未来日期
    data = df.iloc[3:].copy()
    data.columns = ['date'] + [f'c{c}' for c in range(1, df.shape[1])]
    data['date'] = pd.to_datetime(data['date'])
    today = pd.Timestamp.now().normalize()
    data = data[data['date'] <= today].copy()
    log.info(f"  过滤到今天 ({today.date()}) 后行数: {len(data)}")

    # 构建每只债券时间序列，过滤 0/NaN
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

    # meta.date_range 用实际序列日期（已过滤 0/NaN），避免最新一天值未填入时范围虚高
    if series:
        meta_d0 = min(s['dates'][0] for s in series.values())
        meta_d1 = max(s['dates'][-1] for s in series.values())
    else:
        meta_d0 = meta_d1 = ''

    output = {
        'meta': {
            'data_source': 'FICC原始数据（现券）.xlsx',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'date_range': [meta_d0, meta_d1],
            'terms': terms_sorted,
            'bond_count': len(series),
            'total_records': sum(len(s['dates']) for s in series.values()),
        },
        'series': series,
    }

    tmp = FICC_OUTPUT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, separators=(',', ':'))
    os.replace(tmp, FICC_OUTPUT)
    log.info(f"  输出: {FICC_OUTPUT} ({os.path.getsize(FICC_OUTPUT)/1024:.1f} KB)")
    log.info(f"  债券数: {len(series)}, 总记录: {output['meta']['total_records']}")
    return True


# ── 收益率曲线数据处理 ──
def _simplify_name(raw):
    """将 Wind 指标名称简化为短名"""
    s = str(raw).strip()
    m = re.match(r'^中债国债到期收益率:(\d+)年$', s)
    if m: return f'{m.group(1)}年国债'
    m = re.match(r'^中债国开债到期收益率:(\d+)年$', s)
    if m: return f'{m.group(1)}年国开'
    m = re.match(r'^财政部-中国地方政府债券收益率曲线:(\d+)年$', s)
    if m: return f'{m.group(1)}年地方债'
    m = re.match(r'^中国:(\d+)年期国债期货隐含利率:主连$', s)
    if m: return f'{m.group(1)}年期货隐含利率'
    m = re.match(r'^中国:(\d+)年期国债期货IRR:主连$', s)
    if m: return f'{m.group(1)}年期货IRR'
    if s == '中国:回购定盘利率:7天(FR007)': return 'FR007'
    m = re.match(r'^中国:回购定盘利率:7天\(FR007\):(\d+)日移动平均:算术平均$', s)
    if m: return f'FR007-MA{m.group(1)}'
    if s == '中国:存款类机构质押式回购加权利率:1天': return 'DR001'
    m = re.match(r'^中国:存款类机构质押式回购加权利率:1天:(\d+)日移动平均:算术平均$', s)
    if m: return f'DR001-MA{m.group(1)}'
    m = re.match(r'^(\d+)YREPO$', s)
    if m: return f'IRS-{m.group(1)}Y'
    # 期货主力合约价格（后复权）
    if s == 'T.CFE': return 'T主力'
    if s == 'TF.CFE': return 'TF主力'
    if s == 'TS.CFE': return 'TS主力'
    if s == 'TL.CFE': return 'TL主力'
    return s


def _parse_curve_sheet(df):
    """解析收益率曲线 sheet，返回 (series_dict, date_min, date_max)"""
    first_cell = str(df.iloc[0, 0]).strip() if pd.notna(df.iloc[0, 0]) else ''
    if first_cell == 'Wind':
        header_row = 1
        data_start = 2
    elif first_cell == '日期' or '日期' in first_cell:
        header_row = 0
        data_start = 1
    else:
        header_row = 1 if pd.notna(df.iloc[1, 0]) and df.iloc[1, 0] == '指标名称' else 0
        data_start = header_row + 1

    col_names = {}
    for c in range(1, df.shape[1]):
        raw = df.iloc[header_row, c]
        if pd.notna(raw):
            col_names[c] = _simplify_name(raw)

    data = df.iloc[data_start:].copy()
    data.columns = ['date'] + [f'c{c}' for c in range(1, df.shape[1])]
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data[data['date'].notna()].copy()
    today = pd.Timestamp.now().normalize()
    data = data[data['date'] <= today].copy()

    series = {}
    for c, short in col_names.items():
        col_name = f'c{c}'
        sub = data[['date', col_name]].copy()
        sub = sub[sub[col_name].notna() & (sub[col_name] != 0)]
        # 过滤说明行（值非数值，如"当前日期"等）
        try:
            sub[col_name] = sub[col_name].astype(float)
        except (ValueError, TypeError):
            continue
        if len(sub) == 0:
            continue
        dates = sub['date'].dt.strftime('%Y-%m-%d').tolist()
        vals = [round(float(v), 4) for v in sub[col_name].tolist()]
        series[short] = {'dates': dates, 'values': vals}

    # dmin/dmax 用实际序列日期（已过滤 0/NaN），避免最新一天值未填入时 meta 虚高
    if not series:
        return series, None, None
    d0 = min(pd.to_datetime(s['dates'][0]) for s in series.values())
    d1 = max(pd.to_datetime(s['dates'][-1]) for s in series.values())
    return series, d0, d1


def update_yield_curve_data():
    """更新收益率曲线数据（4 个 sheet 合并）"""
    if not os.path.exists(CURVE_EXCEL):
        log.warning(f"收益率曲线数据文件不存在: {CURVE_EXCEL}")
        return False
    if not is_file_today(CURVE_EXCEL):
        log.warning("  FICC原始数据（衍生品、收益率曲线）.xlsx 不是今天下载的，跳过")
        return True

    log.info("  开始处理收益率曲线数据...")
    xl = pd.ExcelFile(CURVE_EXCEL)
    all_series = {}
    categories = {}
    date_min = None
    date_max = None

    for i, sn in enumerate(xl.sheet_names):
        # 跳过"价格数据"sheet（说明性内容，非时间序列）
        if '价格' in sn:
            log.info(f"    Sheet [{sn}]: 跳过（非时间序列）")
            continue
        df = pd.read_excel(CURVE_EXCEL, sheet_name=i, header=None)
        series, dmin, dmax = _parse_curve_sheet(df)
        categories[sn] = list(series.keys())
        for name, s in series.items():
            all_series[name] = s
        if dmin and (date_min is None or dmin < date_min):
            date_min = dmin
        if dmax and (date_max is None or dmax > date_max):
            date_max = dmax
        log.info(f"    Sheet [{sn}]: {len(series)} 条曲线")

    output = {
        'meta': {
            'data_source': 'FICC原始数据（衍生品、收益率曲线）.xlsx',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'date_range': [
                date_min.strftime('%Y-%m-%d') if date_min else '',
                date_max.strftime('%Y-%m-%d') if date_max else '',
            ],
            'categories': categories,
            'series_count': len(all_series),
        },
        'series': all_series,
    }

    tmp = CURVE_OUTPUT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, separators=(',', ':'))
    os.replace(tmp, CURVE_OUTPUT)
    log.info(f"  输出: {CURVE_OUTPUT} ({os.path.getsize(CURVE_OUTPUT)/1024:.1f} KB)")
    log.info(f"  总曲线数: {len(all_series)}")
    return True


# ── 因子数据处理 ──
FACTOR_BOND_TYPE = "国债"  # legacy, 实际 bond_type 在 FACTOR_DEFS 中定义
FACTOR_MATURITY = "20-30年"  # legacy
FACTOR_MA_WINDOW = 10
FACTOR_PERC_WINDOW = 100
FACTOR_PERC_MIN_PERIODS = 60
FACTOR_FINAL_MA = 10
FACTOR_MERGED_DATA = os.path.join(SCRIPT_DIR, "bond_trading_data_merged.json")

# 因子定义（使用合并数据，5年历史）
FACTOR_DEFS = [
    {
        "name": "基金公司及产品·超长国债净买入因子",
        "short_name": "基金·超长债因子",
        "category": "机构行为",
        "institution": "基金公司及产品",
        "bond_type": "国债",
        "maturity": "20-30年",
        "description": "基金公司及产品对20-30年国债净买入的MA10在过去100天的百分位（再MA10）",
    },
    {
        "name": "中小型银行·超长国债净买入因子",
        "short_name": "中小银行·超长债因子",
        "category": "机构行为",
        "institution": "中小型银行",
        "bond_type": "国债",
        "maturity": "20-30年",
        "description": "中小型银行对20-30年国债净买入的MA10在过去100天的百分位（再MA10）",
    },
    {
        "name": "基金公司及产品·7-10年国开债净买入因子",
        "short_name": "基金·国开因子",
        "category": "机构行为",
        "institution": "基金公司及产品",
        "bond_type": "政金债",
        "maturity": "7-10年",
        "description": "基金公司及产品对7-10年政金债净买入的MA10在过去100天的百分位（再MA10）",
    },
    {
        "name": "保险公司·超长国债净买入因子",
        "short_name": "保险·超长债因子",
        "category": "机构行为",
        "institution": "保险公司",
        "bond_type": "国债",
        "maturity": "20-30年",
        "description": "保险公司对20-30年国债净买入的MA10在过去100天的百分位（再MA10）",
    },
    {
        "name": "中小型银行·7-10年国债净买入因子",
        "short_name": "中小银行·7-10年国债因子",
        "category": "机构行为",
        "institution": "中小型银行",
        "bond_type": "国债",
        "maturity": "7-10年",
        "description": "中小型银行对7-10年国债净买入的MA10在过去100天的百分位（再MA10）",
    },
    {
        "name": "基金公司及产品·7-10年国债+国开债净买入合力因子",
        "short_name": "基金·买入力量因子",
        "category": "机构行为",
        "institution": "基金公司及产品",
        "bond_types": ["国债", "政金债"],  # 多券种求和
        "maturity": "7-10年",
        "description": "基金公司及产品对7-10年国债+政金债净买入合计的MA10在过去100天的百分位（再MA10）",
    },
]

# 技术指标因子定义（数据源：yield_curve_data.json 期货主力后复权价，原始值，不做百分位标准化）
TECHNICAL_FACTOR_DEFS = [
    {
        "name": "3T-TL组合价差MACD柱因子",
        "short_name": "3T-TL·MACD柱因子",
        "category": "技术指标因子",
        "underlying": "3T-TL",
        "indicator": "macd_hist",
        "raw_value": True,
        "description": "3T-TL组合价差（3×T主力−1×TL主力，后复权价）的MACD柱（DIF−DEA，参数12/26/9），原始值，0轴以上=多头动能增强，上穿0轴≈金叉",
    },
]

# 估值因子定义（数据源：yield_curve_data.json，利差→MA10→100天百分位→再MA10）
VALUATION_FACTOR_DEFS = [
    {
        "name": "资金利差因子（10年国债-DR001-MA10）",
        "short_name": "资金利差因子",
        "category": "估值因子",
        "spread_components": ["10年国债", "DR001-MA10"],
        "description": "10年国债收益率与DR001-MA10之差的MA10在过去100天的百分位（再MA10），反映长债相对隔夜资金中枢的carry空间，百分位越高=长债相对资金面越便宜",
    },
    {
        "name": "期限利差因子（10年国债-SHIBOR3M）",
        "short_name": "期限利差因子",
        "category": "估值因子",
        "spread_components": ["10年国债", "SHIBOR:3个月"],
        "description": "10年国债收益率与SHIBOR3M之差的MA10在过去100天的百分位（再MA10），反映10Y-3M期限结构估值，百分位越高=期限利差处于历史高位",
    },
]


def _percentile_rank(series):
    if len(series) < 2:
        return 50.0
    current = series.iloc[-1]
    rank = (series < current).sum()
    return rank / (len(series) - 1) * 100


def _compute_factor_series(df_detail, factor_def):
    inst = factor_def["institution"]
    mat = factor_def.get("maturity", "20-30年")
    # 支持单券种（bond_type）和多券种求和（bond_types）
    bond_types = factor_def.get("bond_types")
    if bond_types:
        mask = (
            (df_detail["institution"] == inst) &
            (df_detail["bond_type"].isin(bond_types)) &
            (df_detail["maturity"] == mat)
        )
    else:
        bt = factor_def.get("bond_type", "国债")
        mask = (
            (df_detail["institution"] == inst) &
            (df_detail["bond_type"] == bt) &
            (df_detail["maturity"] == mat)
        )
    sub = df_detail[mask].copy()
    if len(sub) == 0:
        return {"dates": [], "values": [], "net_buys": [], "ma10": [], "percentile": []}

    daily = sub.groupby("date")["value"].sum().sort_index()
    ma10 = daily.rolling(window=FACTOR_MA_WINDOW, min_periods=FACTOR_MA_WINDOW).mean()
    perc = ma10.rolling(window=FACTOR_PERC_WINDOW, min_periods=FACTOR_PERC_MIN_PERIODS).apply(_percentile_rank, raw=False)
    factor = perc.rolling(window=FACTOR_FINAL_MA, min_periods=1).mean()

    valid = factor.dropna()
    if len(valid) == 0:
        return {"dates": [], "values": [], "net_buys": [], "ma10": [], "percentile": []}

    df_all = pd.concat([daily, ma10, perc, factor], axis=1)
    df_all.columns = ["net_buy", "ma10", "percentile", "factor"]
    df_valid = df_all.loc[valid.index]

    return {
        "dates": valid.index.tolist(),
        "values": [round(float(v), 4) for v in valid.values],
        "net_buys": [round(float(v), 4) if pd.notna(v) else None for v in df_valid["net_buy"].tolist()],
        "ma10": [round(float(v), 4) if pd.notna(v) else None for v in df_valid["ma10"].tolist()],
        "percentile": [round(float(v), 4) if pd.notna(v) else None for v in df_valid["percentile"].tolist()],
    }


def _compute_valuation_factors():
    """从 yield_curve_data.json 计算估值因子（利差→MA10→100天百分位→再MA10）
    返回 (factors_dict, factor_defs_used)
    """
    if not os.path.exists(CURVE_OUTPUT):
        log.warning(f"  收益率曲线数据不存在，跳过估值因子: {CURVE_OUTPUT}")
        return {}, []
    with open(CURVE_OUTPUT, "r", encoding="utf-8") as f:
        curve = json.load(f)
    series = curve.get("series", {})

    factors = {}
    defs_used = []
    for fd in VALUATION_FACTOR_DEFS:
        comps = fd["spread_components"]
        a_name, b_name = comps[0], comps[1]
        a = series.get(a_name)
        b = series.get(b_name)
        if not a or not b:
            log.warning(f"    {fd['short_name']}: 缺少序列 {a_name}/{b_name}，跳过")
            continue
        a_s = pd.Series(a["values"], index=pd.to_datetime(a["dates"]))
        b_s = pd.Series(b["values"], index=pd.to_datetime(b["dates"]))
        # 按日期对齐取交集
        spread = (a_s - b_s).dropna()
        if len(spread) == 0:
            log.warning(f"    {fd['short_name']}: 利差为空，跳过")
            continue
        ma10 = spread.rolling(window=FACTOR_MA_WINDOW, min_periods=FACTOR_MA_WINDOW).mean()
        perc = ma10.rolling(window=FACTOR_PERC_WINDOW, min_periods=FACTOR_PERC_MIN_PERIODS).apply(_percentile_rank, raw=False)
        factor = perc.rolling(window=FACTOR_FINAL_MA, min_periods=1).mean()
        valid = factor.dropna()
        if len(valid) == 0:
            log.warning(f"    {fd['short_name']}: 因子无有效值，跳过")
            continue
        df_all = pd.concat([spread, ma10, perc, factor], axis=1)
        df_all.columns = ["spread", "ma10", "percentile", "factor"]
        df_valid = df_all.loc[valid.index]
        factors[fd["short_name"]] = {
            "dates": [d.strftime("%Y-%m-%d") for d in valid.index],
            "values": [round(float(v), 4) for v in valid.values],
            "spreads": [round(float(v), 4) if pd.notna(v) else None for v in df_valid["spread"].tolist()],
            "ma10": [round(float(v), 4) if pd.notna(v) else None for v in df_valid["ma10"].tolist()],
            "percentile": [round(float(v), 4) if pd.notna(v) else None for v in df_valid["percentile"].tolist()],
        }
        defs_used.append(fd)
        log.info(f"    估值因子 {fd['short_name']}: 有效天数 {len(valid)}, 日期 {valid.index[0].strftime('%Y-%m-%d')} ~ {valid.index[-1].strftime('%Y-%m-%d')}")
    return factors, defs_used


def _compute_technical_factors():
    """从 yield_curve_data.json 计算技术指标因子（原始值，不做百分位标准化）
    目前支持：3T-TL 组合价差（3×T主力−1×TL主力）的 MACD 柱（DIF−DEA，12/26/9）
    返回 (factors_dict, factor_defs_used)
    """
    if not os.path.exists(CURVE_OUTPUT):
        log.warning(f"  收益率曲线数据不存在，跳过技术指标因子: {CURVE_OUTPUT}")
        return {}, []
    with open(CURVE_OUTPUT, "r", encoding="utf-8") as f:
        curve = json.load(f)
    series = curve.get("series", {})

    factors = {}
    defs_used = []
    for fd in TECHNICAL_FACTOR_DEFS:
        if fd.get("underlying") == "3T-TL" and fd.get("indicator") == "macd_hist":
            t = series.get("T主力")
            tl = series.get("TL主力")
            if not t or not tl:
                log.warning(f"    {fd['short_name']}: 缺少 T主力/TL主力 序列，跳过")
                continue
            t_s = pd.Series(t["values"], index=pd.to_datetime(t["dates"]))
            tl_s = pd.Series(tl["values"], index=pd.to_datetime(tl["dates"]))
            # 3手T − 1手TL，按日期交集对齐（TL 2023-04-21 上市，序列自该日起）
            spread = (3 * t_s - tl_s).dropna()
            if len(spread) < 40:
                log.warning(f"    {fd['short_name']}: 3T-TL 价差数据过少（{len(spread)}天），跳过")
                continue
            ema_fast = spread.ewm(span=12, adjust=False).mean()
            ema_slow = spread.ewm(span=26, adjust=False).mean()
            dif = ema_fast - ema_slow
            dea = dif.ewm(span=9, adjust=False).mean()
            hist = dif - dea
            factors[fd["short_name"]] = {
                "dates": [d.strftime("%Y-%m-%d") for d in hist.index],
                "values": [round(float(v), 4) for v in hist.values],
                "spreads": [round(float(v), 4) for v in spread.values],
                "dif": [round(float(v), 4) for v in dif.values],
                "dea": [round(float(v), 4) for v in dea.values],
            }
            defs_used.append(fd)
            log.info(f"    技术指标因子 {fd['short_name']}: 有效天数 {len(hist)}, "
                     f"{hist.index[0].strftime('%Y-%m-%d')} ~ {hist.index[-1].strftime('%Y-%m-%d')}, "
                     f"最新值 {round(float(hist.iloc[-1]), 4)}")
        else:
            log.warning(f"    未知技术指标因子定义: {fd.get('short_name')}，跳过")
    return factors, defs_used


def update_merged_data():
    """追加新数据到合并后的 JSON（增量更新，不重新解析旧文件）"""
    if not os.path.exists(FACTOR_MERGED_DATA):
        log.warning("  合并数据不存在，请先运行 prepare_merged_data.py 生成")
        return False
    if not os.path.exists(BOND_DATA_EXCEL):
        log.warning("  新文件不存在，跳过合并数据更新")
        return False

    with open(FACTOR_MERGED_DATA, "r", encoding="utf-8") as f:
        merged = json.load(f)
    existing_dates = set(r["date"] for r in merged["detail"])
    log.info(f"  现有合并数据: {len(merged['detail'])} 条, 日期到 {merged['meta']['date_range'][1]}")

    df = pd.read_excel(BOND_DATA_EXCEL, sheet_name=11)
    date_counts = df.iloc[:, 0].value_counts().sort_index()
    new_records = []

    for day_offset, date_val in enumerate(date_counts.index):
        date_str = format_date(date_val)
        if date_str in existing_dates:
            continue
        start_idx = day_offset * 5
        for bond_offset in range(5):
            row_idx = start_idx + bond_offset
            if row_idx >= len(df):
                break
            row = df.iloc[row_idx]
            bond_type = BOND_TYPES_DETAIL[bond_offset]
            for inst_idx, inst_name in enumerate(INSTITUTIONS):
                col_start = inst_idx * 10 + 1
                for mat_idx, mat_name in enumerate(MATURITIES):
                    val = row.iloc[col_start + mat_idx]
                    if pd.notna(val) and float(val) != 0:
                        new_records.append({
                            "date": date_str, "bond_type": bond_type,
                            "institution": inst_name, "maturity": mat_name,
                            "value": round(float(val), 4),
                        })
                total_val = row.iloc[col_start + 9]
                if pd.notna(total_val):
                    new_records.append({
                        "date": date_str, "bond_type": bond_type,
                        "institution": inst_name, "maturity": "合计",
                        "value": round(float(total_val), 4),
                    })

    if not new_records:
        log.info("  无新日期数据，合并数据无需更新")
        return True

    log.info(f"  新增 {len(new_records)} 条记录（{len(set(r['date'] for r in new_records))} 天）")
    merged["detail"].extend(new_records)
    merged["detail"].sort(key=lambda r: (r["date"], r["bond_type"], r["institution"], r["maturity"]))
    all_dates = sorted(set(r["date"] for r in merged["detail"]))
    merged["meta"]["date_range"] = [all_dates[0], all_dates[-1]]
    merged["meta"]["total_records"] = len(merged["detail"])
    merged["meta"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    tmp = FACTOR_MERGED_DATA + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, separators=(',', ':'))
    os.replace(tmp, FACTOR_MERGED_DATA)
    log.info(f"  合并数据已更新: {all_dates[0]} ~ {all_dates[-1]}, 共 {len(merged['detail'])} 条")
    return True


def update_factor_data():
    """计算机构行为因子（依赖合并后的 bond_trading_data_merged.json）"""
    if not os.path.exists(FACTOR_MERGED_DATA):
        log.warning(f"合并数据不存在，跳过因子计算: {FACTOR_MERGED_DATA}")
        log.warning("  请先运行 prepare_merged_data.py 生成合并数据")
        return False

    log.info("  开始计算因子数据...")
    with open(FACTOR_MERGED_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data["detail"])

    factors = {}
    categories = {}
    all_factor_defs = []
    for fd in FACTOR_DEFS:
        log.info(f"    计算: {fd['short_name']} ({fd['institution']})")
        result = _compute_factor_series(df, fd)
        factors[fd["short_name"]] = result
        categories.setdefault(fd["category"], []).append(fd["short_name"])
        all_factor_defs.append(fd)
        log.info(f"      有效天数: {len(result['dates'])}")

    # 估值因子（数据源：yield_curve_data.json）
    log.info("    计算估值因子（利差百分位）...")
    val_factors, val_defs = _compute_valuation_factors()
    for fd in val_defs:
        factors[fd["short_name"]] = val_factors[fd["short_name"]]
        categories.setdefault(fd["category"], []).append(fd["short_name"])
        all_factor_defs.append(fd)

    # 技术指标因子（数据源：yield_curve_data.json，原始值不标准化）
    log.info("    计算技术指标因子（MACD 原始值）...")
    tech_factors, tech_defs = _compute_technical_factors()
    for fd in tech_defs:
        factors[fd["short_name"]] = tech_factors[fd["short_name"]]
        categories.setdefault(fd["category"], []).append(fd["short_name"])
        all_factor_defs.append(fd)

    all_dates = set()
    for f in factors.values():
        all_dates.update(f["dates"])
    all_dates = sorted(all_dates)

    output = {
        "meta": {
            "data_source": "bond_trading_data_merged.json + yield_curve_data.json",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date_range": [all_dates[0] if all_dates else "", all_dates[-1] if all_dates else ""],
            "categories": categories,
            "factor_defs": all_factor_defs,
            "params": {
                "target_bond_type": FACTOR_BOND_TYPE,
                "target_maturity": FACTOR_MATURITY,
                "ma_window": FACTOR_MA_WINDOW,
                "percentile_window": FACTOR_PERC_WINDOW,
                "percentile_min_periods": FACTOR_PERC_MIN_PERIODS,
                "final_ma": FACTOR_FINAL_MA,
            },
        },
        "series": factors,
    }

    tmp = FACTOR_OUTPUT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, separators=(',', ':'))
    os.replace(tmp, FACTOR_OUTPUT)
    log.info(f"  输出: {FACTOR_OUTPUT} ({os.path.getsize(FACTOR_OUTPUT)/1024:.1f} KB)")
    log.info(f"  因子数: {len(factors)}")
    return True


def run_user_preprocess():
    """运行用户的预处理脚本，从原始日报生成 bond_data.xlsx"""
    if not os.path.exists(USER_PREPROCESS_SCRIPT):
        log.warning(f"用户预处理脚本不存在: {USER_PREPROCESS_SCRIPT}")
        return False
    if not os.path.exists(PYTHON_EXE):
        log.warning(f"Python 解释器不存在: {PYTHON_EXE}")
        return False

    log.info(f"  运行用户预处理脚本: {os.path.basename(USER_PREPROCESS_SCRIPT)}")
    log.info(f"  工作目录: {USER_PREPROCESS_CWD}")
    try:
        result = subprocess.run(
            [PYTHON_EXE, USER_PREPROCESS_SCRIPT],
            cwd=USER_PREPROCESS_CWD,
            capture_output=True,
            text=True,
            timeout=300,  # 5 分钟超时
            encoding='utf-8',
            errors='replace',
        )
        if result.returncode == 0:
            log.info("  用户预处理脚本执行成功")
            # 打印最后几行输出
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    log.info(f"    [用户脚本] {line}")
            return True
        else:
            log.error(f"  用户预处理脚本失败 (返回码 {result.returncode})")
            if result.stderr:
                log.error(f"  错误输出: {result.stderr[-2000:]}")
            return False
    except subprocess.TimeoutExpired:
        log.error("  用户预处理脚本超时（5分钟）")
        return False
    except Exception as e:
        log.exception(f"  运行用户预处理脚本异常: {e}")
        return False


def main():
    log.info("=" * 50)
    log.info("每日数据更新任务启动")

    ok0 = True
    try:
        log.info("[0/5] 运行用户预处理脚本 (现券数据处理_2026.py)")
        ok0 = run_user_preprocess()
    except Exception as e:
        log.exception(f"用户预处理脚本异常: {e}")
        ok0 = False

    ok1 = True
    try:
        log.info("[1/5] 机构行为数据 (bond_data.xlsx)")
        ok1 = update_bond_trading_data()
    except Exception as e:
        log.exception(f"机构行为数据处理失败: {e}")
        ok1 = False

    ok2 = True
    try:
        log.info("[2/5] FICC 现券收益率 (FICC原始数据（现券）.xlsx)")
        ok2 = update_ficc_yield_data()
    except Exception as e:
        log.exception(f"FICC 数据处理失败: {e}")
        ok2 = False

    ok3 = True
    try:
        log.info("[3/5] 收益率曲线 (FICC原始数据（衍生品、收益率曲线）.xlsx)")
        ok3 = update_yield_curve_data()
    except Exception as e:
        log.exception(f"收益率曲线数据处理失败: {e}")
        ok3 = False

    ok4 = True
    try:
        log.info("[4/5] 增量更新合并数据 (追加新日期)")
        ok4 = update_merged_data()
    except Exception as e:
        log.exception(f"合并数据更新失败: {e}")
        ok4 = False

    ok5 = True
    try:
        log.info("[5/5] 因子数据 (依赖合并数据)")
        ok5 = update_factor_data()
    except Exception as e:
        log.exception(f"因子数据处理失败: {e}")
        ok5 = False

    ok6 = True
    try:
        log.info("[6/6] 曲线偏离度 (依赖现券+曲线数据)")
        import curve_deviation
        curve_deviation.main()
    except Exception as e:
        log.exception(f"曲线偏离度处理失败: {e}")
        ok6 = False

    log.info("=" * 50)
    log.info(f"完成: 预处理={'成功' if ok0 else '失败'}, "
             f"机构行为={'成功' if ok1 else '失败'}, "
             f"FICC={'成功' if ok2 else '失败'}, "
             f"曲线={'成功' if ok3 else '失败'}, "
             f"合并={'成功' if ok4 else '失败'}, "
             f"因子={'成功' if ok5 else '失败'}, "
             f"偏离度={'成功' if ok6 else '失败'}")

    # push 到 GitHub（唯一远程；gitee/gitcode 已于 2026-08-06 废弃，不再同步）
    if ok1 or ok2 or ok3 or ok5:
        try:
            log.info("推送数据到远程仓库...")
            git_push_data()
        except Exception as e:
            log.warning(f"Git push 失败（不影响本地数据）: {e}")

    return ok0 and ok1 and ok2 and ok3 and ok4 and ok5 and ok6


def git_push_data():
    """commit 并 push 更新的 JSON 数据 + HTML 网页到 GitHub（唯一远程；gitee/gitcode 已废弃）"""
    repo_dir = Path(SCRIPT_DIR)
    today_str = datetime.now().strftime("%Y-%m-%d")

    env = os.environ.copy()
    env['GIT_TERMINAL_PROMPT'] = '0'
    # 确保 SSH 使用正确的密钥
    env['GIT_SSH_COMMAND'] = 'ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519'

    def run_git(*args):
        try:
            result = subprocess.run(
                ['git'] + list(args),
                cwd=repo_dir,
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
            )
            if result.returncode != 0:
                log.warning(f"  git {args[0]} 返回码 {result.returncode}: {result.stderr[:200]}")
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            log.warning(f"  git {args[0]} 超时")
            return False

    # add JSON 数据文件
    json_files = [
        'bond_trading_data.json',
        'bond_yield_data.json',
        'yield_curve_data.json',
        'factor_data.json',
        'curve_deviation.json',
    ]
    for f in json_files:
        run_git('add', f)

    # add HTML 网页文件（网页代码改动时同步推送）
    html_files = [
        'index.html',
        'bond_trading_dashboard.html',
        'bond_spread_dashboard.html',
        'yield_curve_dashboard.html',
        'factor_dashboard.html',
        'backtest_dashboard.html',
        'bond_curve_deviation.html',
    ]
    for f in html_files:
        if (repo_dir / f).exists():
            run_git('add', f)

    # 检查是否有变化
    result = subprocess.run(
        ['git', 'diff', '--cached', '--quiet'],
        cwd=repo_dir, capture_output=True, text=True, env=env,
    )
    if result.returncode == 0:
        log.info("  无数据变化，跳过 push")
        return

    # commit
    if not run_git('commit', '-m', f'data: {today_str} 每日数据更新'):
        log.warning("  commit 失败")
        return

    # push 到 GitHub
    log.info("  推送到 GitHub...")
    if run_git('push', 'github', 'main'):
        log.info("  GitHub 推送成功")
    else:
        log.warning("  GitHub 推送失败")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
