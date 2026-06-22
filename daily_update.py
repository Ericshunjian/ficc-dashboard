"""
每日定时更新债券交易数据 + FICC 现券收益率数据 + 收益率曲线数据
- bond_data.xlsx → bond_trading_data.json（机构行为）
- FICC原始数据（现券）.xlsx → bond_yield_data.json（个券收益率）
- FICC原始数据（衍生品、收益率曲线）.xlsx → yield_curve_data.json（收益率曲线走势）
- 仅当对应 xlsx 修改时间是今天时才处理该数据源
"""
import pandas as pd
import json
import os
import sys
import re
import logging
from datetime import datetime, date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BOND_DATA_EXCEL = r"C:\Users\lihaoran\Documents\工作\现券交易\bond_data.xlsx"
BOND_DATA_OUTPUT = os.path.join(SCRIPT_DIR, "bond_trading_data.json")
FICC_EXCEL = r"D:\工作1\研究课题\收益率曲线\FICC原始数据（现券）.xlsx"
FICC_OUTPUT = os.path.join(SCRIPT_DIR, "bond_yield_data.json")
CURVE_EXCEL = r"D:\工作1\研究课题\收益率曲线\FICC原始数据（衍生品、收益率曲线）.xlsx"
CURVE_OUTPUT = os.path.join(SCRIPT_DIR, "yield_curve_data.json")
FACTOR_OUTPUT = os.path.join(SCRIPT_DIR, "factor_data.json")
LOG_PATH = os.path.join(SCRIPT_DIR, "daily_update.log")

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


def update_bond_trading_data():
    """更新机构行为数据"""
    if not os.path.exists(BOND_DATA_EXCEL):
        log.warning(f"机构行为数据文件不存在: {BOND_DATA_EXCEL}")
        return False
    if not is_file_today(BOND_DATA_EXCEL):
        log.warning("  bond_data.xlsx 不是今天下载的，跳过")
        return True

    log.info("  开始处理机构行为数据...")
    df_detail = build_detail()
    df_summary = build_summary()
    log.info(f"  明细记录: {len(df_detail)} 条, 汇总记录: {len(df_summary)} 条")
    log.info(f"  日期范围: {df_detail['date'].min()} ~ {df_detail['date'].max()}")

    output = {
        "meta": {
            "data_source": "bond_data.xlsx",
            "date_range": [df_detail['date'].min(), df_detail['date'].max()],
            "bond_types": sorted(df_detail['bond_type'].unique().tolist()),
            "institutions": INSTITUTIONS,
            "maturities": MATURITIES + ["合计"],
            "total_records": len(df_detail),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "detail": df_detail.to_dict(orient="records"),
        "summary": df_summary.to_dict(orient="records"),
    }
    tmp = BOND_DATA_OUTPUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
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
    for c in range(1, df.shape[1]):
        code = code_row[c]
        if pd.notna(code):
            code_str = str(code).strip()
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

    tmp = FICC_OUTPUT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
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
        if len(sub) == 0:
            continue
        dates = sub['date'].dt.strftime('%Y-%m-%d').tolist()
        vals = [round(float(v), 4) for v in sub[col_name].tolist()]
        series[short] = {'dates': dates, 'values': vals}

    if len(data) == 0:
        return series, None, None
    return series, data['date'].min(), data['date'].max()


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
        json.dump(output, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CURVE_OUTPUT)
    log.info(f"  输出: {CURVE_OUTPUT} ({os.path.getsize(CURVE_OUTPUT)/1024:.1f} KB)")
    log.info(f"  总曲线数: {len(all_series)}")
    return True


# ── 因子数据处理 ──
FACTOR_DEFS = [
    {
        "name": "基金公司及产品·超长国债净买入因子",
        "short_name": "基金·超长债因子",
        "category": "机构行为",
        "institution": "基金公司及产品",
        "description": "基金公司及产品对20-30年国债净买入的MA10在过去100天的百分位（再MA10）",
    },
    {
        "name": "中小型银行·超长国债净买入因子",
        "short_name": "中小银行·超长债因子",
        "category": "机构行为",
        "institution": "中小型银行",
        "description": "中小型银行对20-30年国债净买入的MA10在过去100天的百分位（再MA10）",
    },
]
FACTOR_BOND_TYPE = "国债"
FACTOR_MATURITY = "20-30年"
FACTOR_MA_WINDOW = 10
FACTOR_PERC_WINDOW = 100
FACTOR_PERC_MIN_PERIODS = 60
FACTOR_FINAL_MA = 10


def _percentile_rank(series):
    if len(series) < 2:
        return 50.0
    current = series.iloc[-1]
    rank = (series < current).sum()
    return rank / (len(series) - 1) * 100


def _compute_factor_series(df_detail, factor_def):
    inst = factor_def["institution"]
    mask = (
        (df_detail["institution"] == inst) &
        (df_detail["bond_type"] == FACTOR_BOND_TYPE) &
        (df_detail["maturity"] == FACTOR_MATURITY)
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


def update_factor_data():
    """计算机构行为因子（依赖 bond_trading_data.json）"""
    if not os.path.exists(BOND_DATA_OUTPUT):
        log.warning(f"机构行为数据不存在，跳过因子计算: {BOND_DATA_OUTPUT}")
        return False

    log.info("  开始计算因子数据...")
    with open(BOND_DATA_OUTPUT, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data["detail"])

    factors = {}
    categories = {}
    for fd in FACTOR_DEFS:
        log.info(f"    计算: {fd['short_name']} ({fd['institution']})")
        result = _compute_factor_series(df, fd)
        factors[fd["short_name"]] = result
        categories.setdefault(fd["category"], []).append(fd["short_name"])
        log.info(f"      有效天数: {len(result['dates'])}")

    all_dates = set()
    for f in factors.values():
        all_dates.update(f["dates"])
    all_dates = sorted(all_dates)

    output = {
        "meta": {
            "data_source": "bond_trading_data.json",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date_range": [all_dates[0] if all_dates else "", all_dates[-1] if all_dates else ""],
            "categories": categories,
            "factor_defs": FACTOR_DEFS,
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
        json.dump(output, f, ensure_ascii=False, indent=2)
    os.replace(tmp, FACTOR_OUTPUT)
    log.info(f"  输出: {FACTOR_OUTPUT} ({os.path.getsize(FACTOR_OUTPUT)/1024:.1f} KB)")
    log.info(f"  因子数: {len(factors)}")
    return True


def main():
    log.info("=" * 50)
    log.info("每日数据更新任务启动")

    ok1 = True
    try:
        log.info("[1/4] 机构行为数据 (bond_data.xlsx)")
        ok1 = update_bond_trading_data()
    except Exception as e:
        log.exception(f"机构行为数据处理失败: {e}")
        ok1 = False

    ok2 = True
    try:
        log.info("[2/4] FICC 现券收益率 (FICC原始数据（现券）.xlsx)")
        ok2 = update_ficc_yield_data()
    except Exception as e:
        log.exception(f"FICC 数据处理失败: {e}")
        ok2 = False

    ok3 = True
    try:
        log.info("[3/4] 收益率曲线 (FICC原始数据（衍生品、收益率曲线）.xlsx)")
        ok3 = update_yield_curve_data()
    except Exception as e:
        log.exception(f"收益率曲线数据处理失败: {e}")
        ok3 = False

    ok4 = True
    try:
        log.info("[4/4] 因子数据 (依赖 bond_trading_data.json)")
        ok4 = update_factor_data()
    except Exception as e:
        log.exception(f"因子数据处理失败: {e}")
        ok4 = False

    log.info("=" * 50)
    log.info(f"完成: 机构行为={'成功' if ok1 else '失败'}, "
             f"FICC={'成功' if ok2 else '失败'}, "
             f"曲线={'成功' if ok3 else '失败'}, "
             f"因子={'成功' if ok4 else '失败'}")
    return ok1 and ok2 and ok3 and ok4


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
