"""
合并新旧 bond_data.xlsx → bond_trading_data_merged.json
仅用于因子分析，机构行为分析子网页仍用原 bond_trading_data.json

合并规则：
- 旧文件（2021-06-03 ~ 2025-12-30）+ 新文件（2026-01-04 ~）
- 机构名称映射到新文件口径：
    大型商业银行/政策性银行 → 大型银行
    股份制商业银行 + 城市商业银行 + 农村金融机构 + 外资银行 → 中小型银行（加总）
    其他机构名称一致的直接映射
    旧文件多出的机构（其他产品类、境外机构）忽略
- 期限结构两边一致：1年/1-3年/3-5年/5-7年/7-10年/10-15年/15-20年/20-30年/30年/合计
- 每天都是 5 行（国债/政金债/地方债/同业存单/信用债）
"""
import pandas as pd
import json
import os
import sys
import logging
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_EXCEL = r"C:\Users\lihaoran\Documents\工作\现券交易\bond_data.xlsx"
OLD_EXCEL = r"C:\Users\lihaoran\Documents\工作\现券交易\bond_data_备份截至2025年.xlsx"
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "bond_trading_data_merged.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# ── 统一机构口径（新文件标准）──
INSTITUTIONS_NEW = [
    "大型银行", "中小型银行", "证券公司", "保险公司",
    "基金公司及产品", "货币市场基金", "理财子公司及理财类产品", "其他",
]

# 旧文件 → 新文件 机构映射
# 旧文件机构列顺序：大型商业银行/政策性银行, 股份制商业银行, 城市商业银行, 农村金融机构, 外资银行,
#                   证券公司, 保险公司, 基金公司及产品, 理财子公司及理财类产品, 其他产品类, 境外机构,
#                   货币市场基金, 其他
OLD_INSTITUTIONS = [
    "大型商业银行/政策性银行",  # → 大型银行
    "股份制商业银行",            # → 中小型银行（与其他3个相加）
    "城市商业银行",              # → 中小型银行
    "农村金融机构",              # → 中小型银行
    "外资银行",                  # → 中小型银行
    "证券公司",                  # → 证券公司
    "保险公司",                  # → 保险公司
    "基金公司及产品",            # → 基金公司及产品
    "理财子公司及理财类产品",    # → 理财子公司及理财类产品
    "其他产品类",                # → 忽略
    "境外机构",                  # → 忽略
    "货币市场基金",              # → 货币市场基金
    "其他",                      # → 其他
]

# 映射到新口径
OLD_TO_NEW_MAP = {
    "大型商业银行/政策性银行": "大型银行",
    "股份制商业银行": "中小型银行",
    "城市商业银行": "中小型银行",
    "农村金融机构": "中小型银行",
    "外资银行": "中小型银行",
    "证券公司": "证券公司",
    "保险公司": "保险公司",
    "基金公司及产品": "基金公司及产品",
    "理财子公司及理财类产品": "理财子公司及理财类产品",
    "货币市场基金": "货币市场基金",
    "其他": "其他",
    # 以下忽略
    "其他产品类": None,
    "境外机构": None,
}

MATURITIES = ["≤1年", "1-3年", "3-5年", "5-7年", "7-10年",
              "10-15年", "15-20年", "20-30年", ">30年"]
BOND_TYPES_DETAIL = ["国债", "政金债", "地方债", "同业存单", "信用债"]


def format_date(val):
    s = str(int(val))
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def parse_new_file():
    """解析新文件 sheet 11（8 机构 × 10 列）"""
    df = pd.read_excel(NEW_EXCEL, sheet_name=11)
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
            for inst_idx, inst_name in enumerate(INSTITUTIONS_NEW):
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


def parse_old_file():
    """解析旧文件 sheet 11（13 机构 × 10 列，需映射到新口径）"""
    df = pd.read_excel(OLD_EXCEL, sheet_name=11)
    date_counts = df.iloc[:, 0].value_counts().sort_index()
    records = []

    # 旧文件每个机构 10 列，13 个机构 = 130 列 + 日期列 + 债券类型列
    # 额外的合计列在 col 132+，忽略

    for day_offset, date_val in enumerate(date_counts.index):
        start_idx = day_offset * 5
        for bond_offset in range(5):
            row_idx = start_idx + bond_offset
            if row_idx >= len(df):
                break
            row = df.iloc[row_idx]
            date_str = format_date(date_val)
            bond_type = BOND_TYPES_DETAIL[bond_offset]

            # 按新口径机构聚合
            # 先收集每个新口径机构的值
            new_inst_values = {inst: [0.0] * 10 for inst in INSTITUTIONS_NEW}  # 9 期限 + 合计

            for old_idx, old_inst in enumerate(OLD_INSTITUTIONS):
                new_inst = OLD_TO_NEW_MAP.get(old_inst)
                if new_inst is None:
                    continue  # 忽略
                col_start = old_idx * 10 + 1
                for mat_idx in range(10):  # 9 期限 + 合计
                    val = row.iloc[col_start + mat_idx]
                    if pd.notna(val):
                        new_inst_values[new_inst][mat_idx] += float(val)

            # 写入记录
            for inst_name in INSTITUTIONS_NEW:
                vals = new_inst_values[inst_name]
                for mat_idx, mat_name in enumerate(MATURITIES):
                    v = vals[mat_idx]
                    if v != 0:
                        records.append({
                            "date": date_str, "bond_type": bond_type,
                            "institution": inst_name, "maturity": mat_name,
                            "value": round(v, 4),
                        })
                # 合计
                total_v = vals[9]
                if total_v != 0:
                    records.append({
                        "date": date_str, "bond_type": bond_type,
                        "institution": inst_name, "maturity": "合计",
                        "value": round(total_v, 4),
                    })

    return pd.DataFrame(records)


def main():
    log.info("=" * 50)
    log.info("合并新旧 bond_data.xlsx")

    # 检查文件
    for p in [NEW_EXCEL, OLD_EXCEL]:
        if not os.path.exists(p):
            log.error(f"文件不存在: {p}")
            return False

    # 解析新文件
    log.info("[1/3] 解析新文件 (2026-01-04 ~)")
    df_new = parse_new_file()
    log.info(f"  新文件记录数: {len(df_new)}")
    log.info(f"  日期范围: {df_new['date'].min()} ~ {df_new['date'].max()}")

    # 解析旧文件
    log.info("[2/3] 解析旧文件 (2021-06-03 ~ 2025-12-30)")
    df_old = parse_old_file()
    log.info(f"  旧文件记录数: {len(df_old)}")
    log.info(f"  日期范围: {df_old['date'].min()} ~ {df_old['date'].max()}")

    # 合并
    log.info("[3/3] 合并数据")
    df_merged = pd.concat([df_old, df_new], ignore_index=True)
    # 去重（如果日期有重叠，保留新文件的）
    df_merged = df_merged.drop_duplicates(subset=['date', 'bond_type', 'institution', 'maturity'], keep='last')
    df_merged = df_merged.sort_values(['date', 'bond_type', 'institution', 'maturity']).reset_index(drop=True)

    log.info(f"  合并后记录数: {len(df_merged)}")
    log.info(f"  合并后日期范围: {df_merged['date'].min()} ~ {df_merged['date'].max()}")
    log.info(f"  机构列表: {sorted(df_merged['institution'].unique())}")
    log.info(f"  券种列表: {sorted(df_merged['bond_type'].unique())}")

    # 验证关键机构
    for inst in ["基金公司及产品", "中小型银行", "保险公司"]:
        sub = df_merged[df_merged['institution'] == inst]
        log.info(f"  {inst}: {len(sub)} 条记录, 日期 {sub['date'].min()} ~ {sub['date'].max()}")

    output = {
        "meta": {
            "data_source": "bond_data.xlsx + bond_data_备份截至2025年.xlsx (merged)",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date_range": [df_merged['date'].min(), df_merged['date'].max()],
            "bond_types": sorted(df_merged['bond_type'].unique().tolist()),
            "institutions": INSTITUTIONS_NEW,
            "maturities": MATURITIES + ["合计"],
            "total_records": len(df_merged),
            "note": "合并数据，仅用于因子分析。机构映射：大型商业银行/政策性银行→大型银行，股份制+城市+农村+外资→中小型银行",
        },
        "detail": df_merged.to_dict(orient="records"),
    }

    tmp = OUTPUT_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUTPUT_PATH)

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    log.info(f"输出: {OUTPUT_PATH} ({size_mb:.1f} MB)")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
