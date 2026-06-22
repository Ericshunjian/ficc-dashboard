"""
债券交易数据预处理脚本 v2
bond_data.xlsx 中文列名因编码问题乱码，改用纯位置映射硬编码所有中文标签
"""
import pandas as pd
import json
import os

EXCEL_PATH = r"C:\Users\lihaoran\Documents\工作\现券交易\bond_data.xlsx"
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bond_trading_data.json")

# ── 机构标签 ──
# 注意：汇总sheet(0-5)与明细sheet(11)中机构列顺序不同，需分别映射
# 明细sheet(11)列顺序: 大型银行,中小型银行,证券公司,保险公司,基金公司及产品,货币市场基金,理财子公司及理财类产品,其他
INSTITUTIONS = [
    "大型银行",
    "中小型银行",
    "证券公司",
    "保险公司",
    "基金公司及产品",
    "货币市场基金",
    "理财子公司及理财类产品",
    "其他",
]

# 汇总sheet(0-5)列顺序: 大型银行,中小型银行,证券公司,保险公司,基金公司及产品,理财子公司及理财类产品,货币市场基金,其他
INSTITUTIONS_SUMMARY = [
    "大型银行",
    "中小型银行",
    "证券公司",
    "保险公司",
    "基金公司及产品",
    "理财子公司及理财类产品",
    "货币市场基金",
    "其他",
]

MATURITIES = [
    "≤1年",
    "1-3年",
    "3-5年",
    "5-7年",
    "7-10年",
    "10-15年",
    "15-20年",
    "20-30年",
    ">30年",
]

# Sheet 11 中数据排列：国债→政金债→地方债→同业存单→信用债（每天5行）
BOND_TYPES_DETAIL = ["国债", "政金债", "地方债", "同业存单", "信用债"]

# Sheets 0-5 顺序
BOND_TYPES_SUMMARY = ["国债", "政金债", "地方债", "同业存单", "信用债", "利率债"]


def format_date(val):
    s = str(int(val))
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def build_detail():
    """
    从 sheet 11 (汇总) 构建期限细分数据
    列位置映射:
      [0] 日期
      [1-9]   大型银行: ≤1年,1-3年,3-5年,5-7年,7-10年,10-15年,15-20年,20-30年,>30年
      [10]    大型银行合计
      [11-19] 中小型银行 (同上结构)
      [20]    中小型银行合计
      [21-29] 证券公司
      [30]    证券公司合计
      [31-39] 保险公司
      [40]    保险公司合计
      [41-49] 理财公司及产品
      [50]    理财公司及产品合计
      [51-59] 境外市场机构
      [60]    境外市场机构合计
      [61-69] 银行子公司及基金管理产品
      [70]    银行子公司及基金管理产品合计
      [71-79] 其他
      [80]    其他合计
      [81]    债券类型 (乱码，不用)
    """
    df = pd.read_excel(EXCEL_PATH, sheet_name=11)
    
    # 确定每天 5 行对应 5 种债券
    # 检测: 每天实际有几行数据
    date_counts = df.iloc[:, 0].value_counts().sort_index()
    print(f"  每天数据行数分布: {dict(date_counts)}")
    
    records = []
    for day_offset, date_val in enumerate(date_counts.index):
        # 这一天在 df 中的起始行
        start_idx = day_offset * 5  # 每天5行
        
        for bond_offset in range(5):
            row_idx = start_idx + bond_offset
            if row_idx >= len(df):
                break
            
            row = df.iloc[row_idx]
            date_str = format_date(date_val)
            bond_type = BOND_TYPES_DETAIL[bond_offset]
            
            for inst_idx, inst_name in enumerate(INSTITUTIONS):
                col_start = inst_idx * 10 + 1
                
                # 9个期限细分
                for mat_idx, mat_name in enumerate(MATURITIES):
                    val = row.iloc[col_start + mat_idx]
                    if pd.notna(val) and float(val) != 0:
                        records.append({
                            "date": date_str,
                            "bond_type": bond_type,
                            "institution": inst_name,
                            "maturity": mat_name,
                            "value": round(float(val), 4),
                        })
                
                # 合计
                total_val = row.iloc[col_start + 9]
                if pd.notna(total_val):
                    records.append({
                        "date": date_str,
                        "bond_type": bond_type,
                        "institution": inst_name,
                        "maturity": "合计",
                        "value": round(float(total_val), 4),
                    })
    
    df_out = pd.DataFrame(records)
    return df_out


def build_summary():
    """
    从 sheets 0-5 构建净买入汇总
    每 sheet 列结构: [0]日期, [1]大型银行合计, [2]中小型银行合计, [3]证券公司合计,
                   [4]保险公司合计, [5]基金公司及产品合计, [6]理财子公司及理财类产品合计,
                   [7]货币市场基金合计, [8]其他合计
    注意：汇总sheet的机构列顺序与明细sheet不同（理财与货币基金位置互换）
    """
    records = []
    for sheet_idx, bond_type in enumerate(BOND_TYPES_SUMMARY):
        # header=None 因为第一行是乱码中文表头
        df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_idx, header=None)
        # 从第1行开始是数据
        for _, row in df.iloc[1:].iterrows():
            date_str = format_date(row.iloc[0])
            for inst_idx, inst_name in enumerate(INSTITUTIONS_SUMMARY):
                val = row.iloc[inst_idx + 1]
                if pd.notna(val):
                    records.append({
                        "date": date_str,
                        "bond_type": bond_type,
                        "institution": inst_name,
                        "maturity": "合计",
                        "value": round(float(val), 4),
                    })

    return pd.DataFrame(records)


def main():
    print("=" * 50)
    print("Bond Trading Data Preprocessing v2")
    print("=" * 50)
    
    print("\n[1/2] Processing detail sheet (with maturity breakdown)...")
    df_detail = build_detail()
    print(f"  Total records: {len(df_detail)}")
    print(f"  Date range: {df_detail['date'].min()} ~ {df_detail['date'].max()}")
    print(f"  Bond types: {sorted(df_detail['bond_type'].unique())}")
    print(f"  Institutions: {sorted(df_detail['institution'].unique())}")
    print(f"  Maturities: {sorted(df_detail['maturity'].unique())}")
    
    print("\n[2/2] Processing summary sheets (total only)...")
    df_summary = build_summary()
    print(f"  Total records: {len(df_summary)}")
    print(f"  Date range: {df_summary['date'].min()} ~ {df_summary['date'].max()}")
    
    output = {
        "meta": {
            "data_source": "bond_data.xlsx",
            "date_range": [df_detail['date'].min(), df_detail['date'].max()],
            "bond_types": sorted(df_detail['bond_type'].unique().tolist()),
            "institutions": INSTITUTIONS,
            "maturities": MATURITIES + ["合计"],
            "total_records": len(df_detail),
        },
        "detail": df_detail.to_dict(orient="records"),
        "summary": df_summary.to_dict(orient="records"),
    }
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Size: {size_mb:.1f} MB")
    print("Done!")


if __name__ == "__main__":
    main()
