"""
机构行为因子计算（基于合并后的历史数据）
使用 bond_trading_data_merged.json（2021-06 ~ 至今）

5 个机构行为因子：
  1. 基金·超长债因子      - 基金公司及产品 / 国债 / 20-30年
  2. 中小银行·超长债因子   - 中小型银行 / 国债 / 20-30年
  3. 基金·国开因子         - 基金公司及产品 / 政金债 / 7-10年
  4. 保险·超长债因子       - 保险公司 / 国债 / 20-30年
  5. 基金·超长国开因子     - 基金公司及产品 / 政金债 / 20-30年（备选，暂不加）

计算流程（每个因子独立）：
  净买入 → MA10 → 100天百分位(min_periods=60) → MA10 = 因子
"""
import pandas as pd
import json
import os
import sys
import logging
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "bond_trading_data_merged.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "factor_data.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# ── 参数 ──
MA_WINDOW = 10
PERC_WINDOW = 100
PERC_MIN_PERIODS = 60
FINAL_MA = 10

# ── 因子定义 ──
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


def percentile_rank(series):
    if len(series) < 2:
        return 50.0
    current = series.iloc[-1]
    rank = (series < current).sum()
    return rank / (len(series) - 1) * 100


def compute_factor(df_detail, factor_def):
    inst = factor_def["institution"]
    mat = factor_def["maturity"]
    # 支持单券种（bond_type）和多券种求和（bond_types）
    bond_types = factor_def.get("bond_types")
    if bond_types:
        mask = (
            (df_detail["institution"] == inst) &
            (df_detail["bond_type"].isin(bond_types)) &
            (df_detail["maturity"] == mat)
        )
    else:
        bt = factor_def["bond_type"]
        mask = (
            (df_detail["institution"] == inst) &
            (df_detail["bond_type"] == bt) &
            (df_detail["maturity"] == mat)
        )
    sub = df_detail[mask].copy()
    if len(sub) == 0:
        log.warning(f"  {factor_def['short_name']}: 无数据")
        return {"dates": [], "values": [], "net_buys": [], "ma10": [], "percentile": []}

    daily = sub.groupby("date")["value"].sum().sort_index()
    ma10 = daily.rolling(window=MA_WINDOW, min_periods=MA_WINDOW).mean()
    perc = ma10.rolling(window=PERC_WINDOW, min_periods=PERC_MIN_PERIODS).apply(percentile_rank, raw=False)
    factor = perc.rolling(window=FINAL_MA, min_periods=1).mean()

    valid = factor.dropna()
    if len(valid) == 0:
        log.warning(f"  {factor_def['short_name']}: 因子无有效值")
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


def main():
    if not os.path.exists(DATA_PATH):
        log.error(f"数据文件不存在: {DATA_PATH}")
        return False

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data["detail"])
    log.info(f"原始记录数: {len(df)}")
    log.info(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")

    factors = {}
    categories = {}
    for fd in FACTOR_DEFS:
        bt_desc = fd.get("bond_types", [fd.get("bond_type", "?")])
        log.info(f"计算因子: {fd['short_name']} ({fd['institution']}/{'+'.join(bt_desc)}/{fd['maturity']})")
        result = compute_factor(df, fd)
        factors[fd["short_name"]] = result
        categories.setdefault(fd["category"], []).append(fd["short_name"])
        log.info(f"  有效天数: {len(result['dates'])}")
        if result["dates"]:
            log.info(f"  日期范围: {result['dates'][0]} ~ {result['dates'][-1]}")
            log.info(f"  最新因子值: {result['values'][-1]}")

    all_dates = set()
    for f in factors.values():
        all_dates.update(f["dates"])
    all_dates = sorted(all_dates)

    output = {
        "meta": {
            "data_source": "bond_trading_data_merged.json",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date_range": [all_dates[0] if all_dates else "", all_dates[-1] if all_dates else ""],
            "categories": categories,
            "factor_defs": FACTOR_DEFS,
            "params": {
                "ma_window": MA_WINDOW,
                "percentile_window": PERC_WINDOW,
                "percentile_min_periods": PERC_MIN_PERIODS,
                "final_ma": FINAL_MA,
            },
        },
        "series": factors,
    }

    tmp = OUTPUT_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUTPUT_PATH)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    log.info(f"输出: {OUTPUT_PATH} ({size_kb:.1f} KB)")
    log.info(f"因子数: {len(factors)}")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
