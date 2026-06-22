"""
机构行为因子计算
分别计算两个机构对 20-30年国债 的净买入行为因子：
  - 基金公司及产品
  - 中小型银行

计算流程（每个机构独立）：
  1. 取该机构对【国债 20-30年】的日度净买入
  2. 做 10 日滚动平均（含当天，window=10）
  3. 计算 MA10 在过去 100 天的百分位（min_periods=60）
  4. 对百分位再做 10 日平均 → 因子
"""
import pandas as pd
import json
import os
import sys
import logging
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "bond_trading_data.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "factor_data.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# ── 参数 ──
TARGET_BOND_TYPE = "国债"
TARGET_MATURITY = "20-30年"
MA_WINDOW = 10
PERC_WINDOW = 100
PERC_MIN_PERIODS = 60
FINAL_MA = 10

# 因子定义：机构行为类因子
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


def percentile_rank(series):
    """当前值在过去窗口中的百分位 (0-100)"""
    if len(series) < 2:
        return 50.0
    current = series.iloc[-1]
    rank = (series < current).sum()
    return rank / (len(series) - 1) * 100


def compute_factor(df_detail, factor_def):
    """计算单个机构的因子时间序列"""
    inst = factor_def["institution"]
    mask = (
        (df_detail["institution"] == inst) &
        (df_detail["bond_type"] == TARGET_BOND_TYPE) &
        (df_detail["maturity"] == TARGET_MATURITY)
    )
    sub = df_detail[mask].copy()
    if len(sub) == 0:
        log.warning(f"  {inst}: 无数据")
        return {"dates": [], "values": [], "net_buys": [], "ma10": [], "percentile": []}

    daily = sub.groupby("date")["value"].sum().sort_index()

    ma10 = daily.rolling(window=MA_WINDOW, min_periods=MA_WINDOW).mean()
    perc = ma10.rolling(window=PERC_WINDOW, min_periods=PERC_MIN_PERIODS).apply(percentile_rank, raw=False)
    factor = perc.rolling(window=FINAL_MA, min_periods=1).mean()

    # 只保留因子有值的日期
    valid = factor.dropna()
    if len(valid) == 0:
        log.warning(f"  {inst}: 因子无有效值")
        return {"dates": [], "values": [], "net_buys": [], "ma10": [], "percentile": []}

    dates = valid.index.tolist()
    values = [round(float(v), 4) for v in valid.values]

    # 附带中间量（用于详情查看）
    df_all = pd.concat([daily, ma10, perc, factor], axis=1)
    df_all.columns = ["net_buy", "ma10", "percentile", "factor"]
    df_valid = df_all.loc[valid.index]

    return {
        "dates": dates,
        "values": values,
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

    factors = {}
    categories = {}
    for fd in FACTOR_DEFS:
        log.info(f"计算因子: {fd['short_name']} ({fd['institution']})")
        result = compute_factor(df, fd)
        factors[fd["short_name"]] = result
        categories.setdefault(fd["category"], []).append(fd["short_name"])
        log.info(f"  有效天数: {len(result['dates'])}")
        if result["dates"]:
            log.info(f"  日期范围: {result['dates'][0]} ~ {result['dates'][-1]}")
            log.info(f"  最新因子值: {result['values'][-1]}")

    # 合并所有因子的日期并集，用于对齐
    all_dates = set()
    for f in factors.values():
        all_dates.update(f["dates"])
    all_dates = sorted(all_dates)
    if all_dates:
        log.info(f"总日期范围: {all_dates[0]} ~ {all_dates[-1]}, 共 {len(all_dates)} 天")

    output = {
        "meta": {
            "data_source": "bond_trading_data.json",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date_range": [all_dates[0] if all_dates else "", all_dates[-1] if all_dates else ""],
            "categories": categories,
            "factor_defs": FACTOR_DEFS,
            "params": {
                "target_bond_type": TARGET_BOND_TYPE,
                "target_maturity": TARGET_MATURITY,
                "ma_window": MA_WINDOW,
                "percentile_window": PERC_WINDOW,
                "percentile_min_periods": PERC_MIN_PERIODS,
                "final_ma": FINAL_MA,
            },
        },
        "series": factors,
    }

    tmp = OUTPUT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUTPUT_PATH)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    log.info(f"输出: {OUTPUT_PATH} ({size_kb:.1f} KB)")
    log.info(f"因子数: {len(factors)}")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
