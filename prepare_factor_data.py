"""
因子计算（基于合并后的历史数据 + 收益率曲线数据）

两大类因子：
  A. 机构行为因子（数据源：bond_trading_data_merged.json，2021-06 ~ 至今）
     净买入 → MA10 → 100天百分位(min_periods=60) → MA10 = 因子
  B. 估值因子（数据源：yield_curve_data.json）
     利差 → MA10 → 100天百分位(min_periods=60) → MA10 = 因子
     - 资金利差因子：10年国债 - DR001-MA10
     - 期限利差因子：10年国债 - SHIBOR3M
"""
import pandas as pd
import json
import os
import sys
import logging
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "bond_trading_data_merged.json")
CURVE_PATH = os.path.join(SCRIPT_DIR, "yield_curve_data.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "factor_data.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# ── 参数（与 daily_update.py 保持一致）──
MA_WINDOW = 10
PERC_WINDOW = 100
PERC_MIN_PERIODS = 60
FINAL_MA = 10

# ── 机构行为因子定义 ──
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

# ── 技术指标因子定义（数据源：yield_curve_data.json 期货主力后复权价，原始值不标准化）──
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

# ── 估值因子定义（数据源：yield_curve_data.json）──
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


def compute_valuation_factors():
    """从 yield_curve_data.json 计算估值因子（利差→MA10→100天百分位→再MA10）
    返回 (factors_dict, factor_defs_used)
    """
    if not os.path.exists(CURVE_PATH):
        log.warning(f"收益率曲线数据不存在，跳过估值因子: {CURVE_PATH}")
        return {}, []
    with open(CURVE_PATH, "r", encoding="utf-8") as f:
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
            log.warning(f"  估值因子 {fd['short_name']}: 缺少序列 {a_name}/{b_name}，跳过")
            continue
        a_s = pd.Series(a["values"], index=pd.to_datetime(a["dates"]))
        b_s = pd.Series(b["values"], index=pd.to_datetime(b["dates"]))
        # 按日期对齐取交集
        spread = (a_s - b_s).dropna()
        if len(spread) == 0:
            log.warning(f"  估值因子 {fd['short_name']}: 利差为空，跳过")
            continue
        ma10 = spread.rolling(window=MA_WINDOW, min_periods=MA_WINDOW).mean()
        perc = ma10.rolling(window=PERC_WINDOW, min_periods=PERC_MIN_PERIODS).apply(percentile_rank, raw=False)
        factor = perc.rolling(window=FINAL_MA, min_periods=1).mean()
        valid = factor.dropna()
        if len(valid) == 0:
            log.warning(f"  估值因子 {fd['short_name']}: 因子无有效值，跳过")
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
        log.info(f"  估值因子 {fd['short_name']}: 有效天数 {len(valid)}, "
                 f"{valid.index[0].strftime('%Y-%m-%d')} ~ {valid.index[-1].strftime('%Y-%m-%d')}, "
                 f"最新值 {round(float(valid.iloc[-1]), 2)}")
    return factors, defs_used


def compute_technical_factors():
    """从 yield_curve_data.json 计算技术指标因子（原始值，不做百分位标准化）
    目前支持：3T-TL 组合价差（3×T主力−1×TL主力）的 MACD 柱（DIF−DEA，12/26/9）
    返回 (factors_dict, factor_defs_used)
    """
    if not os.path.exists(CURVE_PATH):
        log.warning(f"收益率曲线数据不存在，跳过技术指标因子: {CURVE_PATH}")
        return {}, []
    with open(CURVE_PATH, "r", encoding="utf-8") as f:
        curve = json.load(f)
    series = curve.get("series", {})

    factors = {}
    defs_used = []
    for fd in TECHNICAL_FACTOR_DEFS:
        if fd.get("underlying") == "3T-TL" and fd.get("indicator") == "macd_hist":
            t = series.get("T主力")
            tl = series.get("TL主力")
            if not t or not tl:
                log.warning(f"  技术指标因子 {fd['short_name']}: 缺少 T主力/TL主力 序列，跳过")
                continue
            t_s = pd.Series(t["values"], index=pd.to_datetime(t["dates"]))
            tl_s = pd.Series(tl["values"], index=pd.to_datetime(tl["dates"]))
            spread = (3 * t_s - tl_s).dropna()
            if len(spread) < 40:
                log.warning(f"  技术指标因子 {fd['short_name']}: 3T-TL 价差数据过少（{len(spread)}天），跳过")
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
            log.info(f"  技术指标因子 {fd['short_name']}: 有效天数 {len(hist)}, "
                     f"{hist.index[0].strftime('%Y-%m-%d')} ~ {hist.index[-1].strftime('%Y-%m-%d')}, "
                     f"最新值 {round(float(hist.iloc[-1]), 4)}")
        else:
            log.warning(f"  未知技术指标因子定义: {fd.get('short_name')}，跳过")
    return factors, defs_used


def main():
    factors = {}
    categories = {}
    all_factor_defs = []

    # ── A. 机构行为因子 ──
    if not os.path.exists(DATA_PATH):
        log.error(f"数据文件不存在: {DATA_PATH}")
        return False

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data["detail"])
    log.info(f"机构行为数据: {len(df)} 条, {df['date'].min()} ~ {df['date'].max()}")

    for fd in FACTOR_DEFS:
        bt_desc = fd.get("bond_types", [fd.get("bond_type", "?")])
        log.info(f"计算因子: {fd['short_name']} ({fd['institution']}/{'+'.join(bt_desc)}/{fd['maturity']})")
        result = compute_factor(df, fd)
        factors[fd["short_name"]] = result
        categories.setdefault(fd["category"], []).append(fd["short_name"])
        all_factor_defs.append(fd)
        log.info(f"  有效天数: {len(result['dates'])}")
        if result["dates"]:
            log.info(f"  日期范围: {result['dates'][0]} ~ {result['dates'][-1]}")
            log.info(f"  最新因子值: {result['values'][-1]}")

    # ── B. 估值因子 ──
    log.info("计算估值因子（利差百分位）...")
    val_factors, val_defs = compute_valuation_factors()
    for fd in val_defs:
        factors[fd["short_name"]] = val_factors[fd["short_name"]]
        categories.setdefault(fd["category"], []).append(fd["short_name"])
        all_factor_defs.append(fd)

    # ── C. 技术指标因子（原始值不标准化）──
    log.info("计算技术指标因子（MACD 原始值）...")
    tech_factors, tech_defs = compute_technical_factors()
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
    log.info(f"因子数: {len(factors)}（机构行为 {len(FACTOR_DEFS)} + 估值 {len(val_defs)} + 技术 {len(tech_defs)}）")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
