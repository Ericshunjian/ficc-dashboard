"""Fetch CSI 300 daily closes and rebuild rolling-volatility data.

数据源演进
----------
- 原实现依赖 ``akshare``。但该包未安装，且其依赖树庞大，装进主流程所用的
  Python 环境可能牵动 pandas / numpy 等核心依赖，风险高于收益。
- 本脚本仅需要「一个指数的日线收盘价」，因此 **2026-09-02 起改用腾讯行情公开
  接口**，只用到主流程已有的 ``requests``，零新增依赖。

口径一致性（切换前已验证，勿再重复排查）
----------------------------------------
- 腾讯 vs akshare：792 个重叠交易日，最大绝对差 **0.005**（腾讯保留 3 位小数
  vs akshare 4 位的舍入差异），差异 > 0.01 的天数为 **0** → 口径一致。
- 用现有 close 序列重算 hv20/40/60 与原文件对比：最大差 **0.00005**，属浮点
  舍入 → 重算等价于原值。

更新策略：增量合并（历史零变更）
------------------------------
- 以现有 JSON 的 close 序列为基座，**只追加缺失的新日期**，已有日期的数值
  一律不动。
- hv 用合并后的完整序列重算。但序列起点为 2016-01-04，前 ``window`` 行缺少
  2016 年之前的历史而无法重算，这部分**沿用原文件值**（已验证等价）。
- 保留「行情源日期回退则拒绝更新」的保护。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "hs300_volatility_data.json"
START_DATE = pd.Timestamp("2016-01-01")
ANNUALIZATION = 246
WINDOWS = (20, 40, 60)

# 腾讯行情：单次最多约 800 个交易日；带日期范围参数会返回空，只能靠数量截断
TX_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
TX_PARAMS = {"param": "sh000300,day,,,800,qfq"}
TX_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
TX_TIMEOUT = 40


def _load_existing() -> dict | None:
    """读取现有 JSON 作为历史基座；不存在或损坏则返回 None。"""
    if not OUTPUT.exists():
        return None
    try:
        data = json.loads(OUTPUT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not {"dates", "close"}.issubset(data):
        return None
    return data


def _fetch_closes() -> dict[str, float]:
    """从腾讯行情取沪深300日线收盘价，返回 {date_str: close}。

    返回行格式为 ``[日期, 开, 收, 高, 低, 成交量]``，收盘价在索引 2。
    """
    resp = requests.get(TX_URL, params=TX_PARAMS, headers=TX_HEADERS, timeout=TX_TIMEOUT)
    payload = resp.json()
    node = payload.get("data")
    if not isinstance(node, dict) or "sh000300" not in node:
        raise RuntimeError(f"腾讯行情返回结构异常: {str(payload)[:200]}")
    klines = node["sh000300"].get("day") or node["sh000300"].get("qfqday")
    if not klines:
        raise RuntimeError("腾讯行情 K 线为空")

    out: dict[str, float] = {}
    for row in klines:
        date, close = str(row[0]), float(row[2])
        if close > 0:
            out[date] = close
    if not out:
        raise RuntimeError("沪深300有效收盘价为空")
    return out


def build_payload() -> dict:
    fetched = _fetch_closes()
    fetched_last = max(fetched)

    existing = _load_existing()
    if existing is None:
        raise RuntimeError(
            f"{OUTPUT.name} 缺失，且公开接口单次最多约 800 个交易日，无法重建 2016 年以来的"
            "完整历史。请从 git 恢复该文件，或临时安装 akshare 做一次全量重建。"
        )

    existing_last = existing.get("last_date") or existing["dates"][-1]
    if fetched_last < existing_last:
        raise RuntimeError(f"行情源最新日期{fetched_last}早于现有文件{existing_last}，拒绝回退")

    # 只追加现有文件缺失的日期，已有日期保持原值（历史零变更）
    hist_close = dict(zip(existing["dates"], existing["close"]))
    added = sorted(d for d in fetched if d not in hist_close)

    merged = dict(hist_close)
    for d in added:
        merged[d] = fetched[d]

    dates = sorted(merged)
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "close": [merged[d] for d in dates]})
    log_return = np.log(frame["close"]).diff()
    for window in WINDOWS:
        frame[f"hv{window}"] = (
            log_return.rolling(window).std(ddof=1) * math.sqrt(ANNUALIZATION) * 100
        )

    # 前 window 行缺少 2016 年之前的历史，无法重算 → 沿用原文件值（已验证等价）
    n_hist = len(existing["dates"])
    for window in WINDOWS:
        col = frame[f"hv{window}"].tolist()
        old = dict(zip(existing["dates"], existing.get(f"hv{window}", [])))
        for i in range(min(window, n_hist)):
            if pd.isna(col[i]):
                col[i] = old.get(dates[i])
        frame[f"hv{window}"] = col

    frame = frame.loc[frame["date"] >= START_DATE].copy()
    frame = frame.dropna(subset=[f"hv{window}" for window in WINDOWS])
    if frame.empty:
        raise RuntimeError("2016年以来的HV数据为空")

    return {
        "symbol": existing.get("symbol", "000300.SH"),
        "name": existing.get("name", "沪深300指数"),
        "annualization": ANNUALIZATION,
        "windows": list(WINDOWS),
        "first_date": frame["date"].iloc[0].strftime("%Y-%m-%d"),
        "last_date": frame["date"].iloc[-1].strftime("%Y-%m-%d"),
        "dates": frame["date"].dt.strftime("%Y-%m-%d").tolist(),
        "close": frame["close"].round(4).tolist(),
        **{f"hv{window}": frame[f"hv{window}"].round(4).tolist() for window in WINDOWS},
    }


def main() -> bool:
    before = _load_existing()
    before_last = before.get("last_date") if before else None
    payload = build_payload()
    temp = OUTPUT.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(OUTPUT)
    print(
        f"updated {OUTPUT.name}: {payload['first_date']} -> {payload['last_date']}, "
        f"{len(payload['dates'])} observations"
        + (f" (新增 {len(payload['dates']) - len(before['dates'])} 天，原止于 {before_last})" if before else "")
    )
    return True


if __name__ == "__main__":
    main()
