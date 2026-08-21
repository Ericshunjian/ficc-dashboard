"""Fetch CSI 300 daily closes and rebuild rolling-volatility data."""
from __future__ import annotations

import json
import math
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "hs300_volatility_data.json"
START_DATE = pd.Timestamp("2016-01-01")
ANNUALIZATION = 246
WINDOWS = (20, 40, 60)


def _existing_last_date() -> str | None:
    if not OUTPUT.exists():
        return None
    try:
        return json.loads(OUTPUT.read_text(encoding="utf-8")).get("last_date")
    except (OSError, ValueError, TypeError):
        return None


def build_payload() -> dict:
    raw = ak.stock_zh_index_daily(symbol="sh000300")
    if raw.empty or not {"date", "close"}.issubset(raw.columns):
        raise RuntimeError("沪深300行情为空或缺少date/close列")

    frame = raw.loc[:, ["date", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna().drop_duplicates("date", keep="last").sort_values("date")
    frame = frame.loc[frame["close"] > 0].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("沪深300有效收盘价为空")

    log_return = np.log(frame["close"]).diff()
    for window in WINDOWS:
        frame[f"hv{window}"] = log_return.rolling(window).std(ddof=1) * math.sqrt(ANNUALIZATION) * 100

    frame = frame.loc[frame["date"] >= START_DATE].copy()
    frame = frame.dropna(subset=[f"hv{window}" for window in WINDOWS])
    if frame.empty:
        raise RuntimeError("2016年以来的HV数据为空")

    last_date = frame["date"].iloc[-1].strftime("%Y-%m-%d")
    existing_last = _existing_last_date()
    if existing_last and last_date < existing_last:
        raise RuntimeError(f"行情源最新日期{last_date}早于现有文件{existing_last}，拒绝回退")

    return {
        "symbol": "000300.SH",
        "name": "沪深300指数",
        "annualization": ANNUALIZATION,
        "windows": list(WINDOWS),
        "first_date": frame["date"].iloc[0].strftime("%Y-%m-%d"),
        "last_date": last_date,
        "dates": frame["date"].dt.strftime("%Y-%m-%d").tolist(),
        "close": frame["close"].round(4).tolist(),
        **{
            f"hv{window}": frame[f"hv{window}"].round(4).tolist()
            for window in WINDOWS
        },
    }


def main() -> bool:
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
    )
    return True


if __name__ == "__main__":
    main()
