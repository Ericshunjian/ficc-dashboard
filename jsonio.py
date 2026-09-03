# -*- coding: utf-8 -*-
"""JSON 写入辅助：无实质变化时跳过重写。

背景（2026-09-03）：每日更新脚本在**没有新交易日数据**时仍会重写 JSON，
唯一变化是 meta.last_updated / meta.generated 时间戳。带来的两个问题：
  1. 每天产生一个只含时间戳的"虚假 commit"；
  2. 前端 IndexedDB 以 last_updated 判定是否有更新，时间戳一变就重新下载整个文件
     （bond_trading_data.json 6.27MB），无谓消耗。

策略：写入前与现有文件比对，忽略时间戳类字段；实质等价则跳过重写。
与 repo_data_update.py 的做法保持一致。
"""
import json
import logging
import os

logger = logging.getLogger(__name__)

# 这些字段每次运行必然变化，比对时忽略
IGNORE_KEYS = ("last_updated", "generated")


def _strip(obj):
    """递归去掉时间戳类字段的值（置 None 占位，保持键结构完整）。"""
    if isinstance(obj, dict):
        return {k: (None if k in IGNORE_KEYS else _strip(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


def write_json_skip_unchanged(path, payload, log=None):
    """写入 JSON；若与现有文件实质等价则跳过。

    返回 True=已重写，False=跳过（文件保持原样）。
    """
    log = log or logger
    name = os.path.basename(path)

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                old = json.load(f)
            if _strip(old) == _strip(payload):
                log.info(f"  数据无变化，跳过重写 {name}")
                return False
        except Exception as e:  # 读不出/解析失败 → 保守重写
            log.warning(f"  比对现有 {name} 失败（{e}），直接重写")

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    log.info(f"  输出: {name} ({os.path.getsize(path) / 1024:.1f} KB)")
    return True
