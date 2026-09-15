# -*- coding: utf-8 -*-
"""
因子库（L0 原始序列池）生成器

产出 factor_library.json，供 factor_library.html 前端做 L1 实时变换（MA / 百分位 / Z-score / 差分）。

数据来源（全部为本仓库已生成的 JSON，不新增外部依赖）：
  - bond_trading_data_merged.json  现券机构净买入（亿元）
  - repo_trading_data.json         质押式回购（亿元 / %）
  - yield_curve_data.json          收益率曲线与期货（%，注意：内部存百分数）

入选口径（与 2026-09-15 约定一致）：
  - 现券：券种 = 国债 / 政金债 / 地方债（信用债、同业存单因无配置含义剔除）
          8 机构 × 全部期限档（含"合计"）
  - 回购：8 机构 × [净融出 net / 逆回购余额 / 正回购余额 / 正回购加权利率-DR001]
          （杠杆代理不做：无债券托管数据）
  - 曲线：52 条全留（基准 / IRS / 期货 / 债券）

存储格式：全局日期轴 + 每条序列 {i0: 起始索引, v: [...]}，序列前端缺失的部分不占位。
"""
import json
import os
import sys
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

CASH_BOND_TYPES = ["国债", "政金债", "地方债"]

REPO_FIELDS = [
    ("net", "净融出"),
    ("rev_bal", "逆回购余额"),
    ("repo_bal", "正回购余额"),
    ("rate_spread", "正回购利率-DR001"),
]

F_PATH = {
    "merged": "bond_trading_data_merged.json",
    "repo": "repo_trading_data.json",
    "curve": "yield_curve_data.json",
    "out": "factor_library.json",
}


def _p(name):
    return os.path.join(BASE, name)


def _load(name):
    with open(_p(name), encoding="utf-8") as f:
        return json.load(f)


def build():
    merged = _load(F_PATH["merged"])
    repo = _load(F_PATH["repo"])
    curve = _load(F_PATH["curve"])

    # ---------- 全局日期轴：三源并集 ----------
    all_dates = set(repo["dates"])
    for r in merged["detail"]:
        all_dates.add(r["date"])
    for name, ser in curve["series"].items():
        all_dates.update(ser["dates"] if isinstance(ser, dict) else [])
    dates = sorted(all_dates)
    idx = {d: i for i, d in enumerate(dates)}
    n = len(dates)

    # ---------- 现券 ----------
    cash_map = {}
    for r in merged["detail"]:
        if r["bond_type"] not in CASH_BOND_TYPES:
            continue
        key = (r["bond_type"], r["institution"], r["maturity"])
        cash_map.setdefault(key, {})[r["date"]] = r["value"]

    cash = []
    for (bt, inst, mat), dv in sorted(cash_map.items()):
        items = sorted((idx[d], v) for d, v in dv.items() if d in idx)
        if not items:
            continue
        i0 = items[0][0]
        # 中间缺失补 None，尾部截断
        vals = [None] * (items[-1][0] - i0 + 1)
        for i, v in items:
            vals[i - i0] = round(v, 2)
        cash.append({"bt": bt, "inst": inst, "mat": mat, "i0": i0, "v": vals})

    # ---------- 回购 ----------
    # DR001 用于计算利率利差
    dr001 = {}
    s = curve["series"].get("DR001")
    if isinstance(s, dict):
        dr001 = dict(zip(s["dates"], s["values"]))
    else:
        dr001 = dict(zip(curve["dates"], s or []))

    repo_out = []
    for inst, fields in repo["inst"].items():
        for fkey, flabel in REPO_FIELDS:
            if fkey == "rate_spread":
                base = fields.get("repo_rate") or []
                vals_src = [
                    (round(base[i] - dr001[d], 4) if (d in dr001 and base[i] is not None) else None)
                    for i, d in enumerate(repo["dates"])
                ]
            else:
                vals_src = fields.get(fkey) or []
            arr = []
            for i, d in enumerate(repo["dates"]):
                if d not in idx:
                    continue
                v = vals_src[i] if i < len(vals_src) else None
                arr.append((idx[d], None if v is None else round(v, 4)))
            arr = [x for x in arr if x[1] is not None]
            if not arr:
                continue
            i0 = arr[0][0]
            vals = [None] * (arr[-1][0] - i0 + 1)
            for i, v in arr:
                vals[i - i0] = v
            repo_out.append({"inst": inst, "field": fkey, "label": flabel, "i0": i0, "v": vals})

    # ---------- 曲线 ----------
    curve_out = []
    curve_cats = curve["meta"].get("categories", {})
    cat_of = {}
    for c, names in curve_cats.items():
        for nm in names:
            cat_of[nm] = c
    for name, ser in curve["series"].items():
        if isinstance(ser, dict):
            sdates, svals = ser["dates"], ser["values"]
        else:
            sdates, svals = curve["dates"], ser
        arr = [(idx[d], v) for d, v in zip(sdates, svals) if d in idx and v is not None]
        if not arr:
            continue
        i0 = arr[0][0]
        vals = [None] * (arr[-1][0] - i0 + 1)
        for i, v in arr:
            vals[i - i0] = round(v, 4)
        curve_out.append({"name": name, "cat": cat_of.get(name, "其他"), "i0": i0, "v": vals})
    curve_out.sort(key=lambda x: (x["cat"], x["name"]))

    insts = merged["meta"]["institutions"]
    mats = merged["meta"]["maturities"]

    payload = {
        "meta": {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date_range": [dates[0], dates[-1]],
            "n_dates": n,
            "sources": {
                "cash": "bond_trading_data_merged.json（现券机构净买入，亿元）",
                "repo": "repo_trading_data.json（质押式回购，亿元 / %）",
                "curve": "yield_curve_data.json（%，内部存百分数：123.02 = 1.2302%）",
            },
            "units": {"cash": "亿元", "repo": "亿元 / 百分点", "curve": "%（百分数存储）"},
            "groups": [
                {"key": "cash", "name": "机构行为 · 现券", "count": len(cash),
                 "dims": {"bond_type": CASH_BOND_TYPES, "institution": insts, "maturity": mats}},
                {"key": "repo", "name": "机构行为 · 质押式回购", "count": len(repo_out),
                 "dims": {"institution": insts, "field": [f[1] for f in REPO_FIELDS]}},
                {"key": "curve", "name": "估值 · 曲线与期货", "count": len(curve_out),
                 "dims": {"cat": list(curve_cats.keys())}},
            ],
            "note": "L0 原始序列池。变换（MA / 滚动百分位 / Z-score / 差分）在前端实时计算，不预存。",
        },
        "dates": dates,
        "cash": cash,
        "repo": repo_out,
        "curve": curve_out,
    }
    return payload


def main():
    payload = build()
    out = _p(F_PATH["out"])
    # 幂等写入：数据无实质变化时不重写（避免虚假 commit 与前端无谓重下）
    try:
        import jsonio
        jsonio.write_json_skip_unchanged(out, payload)
    except Exception:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    size = os.path.getsize(out)
    m = payload["meta"]
    print(f"写出 {out}")
    print(f"  {size/1e6:.2f} MB   日期 {m['date_range'][0]} ~ {m['date_range'][1]}（{m['n_dates']} 天）")
    for g in m["groups"]:
        print(f"  {g['name']}: {g['count']} 条")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
