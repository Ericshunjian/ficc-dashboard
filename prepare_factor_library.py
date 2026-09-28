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

# ---------------- Carry 结构因子定义（2026-09-28 加） ----------------
# 含义：按期货组合的 **名义金额** 加权的「券面收益 − 融资成本」。
#   期货价格 = 现货 − 持有成本 ⇒ carry 决定期货相对现货的贴水幅度，
#   也就给出了价差随时间漂移的方向。这是价差类目标唯一有明确理论依据的驱动项。
#
#   4TS−T：名义 400 万 TS（2 手 × 200 万/手） − 100 万 T，故权重 4 : 1；
#          T 的可交割券里 CTD 通常在 7 年附近，故用 7 年国债代表 T 腿。
#   3T−TL ：3 手 T − 1 手 TL，权重 3 : 1，腿分别取 10 年 / 30 年国债。
#
# 融资利率默认 **DR001-MA10**（用户指定：用平滑后的资金成本，剔除单日脉冲）。
# 4TS−T 额外给一份 DR001 原值版本，用于检验「MA10 平滑是否必要」。
# 单边界（TS/T/TL/10Y）是价差腿的分解，用来定位 carry 变化来自哪一端。
#
# 输出单位 **bp**（百分点 ×100）。
CARRY_DEFS = [
    ("ts4t", "价差", "Carry·4TS−T（DR001-MA10）",
     "4×(2年国债−DR001_MA10) − (7年国债−DR001_MA10)",
     [("2年国债", 4), ("7年国债", -1)], "DR001-MA10"),
    ("ts4t_raw", "价差", "Carry·4TS−T（DR001原值）",
     "4×(2年国债−DR001) − (7年国债−DR001)",
     [("2年国债", 4), ("7年国债", -1)], "DR001"),
    ("tl3t", "价差", "Carry·3T−TL（DR001-MA10）",
     "3×(10年国债−DR001_MA10) − (30年国债−DR001_MA10)",
     [("10年国债", 3), ("30年国债", -1)], "DR001-MA10"),
    ("s3010", "价差", "Carry·30Y−10Y（融资项抵消）",
     "(30年国债−DR001_MA10) − (10年国债−DR001_MA10)  ⇒ 等于期限利差",
     [("30年国债", 1), ("10年国债", -1)], "DR001-MA10"),
    # 分解项（2026-09-28 实测补入）：carry = (4y2 − y7) − 3r，两项对 4TS−T 的贡献相反，
    # 融资项实测为负贡献（IC −0.02 ~ −0.08），故单独入库以便直接对比"带融资 / 不带融资"。
    ("ts4t_curve", "分解", "Carry·4TS−T 曲线项（无融资）",
     "4×2年国债 − 7年国债", [("2年国债", 4), ("7年国债", -1)], None),
    ("tl3t_curve", "分解", "Carry·3T−TL 曲线项（无融资）",
     "3×10年国债 − 30年国债", [("10年国债", 3), ("30年国债", -1)], None),
    ("ts", "单边", "Carry·TS单边（2年−DR_MA10）",
     "2年国债 − DR001_MA10", [("2年国债", 1)], "DR001-MA10"),
    ("t", "单边", "Carry·T单边（7年−DR_MA10）",
     "7年国债 − DR001_MA10", [("7年国债", 1)], "DR001-MA10"),
    ("y10", "单边", "Carry·10Y单边（10年−DR_MA10）",
     "10年国债 − DR001_MA10", [("10年国债", 1)], "DR001-MA10"),
    ("tl", "单边", "Carry·TL单边（30年−DR_MA10）",
     "30年国债 − DR001_MA10", [("30年国债", 1)], "DR001-MA10"),
]


def _p(name):
    return os.path.join(BASE, name)


def _load(name):
    with open(_p(name), encoding="utf-8") as f:
        return json.load(f)


def _load_merged():
    """现券合并明细。

    优先用本机生成的 `bond_trading_data_merged.json`（~44MB，.gitignore 不入库）。
    若不存在（例如在家里 clone 的干净工作区），从入库的 `bond_trading_data.json`
    解码还原——后者是同一份数据的紧凑编码版（idx + 编码 detail），口径完全一致。
    """
    p = _p(F_PATH["merged"])
    if os.path.exists(p):
        return _load(F_PATH["merged"])
    alt = "bond_trading_data.json"
    if not os.path.exists(_p(alt)):
        raise FileNotFoundError(
            "缺少 %s（本机生成，不入库）且无 %s（入库）无法还原现券明细" % (F_PATH["merged"], alt)
        )
    b = _load(alt)
    ix = b["idx"]
    detail = [
        {
            "date": ix["dates"][r[0]],
            "bond_type": ix["bond_types"][r[1]],
            "institution": ix["institutions"][r[2]],
            "maturity": ix["maturities"][r[3]],
            "value": r[4],
        }
        for r in b["detail"]
    ]
    print("  [info] %s 不在工作区，已从 %s 还原 %d 条明细（同源同口径）"
          % (F_PATH["merged"], alt, len(detail)))
    return {"meta": dict(b["meta"]), "detail": detail}


def build():
    merged = _load_merged()
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

    # ---------- Carry 结构因子 ----------
    # 纯 Python 实现（不引 numpy）：n≈1500、8 条定义，逐日循环开销可忽略。
    _cache = {}

    def _series(name):
        """把 curve_out 的紧凑序列展开成全轴列表，缺日为 None"""
        if name in _cache:
            return _cache[name]
        full = None
        for s in curve_out:
            if s["name"] == name:
                full = [None] * n
                for i, v in enumerate(s["v"]):
                    full[s["i0"] + i] = v
                break
        _cache[name] = full
        return full

    carry_out = []
    for key, cat, cname, formula, legs, fund_name in CARRY_DEFS:
        if fund_name is None:          # 纯曲线项：不减融资成本
            fund = [0.0] * n
        else:
            fund = _series(fund_name)
            if fund is None:
                print("  [warn] carry 融资序列缺失：%s" % fund_name)
                continue
        acc = [None] * n
        ok = True
        for nm, w in legs:
            s = _series(nm)
            if s is None:
                print("  [warn] carry 腿缺失：%s" % nm)
                ok = False
                break
            for i in range(n):
                # 任一腿缺失该日即不可算（缺失不当 0）
                if s[i] is None or fund[i] is None:
                    acc[i] = None
                else:
                    d = w * (s[i] - fund[i]) * 100      # 百分点 → bp
                    acc[i] = d if acc[i] is None else acc[i] + d
        if not ok:
            continue
        idxs = [i for i, v in enumerate(acc) if v is not None]
        if len(idxs) < 30:
            continue
        i0 = idxs[0]
        vals = [None if v is None else round(v, 4) for v in acc[i0:idxs[-1] + 1]]
        carry_out.append({"key": key, "cat": cat, "name": cname, "formula": formula,
                          "i0": i0, "v": vals})
    print("  carry 因子 %d 条" % len(carry_out))

    # ---------- 现券衍生因子（机构行为三视角变换） ----------
    derived, dmeta = [], {"institutions": [], "tenors": [], "classes": []}
    try:
        import factor_derived
        t_ser = curve["series"]["T主力"]
        futures_dates = [d for d, v in zip(t_ser["dates"], t_ser["values"])
                         if v is not None]
        derived, dmeta = factor_derived.build(merged, idx, n, session_dates=futures_dates)
    except Exception as e:  # 衍生层失败不阻断主流程
        print("  [warn] 现券衍生因子生成失败：%r" % (e,))

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
            "units": {"cash": "亿元", "repo": "亿元 / 百分点", "curve": "%（如 1.6829 = 1.6829%）",
                      "carry": "bp（券面收益 − 融资成本，按期货组合名义金额加权）"},
            "groups": [
                {"key": "cash", "name": "机构行为 · 现券", "count": len(cash),
                 "dims": {"bond_type": CASH_BOND_TYPES, "institution": insts, "maturity": mats}},
                {"key": "repo", "name": "机构行为 · 质押式回购", "count": len(repo_out),
                 "dims": {"institution": insts, "field": [f[1] for f in REPO_FIELDS]}},
                {"key": "curve", "name": "估值 · 曲线与期货", "count": len(curve_out),
                 "dims": {"cat": list(curve_cats.keys())}},
                {"key": "derived", "name": "机构行为 · 现券衍生因子", "count": len(derived),
                 "dims": {"cls": dmeta["classes"], "institution": dmeta["institutions"],
                          "tenor": dmeta["tenors"]},
                 # 配置盘/交易盘的机构归属随期限档变化，页面据此展示
                 "cfg_sides": dmeta.get("cfg_sides", {})},
                # ★ carry 追加在末尾：页面多处按 groups[0..3] 硬编码取前四组，插队会错位
                {"key": "carry", "name": "估值 · Carry 结构", "count": len(carry_out),
                 "dims": {"cat": ["价差", "单边", "分解"]}},
            ],
            "note": "L0 原始序列池。变换（MA / 滚动百分位 / Z-score / 差分）在前端实时计算，不预存。",
        },
        "dates": dates,
        "cash": cash,
        "repo": repo_out,
        "curve": curve_out,
        "derived": derived,
        "carry": carry_out,
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
