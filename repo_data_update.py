# -*- coding: utf-8 -*-
"""
质押式回购数据更新：扫描日报 → 解析 → repo_trading_data.json (+ repo_data_test.xlsx)

数据源（3 个目录，自动识别新旧格式）：
  C:\\Users\\lihaoran\\Documents\\工作\\质押式回购\\           2021-01-04 ~ 2025-04-02 旧格式(.xls, 11机构)
  C:\\Users\\lihaoran\\Documents\\工作\\质押式回购\\2025年\\     2025-04-03 ~ 2025-12-31 待用户补充（新旧格式均可）
  C:\\Users\\lihaoran\\Documents\\工作\\质押式回购\\2026年\\     2026-01-05 ~ 今        新格式(.xlsx, 8机构)

格式差异（列布局完全一致，只有行位置/机构数不同）：
  旧格式 (207行)：表二表头在 header=None 行 8，机构×11期限 行 9-129（11机构），表三押品表头行 132 起（11机构×3押品）
  新格式 (128行)：表二表头在 header=None 行 6，机构×11期限 行 7-94（8机构），表三押品表头行 97 起（8机构×3押品）
  → 解析按「内容定位表头」而非固定行号，两种格式通吃；未来用户补的 2025 年文件无论哪种格式都能读

表二列布局：0=机构类型 1=期限品种 2=正回购加权利率 3=正回购-剔除超50BP 4=正回购金额
            5=逆回购加权利率 6=逆回购-剔除超50BP 7=逆回购金额 8=净融入金额 9=正回购余额 10=逆回购余额
表三列布局：0=机构类型 1=债券类型 2=正回购加权利率 3=正回购金额 4=逆回购加权利率 5=逆回购金额

机构映射（旧 11 → 新 8，按 2026 新格式官方注释口径：
  大型银行=大商行+政策性+股份制；中小型银行=城商行+农商行等）：
  大型商业/政策性银行 → 大型银行；股份制商业银行 → 大型银行
  城市商业银行 → 中小型银行；农村金融机构 → 中小型银行
  其他产品类 → 其他；其余同名直映

输出：
  repo_trading_data.json（FICC 项目目录，git 入库，看板数据源）
  repo_data_test.xlsx（质押式回购\\2026年\\，用户人工查看用，9 个 sheet）
"""
import json
import os
import re
import sys
import logging
from datetime import datetime

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_BASE = r"C:\Users\lihaoran\Documents\工作\质押式回购"
REPO_DIR_OLD = REPO_BASE                                   # 2021-01-04 ~ 2025-04-02 旧格式
REPO_DIR_2025 = os.path.join(REPO_BASE, "2025年")          # 2025-04-03 ~ 2025-12-31（待补）
REPO_DIR_2026 = os.path.join(REPO_BASE, "2026年")          # 2026-01-05 ~ 新格式
REPO_JSON_OUT = os.path.join(SCRIPT_DIR, "repo_trading_data.json")
REPO_XLSX_OUT = os.path.join(REPO_DIR_2026, "repo_data_test.xlsx")

REPORT_PATTERN = re.compile(r"质押式回购市场交易情况总结日报_(\d{8})\.(xls|xlsx)$", re.IGNORECASE)

INSTITUTIONS = [
    "大型银行", "中小型银行", "证券公司", "保险公司",
    "基金公司及产品", "货币市场基金", "理财子公司及理财类产品", "其他",
]
COLLATERALS = ["利率债", "信用债", "同业存单"]

# 旧格式 11 机构 → 新 8 机构（按新格式官方注释口径，与现券页历史映射不同：股份制归大型银行）
OLD_TO_NEW = {
    "大型商业/政策性银行": "大型银行",
    "股份制商业银行": "大型银行",
    "城市商业银行": "中小型银行",
    "农村金融机构": "中小型银行",
    "证券公司": "证券公司",
    "保险公司": "保险公司",
    "基金公司及产品": "基金公司及产品",
    "货币市场基金": "货币市场基金",
    "理财子公司及理财类产品": "理财子公司及理财类产品",
    "其他产品类": "其他",
    "其他": "其他",
}

# 数值字段（单位换算：百万 → 亿）
AMOUNT_FIELDS = ["repo_amt", "rev_amt", "repo_bal", "rev_bal"]  # /100
RATE_FIELDS = ["repo_rate", "rev_rate"]                          # % 原样

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(SCRIPT_DIR, "repo_update.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def _num(v):
    """单元格 → float；'-'/空/非数值 → None"""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s in ("-", "", "—", "--"):
            return None
        try:
            return float(s)
        except ValueError:
            return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _add(dst, key, v, scale=1.0):
    if v is None:
        return
    dst[key] = dst.get(key, 0.0) + v * scale


def _new_inst_rec():
    return {"repo_amt": 0.0, "rev_amt": 0.0, "repo_bal": 0.0, "rev_bal": 0.0,
            "rate_num": 0.0, "rate_den": 0.0, "rev_rate_num": 0.0, "rev_rate_den": 0.0,
            "coll": {c: {"repo": 0.0, "rev": 0.0} for c in COLLATERALS}}


def parse_report(path):
    """解析一份日报 → {新机构名: 字段dict}；失败返回 None"""
    try:
        df = pd.read_excel(path, header=None)
    except Exception as e:
        log.warning(f"  读取失败 {os.path.basename(path)}: {e}")
        return None

    n = len(df)
    # ── 内容定位两张表的表头行（按第2列关键词；个别文件首列"机构类型"空缺，如 20230728）──
    t2_head = t3_head = None
    for i in range(n):
        c1 = df.iloc[i, 1]
        if not isinstance(c1, str):
            continue
        c1 = c1.strip()
        if c1 == "期限品种" and t2_head is None:
            t2_head = i
        elif c1 == "债券类型" and t3_head is None:
            t3_head = i
    if t2_head is None or t3_head is None:
        log.warning(f"  未找到表头（表二={t2_head}, 表三={t3_head}）: {os.path.basename(path)}")
        return None
    if t3_head <= t2_head:
        t3_head = n  # 容错：表三缺失则表二解析到文件尾

    recs = {}

    # ── 表二：机构 × 期限 ──
    cur = None
    for i in range(t2_head + 1, t3_head):
        c0 = df.iloc[i, 0]
        c1 = df.iloc[i, 1]
        c0s = str(c0).strip() if isinstance(c0, str) and c0.strip() else None
        if c0s == "机构类型":  # 又一张表（不应发生）
            break
        if c0s:
            cur = OLD_TO_NEW.get(c0s, c0s)
            if cur not in recs:
                recs[cur] = _new_inst_rec()
        if cur is None:
            continue
        tenor = str(c1).strip() if isinstance(c1, str) and c1.strip() else None
        if not tenor:
            continue
        r = recs[cur]
        repo_amt = _num(df.iloc[i, 4])
        rev_amt = _num(df.iloc[i, 7])
        repo_bal = _num(df.iloc[i, 9])
        rev_bal = _num(df.iloc[i, 10])
        _add(r, "repo_amt", repo_amt, 0.01)
        _add(r, "rev_amt", rev_amt, 0.01)
        _add(r, "repo_bal", repo_bal, 0.01)
        _add(r, "rev_bal", rev_bal, 0.01)
        # 加权利率：按成交金额加权（利率缺失的行不计入分子分母）
        repo_rate = _num(df.iloc[i, 2])
        rev_rate = _num(df.iloc[i, 5])
        if repo_rate is not None and repo_amt:
            r["rate_num"] += repo_rate * repo_amt
            r["rate_den"] += repo_amt
        if rev_rate is not None and rev_amt:
            r["rev_rate_num"] += rev_rate * rev_amt
            r["rev_rate_den"] += rev_amt

    # ── 表三：机构 × 押品 ──
    cur = None
    for i in range(t3_head + 1, n):
        c0 = df.iloc[i, 0]
        c1 = df.iloc[i, 1]
        c0s = str(c0).strip() if isinstance(c0, str) and c0.strip() else None
        if c0s == "机构类型":
            break
        if c0s:
            cur = OLD_TO_NEW.get(c0s, c0s)
            if cur not in recs:
                recs[cur] = _new_inst_rec()
        coll = str(c1).strip() if isinstance(c1, str) and c1.strip() else None
        if cur is None or coll not in COLLATERALS:
            continue
        r = recs[cur]
        _add(r["coll"][coll], "repo", _num(df.iloc[i, 3]), 0.01)
        _add(r["coll"][coll], "rev", _num(df.iloc[i, 5]), 0.01)

    # 派生利率
    out = {}
    for inst, r in recs.items():
        out[inst] = {
            "repo_amt": r["repo_amt"], "rev_amt": r["rev_amt"],
            "repo_bal": r["repo_bal"], "rev_bal": r["rev_bal"],
            "repo_rate": (r["rate_num"] / r["rate_den"]) if r["rate_den"] > 0 else None,
            "rev_rate": (r["rev_rate_num"] / r["rev_rate_den"]) if r["rev_rate_den"] > 0 else None,
            "coll": r["coll"],
        }
    return out


def scan_source_files():
    """扫描 3 个目录 → {date_int: path}（同日多文件取序号最大者=最新目录）"""
    found = {}
    for d, order in ((REPO_DIR_OLD, 0), (REPO_DIR_2025, 1), (REPO_DIR_2026, 2)):
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            m = REPORT_PATTERN.match(f)
            if not m:
                continue
            di = int(m.group(1))
            p = os.path.join(d, f)
            if di not in found or order >= found[di][0]:
                found[di] = (order, p)
    return {di: p for di, (order, p) in found.items()}


def _di_to_str(di):
    s = str(di)
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def load_master():
    """读取现有 JSON master → (records, meta)；records = {date_str: {inst: fields}}"""
    if not os.path.exists(REPO_JSON_OUT):
        return {}, None
    try:
        with open(REPO_JSON_OUT, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        log.warning(f"读取现有 JSON 失败（将全量重建）: {e}")
        return {}, None
    records = {}
    dates = raw.get("dates", [])
    for inst, blk in raw.get("inst", {}).items():
        for k, arr in blk.items():
            if k == "coll":
                for coll, cb in blk["coll"].items():
                    for direction in ("repo", "rev"):
                        a = cb.get(direction) or []
                        for i, v in enumerate(a):
                            if v is None:
                                continue
                            d = dates[i]
                            records.setdefault(d, {}).setdefault(inst, {}).setdefault("coll", {}).setdefault(coll, {})[direction] = v
            else:
                for i, v in enumerate(arr or []):
                    if v is None:
                        continue
                    records.setdefault(dates[i], {}).setdefault(inst, {})[k] = v
    return records, raw.get("meta", {})


def build_output(records):
    """records → 列式结构 dict"""
    dates = sorted(records.keys())
    inst_blk = {inst: {"net": [], "repo_bal": [], "rev_bal": [], "repo_amt": [], "rev_amt": [],
                       "repo_rate": [], "rev_rate": [],
                       "coll": {c: {"repo": [], "rev": []} for c in COLLATERALS}}
                for inst in INSTITUTIONS}
    for d in dates:
        day = records[d]
        for inst in INSTITUTIONS:
            r = day.get(inst) or {}
            blk = inst_blk[inst]
            rb, vb = r.get("repo_bal"), r.get("rev_bal")
            blk["repo_bal"].append(None if rb is None else round(rb, 2))
            blk["rev_bal"].append(None if vb is None else round(vb, 2))
            # net 优先用已存储值（master 往返保真）；仅新解析日期（recs 无 net）才从余额重算。
            # 若无条件重算：第一次从原始全精度、之后从 round 后余额算，.005 边界会漂移 0.01（2023-07-28 实测 3 处）
            net = r.get("net")
            if net is None and (rb is not None or vb is not None):
                net = (vb or 0.0) - (rb or 0.0)
            blk["net"].append(None if net is None else round(net, 2))
            for k in ("repo_amt", "rev_amt"):
                v = r.get(k)
                blk[k].append(None if v is None else round(v, 2))
            for k in ("repo_rate", "rev_rate"):
                v = r.get(k)
                blk[k].append(None if v is None else round(v, 4))
            coll = r.get("coll") or {}
            for c in COLLATERALS:
                cb = coll.get(c) or {}
                for direction in ("repo", "rev"):
                    v = cb.get(direction)
                    blk["coll"][c][direction].append(None if v is None else round(v, 2))
    return dates, inst_blk


def write_xlsx(dates, inst_blk):
    """生成 repo_data_test.xlsx（9 sheet，供人工查看）"""
    idx = pd.Index(dates, name="日期")
    sheets = {}
    sheets["净融出"] = pd.DataFrame({i: inst_blk[i]["net"] for i in INSTITUTIONS}, index=idx)
    sheets["正回购余额"] = pd.DataFrame({i: inst_blk[i]["repo_bal"] for i in INSTITUTIONS}, index=idx)
    sheets["逆回购余额"] = pd.DataFrame({i: inst_blk[i]["rev_bal"] for i in INSTITUTIONS}, index=idx)
    sheets["正回购金额"] = pd.DataFrame({i: inst_blk[i]["repo_amt"] for i in INSTITUTIONS}, index=idx)
    sheets["逆回购金额"] = pd.DataFrame({i: inst_blk[i]["rev_amt"] for i in INSTITUTIONS}, index=idx)
    sheets["正回购加权利率"] = pd.DataFrame({i: inst_blk[i]["repo_rate"] for i in INSTITUTIONS}, index=idx)
    sheets["逆回购加权利率"] = pd.DataFrame({i: inst_blk[i]["rev_rate"] for i in INSTITUTIONS}, index=idx)
    for direction, name in (("repo", "押品正回购金额"), ("rev", "押品逆回购金额")):
        cols = {}
        for i in INSTITUTIONS:
            for c in COLLATERALS:
                cols[f"{i}_{c}"] = inst_blk[i]["coll"][c][direction]
        sheets[name] = pd.DataFrame(cols, index=idx)
    try:
        with pd.ExcelWriter(REPO_XLSX_OUT, engine="openpyxl") as w:
            for name, df in sheets.items():
                df.to_excel(w, sheet_name=name)
        log.info(f"  已生成 {os.path.basename(REPO_XLSX_OUT)}（{len(dates)} 天 × {len(sheets)} sheet）")
    except Exception as e:
        log.warning(f"  生成 xlsx 失败（不影响 JSON）: {e}")


def _payloads_equal(old_path, new_payload):
    """比较新旧数据是否实质等价（忽略 last_updated 时间戳）"""
    try:
        with open(old_path, encoding="utf-8") as f:
            old = json.load(f)
    except Exception:
        return False
    if old.get("dates") != new_payload["dates"]:
        return False
    om = dict(old.get("meta") or {}); om.pop("last_updated", None)
    nm = dict(new_payload["meta"]); nm.pop("last_updated", None)
    return om == nm and old.get("inst") == new_payload["inst"]


def main():
    log.info("质押式回购数据更新开始")
    records, old_meta = load_master()
    if old_meta:
        log.info(f"  现有 master: {len(records)} 天（{old_meta.get('date_range')}）")

    files = scan_source_files()
    if not files:
        log.warning("  未扫描到任何日报文件")
        return False

    missing = {di: p for di, p in files.items() if _di_to_str(di) not in records}
    log.info(f"  扫描到 {len(files)} 份日报，待解析 {len(missing)} 份")
    n_fail = 0
    for di in sorted(missing):
        recs = parse_report(missing[di])
        if not recs:
            n_fail += 1
            continue
        d = _di_to_str(di)
        if d in records:
            continue
        # 校验：8 机构是否齐全（缺则记 warning，不阻断）
        absent = [i for i in INSTITUTIONS if i not in recs]
        if absent:
            log.warning(f"  {d} 缺机构: {absent}")
        records[d] = recs

    if n_fail:
        log.warning(f"  {n_fail} 份文件解析失败（详见上方 warning）")
    if not records:
        log.error("  无任何有效数据")
        return False

    dates, inst_blk = build_output(records)

    # 缺口提示（连续交易日缺口 > 10 天时列出，方便用户发现该补数据）
    try:
        dts = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
        gaps = []
        for a, b in zip(dts, dts[1:]):
            span = (b - a).days
            if span >= 10:
                gaps.append(f"{a.date()}~{b.date()}（{span}天）")
        if gaps:
            log.info("  数据缺口（自然日≥10天）: " + "; ".join(gaps))
    except Exception:
        pass

    meta = {
        "format": "repo_v1",
        "data_source": "质押式回购市场交易情况总结日报（外汇交易中心）",
        "institutions": INSTITUTIONS,
        "collaterals": COLLATERALS,
        "date_range": [dates[0], dates[-1]],
        "total_days": len(dates),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "旧格式11→8机构映射（股份制并入大型银行，按新格式官方口径）；"
                "金额/余额单位亿元；净融出=逆回购余额-正回购余额；"
                "加权利率按当日成交金额加权",
    }

    payload = {"meta": meta, "dates": dates, "inst": inst_blk}

    # 幂等保护：无实质变化时不重写（保持 last_updated 不变 → 不产生虚假 commit，
    # 前端 IndexedDB 也不会因时间戳变化而重新下载 1MB）
    if os.path.exists(REPO_JSON_OUT) and _payloads_equal(REPO_JSON_OUT, payload):
        log.info("  数据无变化，跳过重写 JSON / xlsx")
        log.info("质押式回购数据更新完成（无变化）")
        return True

    with open(REPO_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    size_mb = os.path.getsize(REPO_JSON_OUT) / 1048576
    log.info(f"  已写出 repo_trading_data.json（{len(dates)} 天, {size_mb:.2f} MB, "
             f"区间 {dates[0]} ~ {dates[-1]}）")

    write_xlsx(dates, inst_blk)
    log.info("质押式回购数据更新完成")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
