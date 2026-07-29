# -*- coding: utf-8 -*-
"""
个券收益率对拟合曲线偏离度计算
- 曲线：yield_curve_data.json 中"债券"类 30 条期限点曲线（国债/国开/地方债）
- 个券：bond_yield_data.json 41 只券，duration 字段 = 数据起始日剩余期限
- 偏离度 dev_bp = (个券YTM - 同类别曲线当日线性插值[按当日剩余期限]) * 100
- 50 年国债超出曲线最长节点(30Y)：dev_bp=None，另给 spread30_bp（对 30Y 节点利差）
输出：curve_deviation.json
"""
import json
import logging
import os
import re
from datetime import date, datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CURVE_RE = re.compile(r"^(\d+)年(国债|国开|地方债)$")
CATEGORY_TO_CURVE = {
    "50年国债": "国债", "30年国债": "国债", "7-10年国债": "国债", "2-7年国债": "国债",
    "7-10年国开": "国开", "2-7年国开": "国开",
    "地方债": "地方债",
}


def build_curve_nodes(yc_series):
    """从 yield_curve_data.json 的 series key 动态发现曲线节点（如 15年国债 加列后自动生效）。"""
    nodes = {}
    for key in yc_series:
        m = CURVE_RE.match(key)
        if m:
            nodes.setdefault(m.group(2), []).append(int(m.group(1)))
    for k in nodes:
        nodes[k] = sorted(nodes[k])
    return nodes


def interp(x, xs, ys):
    """线性插值；超出节点范围返回 None。"""
    if x < xs[0] or x > xs[-1]:
        return None
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def main():
    yc = json.load(open(os.path.join(SCRIPT_DIR, "yield_curve_data.json"), encoding="utf-8"))
    bd = json.load(open(os.path.join(SCRIPT_DIR, "bond_yield_data.json"), encoding="utf-8"))

    # 曲线节点动态发现（Excel 加列如 15年/20年/50年国债 后自动生效）
    CURVE_NODES = build_curve_nodes(yc["series"])
    log.info(f"  曲线节点: " + ", ".join(f"{k}={v}" for k, v in CURVE_NODES.items()))

    # 曲线按日期重排：{类别: {日期: {期限: 收益率}}}
    curve_by_date = {}
    for cname, nodes in CURVE_NODES.items():
        per_date = {}
        for n in nodes:
            key = f"{n}年{cname}"
            ser = yc["series"].get(key)
            if not ser:
                log.warning(f"  缺少曲线 {key}")
                continue
            for ds, v in zip(ser["dates"], ser["values"]):
                if v is None:
                    continue
                per_date.setdefault(ds, {})[n] = v
        curve_by_date[cname] = per_date
        log.info(f"  曲线 {cname}: 节点 {len(nodes)} 个, 日期 {len(per_date)} 天")

    out_bonds = {}
    all_dates = set()
    n_out_of_range = 0
    for code, it in bd["series"].items():
        cat = it["category"]
        cname = CATEGORY_TO_CURVE[cat]
        nodes = CURVE_NODES[cname]
        d0 = date.fromisoformat(it["dates"][0])
        dur0 = float(it["duration"])
        dates, yields, rems, devs, spr30 = [], [], [], [], []
        for ds, y in zip(it["dates"], it["yields"]):
            if y is None:
                continue
            t = date.fromisoformat(ds)
            rem = dur0 - (t - d0).days / 365.0
            cmap = curve_by_date[cname].get(ds)
            if not cmap:
                continue
            ys = [cmap.get(n) for n in nodes]
            if any(v is None for v in ys):
                continue
            cy = interp(rem, nodes, ys)
            if cy is None:
                n_out_of_range += 1
                dev = None
                if rem > nodes[-1]:
                    # 超出曲线最长端（50年国债）：对30Y节点利差有参考意义
                    s30 = round((y - cmap[nodes[-1]]) * 100, 2)
                else:
                    # 低于最短节点（<1Y）：无可比基准，不给兜底值
                    s30 = None
            else:
                dev = round((y - cy) * 100, 2)
                s30 = None
            dates.append(ds)
            yields.append(y)
            rems.append(round(rem, 3))
            devs.append(dev)
            spr30.append(s30)
            all_dates.add(ds)
        out_bonds[code] = {
            "category": cat, "curve": cname,
            "dates": dates, "yields": yields, "rem": rems,
            "dev_bp": devs, "spread30_bp": spr30,
        }

    # 曲线节点序列（仅保留个券覆盖日期，控制体积）
    out_curves = {}
    for cname, per_date in curve_by_date.items():
        nodes = CURVE_NODES[cname]
        dates = sorted(d for d in per_date if d in all_dates)
        out_curves[cname] = {
            "nodes": nodes,
            "dates": dates,
            "values": [[per_date[d].get(n) for n in nodes] for d in dates],
        }

    tn = CURVE_NODES.get("国债", [])
    gap_note = ""
    if tn and not any(10 < n < 30 for n in tn):
        gap_note = "国债曲线10-30Y之间暂无中间节点，该区间为两点线性插值；真实曲线呈凸形，15-25Y券偏离度系统性偏高（含插值误差），同剩余期限券之间相对比较仍有效。"
    out = {
        "meta": {
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dev_definition": "dev_bp = (个券YTM - 同类别曲线按当日剩余期限线性插值) × 100，单位bp；正值=收益率高于曲线=现券价格偏便宜",
            "curve_nodes": CURVE_NODES,
            "note_50y": "剩余期限超出曲线最长节点的券（如50年国债>30Y节点）：dev_bp=null，以spread30_bp（对最长节点利差）参考；若曲线补充50年节点则自动转为正常偏离度",
            "note_interp": ("节点间线性插值。" + gap_note) if gap_note else "节点间线性插值。",
            "note_short": "剩余期限<曲线最短节点的券：dev_bp与spread30_bp均为null",
            "bond_date_range": bd["meta"]["date_range"],
        },
        "curves": out_curves,
        "bonds": out_bonds,
    }
    out_path = os.path.join(SCRIPT_DIR, "curve_deviation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    log.info(f"  输出 curve_deviation.json ({os.path.getsize(out_path)/1024:.1f} KB)")
    log.info(f"  个券 {len(out_bonds)} 只, 曲线外插值(50Y)记录 {n_out_of_range} 条")

    # 最新一天偏离度一览
    latest = max(all_dates)
    log.info(f"  最新日期 {latest} 偏离度(bp):")
    rows = []
    for code, b in out_bonds.items():
        if b["dates"] and b["dates"][-1] == latest:
            d = b["dev_bp"][-1]
            s30 = b["spread30_bp"][-1]
            rows.append((code, b["category"], b["rem"][-1], b["yields"][-1],
                         d if d is not None else s30,
                         d is None and s30 is not None))
    rows.sort(key=lambda r: (r[5], -(r[4] if r[4] is not None else -999)))
    for code, cat, rem, y, d, is50 in rows:
        tag = "(对30Y)" if is50 else ""
        ds = f"{d:+7.2f}bp" if d is not None else "    N/A(<1Y)"
        log.info(f"    {code:12} {cat:8} {rem:6.2f}Y  {y:7.4f}%  {ds}{tag}")


if __name__ == "__main__":
    main()
