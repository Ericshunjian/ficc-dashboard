# -*- coding: utf-8 -*-
"""L2 因子体检层 —— 对 factor_library 的每条序列做时序有效性检验。

设计要点（防坑清单）：
1. **机构行为 T+1 发布** → 因子默认 LAG=1 使用，t 日只能看到 t-1 的因子值。
2. **分组用滚动 250 日 Z-score**，绝不用全样本分位（那是未来函数）。
3. **t 值一律 Newey-West 修正**（N 日收益相邻样本重叠 N-1/N）。
4. **显著性用 FFT 循环置换检验**：保留 y 的完整自相关结构，一次算出全部 n 个
   循环移位的分布，得到精确 p 值。确定性、可复现、无随机种子，O(n log n)。
5. **多重检验 BH-FDR 5%**：每个 (形态×目标×周期) 组合内独立校正。
6. 目标定义配置化（TARGETS），以后加 4TS-T / TL 单边等只需加一行。

输出 factor_checkup.json，列为紧凑数组（列名见 meta.cols）以控制体积。
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import rankdata, kurtosis, t as tdist

BASE = os.path.dirname(os.path.abspath(__file__))
try:
    import jsonio
except Exception:
    jsonio = None

LIB_PATH = os.path.join(BASE, "factor_library.json")
OUT_PATH = os.path.join(BASE, "factor_checkup.json")
DETAIL_PATH = os.path.join(BASE, "factor_checkup_detail.json")

# ---------------- 配置 ----------------

FORMS = [("raw", "原值"), ("pct", "MA10→60日百分位"), ("diff", "5日差分")]

# 预测目标：配置化。以后扩展只需追加一行。
#   mode=yield   : legs 为 (曲线名, 权重)，y = Σ w·Y，取 N 日变动 ×100 → bp
#   mode=futures : legs 为 (期货主力名, 手数[, 面值百万])，默认面值 100 万
#                  y = Σ w·fv·ΔP / Σ|w|·fv·P × 100 → %
#                  若加 "as": "price"，则直接输出价差报价的点数变动 → 点
#   可选 "maxN"  : 该目标最长预测周期（样本短的品种用它限制，如 TL 仅 2023-04 起）
# ★ 方向约定：符号必须与该目标对应的收益率利差同向（3T−TL 与「30Y−10Y 走阔」同向）
TARGETS = [
    {"id": "y10", "name": "10Y国债", "mode": "yield",
     "legs": [("10年国债", 1)], "unit": "bp"},
    {"id": "s3010", "name": "30Y−10Y利差", "mode": "yield",
     "legs": [("30年国债", 1), ("10年国债", -1)], "unit": "bp"},
    {"id": "tl3t", "name": "3T−TL", "mode": "futures",
     "legs": [("T主力", 3), ("TL主力", -1)], "unit": "%", "maxN": 10},
    # ★ 4TS−T 的「4」= 名义 400 万 = 2 手 TS(200万/手) − 1 手 T(100万)
    {"id": "ts4t", "name": "4TS−T", "mode": "futures",
     "legs": [("TS主力", 2, 200), ("T主力", -1, 100)], "unit": "%"},
    {"id": "t", "name": "T单边", "mode": "futures",
     "legs": [("T主力", 1)], "unit": "点", "as": "price"},
    {"id": "tl", "name": "TL单边", "mode": "futures",
     "legs": [("TL主力", 1)], "unit": "点", "as": "price", "maxN": 10},
]

HORIZONS = [1, 5, 10, 20]
LAG = 1                 # 因子滞后使用天数（机构行为 T+1 发布）
MIN_OBS = 200           # 最小有效样本
Z_WIN = 250             # 分组所用滚动 Z-score 窗口
ICIR_WIN = 60           # 滚动 IC 窗口
FDR_ALPHA = 0.05
MIN_IC_STRONG = 0.05     # 判定显著所要求的最小 |IC|（防极端值撑高的假 t 值）
SIGNAL_Z = 1.0          # 胜率/盈亏比的开仓阈值（|Z|>1 才持仓）
TAIL_Z = 1.5            # 尾部赔率阈值
RECENT6 = 125           # "近 6 个月"交易日数
CURVE_PTS = 20          # 详情曲线降采样点数（控制 factor_checkup_detail.json 体积）

# 是否把「机构行为·现券衍生因子」纳入周一全量体检。
# 默认关闭：衍生层与原始 303 条同源（同一批净买入的变换），纳入后会显著加重
# 多重检验负担、并把周一耗时从 ~130s 拉长到 ~210s。
# 需要单独检验衍生层时，用环境变量临时打开，不改代码：
#     FICC_CHECKUP_DERIVED=1 python prepare_factor_checkup.py
INCLUDE_DERIVED = os.environ.get("FICC_CHECKUP_DERIVED", "0") == "1"
COND_EDGES = [0, 10, 25, 50, 75, 90, 100]   # 条件分布：滚动经验分位切点

COLS = ["g", "k", "label", "n", "ic", "icp", "icir", "t", "ti", "pind", "win", "wl", "ws",
        "pl", "tail", "q1", "q2", "q3", "q4", "q5", "qd", "qt", "mono", "turn",
        "kurt", "acf", "p", "pb", "sig", "h6",
        "yr21", "yr22", "yr23", "yr24", "yr25", "yr26",
        "rgu", "rgd", "rghv", "rglv"]


# ---------------- 基础工具 ----------------

def _r(x, nd=4):
    """round 且把 nan/inf 转成 None（JSON 合法）"""
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return round(v, nd)


def sliding_mean(a, w):
    """O(n) 滑动均值，前 w-1 个为 nan"""
    n = len(a)
    if n < w:
        return np.full(n, np.nan)
    c = np.concatenate([[0.0], np.cumsum(a)])
    out = np.full(n, np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def ma_nan(a, w, min_ratio=0.7):
    """窗口内非空比例 >= min_ratio 才输出均值，否则 nan"""
    n = len(a)
    if n < w:
        return np.full(n, np.nan)
    win = sliding_window_view(a, w)
    need = max(1, int(np.ceil(w * min_ratio)))
    with np.errstate(invalid="ignore"):
        s = np.nansum(np.where(np.isnan(win), 0.0, win), axis=1)
    cnt = (~np.isnan(win)).sum(axis=1)
    out = np.full(n, np.nan)
    val = np.where(cnt >= need, s / np.maximum(cnt, 1), np.nan)
    out[w - 1:] = val
    return out


def rolling_pct(a, w):
    """滚动 w 窗口内的百分位 (0-100)，忽略 NaN"""
    n = len(a)
    out = np.full(n, np.nan)
    if n < w:
        return out
    win = sliding_window_view(a, w)
    cur = a[w - 1:]
    m = ~np.isnan(win)
    le = ((win < cur[:, None]) & m).sum(axis=1)
    eq = ((win == cur[:, None]) & m).sum(axis=1)
    cnt = np.maximum(m.sum(axis=1), 1)
    out[w - 1:] = (le + 0.5 * eq) / cnt * 100.0
    return out


def rolling_z(a, w):
    """滚动 Z-score（nan-aware）"""
    n = len(a)
    out = np.full(n, np.nan)
    if n < w:
        return out
    win = sliding_window_view(a, w)
    m = np.nanmean(np.where(np.isnan(win), np.nan, win), axis=1)
    s = np.nanstd(np.where(np.isnan(win), np.nan, win), axis=1, ddof=1)
    out[w - 1:] = np.where(s > 0, (a[w - 1:] - m) / s, np.nan)
    return out


def pearson(x, y):
    xc = x - x.mean()
    yc = y - y.mean()
    sx = np.sqrt((xc * xc).sum())
    sy = np.sqrt((yc * yc).sum())
    if sx <= 0 or sy <= 0:
        return np.nan
    return float((xc * yc).sum() / (sx * sy))


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def rolling_ic(x, y, w):
    """滚动 Pearson IC 序列"""
    n = len(x)
    if n < w + 5:
        return np.array([])
    mx = sliding_mean(x, w)
    my = sliding_mean(y, w)
    mxy = sliding_mean(x * y, w)
    mxx = sliding_mean(x * x, w)
    myy = sliding_mean(y * y, w)
    cov = mxy - mx * my
    vx = mxx - mx * mx
    vy = myy - my * my
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(np.maximum(vx, 0) * np.maximum(vy, 0))
    r = r[w - 1:]
    return r[np.isfinite(r)]


def nw_t(x, y, lag):
    """OLS y = a + b·x，b 的 Newey-West 修正 t 值"""
    n = len(x)
    if n < 30:
        return np.nan
    xc = x - x.mean()
    sxx = (xc * xc).sum()
    if sxx <= 0:
        return np.nan
    b = (xc * (y - y.mean())).sum() / sxx
    a = y.mean() - b * x.mean()
    e = y - (a + b * x)
    u = xc * e
    L = max(int(lag), 1)
    V = float((u * u).sum())
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        V += 2.0 * w * float((u[l:] * u[:-l]).sum())
    if V <= 0:
        return np.nan
    se = np.sqrt(V) / sxx
    return float(b / se) if se > 0 else np.nan


def indep_t(x, y, N):
    """不重叠子样本 t 检验：按 N 日步长抽稀，使样本近似独立。

    NW-t 在 N 大、样本小时会因滞后阶数过高而低估标准误（t 虚高）。
    这里用抽稀后的独立样本做普通 OLS t，作为不依赖渐近假设的交叉验证。
    """
    n = len(x)
    idx = np.arange(0, n, max(int(N), 1))
    xs, ys = x[idx], y[idx]
    m = len(xs)
    if m < 15:
        return np.nan, m
    xc = xs - xs.mean()
    sxx = float((xc * xc).sum())
    if sxx <= 0:
        return np.nan, m
    b = float((xc * (ys - ys.mean())).sum() / sxx)
    a = ys.mean() - b * xs.mean()
    e = ys - (a + b * xs)
    s2 = float((e * e).sum()) / (m - 2)
    if s2 <= 0:
        return np.nan, m
    se = np.sqrt(s2 / sxx)
    return (b / se if se > 0 else np.nan), m


def circ_perm_p(rx, ry):
    """循环置换检验：把 y 循环移位 n 次，保留 y 全部自相关结构。

    返回 (r0, p)。p = |{k: |r_k| >= |r_0|}| / n，精确、确定性、无随机种子。
    rx, ry 应同为 rank（平均秩）序列，此时 r0 即 Spearman。
    原理：y 循环移位时其秩序列同步移位，故 Spearman 的置换分布
          等于 rank(x) 与 rank(y) 的循环互相关，可用 FFT 一次算完。
    """
    n = len(rx)
    rx_c = rx - rx.mean()
    y_c = ry - ry.mean()
    Sxx = float((rx_c * rx_c).sum())
    Syy = float((y_c * y_c).sum())
    if Sxx <= 0 or Syy <= 0:
        return np.nan, np.nan
    denom = np.sqrt(Sxx * Syy)
    X = np.fft.rfft(rx_c)
    Y = np.fft.rfft(y_c)
    c = np.fft.irfft(Y * np.conj(X), n)
    r_all = c / denom
    r0 = float(r_all[0])
    p = float((np.abs(r_all) >= abs(r0) - 1e-12).sum()) / n
    return r0, p


def bh_fdr(pvals, alpha=FDR_ALPHA):
    """Benjamini-Hochberg：返回显著布尔数组"""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    sig = np.zeros(len(p), dtype=bool)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return sig
    pv = p[idx]
    order = np.argsort(pv)
    m = len(pv)
    thr = alpha * (np.arange(1, m + 1) / m)
    passed = pv[order] <= thr
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        sig[idx[order[:kmax + 1]]] = True
    return sig


# ---------------- 数据装载 ----------------

def load_library():
    with open(LIB_PATH, encoding="utf-8") as f:
        lib = json.load(f)
    dates = lib["dates"]
    D = len(dates)
    series = []          # (group, key, label, full_array)
    for s in lib["cash"]:
        v = np.full(D, np.nan)
        i0, arr = s["i0"], np.asarray(s["v"], dtype=float)
        v[i0:i0 + len(arr)] = arr
        series.append(("cash", f"{s['bt']}|{s['inst']}|{s['mat']}",
                       f"现券·{s['bt']}·{s['inst']}·{s['mat']}", v))
    for s in lib["repo"]:
        v = np.full(D, np.nan)
        i0, arr = s["i0"], np.asarray(s["v"], dtype=float)
        v[i0:i0 + len(arr)] = arr
        series.append(("repo", f"{s['inst']}|{s['field']}",
                       f"回购·{s['inst']}·{s['label']}", v))
    cmap = {}
    for s in lib["curve"]:
        v = np.full(D, np.nan)
        i0, arr = s["i0"], np.asarray(s["v"], dtype=float)
        v[i0:i0 + len(arr)] = arr
        cmap[s["name"]] = v
        series.append(("curve", f"{s['name']}", f"{s['cat']}·{s['name']}", v))
    if INCLUDE_DERIVED:
        for s in lib.get("derived", []):
            v = np.full(D, np.nan)
            i0, arr = s["i0"], np.asarray(s["v"], dtype=float)
            v[i0:i0 + len(arr)] = arr
            series.append(("derived", f"{s['op']}|{s['inst']}|{s['tenor']}", s["name"], v))
    return lib, dates, series, cmap


def build_target(spec, cmap, D):
    """返回 (level 数组, 收益生成函数所需的价格/收益率数组, denom 数组)"""
    if spec["mode"] == "yield":
        lvl = np.zeros(D)
        for name, w in spec["legs"]:
            lvl = lvl + w * cmap[name]
        return lvl, None
    else:
        pnl = np.zeros(D)
        denom = np.zeros(D)
        notional = 0.0
        for leg in spec["legs"]:
            name, w = leg[0], leg[1]
            fv = leg[2] if len(leg) > 2 else 100.0   # 面值（百万），TS=200 其余=100
            p = cmap[name]
            pnl = pnl + w * fv * p
            denom = denom + abs(w) * fv * p
            notional += abs(w) * fv
        if spec.get("as") == "price":
            # 价差报价（每百元面值）：除以名义金额而非价格加权分母
            return pnl, np.full(D, notional if notional > 0 else np.nan)
        return pnl, denom


def target_change(spec, lvl, denom, N, D):
    """未来 N 个交易日的变动序列（索引 t 对应 t→t+N）"""
    out = np.full(D, np.nan)
    nxt = np.full(D, np.nan)
    if D > N:
        nxt[:D - N] = lvl[N:]
    valid = np.isfinite(lvl) & np.isfinite(nxt)
    if spec["mode"] == "yield":
        out[valid] = (nxt[valid] - lvl[valid]) * 100.0      # bp
    elif spec.get("as") == "price":
        d = np.where((denom > 0) & np.isfinite(denom), denom, np.nan)
        out[valid] = (nxt[valid] - lvl[valid]) / d[valid]   # 点（价差报价的涨跌，几分几毛）
    else:
        d = np.where(denom > 0, denom, np.nan)
        out[valid] = (nxt[valid] - lvl[valid]) / d[valid] * 100.0   # %
    return out


def _downsample(a, m):
    """降采样到最多 m 点，只保留有限值"""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return []
    if len(a) <= m:
        return [round(float(v), 3) for v in a]
    idx = np.linspace(0, len(a) - 1, m).round().astype(int)
    return [round(float(v), 3) for v in a[idx]]


def curve_pair(xs, ys, z, N):
    """非重叠抽稀的 (累计 IC, 信号累计净值)。
    y 是未来 N 日累计变动，重叠长度即 N，故按步长 N 抽稀后各期近似独立，
    累计曲线才有意义；直接对逐日滚动 IC 累加只会得到一条假稳定的线。"""
    step = max(int(N), 1)
    icseq = rolling_ic(xs, ys, ICIR_WIN)
    cumic = np.cumsum(icseq[::step]) if len(icseq) > 5 else np.array([])
    pos = np.where(z > SIGNAL_Z, 1.0, np.where(z < -SIGNAL_Z, -1.0, 0.0))
    cumpnl = np.cumsum((pos * ys)[::step])
    # ★ 必须返回 list 不能是 tuple：json.dump 会把 tuple 写成数组、读回来却变成 list，
    #   导致 jsonio 的幂等比对 (tuple != list) 永远判定"有变化"，
    #   每周一体检重跑都会白写一个 8.67MB 的新 blob。详见 2026-09-21 排查。
    return [_downsample(cumic, CURVE_PTS), _downsample(cumpnl, CURVE_PTS)]


def cond_dist(p, ys):
    """按滚动经验分位切 6 区间的条件分布：
    每区间 [n, 均值, 中位, 标准差, 上行概率%, 5%分位]
    用于捞 B/C 类因子：线性 IC 不显著，但区间间分布差异可能很大（非单调或尾部差异）。"""
    out = []
    for i in range(len(COND_EDGES) - 1):
        lo, hi = COND_EDGES[i], COND_EDGES[i + 1]
        s = (p >= lo) & ((p < hi) if hi < 100 else (p <= 100))
        yy = ys[s & np.isfinite(ys)]
        if len(yy) < 10:
            out.append([0, None, None, None, None, None])
            continue
        out.append([int(len(yy)), round(float(yy.mean()), 3), round(float(np.median(yy)), 3),
                    round(float(yy.std(ddof=1)), 3), round(float((yy > 0).mean() * 100), 1),
                    round(float(np.percentile(yy, 5)), 3)])
    return out


def horizons_for(spec):
    """该目标允许的预测周期（样本短的品种用 maxN 限制，如 TL 仅 2023-04 起）"""
    mx = spec.get("maxN")
    return [N for N in HORIZONS if mx is None or N <= mx]


def make_form(x, form):
    if form == "raw":
        return x
    if form == "pct":
        return rolling_pct(ma_nan(x, 10), 60)
    if form == "diff":
        out = np.full(len(x), np.nan)
        out[5:] = x[5:] - x[:-5]
        return out
    return x


# ---------------- 主体 ----------------

def main():
    t0 = time.time()
    lib, dates, series, cmap = load_library()
    D = len(dates)
    years = np.array([int(d[:4]) for d in dates])
    print(f"载入 {len(series)} 条序列，日期轴 {dates[0]} ~ {dates[-1]}（{D} 天）")

    # 预生成每个目标的 level / 各周期收益 / regime 切分
    tgt_cache = {}
    for spec in TARGETS:
        lvl, denom = build_target(spec, cmap, D)
        ys = {N: target_change(spec, lvl, denom, N, D) for N in HORIZONS}
        # regime：60 日趋势方向
        trend = np.full(D, np.nan)
        trend[60:] = lvl[60:] - lvl[:-60]
        # regime：20 日已实现波动率
        d1 = np.full(D, np.nan)
        d1[1:] = lvl[1:] - lvl[:-1]
        vol = np.full(D, np.nan)
        if D > 20:
            w = sliding_window_view(d1, 20)
            vol[19:] = np.nanstd(np.where(np.isnan(w), np.nan, w), axis=1, ddof=1)
        vmed = np.nanmedian(vol)
        tgt_cache[spec["id"]] = {
            "spec": spec, "lvl": lvl, "ys": ys,
            "up": trend > 0, "dn": trend < 0,
            "hv": vol > vmed, "lv": vol <= vmed,
        }
        print(f"  目标 {spec['id']} ({spec['name']}, {spec['unit']}) 就绪")

    # 因子形态预计算（含滚动 Z，避免在主循环里重复算 3600 次）
    forms_cache = {}
    for fid, _ in FORMS:
        lst = []
        for g, k, label, v in series:
            xf = make_form(v, fid)
            zf = rolling_z(xf, Z_WIN)
            pf = rolling_pct(xf, Z_WIN)      # 滚动经验分位（条件分布分区用，防前视）
            if LAG > 0:
                xl = np.full(D, np.nan)
                xl[LAG:] = xf[:-LAG]
                zl = np.full(D, np.nan)
                zl[LAG:] = zf[:-LAG]
                pl = np.full(D, np.nan)
                pl[LAG:] = pf[:-LAG]
            else:
                xl, zl, pl = xf, zf, pf
            lst.append((xl, zl, pl))
        forms_cache[fid] = lst
        print(f"  形态 {fid} 预计算完成")

    blocks = {}
    dblocks = {}
    n_tests = 0
    for fid, _fname in FORMS:
        xlist = forms_cache[fid]
        for spec in TARGETS:
            tc = tgt_cache[spec["id"]]
            for N in horizons_for(spec):
                y_full = tc["ys"][N]
                rows = []
                drows = []
                pvals = []
                pperms = []
                ic_list = []
                for si, (g, k, label, _v) in enumerate(series):
                    x, zlag, plag = xlist[si]
                    m = np.isfinite(x) & np.isfinite(y_full)
                    n = int(m.sum())
                    if n < MIN_OBS:
                        continue
                    xs = x[m]
                    ys = y_full[m]
                    yr = years[m]

                    ic = spearman(xs, ys)
                    icp = pearson(xs, ys)
                    icseq = rolling_ic(xs, ys, ICIR_WIN)
                    icir = float(icseq.mean() / icseq.std(ddof=1)) if len(icseq) > 10 and icseq.std(ddof=1) > 0 else np.nan

                    # NW-t：滞后取 max(N-1, 自动规则)
                    autoL = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
                    tval = nw_t(xs, ys, max(N - 1, autoL, 1))

                    # 分组（滚动 250 日 Z-score，已预计算并滞后，防未来函数）
                    z = zlag[m]
                    pctl = plag[m]
                    qb = [-0.8416, -0.2533, 0.2533, 0.8416]
                    qs, mono = [], 0
                    edges = [-np.inf] + qb + [np.inf]
                    for qi in range(5):
                        sel = (z > edges[qi]) & (z <= edges[qi + 1])
                        qs.append(float(ys[sel].mean()) if sel.sum() >= 20 else np.nan)
                    qarr = np.array(qs, dtype=float)
                    if np.isfinite(qarr).all():
                        d = np.diff(qarr)
                        if (d < 0).all():
                            mono = -1
                        elif (d > 0).all():
                            mono = 1
                    qd = float(qarr[4] - qarr[0]) if np.isfinite(qarr[4]) and np.isfinite(qarr[0]) else np.nan
                    # Q5-Q1 差的 NW-t（两组均值差）
                    selA = z > qb[3]
                    selB = z <= qb[0]
                    qt = np.nan
                    if selA.sum() >= 20 and selB.sum() >= 20:
                        yy = np.concatenate([ys[selA], ys[selB]])
                        xx = np.concatenate([np.ones(selA.sum()), np.zeros(selB.sum())])
                        qt = nw_t(xx, yy, max(N - 1, 1))

                    # 信号化：|Z|>SIGNAL_Z 持仓
                    pos = np.where(z > SIGNAL_Z, 1.0, np.where(z < -SIGNAL_Z, -1.0, 0.0))
                    on = pos != 0
                    win = float(((pos[on] * ys[on]) > 0).mean()) if on.sum() >= 20 else np.nan
                    lg = pos > 0
                    sh = pos < 0
                    wl = float((ys[lg] > 0).mean()) if lg.sum() >= 20 else np.nan
                    ws = float((ys[sh] < 0).mean()) if sh.sum() >= 20 else np.nan
                    pnl_on = pos[on] * ys[on]
                    gp = pnl_on[pnl_on > 0]
                    gl = pnl_on[pnl_on < 0]
                    pl = float(gp.mean() / abs(gl.mean())) if len(gp) >= 5 and len(gl) >= 5 and gl.mean() != 0 else np.nan
                    # 尾部赔率：|Z|>TAIL_Z
                    tz = np.abs(z) > TAIL_Z
                    tp = (pos[tz] * ys[tz])
                    tgp = tp[tp > 0]
                    tgl = tp[tp < 0]
                    tail = float(tgp.mean() / abs(tgl.mean())) if len(tgp) >= 5 and len(tgl) >= 5 and tgl.mean() != 0 else np.nan

                    turn = float((np.diff(pos) != 0).mean()) if n > 2 else np.nan
                    kt = float(kurtosis(xs, nan_policy="omit")) if n > 10 else np.nan
                    acf = np.nan
                    if n > 30:
                        a = xs - xs.mean()
                        d0 = (a * a).sum()
                        acf = float((a[1:] * a[:-1]).sum() / d0) if d0 > 0 else np.nan

                    _, pv = circ_perm_p(rankdata(xs), rankdata(ys))

                    # 分年 IC
                    yrd = {}
                    for yy_ in range(2021, 2027):
                        sel = yr == yy_
                        yrd[yy_] = spearman(xs[sel], ys[sel]) if sel.sum() >= 40 else np.nan
                    h6 = spearman(xs[-RECENT6:], ys[-RECENT6:]) if n >= RECENT6 else np.nan

                    # regime（mask 均已按 m 降维到 dense）
                    def _rg(mask_full):
                        mm = mask_full[m]
                        if mm.sum() < 40:
                            return np.nan
                        return spearman(xs[mm], ys[mm])

                    rgu = _rg(tc["up"])
                    rgd = _rg(tc["dn"])
                    rghv = _rg(tc["hv"])
                    rglv = _rg(tc["lv"])

                    tind, m_ind = indep_t(xs, ys, N)
                    pind = float(2 * tdist.sf(abs(tind), max(m_ind - 2, 1))) if np.isfinite(tind) else np.nan
                    rows.append([
                        g, k, label, n, ic, icp, icir, tval, tind, pind, win, wl, ws, pl, tail,
                        qs[0], qs[1], qs[2], qs[3], qs[4], qd, qt, mono, turn, kt, acf,
                        pv, None, 0, h6,
                        yrd[2021], yrd[2022], yrd[2023], yrd[2024], yrd[2025], yrd[2026],
                        rgu, rgd, rghv, rglv,
                    ])
                    drows.append([curve_pair(xs, ys, z, N), cond_dist(pctl, ys)])
                    pvals.append(pind)
                    pperms.append(pv)
                    ic_list.append(ic if np.isfinite(ic) else 0.0)

                # BH-FDR：主判据用不重叠子样本 t 的 p（不依赖渐近假设）
                sig = bh_fdr(pvals)
                sigp = bh_fdr(pperms)
                aic = np.asarray(ic_list, dtype=float)
                strong = np.where(np.isfinite(aic), np.abs(aic) >= MIN_IC_STRONG, False)
                for i in range(len(rows)):
                    # 显著还要求 |IC| 达标：防止 IC≈0 却被极端值撑高的假 t 值
                    if not strong[i]:
                        rows[i][28] = 0
                    else:
                        rows[i][28] = 2 if sigp[i] else (1 if sig[i] else 0)
                key = f"{fid}|{spec['id']}|{N}"
                blocks[key] = [[_r(c, 4) if isinstance(c, float) else c for c in r] for r in rows]
                dblocks[key] = drows
                n_tests += len(rows)
                print(f"  {key:16s} {len(rows):4d} 行，独立样本显著 {int(sig.sum())}，置换也认可 {int(sigp.sum())}")

    meta = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date_range": [dates[0], dates[-1]],
        "n_dates": D,
        "lag": LAG,
        "min_obs": MIN_OBS,
        "z_win": Z_WIN,
        "icir_win": ICIR_WIN,
        "signal_z": SIGNAL_Z,
        "tail_z": TAIL_Z,
        "recent6_days": RECENT6,
        "fdr_alpha": FDR_ALPHA,
        "bootstrap": "circular permutation (FFT, exact, deterministic)",
        "n_tests": n_tests,
        "cols": COLS,
        "forms": [{"id": a, "name": b} for a, b in FORMS],
        "targets": [{"id": t["id"], "name": t["name"], "unit": t["unit"],
                     "legs": [leg[0] for leg in t["legs"]],
                     "maxN": t.get("maxN"), "as": t.get("as")} for t in TARGETS],
        "cond_edges": COND_EDGES,
        "curve_pts": CURVE_PTS,
        "detail_file": "factor_checkup_detail.json",
        "min_ic_strong": MIN_IC_STRONG,
        "horizons": HORIZONS,
        "notes": [
            "ic = Spearman(因子, 未来N日目标变动)；icp = Pearson",
            "t = Newey-West 修正 t 值（滞后 max(N-1, 自动规则)），功效高但 N 大时易虚高",
            "ti = 不重叠子样本（按 N 日抽稀）OLS t 值，不依赖渐近假设，为交叉验证判据",
            "pind = ti 对应的 t 分布 p 值；FDR 主判据用 pind",
            "p = 循环置换检验精确 p（最保守：保留 y 全部自相关，持久序列下功效低）",
            "sig: 0=不显著 1=独立样本 FDR 显著 2=连置换检验也显著（最可信）",
            "t 显著而 ti/p 不显著 ⇒ 信号多半来自共同低频趋势，慎用",
            "q1..q5 = 按滚动250日Z-score 五分组的未来N日平均收益（目标单位）",
            "qd = Q5-Q1；qt = Q5-Q1 的 NW-t；mono = 单调性(1递增/-1递减/0非单调)",
            f"因子滞后 {LAG} 日使用（机构行为 T+1 发布）",
            "turn = 持仓方向日均变化频率；kurt/acf 为形态标签，不参与筛选",
        ],
    }
    # 全局 p 值分布：若均匀则说明整批因子无系统性信号（硬证据）
    pi = COLS.index("p")
    allp = np.array([r[pi] for b in blocks.values() for r in b
                     if r[pi] is not None], dtype=float)
    if len(allp):
        f05 = float((allp < 0.05).mean())
        f10 = float((allp < 0.10).mean())
        med = float(np.median(allp))
        if 0.03 <= f05 <= 0.09 and 0.35 <= med <= 0.65:
            verdict = "均匀分布：整批因子无系统性信号，个别显著多半是多重检验下的偶然"
        elif f05 > 0.15:
            verdict = "小 p 值富集：存在系统性信号"
        else:
            verdict = "偏离均匀：需人工复核"
        meta["p_dist"] = {
            "n": int(len(allp)),
            "p_lt_05": round(f05, 4), "p_lt_10": round(f10, 4),
            "median": round(med, 4),
            "p05_expected": 0.05,
            "verdict": verdict,
        }
        print(f"\n全局 p(置换) 分布：n={len(allp)}  P(p<0.05)={f05:.3f}  "
              f"P(p<0.10)={f10:.3f}  中位数={med:.3f}")
        print(f"  → {verdict}")

    payload = {"meta": meta, "blocks": blocks}

    if jsonio is not None:
        changed = jsonio.write_json_skip_unchanged(OUT_PATH, payload)
    else:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        changed = True

    size = os.path.getsize(OUT_PATH) / 1e6
    print(f"\n输出 {OUT_PATH}  {size:.2f} MB  {'[已更新]' if changed else '[无变化，跳过重写]'}")

    # 详情（累计 IC / 信号累计净值 / 分位条件分布）：独立文件，前端按需 fetch，主表不膨胀
    dpayload = {
        "generated": meta["generated"],
        "cond_edges": COND_EDGES,
        "curve_pts": CURVE_PTS,
        "blocks": dblocks,
    }
    if jsonio is not None:
        dch = jsonio.write_json_skip_unchanged(DETAIL_PATH, dpayload)
    else:
        with open(DETAIL_PATH, "w", encoding="utf-8") as f:
            json.dump(dpayload, f, ensure_ascii=False, separators=(",", ":"))
        dch = True
    dsize = os.path.getsize(DETAIL_PATH) / 1e6
    print(f"输出 {DETAIL_PATH}  {dsize:.2f} MB  {'[已更新]' if dch else '[无变化，跳过重写]'}")

    print(f"总检验 {n_tests} 套，耗时 {time.time()-t0:.1f}s")
    return payload


if __name__ == "__main__":
    main()
