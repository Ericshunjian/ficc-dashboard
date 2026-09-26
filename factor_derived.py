# -*- coding: utf-8 -*-
"""
机构行为 · 现券衍生因子生成器（L0 衍生层）

方法来源：国泰海通证券《债市因子图鉴（一）（二）》的三视角框架
    趋势类（买在一致） / 截面类（买在分歧） / 极值类（卖在一致）
另加一组跨机构结构指标（分歧度、共识度、配置盘 vs 交易盘），用于刻画
"一致 / 分歧"这一对状态本身，而不是单家机构的绝对买卖量。

基础序列：国债现券净买入（亿元），按 8 机构 × 9 个原始期限档分别计算
衍生原语：
    趋势 trend   days_ratio_N20   过去 N 日净买入为正的天数占比（0~100）
    趋势 trend   accel_N5_T20     MA5 − MA20，净买卖加速度（亿元）
    趋势 trend   slope_N10        累积净买入的 N 日 OLS 斜率（亿元/日）
    截面 cross   rank_N20         N 日动量在同期限 8 机构中的横截面分位（0~100）
    极值 extreme pos_util_N20     仓位占用度：N 日累计净买入处于 250 日区间的分位（0~100）
    极值 extreme act_anom_N20     交易活跃度异常：N 日日均买卖量 / 250 日常态（倍）
    结构 struct  cons_N20         共识度：动量>0 与 <0 的机构数净占比（−100~100）
    结构 struct  hhi_N20          活跃集中度：Σ(机构|动量|占比²)×100，越大越被少数机构主导
    结构 struct  cfg_trd_N20      配置盘 − 交易盘 的 N 日动量差（亿元，机构归属按期限档）
    结构 struct  cfg_trd_accel    配置盘 − 交易盘 的加速度差（亿元，同上）
    跨期限 tenor  tenor_diff_N5    同机构两期限严格 5 日累计净买入之差（亿元）
    跨期限 tenor  tenor_zdiff     上述两端分别做 60 日 Z-score，再相减

§分组依据（配置盘/交易盘的机构归属随期限变化，见 sides_for）：
    10 年以上     配置盘 = 大型银行 + 保险公司      交易盘 = 基金公司及产品 + 其他
    10 年及以下   配置盘 = 大型银行 + 理财子公司 + 保险公司  交易盘 = 基金公司及产品
数据口径提示：本套数据是【二级现券交易】净买入。机构在一级市场的认购不计入，
因此券商、大型银行等机构的二级净买入并不等于其真实配置力度，解读时需留意。

注意：机构净买入是全市场零和的（买入必有对手方卖出），因此"全市场加总"型
指标（Σ动量）恒等于 0、无信息量。跨机构指标必须建立在方向、离散度或子集差
之上，本模块的 cons / hhi / cfg_trd 均按此原则设计。

缺失处理：一般窗口内有效观测不足 70% 时输出 null；10 日斜率要求历史完整，绝不把缺失当 0。
机构净买入当日收盘后可得；用于期货交易时最早从下一期货交易日开始。
"""
import numpy as np

BOND_TYPE = "国债"

TENORS = [(m, [m]) for m in (
    "≤1年", "1-3年", "3-5年", "5-7年", "7-10年",
    "10-15年", "15-20年", "20-30年", ">30年",
)]
TENOR_PAIRS = (
    ("7-10年", "20-30年", "3T−TL"),
    ("1-3年", "7-10年", "4TS−T"),
)
TENOR_Z_WIN = 60

# 配置盘：负债端稳定、买入偏战略；交易盘：波段与流动性博弈
# ★ 配置盘/交易盘的机构归属【随原始期限变化】（用户 2026-09-23 指定，见 §分组依据）：
#   10 年以上     配置盘 = 大行 + 保险      交易盘 = 基金 + 其他（券商资管等资管机构）
#   10 年及以下   配置盘 = 大行 + 理财 + 保险  交易盘 = 基金
#   理由：超长端理财参与度低，资管类（"其他"）反而是主要交易对手；
#        中短端理财是配置主力，故归入配置盘。
CONFIG_SIDES = {
    m: (["大型银行", "保险公司"], ["基金公司及产品", "其他"])
    for m in ("10-15年", "15-20年", "20-30年", ">30年")
}
DEFAULT_SIDES = (["大型银行", "理财子公司及理财类产品", "保险公司"], ["基金公司及产品"])


def sides_for(tname):
    """返回 (配置盘机构列表, 交易盘机构列表)"""
    return CONFIG_SIDES.get(tname, DEFAULT_SIDES)

N_FAST, N_SLOW = 5, 20
N_MID = 10
HIST = 250
MINP = 0.7

CLS_LABEL = {
    "trend": "趋势·买在一致",
    "cross": "截面·买在分歧",
    "extreme": "极值·卖在一致",
    "struct": "结构·跨机构",
    "tenor": "结构·跨期限",
}

_OP_LABEL = {
    "days_ratio": "净买入天数占比",
    "accel": "净买卖加速度",
    "slope": "累积净买入斜率",
    "rank": "截面动量排名",
    "pos_util": "仓位占用度",
    "act_anom": "交易活跃度异常",
    "cons": "方向共识度",
    "hhi": "活跃集中度",
    "cfg_trd": "配置盘−交易盘",
    "cfg_trd_accel": "配置盘−交易盘加速度",
    "tenor_diff": "跨期限净买入差",
    "tenor_zdiff": "跨期限标准化需求差",
}

_OP_UNIT = {
    "days_ratio": "%",
    "accel": "亿元",
    "slope": "亿元/日",
    "rank": "分位",
    "pos_util": "%",
    "act_anom": "倍",
    "cons": "%",
    "hhi": "%",
    "cfg_trd": "亿元",
    "cfg_trd_accel": "亿元",
    "tenor_diff": "亿元",
    "tenor_zdiff": "标准差",
}


def _minp(n):
    return max(1, int(round(n * MINP)))


def _roll_mean(a, n):
    """忽略 nan 的滚动均值；窗口内有效数 < 70% 输出 nan"""
    L = len(a)
    out = np.full(L, np.nan)
    if L < n:
        return out
    filled = np.where(np.isnan(a), 0.0, a)
    cs = np.cumsum(filled)
    cc = np.cumsum(~np.isnan(a))
    mp = _minp(n)
    for i in range(n - 1, L):
        s = cs[i] - (cs[i - n] if i >= n else 0.0)
        c = cc[i] - (cc[i - n] if i >= n else 0)
        if c >= mp:
            out[i] = s / c
    return out


def _roll_absum(a, n):
    """滚动 Σ|x| / n，即日均买卖量"""
    return _roll_mean(np.abs(a), n)


def _roll_sum_strict(a, n):
    """严格累计净买入：窗口内任一天缺失则整点缺失，不把缺失补零。"""
    out = np.full(len(a), np.nan)
    if len(a) >= n:
        from numpy.lib.stride_tricks import sliding_window_view
        out[n - 1:] = np.sum(sliding_window_view(a, n), axis=1)
    return out


def _roll_z(a, n):
    """当前值相对含当天的过去 n 个交易日均值/样本标准差；至少 70% 有效。"""
    out = np.full(len(a), np.nan)
    if len(a) < n:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(a, n)
    valid = np.isfinite(windows)
    count = valid.sum(axis=1)
    safe = np.where(valid, windows, 0.0)
    mean = safe.sum(axis=1) / np.maximum(count, 1)
    var = np.sum(np.where(valid, (windows - mean[:, None]) ** 2, 0.0), axis=1) / np.maximum(count - 1, 1)
    current = a[n - 1:]
    ok = (count >= _minp(n)) & np.isfinite(current) & (var > 1e-12)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[n - 1:] = np.where(ok, (current - mean) / np.sqrt(var), np.nan)
    return out


def _days_ratio(a, n):
    L = len(a)
    out = np.full(L, np.nan)
    pos = np.where(np.isnan(a), 0.0, (a > 0).astype(float))
    valid = (~np.isnan(a)).astype(float)
    cp = np.cumsum(pos)
    cv = np.cumsum(valid)
    mp = _minp(n)
    for i in range(n - 1, L):
        p = cp[i] - (cp[i - n] if i >= n else 0.0)
        v = cv[i] - (cv[i - n] if i >= n else 0.0)
        if v >= mp:
            out[i] = p / v * 100.0
    return out


def _cum_slope(a, n):
    """过去 n 日累积净买入路径对时间做 OLS 的斜率；路径缺日则不可计算。"""
    L = len(a)
    out = np.full(L, np.nan)
    x = np.arange(n, dtype=float)
    xm = x.mean()
    den = ((x - xm) ** 2).sum()
    for i in range(n - 1, L):
        w = a[i - n + 1:i + 1]
        if np.any(np.isnan(w)):
            continue
        y = np.cumsum(w)
        out[i] = ((x - xm) * (y - y.mean())).sum() / den
    return out


def _sliding_minmax(a, hist):
    """滑动窗口 min / max（忽略 nan，窗口内有效数 < 60 输出 nan）"""
    from numpy.lib.stride_tricks import sliding_window_view

    L = len(a)
    mn = np.full(L, np.nan)
    mx = np.full(L, np.nan)
    if L < hist:
        return mn, mx
    win = sliding_window_view(a, hist)
    with np.errstate(invalid="ignore"):
        wmn = np.nanmin(win, axis=1)
        wmx = np.nanmax(win, axis=1)
    cnt = np.count_nonzero(~np.isnan(win), axis=1)
    ok = cnt >= 60
    mn[hist - 1:] = np.where(ok, wmn, np.nan)
    mx[hist - 1:] = np.where(ok, wmx, np.nan)
    return mn, mx


def _pos_util(a, n):
    """N 日累计净买入处于过去 250 日区间的位置（0~100）"""
    cum = _roll_mean(a, n) * n
    mn, mx = _sliding_minmax(cum, HIST)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (cum - mn) / (mx - mn) * 100.0
    out = np.where((mx - mn) <= 1e-9, np.nan, out)
    return np.where(np.isnan(cum), np.nan, out)


def _act_anom(a, n):
    """N 日日均买卖量 / 250 日常态"""
    act = _roll_absum(a, n)
    base = _roll_mean(act, HIST)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = act / base
    return np.where(base <= 1e-9, np.nan, out)


def _avg_rank_pct(vals):
    """并列取平均秩，转成 0~100 分位；不足 5 个有效值返回 None"""
    idx_ok = [i for i, v in enumerate(vals) if v == v]
    if len(idx_ok) < 5:
        return None
    m = len(idx_ok)
    sub = [vals[i] for i in idx_ok]
    order = sorted(range(m), key=lambda k: sub[k])
    ranks = [0.0] * m
    i = 0
    while i < m:
        j = i
        while j + 1 < m and sub[order[j + 1]] == sub[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    out = [np.nan] * len(vals)
    for k, gi in enumerate(idx_ok):
        out[gi] = ranks[k] / (m - 1) * 100.0
    return out


def _pack(name, cls, op, n, inst, tenor, arr, unit=None):
    """裁剪成 {i0, v} 存储格式"""
    ok = np.where(~np.isnan(arr))[0]
    if len(ok) == 0:
        return None
    i0, i1 = int(ok[0]), int(ok[-1])
    vals = arr[i0:i1 + 1]
    v = [None if x != x else round(float(x), 4) for x in vals]
    return {
        "name": name,
        "cls": CLS_LABEL[cls],
        "op": op,
        "n": n,
        "inst": inst,
        "tenor": tenor,
        "unit": unit if unit is not None else _OP_UNIT[op],
        "i0": i0,
        "v": v,
    }


def build(merged, idx, n_dates, session_dates=None):
    """生成衍生因子；滚动窗口按期货交易日计，结果仍映射回因子库日期轴。"""
    global_n_dates = n_dates
    session_positions = np.asarray(
        sorted({idx[d] for d in session_dates if d in idx}) if session_dates is not None
        else range(n_dates), dtype=int)
    session_lookup = {int(pos): j for j, pos in enumerate(session_positions)}
    n_dates = len(session_positions)
    mat2tenor = {}
    for tname, mats in TENORS:
        for m in mats:
            mat2tenor[m] = tname

    insts = list(merged["meta"]["institutions"])
    tenor_names = [t[0] for t in TENORS]

    # ---------- 基础序列：机构 × 期限档 ----------
    base = {}
    for inst in insts:
        for tname in tenor_names:
            base[(inst, tname)] = np.zeros(n_dates)
    filled = {}
    for inst in insts:
        for tname in tenor_names:
            filled[(inst, tname)] = np.zeros(n_dates)

    for r in merged["detail"]:
        if r["bond_type"] != BOND_TYPE:
            continue
        tenor = mat2tenor.get(r["maturity"])
        if not tenor:
            continue
        i = session_lookup.get(idx.get(r["date"]))
        if i is None:
            continue
        key = (r["institution"], tenor)
        if key not in base:
            continue
        v = r.get("value")
        if v is None:
            continue
        base[key][i] += float(v)
        filled[key][i] += 1.0

    # 当天该档完全没有记录 -> 缺失（不是 0）
    for key in base:
        base[key] = np.where(filled[key] > 0, base[key], np.nan)

    out = []
    label_short = {
        "证券公司": "券商", "基金公司及产品": "基金", "货币市场基金": "货基",
        "理财子公司及理财类产品": "理财子", "其他": "其他",
    }

    # ---------- 趋势类：单机构时间序列变换 ----------
    for inst in insts:
        for tname in tenor_names:
            a = base[(inst, tname)]
            nm = label_short.get(inst, inst)

            dr = _days_ratio(a, N_SLOW)
            out.append(_pack(
                "衍生·%s·%s·%s·净买入天数占比N%d" % (BOND_TYPE, nm, tname, N_SLOW),
                "trend", "days_ratio", N_SLOW, inst, tname, dr))

            acc = _roll_mean(a, N_FAST) - _roll_mean(a, N_SLOW)
            out.append(_pack(
                "衍生·%s·%s·%s·净买卖加速度N%d−N%d" % (BOND_TYPE, nm, tname, N_FAST, N_SLOW),
                "trend", "accel", N_SLOW, inst, tname, acc))

            slp = _cum_slope(a, N_MID)
            out.append(_pack(
                "衍生·%s·%s·%s·累积净买入斜率N%d" % (BOND_TYPE, nm, tname, N_MID),
                "trend", "slope", N_MID, inst, tname, slp))

    # ---------- 截面类：同期限跨机构排名 ----------
    mom = {}
    for inst in insts:
        for tname in tenor_names:
            mom[(inst, tname)] = _roll_mean(base[(inst, tname)], N_SLOW)
    rank_mat = {}
    for tname in tenor_names:
        M = np.vstack([mom[(inst, tname)] for inst in insts])  # 8 × L
        R = np.full_like(M, np.nan)
        for t in range(n_dates):
            vals = M[:, t]
            if np.all(np.isnan(vals)):
                continue
            rk = _avg_rank_pct([float(x) for x in vals])
            if rk is None:
                continue
            R[:, t] = rk
        rank_mat[tname] = R

    for k, inst in enumerate(insts):
        for tname in tenor_names:
            arr = rank_mat[tname][k]
            nm = label_short.get(inst, inst)
            out.append(_pack(
                "衍生·%s·%s·%s·截面动量排名N%d" % (BOND_TYPE, nm, tname, N_SLOW),
                "cross", "rank", N_SLOW, inst, tname, arr))

    # ---------- 极值类：拥挤度与异常 ----------
    for inst in insts:
        for tname in tenor_names:
            a = base[(inst, tname)]
            nm = label_short.get(inst, inst)

            pu = _pos_util(a, N_SLOW)
            out.append(_pack(
                "衍生·%s·%s·%s·仓位占用度N%d" % (BOND_TYPE, nm, tname, N_SLOW),
                "extreme", "pos_util", N_SLOW, inst, tname, pu))

            aa = _act_anom(a, N_SLOW)
            out.append(_pack(
                "衍生·%s·%s·%s·交易活跃度异常N%d" % (BOND_TYPE, nm, tname, N_SLOW),
                "extreme", "act_anom", N_SLOW, inst, tname, aa))

    # ---------- 结构类：跨机构合成 ----------
    sides_meta = {}
    for tname in tenor_names:
        M = np.vstack([mom[(inst, tname)] for inst in insts])
        # 零和约束下只有方向 / 离散度 / 子集差才有信息量
        cons = np.full(M.shape[1], np.nan)
        hhi = np.full(M.shape[1], np.nan)
        for t in range(M.shape[1]):
            v = M[:, t]
            v = v[~np.isnan(v)]
            if len(v) < 5:
                continue
            up = int((v > 0).sum())
            dn = int((v < 0).sum())
            if up + dn == 0:
                continue
            cons[t] = (up - dn) / (up + dn) * 100.0
            tot = np.sum(np.abs(v))
            if tot > 1e-9:
                w = np.abs(v) / tot
                hhi[t] = float((w ** 2).sum()) * 100.0
        out.append(_pack(
            "衍生·%s·%s·全机构·方向共识度N%d" % (BOND_TYPE, tname, N_SLOW),
            "struct", "cons", N_SLOW, "全机构", tname, cons))
        out.append(_pack(
            "衍生·%s·%s·全机构·活跃集中度N%d" % (BOND_TYPE, tname, N_SLOW),
            "struct", "hhi", N_SLOW, "全机构", tname, hhi))

        # 配置盘 − 交易盘（机构归属按期限档取，见 sides_for）
        cfg_names, trd_names = sides_for(tname)

        def _side(names, mat):
            return [mat[insts.index(x)] for x in names if x in insts]

        cfg_rows = _side(cfg_names, M)
        trd_rows = _side(trd_names, M)
        # 任一组成机构缺失时，不能把缺口当成 0 净买入。
        cfg = np.sum(np.vstack(cfg_rows), axis=0) if cfg_rows else None
        trd = np.sum(np.vstack(trd_rows), axis=0) if trd_rows else None
        if cfg is not None and trd is not None:
            out.append(_pack(
                "衍生·%s·%s·配置盘−交易盘·动量差N%d" % (BOND_TYPE, tname, N_SLOW),
                "struct", "cfg_trd", N_SLOW, "全机构", tname, cfg - trd))

        acc_rows = []
        for inst in insts:
            a = base[(inst, tname)]
            acc_rows.append(_roll_mean(a, N_FAST) - _roll_mean(a, N_SLOW))
        A = np.vstack(acc_rows)
        cfg_a = np.sum(np.vstack([A[insts.index(x)] for x in cfg_names if x in insts]), axis=0)
        trd_a = np.sum(np.vstack([A[insts.index(x)] for x in trd_names if x in insts]), axis=0)
        out.append(_pack(
            "衍生·%s·%s·配置盘−交易盘·加速度差" % (BOND_TYPE, tname),
            "struct", "cfg_trd_accel", N_SLOW, "全机构", tname, cfg_a - trd_a))
        sides_meta[tname] = {"config": cfg_names, "trading": trd_names}

    # ---------- 结构类：同机构跨期限需求差 ----------
    # 先按期限各算严格 5 日累计；标准化版本逐端做 60 日 Z-score 后相减。
    # 此处不做 DV01 配比：它是需求强弱因子，并非期货头寸。
    for inst in insts:
        nm = label_short.get(inst, inst)
        for near, far, target in TENOR_PAIRS:
            pair = near + "−" + far
            near_sum = _roll_sum_strict(base[(inst, near)], N_FAST)
            far_sum = _roll_sum_strict(base[(inst, far)], N_FAST)
            out.append(_pack(
                "衍生·%s·%s·%s·5日累计净买入差（%s）" % (BOND_TYPE, nm, pair, target),
                "tenor", "tenor_diff", N_FAST, inst, pair, near_sum - far_sum))
            out.append(_pack(
                "衍生·%s·%s·%s·5日累计净买入标准化差Z%d（%s）" %
                (BOND_TYPE, nm, pair, TENOR_Z_WIN, target),
                "tenor", "tenor_zdiff", TENOR_Z_WIN, inst, pair,
                _roll_z(near_sum, TENOR_Z_WIN) - _roll_z(far_sum, TENOR_Z_WIN)))

    out = [x for x in out if x]
    # JSON 使用现券/回购/曲线日期的并集；非期货交易日的衍生值保持 null。
    for item in out:
        first_session = item["i0"]
        values = item["v"]
        first_global = int(session_positions[first_session])
        last_global = int(session_positions[first_session + len(values) - 1])
        expanded = [None] * (last_global - first_global + 1)
        for j, value in enumerate(values):
            if value is not None:
                expanded[int(session_positions[first_session + j]) - first_global] = value
        item["i0"], item["v"] = first_global, expanded
    return out, {
        "institutions": insts,
        "tenors": tenor_names + [a + "−" + b for a, b, _ in TENOR_PAIRS],
        "classes": [CLS_LABEL[c] for c in ("trend", "cross", "extreme", "struct", "tenor")],
        # 配置盘/交易盘的实际机构归属（按期限档），供页面与文档展示
        "cfg_sides": sides_meta,
    }
