# -*- coding: utf-8 -*-
"""现券机构净买入日报 → bond_data.xlsx。



兼容两类源文件：

1. 现券市场交易情况总结日报_YYYYMMDD.xlsx（旧版）

2. 现券成交分机构统计YYYYMMDD - YYYYMMDD.xlsx（2026-10-30 起新版）



新版主要变化：文件名、首行空白、中文表头、期限简称、'--' 空值、机构名逐行重复。

输出结构保持不变，供 daily_update.py / prepare_bond_data.py 使用。

"""
import os

import re

import warnings


import pandas as pd

from openpyxl.styles.stylesheet import Stylesheet



warnings.filterwarnings("ignore", category=UserWarning, module=Stylesheet.__module__)



SOURCE_DIR = r"C:\Users\lihaoran\Documents\工作\现券交易\2026年交易"

MASTER_EXCEL = os.path.join(SOURCE_DIR, "bond_data.xlsx")

OUTPUT_EXCEL = r"C:\Users\lihaoran\Documents\工作\现券交易\bond_data.xlsx"



INVESTOR = [

    "大型银行", "中小型银行", "证券公司", "保险公司", "基金公司及产品",

    "理财子公司及理财类产品", "货币市场基金", "其他",

]

BOND_TYPES = ["国债", "政金债", "地方债", "同业存单", "信用债"]

TENORS = ["1年", "1-3年", "3-5年", "5-7年", "7-10年",

          "10-15年", "15-20年", "20-30年", "30年", "合计\n(Total)"]

DETAIL_COLUMNS = [inst + tenor for inst in INVESTOR for tenor in TENORS]

SUMMARY_COLUMNS = [inst + "合计\n(Total)" for inst in INVESTOR]



REPORT_PATTERNS = (

    re.compile(r"^现券市场交易情况总结日报_(\d{8})\.xlsx$", re.IGNORECASE),

    re.compile(r"^现券成交分机构统计(\d{8})\s*-\s*\d{8}\.xlsx$", re.IGNORECASE),

)



def report_date(path):

    """从新旧文件名提取 YYYYMMDD；不是日报则返回 None。"""

    name = os.path.basename(path)

    for pattern in REPORT_PATTERNS:

        match = pattern.match(name)

        if match:

            return match.group(1)

    return None



def scan_reports():

    """扫描日报；同日新旧文件同时存在时优先新版。"""

    found = {}

    for name in os.listdir(SOURCE_DIR):

        path = os.path.join(SOURCE_DIR, name)

        if not os.path.isfile(path):

            continue

        date = report_date(name)

        if date is None:

            continue

        # 字典后写覆盖先写：文件名排序后旧版在前、新版在后。

        found[date] = path

    return [found[d] for d in sorted(found)]



def normalize_institution(value):

    if not isinstance(value, str):

        return None

    text = value.strip().split("\n", 1)[0]

    text = re.split(r"[（(]", text, maxsplit=1)[0].strip()

    return text if text in INVESTOR else None



def normalize_tenor(value):

    """把新旧期限标签统一成中间表沿用的历史标签。"""

    if not isinstance(value, str):

        return None

    text = value.strip().replace(" ", "")

    if text.startswith("合计"):

        return "合计\n(Total)"

    if text.startswith(("1年及1年以下", "≤1Y", "≦1Y", "<=1Y")):

        return "1年"

    if text.startswith(("30年以上", ">30Y")):

        return "30年"

    for tenor in TENORS[1:9]:

        if text.startswith(tenor):

            return tenor

    # 新版英文期限：1-3Y / 3-5Y / ... / 20-30Y

    match = re.match(r"^(\d+(?:-\d+)?)Y", text, re.IGNORECASE)

    if match:

        candidate = match.group(1) + "年"

        if candidate in TENORS:

            return candidate

    return None



def find_net_sheet(excel):

    for name in excel.sheet_names:

        if name.startswith("机构净买入债券成交金额统计表"):

            return name

    raise ValueError(f"找不到机构净买入工作表: {excel.sheet_names}")



def find_header_row(df):

    for i in range(min(20, len(df))):

        first = str(df.iat[i, 0]).strip() if pd.notna(df.iat[i, 0]) else ""

        second = str(df.iat[i, 1]).strip() if pd.notna(df.iat[i, 1]) else ""

        if first in ("", "nan", "单位:亿元", "机构类型") and second == "期限":

            return i

    raise ValueError("找不到『机构类型/期限』表头")



def parse_report(path, data_initial):

    """解析一份日报并追加到 data_initial，返回新的 DataFrame。"""

    date = report_date(path)

    if date is None:

        raise ValueError(f"无法从文件名提取日期: {path}")

    with pd.ExcelFile(path) as excel:

        sheet = find_net_sheet(excel)

        raw = pd.read_excel(excel, sheet_name=sheet, header=None)

    if raw.shape[1] < 14:

        raise ValueError(f"净买入表只有 {raw.shape[1]} 列，预期至少 14 列: {path}")


    header = find_header_row(raw)

    amounts = raw.iloc[:, 2:14].replace({"-": 0, "--": 0, "—": 0, "": 0})

    amounts = amounts.apply(pd.to_numeric, errors="coerce").fillna(0.0)



    rows = {bond: {} for bond in BOND_TYPES}

    current_inst = None

    n_rows = 0

    for i in range(header + 1, len(raw)):

        inst = normalize_institution(raw.iat[i, 0])

        if inst is not None:

            current_inst = inst

        tenor = normalize_tenor(raw.iat[i, 1])

        if current_inst is None or tenor is None:

            continue

        values = amounts.iloc[i]

        bond_values = {

            "国债": values.iloc[0] + values.iloc[1],

            "政金债": values.iloc[2] + values.iloc[3],

            "地方债": values.iloc[7],

            "同业存单": values.iloc[8],

            # 保持历史口径：其他 + 中期票据 + 短期/超短期融资券。

            "信用债": values.iloc[10] + values.iloc[4] + values.iloc[5],

        }

        for bond, value in bond_values.items():

            rows[bond][current_inst + tenor] = float(value)

        n_rows += 1



    if n_rows != 80:

        raise ValueError(f"应解析 8机构×10期限=80 行，实际 {n_rows} 行: {path}")



    frame = pd.DataFrame([rows[bond] for bond in BOND_TYPES], index=[int(date)] * 5)

    frame = frame.reindex(columns=DETAIL_COLUMNS, fill_value=0.0)

    frame["债券类型"] = BOND_TYPES

    frame.index.name = "日期"

    result = frame.copy() if data_initial.empty else pd.concat([data_initial, frame], sort=False)

    result = result.reindex(columns=DETAIL_COLUMNS + ["债券类型"])

    result.index.name = "日期"

    # 重复运行或新旧文件同日并存时，以本次解析值为准。

    result = result.reset_index().drop_duplicates(["日期", "债券类型"], keep="last").set_index("日期")

    print(f"  {date}: {os.path.basename(path)} → {n_rows} 行, 5 类债券")

    return result



def read_bond_data(path_list, data_initial):

    """保留原函数入口，供手动调用。"""

    for path in path_list:

        data_initial = parse_report(path, data_initial)

    return data_initial



def existing_dates():

    df = pd.read_excel(MASTER_EXCEL, sheet_name="国债汇总")

    dates = set()

    for value in df["日期"].dropna():

        if isinstance(value, pd.Timestamp):

            dates.add(value.strftime("%Y%m%d"))

        else:

            match = re.search(r"(\d{8})", str(value))

            if match:

                dates.add(match.group(1))

    return dates



def write_output(data):

    bond = {name: data[data["债券类型"] == name] for name in BOND_TYPES}

    rate_bond = bond["国债"][DETAIL_COLUMNS] + bond["政金债"][DETAIL_COLUMNS] + bond["地方债"][DETAIL_COLUMNS]

    with pd.ExcelWriter(OUTPUT_EXCEL, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:

        for name in BOND_TYPES:

            bond[name][SUMMARY_COLUMNS].to_excel(writer, sheet_name=f"{name}汇总")

        rate_bond[SUMMARY_COLUMNS].to_excel(writer, sheet_name="利率债汇总")

        for name in BOND_TYPES:

            bond[name].to_excel(writer, sheet_name=f"{name}原数据")

        data.to_excel(writer, sheet_name="汇总")



def main():

    reports = scan_reports()

    have_dates = existing_dates()

    new_reports = [p for p in reports if report_date(p) not in have_dates]

    print(f"扫描到 {len(reports)} 份日报，待解析 {len(new_reports)} 份")

    data = pd.read_excel(MASTER_EXCEL, sheet_name="汇总", index_col=0)

    data.index.name = "日期"

    data = read_bond_data(new_reports, data)

    write_output(data)

    print(f"已更新 {OUTPUT_EXCEL}，末日 {max(data.index)}")



if __name__ == "__main__":

    main()

