# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:11:56 2024

@author: lihaoran
"""


import numpy as np
import pandas as pd

import time 
import matplotlib.pyplot as plt
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib
import os
from datetime import datetime

from openpyxl import load_workbook
import warnings
from openpyxl.styles.stylesheet import Stylesheet

# 精准忽略 openpyxl 的默认样式警告
warnings.filterwarnings("ignore", category=UserWarning, module=Stylesheet.__module__)




path = 'C:/Users/lihaoran/Documents/工作/现券交易/2026年交易'
path = os.listdir(path)
print(path)
path_list = []

for i in path:
    if i[:4] =='现券市场'  :
        path_list.append(i)
        
print(path_list)


file = path_list[0]

investor = ['大型银行', '中小型银行', '证券公司',
       '保险公司', '基金公司及产品', '理财子公司及理财类产品',  '货币市场基金', '其他']

bond_type = ['政金债','国债','地方债','同业存单']


#investor.remove('其他')


sheetname = '机构净买入债券成交金额统计表_'+ file[-12:-4]



k =0 




data_initial = pd.DataFrame()
data_initial_0 = pd.DataFrame()
data_initial_1 = pd.DataFrame()
data_initial_2 = pd.DataFrame()
data_initial_3 = pd.DataFrame()
data_initial_4 = pd.DataFrame()


def read_bond_data(path_list,data_initial):

    for file in path_list[:]:
        file_type = 'old'
        
        
        sheetname_with_date  = '机构净买入债券成交金额统计表_'+ file[-12:-4]
        sheetname_without_date = '机构净买入债券成交金额统计表'
        
        # 智能选择存在的Sheet名称
        with pd.ExcelFile(file) as excel:
            if sheetname_with_date in excel.sheet_names:
                sheetname = sheetname_with_date
            elif sheetname_without_date in excel.sheet_names:
                sheetname = sheetname_without_date
                file_type = 'new'
            else:
                raise ValueError(f"文件 {file} 中不存在有效Sheet")
        
        
        
        data = pd.read_excel(file,sheet_name=sheetname)
        data= data.replace('—',0)
        data= data.replace('-',0)

        
        data.iloc[:,0] = data.iloc[:,0].ffill()
        data.iloc[:,0] = data.iloc[:,0].apply(lambda x :x[:x.index('\n')] if type(x)==str and '\n' in x else x  )
        data.iloc[:,0] = data.iloc[:,0].apply(lambda x :x[:x.index('(')] if type(x)==str and '(' in x else x  )
        data.iloc[:,0] = data.iloc[:,0].apply(lambda x :x[:x.index('（')] if type(x)==str and '（' in x else x  )
        data.iloc[:,1] = data.iloc[:,1].apply(lambda x :x[:x.index('年')+1] if type(x)==str and '年' in x else x  )  
        data = data.replace('合计(Total)','合计\n(Total)')
        
        
        #data[['Unnamed: 0']] = data[['Unnamed: 0']].ffill()
        #data['Unnamed: 0'] = data['Unnamed: 0'].apply(lambda x :x[:x.index('\n')] if type(x)==str and '\n' in x else x  )    
        #data['Unnamed: 1'] = data['Unnamed: 1'].apply(lambda x :x[:x.index('年')+1] if type(x)==str and '年' in x else x  )  
    
        
        if file_type  == 'old':
            data = data.iloc[3:]
            print(file[-12:-4])

            # 122行
        else:
            data = data.iloc[1:]
            print(file[-13:-5])

        
        data.columns = data.iloc[0]
        data['国债'] =  data.iloc[:,2] + data.iloc[:,3]
        data['政金债'] = data.iloc[:,4] + data.iloc[:,5] 
        data['地方债'] = data.iloc[:,9]
        data['同业存单'] = data.iloc[:,10]
        data['信用债'] = data.iloc[:,12] + data.iloc[:,6] +data.iloc[:,7]
    
        data['机构'] = data['单位:亿元'] + data['期限']
        data = data.iloc[1:,-6:]
        data = data.T
        data.columns = data.iloc[-1]
        data = data.iloc[:-1,:-1]
        #原代码 data = data.iloc[:-1,:-11]
        
        
        if file_type =='old':
            data['日期'] =  file[-12:-4]
        else:
            data['日期'] = file[-13:-5]
        
        data['债券类型'] = data.index
        data.index = data['日期']
        del data['日期']
        data_initial = pd.concat([data_initial,data])
        
    
    #data_initial['债券类型'] = data_initial.index
    #data_initial.index = data_initial['日期']
    
    
    data_initial_0 = data_initial[data_initial['债券类型']=='国债']
    data_initial_1 = data_initial[data_initial['债券类型']=='政金债']
    data_initial_2 = data_initial[data_initial['债券类型']=='地方债']
    data_initial_3 = data_initial[data_initial['债券类型']=='同业存单']
    data_initial_4 = data_initial[data_initial['债券类型']=='信用债']
    
    data_initial_5 = data_initial[data_initial['债券类型']=='国债'] +\
                             data_initial[data_initial['债券类型']=='政金债'] + \
                                 data_initial[data_initial['债券类型']=='地方债']
                                 
                                 
    investor_display = []
    
    for i in investor:
         
        investor_display.append(i+'合计\n(Total)')
        
    bond_path = 'C:/Users/lihaoran/Documents/工作/现券交易/bond_data.xlsx'
    with pd.ExcelWriter(bond_path, mode="a", engine="openpyxl",if_sheet_exists='replace') as writer:
        
        data_initial_0[investor_display].to_excel(writer,sheet_name ='国债汇总')
        data_initial_1[investor_display].to_excel(writer,sheet_name ='政金债汇总')
        data_initial_2[investor_display].to_excel(writer,sheet_name ='地方债汇总')
        data_initial_3[investor_display].to_excel(writer,sheet_name ='同业存单汇总')
        data_initial_4[investor_display].to_excel(writer,sheet_name ='信用债汇总')
        data_initial_5[investor_display].to_excel(writer,sheet_name = '利率债汇总')
        
        data_initial_0.to_excel(writer,sheet_name ='国债原数据')
        data_initial_1.to_excel(writer,sheet_name ='政金债原数据')
        data_initial_2.to_excel(writer,sheet_name ='地方债原数据')
        data_initial_3.to_excel(writer,sheet_name ='同业存单原数据')
        data_initial_4.to_excel(writer,sheet_name ='信用债原数据')

        data_initial.to_excel(writer,sheet_name = '汇总')






#如果从头开始读取，可以运行这行代码

#read_bond_data(path_list[:],data_initial)


#每次读取新的文件，在原来dataframe上添加新的行
path = 'C:/Users/lihaoran/Documents/工作/现券交易/2026年交易/'
path = os.listdir(path)
print(path)
path_list = []


for i in path:
    if i[:4] =='现券市场'  :
        path_list.append(i)

bond_path = 'C:/Users/lihaoran/Documents/工作/现券交易/2026年交易/bond_data.xlsx'

original_dataframe = pd.read_excel(bond_path,sheet_name='国债汇总')
original_date_list = list(original_dataframe['日期'])
original_date_list = [str(x) for x in original_date_list]


newcomming_list = []

for i in range(len(path_list)):
    if path_list[i][13:21] in original_date_list:
        continue
    else:
        newcomming_list.append(path_list[i])


file_path  = 'C:/Users/lihaoran/Documents/工作/现券交易/2026年交易/'

newcomming_list = [ file_path+x for x in newcomming_list]
data_initial = pd.read_excel(bond_path,sheet_name='汇总',index_col=0)


read_bond_data(newcomming_list,data_initial)


#计算一些指标，1）30年基金过去5日平均净买，求分位数；2）各类机构利率债过去5日平均净买（移动平均），求分位数;3)
#注：农商、券商、保险30年净买意义不大，其中保险农商偏配置，变化不大。而券商方面有分销，因此关注基金公司行为。





