# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 23:11:08 2021

@author: siris
"""
import pandas as pd
import os
from openpyxl import Workbook

def append_to_excel(data, columns):
    xlspath = r'test_results_ETM.xlsx'
    if not os.path.exists(xlspath):
        wb = Workbook().save(filename=xlspath)
    
    df = pd.DataFrame(data, index=None, columns=columns)   
    df_source = pd.DataFrame(pd.read_excel(xlspath, engine='openpyxl'))
    if df_source is not None:
        df_dest = df_source.append(df)
    else:
        df_dest = df
    df_dest.to_excel(xlspath, engine='openpyxl', index=False, columns=columns)
    
