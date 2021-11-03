# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 23:11:08 2021

@author: siris
"""
import pandas as pd
import os

def append_to_excel(data, columns):
    xlspath = r'test_results_ETM.xlsx'
    with pd.ExcelWriter(xlspath) as writer:
            
        df = pd.DataFrame(data, index=None, columns=columns)
        
        df_source = None
        if os.path.exists(xlspath):
            df_source = pd.DataFrame(pd.read_excel(xlspath))
        if df_source is not None:
            df_dest = df_source.append(df)
        else:
            df_dest = df
        
        df_dest.to_excel(writer, index=False, columns=columns)
    