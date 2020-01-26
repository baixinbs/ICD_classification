import pandas as pd
import pickle
import os

def load_6600():
    pklfilepath_6600 = "E:\\ICD_classification\\pickle\\6600_pkl"
    if not os.path.exists(pklfilepath_6600):
        with open(pklfilepath_6600, 'wb') as pklfile:
            df = pd.read_excel("E:\\ICD_classification\\data\\6600.xlsx", dtype=str)
            pickle.dump(df, pklfile)
    else:
        with open(pklfilepath_6600, 'rb') as pklfile:
            df = pickle.load(pklfile)
    return df

def load_cyxj():
    pklfilepath_cyxj = "E:\\ICD_classification\\pickle\\cyxj_pkl"
    if not os.path.exists(pklfilepath_cyxj):
        with open(pklfilepath_cyxj, 'wb') as pklfile:
            df = pd.read_excel("E:\\ICD_classification\\data\\cyxj.xls", dtype=str)
            pickle.dump(df, pklfile)
    else:
        with open(pklfilepath_cyxj, 'rb') as pklfile:
            df = pickle.load(pklfile)
    return df

df6600 = load_6600()
df6600 = df6600[['病案号','icd编码']]
print(df6600.head())

dfcyxj = load_cyxj()
dfcyxj = dfcyxj[['病案号','内容']]
print(dfcyxj.head())

df = pd.merge(df6600, dfcyxj)
pd.set_option('display.max_columns', None)
print(df.head())
print(df.info())