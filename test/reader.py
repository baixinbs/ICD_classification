import pandas as pd
import pickle
import os

pklfilepath_6600 = "E:\\ICD_classification\\pickle\\6600_pkl"
if not os.path.exists(pklfilepath_6600):
    with open(pklfilepath_6600, 'wb') as pklfile:
        df = pd.read_excel("E:\\ICD_classification\\data\\6600.xlsx", dtype=str)
        pickle.dump(df, pklfile)
else:
    with open(pklfilepath_6600, 'rb') as pklfile:
        df = pickle.load(pklfile)

print(df.head())
print("info")
df.info()
