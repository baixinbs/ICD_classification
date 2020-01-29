import pandas as pd
import pickle
import os

#实例化对象
class FileReader(object):
    def __init__(self,
                 excel_file_6600 = "E:\\ICD_classification\\data\\6600.xlsx",
                 pkl_file_6600 = "E:\\ICD_classification\\pickle\\6600_pkl",
                 excel_file_cyxj = "E:\\ICD_classification\\data\\cyxj.xls",
                 pkl_file_cyxj = "E:\\ICD_classification\\pickle\\cyxj_pkl"
                 ):
        self.excel_file_6600 = excel_file_6600
        self.pkl_file_6600 = pkl_file_6600
        self.excel_file_cyxj = excel_file_cyxj
        self.pkl_file_cyxj = pkl_file_cyxj

    def load_6600(self):
        if not os.path.exists(self.pkl_file_6600):
            with open(self.pkl_file_cyxj, 'wb') as pklfile:
                df = pd.read_excel(self.excel_file_6600, dtype=str)
                pickle.dump(df, pklfile)
        else:
            with open(self.pkl_file_6600, 'rb') as pklfile:
                df = pickle.load(pklfile)
        return df

    def load_cyxj(self):
        if not os.path.exists(self.pkl_file_cyxj):
            with open(self.pkl_file_cyxj, 'wb') as pklfile:
                df = pd.read_excel(self.excel_file_cyxj, dtype=str)
                pickle.dump(df, pklfile)
        else:
            with open(self.pkl_file_cyxj, 'rb') as pklfile:
                df = pickle.load(pklfile)
        return df

    def get_dataframe(self):
        df6600 = self.load_6600()
        df6600 = df6600[['病案号','icd编码']]
        # print(df6600.head())

        dfcyxj = self.load_cyxj()
        dfcyxj = dfcyxj[['病案号','内容']]
        # print(dfcyxj.head())

        df = pd.merge(df6600, dfcyxj)
        # pd.set_option('display.max_columns', None)
        #         # print(df.head())
        #         # print(df.info())
        return df

if __name__ == "__main__":
    excel_file_6600 = "E:\\ICD_classification\\data\\6600.xlsx"
    pkl_file_6600 = "E:\\ICD_classification\\pickle\\6600_pkl"
    excel_file_cyxj = "E:\\ICD_classification\\data\\cyxj.xls"
    pkl_file_cyxj = "E:\\ICD_classification\\pickle\\cyxj_pkl"
    m_reader = FileReader(excel_file_6600,pkl_file_6600,excel_file_cyxj,pkl_file_cyxj)
    df = m_reader.get_dataframe()
    print('ending')