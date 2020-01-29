from core.file_reader import FileReader
from core.stop_words_manager import StopWordsManager
import os
import pickle


class DataClient(object):
    def __init__(self, filepath_stopwords, excel_file_6600, pkl_file_6600, excel_file_cyxj, pkl_file_cyxj):
        self.filepath_stopwords = filepath_stopwords
        self.excel_file_6600 = excel_file_6600
        self.pkl_file_6600 = pkl_file_6600
        self.excel_file_cyxj = excel_file_cyxj
        self.pkl_file_cyxj = pkl_file_cyxj
        pass

    def df_remove_stopwords(self):
        m_reader = FileReader(self.excel_file_6600, self.pkl_file_6600, self.excel_file_cyxj, self.pkl_file_cyxj)
        df = m_reader.get_dataframe()
        m_stopwordsmanager = StopWordsManager(self.filepath_stopwords)
        # add a new column to save the content without the removed stop words
        df['stopwords_removed'] = [''] * df.shape[0]
        count = 0
        for i in range(df.shape[0]):
            count += 1
            print(count)
            df.loc[i, 'stopwords_removed'] = m_stopwordsmanager.remove_stop_words(df.loc[i, '内容'])
        # print(df.head())
        # print(df.loc[0, 'stopwords_removed'])
        return df

    def load_df_without_stopwords(self, pkl_file_df_stopwords_removed):
        if not os.path.exists(pkl_file_df_stopwords_removed):
            with open(pkl_file_df_stopwords_removed, 'wb') as pklfile:
                df = self.df_remove_stopwords()
                pickle.dump(df, pklfile)
        else:
            with open(pkl_file_df_stopwords_removed, 'rb') as pklfile:
                df = pickle.load(pklfile)
        return df

