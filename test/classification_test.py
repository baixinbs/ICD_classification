from core.file_reader import FileReader
from core.stop_words_manager import StopWordsManager
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import jieba
import pandas as pd

filepath_stopwords = 'E:\\ICD_classification\\stopwords\\stop_words.txt'
excel_file_6600 = "E:\\ICD_classification\\data\\6600.xlsx"
pkl_file_6600 = "E:\\ICD_classification\\pickle\\6600_pkl"
excel_file_cyxj = "E:\\ICD_classification\\data\\cyxj.xls"
pkl_file_cyxj = "E:\\ICD_classification\\pickle\\cyxj_pkl"
pd.set_option('display.max_columns', None)

m_reader = FileReader(excel_file_6600, pkl_file_6600, excel_file_cyxj, pkl_file_cyxj)
df = m_reader.get_dataframe()
m_stopwordsmanager = StopWordsManager(filepath_stopwords)
# add a new column to save the prepoocessed content
df['stopwords_removed'] = [''] * df.shape[0]
count = 0
for i in range(20):
    count += 1
    print(count)
    df.loc[i, 'stopwords_removed'] = m_stopwordsmanager.remove_stop_words(df.loc[i, '内容'])
print(df.head())
print(df.loc[0, 'stopwords_removed'])

print("pause")