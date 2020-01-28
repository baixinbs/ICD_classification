from core.file_reader import FileReader
from core.stop_words_manager import StopWordsManager
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import jieba
import pandas as pd
import os
import pickle

filepath_stopwords = 'E:\\ICD_classification\\stopwords\\stop_words.txt'
excel_file_6600 = "E:\\ICD_classification\\data\\6600.xlsx"
pkl_file_6600 = "E:\\ICD_classification\\pickle\\6600_pkl"
excel_file_cyxj = "E:\\ICD_classification\\data\\cyxj.xls"
pkl_file_cyxj = "E:\\ICD_classification\\pickle\\cyxj_pkl"
pd.set_option('display.max_columns', None)

def remove_stopwords():
    m_reader = FileReader(excel_file_6600, pkl_file_6600, excel_file_cyxj, pkl_file_cyxj)
    df = m_reader.get_dataframe()
    m_stopwordsmanager = StopWordsManager(filepath_stopwords)
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

pkl_file_df_stopwords_removed = "E:\\ICD_classification\\pickle\\df_stopwords_removed_pkl"
if not os.path.exists(pkl_file_df_stopwords_removed):
    with open(pkl_file_df_stopwords_removed, 'wb') as pklfile:
        df = remove_stopwords()
        pickle.dump(df, pklfile)
else:
    with open(pkl_file_df_stopwords_removed, 'rb') as pklfile:
        df = pickle.load(pklfile)

# formalize the text list like this: ['我 爱 中国','我 是 中国人']
text_list = df['stopwords_removed'].tolist()
text_list = [' '.join(x) for x in text_list]

m_max_features = 200
vectorizer = CountVectorizer(token_pattern=u'(?u)\w+',max_features=m_max_features)
count0 = vectorizer.fit_transform(text_list)

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(count0)

print(vectorizer.vocabulary_)
print(len(vectorizer.vocabulary_))
names = vectorizer.get_feature_names()
print(names)
print(len(names))
print(count0.toarray())
print(count0.toarray().shape)

print("pause")