from core.file_reader import FileReader
from core.stop_words_manager import StopWordsManager
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score
from collections import  Counter

import pandas as pd
import os
import pickle
import numpy as np

data_dir = "E:\\ICD_classification\\"

filepath_stopwords = data_dir + 'stopwords\\stop_words.txt'
excel_file_6600 = data_dir + 'data\\6600.xlsx'
pkl_file_6600 = data_dir + 'pickle\\6600_pkl'
excel_file_cyxj = data_dir + 'data\\cyxj.xls'
pkl_file_cyxj = data_dir + 'pickle\\cyxj_pkl'
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

pkl_file_df_stopwords_removed = data_dir + 'pickle\\df_stopwords_removed_pkl'
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
# vectorizer = CountVectorizer(token_pattern=u'(?u)\w+')
m_count = vectorizer.fit_transform(text_list)

transformer = TfidfTransformer()
m_tfidf = transformer.fit_transform(m_count)

# print(vectorizer.vocabulary_)
# print(len(vectorizer.vocabulary_))
# names = vectorizer.get_feature_names()
# print(names)
# print(len(names))
# print(m_count.toarray())
# print(m_count.toarray().shape)
# print(m_tfidf.toarray())
# print(m_tfidf.toarray().shape)

x = m_tfidf
y = df['出院主要诊断编码'].tolist()
y = np.array([int(i[-1:]) for i in y])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print(type(y_train))
print(y_train.shape)
print(type(y_train[0]))
print(y_test)
print(y_train[0:800])

y_train = np.where(y_train==1, 0, y_train)
y_train = np.where(y_train==4, 1, y_train)
y_train = np.where(y_train==5, 2, y_train)
y_train = np.where(y_train==6, 3, y_train)

y_test = np.where(y_test==1, 0, y_test)
y_test = np.where(y_test==4, 1, y_test)
y_test = np.where(y_test==5, 2, y_test)
y_test = np.where(y_test==6, 3, y_test)
print(type(y_train))
print(y_train.shape)
print(type(y_train[0]))
print(Counter(y_train))
print(Counter(y_test))
print("hahaha")
print(x_train.shape)
print(len(y_train))
print(x_test.shape)
print(y_test)
# modelnew = MultinomialNB(alpha = 0.1)
# modelnew = DecisionTreeClassifier()
#modelnew = SVC()
# params = {
#     "objective": "multi:softmax",
#     'gamma': 0.1,
#     'max_depth': 3,
#     'lambda': 2,
#     'max_delta_step': 5,
#     'subsample': 0.5,
#     'colsample_bytree': 0.7,
#     'update': 'refresh',
#     'refresh_leaf': True
# }
params = {
    'booster':'gbtree',
    'objective':'multi:softmax',
    'num_class':5,
    'gamma':0.1,
    'max_depth':6,
    'lambda':2,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'min_child_weight':3,
    'slient':1,
    'eta':0.1,
    'seed':500,
    'nthread':4
}
plst = params.items()
dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)
#watchlist = [(test_result, 'eval'), (train_result, 'train')]
num_round = 100
model = xgb.train(plst, dtrain, num_round)
pred = model.predict(dtest)
accuracy = accuracy_score(y_test, pred)
print('accuarcy:%.2f%%'%(accuracy*100))


# modelnew.fit(x_train, y_train)
# scorenew = modelnew.score(x_test, y_test)
# print(scorenew)
print("pause")