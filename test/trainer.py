from core.data_client import DataClient
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import pandas as pd
import numpy as np

data_dir = "E:\\ICD_classification\\"
m_max_features = 100
def prepare_data():
    filepath_stopwords = data_dir + 'stopwords\\stop_words.txt'
    excel_file_6600 = data_dir + 'data\\6600.xlsx'
    pkl_file_6600 = data_dir + 'pickle\\6600_pkl'
    excel_file_cyxj = data_dir + 'data\\cyxj.xls'
    pkl_file_cyxj = data_dir + 'pickle\\cyxj_pkl'
    pkl_file_df_stopwords_removed = data_dir + 'pickle\\df_stopwords_removed_pkl'
    pd.set_option('display.max_columns', None)

    m_data_client = DataClient(filepath_stopwords, excel_file_6600, pkl_file_6600, excel_file_cyxj, pkl_file_cyxj)
    df = m_data_client.load_df_without_stopwords(pkl_file_df_stopwords_removed)

    # formalize the text list like this: ['我 爱 中国','我 是 中国人']
    text_list = df['stopwords_removed'].tolist()
    text_list = [' '.join(x) for x in text_list]

    vectorizer = CountVectorizer(token_pattern=u'(?u)\w+',max_features=m_max_features)
    # vectorizer = CountVectorizer(token_pattern=u'(?u)\w+')
    m_count = vectorizer.fit_transform(text_list)
    transformer = TfidfTransformer()
    m_tfidf = transformer.fit_transform(m_count)

    x = m_tfidf
    y = df['出院主要诊断编码'].tolist()
    y = np.array([int(i[-1:]) for i in y])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = prepare_data()
# m_model = MultinomialNB(alpha = 0.1)
# m_model = DecisionTreeClassifier()
# m_model = SVC()
m_model = XGBClassifier()
m_model.fit(x_train.toarray(),
            np.array(y_train),
            eval_metric='auc'
            )
y_pred = m_model.predict(x_test.toarray())
accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)

# def PRF(y_true, y_pred):
#     acc_test = metrics.accuracy_score(y_true, y_pred)
#     P_test = metrics.precision_score(y_true, y_pred, average='macro')
#     R_test = metrics.recall_score(y_true, y_pred, average='macro')
#     F_test = metrics.f1_score(y_true, y_pred, average='macro')
#     print(acc_test, P_test, R_test, F_test)
# PRF(y_test,y_pred)

print("pause")