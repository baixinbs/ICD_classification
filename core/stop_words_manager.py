import jieba
import pandas as pd

class StopWordsManager(object):
    def __init__(self, filepath_stopwords='E:\\ICD_classification\\stopwords\\stop_words.txt'):
        self.filepath_stopwords = filepath_stopwords
        self.stopwords = self.load_stop_words()

    def load_stop_words(self):
        f = open(self.filepath_stopwords, encoding="utf-8")
        stop = [line.strip() for line in f.readlines()]
        return stop

    def remove_stop_words(self, msg):
        stop_words = self.stopwords
        stop_words.append('\n')
        stop_words.append('\t')
        stop_words.append(' ')
        seg_list = jieba.cut(msg, cut_all=False)
        # key_list = jieba.analyse.extract_tags(msg,20) #get keywords
        leftWords = []
        for x in seg_list:
            if x not in stop_words:
                leftWords.append(x)
        return leftWords

if __name__=="__main__":
    filepath_stopwords = 'E:\\ICD_classification\\stopwords\\stop_words.txt'
    m_stopwordsmanager = StopWordsManager(filepath_stopwords)

    from core.file_reader import FileReader
    pd.set_option('display.max_columns', None)
    m_reader = FileReader()
    df = m_reader.get_dataframe()
    # add a new column to save the prepoocessed content
    df['stopwords_removed'] = [''] * df.shape[0]
    count = 0
    for i in range(20):
        count += 1
        print(count)
        df.loc[i, 'stopwords_removed'] = m_stopwordsmanager.remove_stop_words(df.loc[i, '内容'])
    print(df.head())
    print(df.loc[0, 'stopwords_removed'])