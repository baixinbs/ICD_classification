import jieba
############################################
seg_list = jieba.cut("他来到上海交通大学", cut_all=True)
print("【全模式】：" + "/ ".join(seg_list))
###############################################
def cutWords(msg, stopWords):
    seg_list = jieba.cut(msg, cut_all=True)
    # key_list = jieba.analyse.extract_tags(msg,20) #get keywords
    leftWords = []
    for i in seg_list:
        if (i not in stopWords):
            leftWords.append(i)
    return leftWords

# 获取停用词表
def loadStopWords():
    f = open('E:\\ICD_classification\\stopwords\\中文停用词表.txt', encoding="utf-8")
    stop = [line.strip() for line in f.readlines()]
    return stop

msg = "他来到上海交通大学"
print(cutWords(msg, loadStopWords()))
#########################################################3
# navie bayes classifier
def nbClassifier(trainData, testData, trainLabel, testLabel):
    vectorizer = CountVectorizer(binary=True)
    fea_train = vectorizer.fit_transform(trainData)
    fea_test = vectorizer.transform(testData);
    #     tv=TfidfVectorizer()#该类会统计每个词语的tf-idf权值
    #     fea_train = tv.fit_transform(trainData)    #return feature vector 'fea_train' [n_samples,n_features]
    #     fea_test = tv.transform(testData);
    print('Size of fea_train:' + repr(fea_train.shape))
    print('Size of fea_test:' + repr(fea_test.shape))
    print(fea_train.nnz)
    print(fea_test.nnz)

    clf = MultinomialNB(alpha=0.01)
    clf.fit(fea_train, np.array(trainLabel))
    pred = clf.predict(fea_test)
    totalScore(pred, testData, testLabel)
