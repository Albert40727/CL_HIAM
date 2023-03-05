from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

def setdictionary(keywords_list):
    count = 0
    for i in range(len(keywords_list)): #計算平均長度
        count += len(keywords_list[i])
    answer = count//len(keywords_list)
    print(f"keyword平均長度: {answer}")
    #拿keywords建立字典
    dictionary = corpora.Dictionary(keywords_list)
    return dictionary

# keyword刪除停用詞
def LDAGrouping(dictionary,keywords_list,df,target_column_name):
    corpus = [dictionary.doc2bow(text) for text in keywords_list]
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5)#用字典做LDA
    doc_topic = []
    for e, values in enumerate(lda.inference(corpus)[0]):
        topic_val = 0
        topic_id = 0
        #找出分數最高的group
        for tid, val in enumerate(values):
            if val > topic_val:
                topic_val = val
                topic_id = tid
        doc_topic.append(topic_id)
    df[target_column_name] = doc_topic
    return df

# keyword做詞幹統一
def delete_stopwords(kwlist):
    vectorizer = TfidfVectorizer(stop_words = "english")
    stop_list = list(vectorizer.get_stop_words())
    clean_list = []
    for i in range(len(kwlist)):
        tmp = []      
        for word in kwlist[i]:
            if len(word)>2:
                if word not in stop_list:
                    tmp.append(word)
        if len(tmp)==0 :
            print("空: "+str(i))
        clean_list.append(tmp)
    return clean_list

def stemmer(kwlist):
    porter_stemmer = PorterStemmer()
    all_stem_words=[]
    for i in range (len(kwlist)):
        stem_list =[]
        for word in kwlist[i]:
            stem_list.insert(-1,porter_stemmer.stem(word))
        all_stem_words.append(stem_list)
    return all_stem_words