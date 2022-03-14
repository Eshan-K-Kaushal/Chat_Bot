import numpy as np
import pandas
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

# corpus is a collection of sentences or documents
doc1 = 'The cat sat on my lap'
doc2 = 'The dog sat under a tree'
doc3 = 'Today is a sunny day.'
doc4 = 'The live show was great!'
stopset = set(stopwords.words('english'))
stopset.update(['.', '!', ',', "'", "_", '-']) # adding one more entity to the list of stopwords
# print(stopset)
bow1 = doc1.split(" ") # we are basically tokenizing here
bow2 = doc2.split(" ")
print('Bow1: ',bow1)
print('Bow2: ', bow2)
#---------------------------------------------------
print('USING THE WORD_TOKENIZE FROM NLTK')
print('---------------------------------------------')
bow3_wt = word_tokenize(doc3) # the tokenize takes into consideration the prepositions too
print(bow3_wt)
bow4_wt = word_tokenize(doc4) # the tokenize takes into consideration the prepositions too
print(bow4_wt)
print('---------------------------------------------')
#---------------------------------------------------
'''
REMOVING THE STOP WORDS
'''
bow3_no_stopwords = []
for word in bow3_wt:
    if word not in stopset:
        bow3_no_stopwords.append(word)
print('The sentence without stopwords and prepositions- ', bow3_no_stopwords)

bow4_no_stopwords = []
for word in bow4_wt:
    if word not in stopset:
        bow4_no_stopwords.append(word)
print('The sentence without stopwords and prepositions - ', bow4_no_stopwords)
#---------------------------------------------------
# total wordset
wordset = set(bow1).union(set(bow2)) # unique words and the common words from both the bags
wordset_intersection = set(bow1).intersection(set(bow2))
print('the wordset - ', wordset)
print('The wordset but with intersection - ', wordset_intersection) # only the common words from both the sets

# a dict to keep the count of the words
dict1 = dict.fromkeys(wordset, 0)
dict2 = dict.fromkeys(wordset, 0)

for word in bow1:
    dict1[word] += 1
for word in bow2:
    dict2[word] += 1
print('Dict 1 with the counts - \n', dict1)
print('Dict 2 with the counts - \n', dict2)
print('-----------------------------------------------')
# using pandas to make tables of the data
df = pandas.DataFrame([dict1, dict2])
print(df)
# the above is a numerical representation
# this we will use to make models
'''
Using tfidf to give them scores
'''
# the term frequency
def compute_tf(dict,bow): # term frequency
    tf_dict = {} # a (word, tf_score) dictionary for all the words in a bow
    bow_count = len(bow)
    for word, count in dict1.items():
        tf_dict[word] = count / float(bow_count)
    return tf_dict
print('tf score for bow1 = ', compute_tf(dict1, bow1))
print('tf score for bow2 = ', compute_tf(dict2, bow2))
tf_bow1 = compute_tf(dict1, bow1)
tf_bow2 = compute_tf(dict2, bow2)
# the inverse document frequency
def compute_idfs(doclist):
    import math
    idf_dict = {}
    N = len(doclist) # the total number of documents
    idf_dict = dict.fromkeys(doclist[0].keys(), 0)
    for doc in doclist:
        for word, val in doc.items():
            if val > 0:
                idf_dict[word] += 1

    for word, val in idf_dict.items():
        idf_dict[word] = math.log(N/float(val))
    return idf_dict
print('idf_score for all is: ', compute_idfs([dict1, dict2]))
idf_score = compute_idfs([dict1, dict2])

def compute_tfidf(tf_bow, idf):
    tfidf = {}
    for word,val in tf_bow.items():
        tfidf[word] = val*idf[word]
    return tfidf
tfidf_1 = compute_tfidf(tf_bow1, idf_score)
tfidf_2 = compute_tfidf(tf_bow2, idf_score)
df2 = pandas.DataFrame([tfidf_1, tfidf_2])
print(df2)
# comparing now
tfidf_scorer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)
doc1_mat = np.array([doc1])
docs_fit = tfidf_scorer.fit_transform([doc1, doc2])
print(docs_fit)
feature_names = tfidf_scorer.get_feature_names()
dense = docs_fit.todense()
denselist = dense.tolist()
df3 = pd.DataFrame(denselist, columns=feature_names)
print(df3)

