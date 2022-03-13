import pandas as pd
import sklearn
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('sentiment_analysis.txt', sep='\t', names=['liked', 'txt'])
print(data.head())

stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase='True', stop_words=stopset, strip_accents='ascii')

y = data.liked
X = vectorizer.fit_transform(data.txt) # always use fit transform

print(y.shape, '---', X.shape)
print(X[:,1].shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print(X_test)

clf = naive_bayes.MultinomialNB()
clf.fit(X_train, y_train)

score = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
print('Score is - ', score)

message = ''

print('Running....\n')
while (message!= 'bye now' and message!= 'terminate the code' and message!= 'Bye now'):
    print('Bot: Whats on your mind?')
    message = input('')
    message_mat = np.array([str(message)])
    message_v = vectorizer.transform(message_mat)
    if clf.predict(message_v) == 1:
        print('Bot: Yes I do agree with you')
    elif clf.predict(message_v) == 0:
        print("Bot: I don't agree with you, but ok!")


