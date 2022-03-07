# implemented Naive Bayes
import random
import re
import nltk
import numpy as np
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import swadesh
import translate
import json
import numpy
from nltk.stem import WordNetLemmatizer
import pickle

lemmer = WordNetLemmatizer()
fintents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore = ['?', '.', '!', ',', "'"]

for intent in fintents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list) # in order to add more to the list we want to take a thing and "add"
        # it to the list
        documents.append((word_list, intent['tag'])) # breaking all the patterns into words and then setting
        #them up class/tag wise
        if intent['tag'] not in classes:
            classes.append(intent['tag']) # we make a list of tags too, to keep track of all the classes
            #we have encountered so far

words = [lemmer.lemmatize(word) for word in words if word not in ignore]
words = sorted(set(words))
classes = sorted(set(classes)) # making a sorted list of classes - 'set' removes all the repeating

pickle.dump(words, open('words.pkl', 'wb')) #store them in a file
pickle.dump(classes, open('classes.pkl', 'wb')) #store them in a file

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1 #set 1 for true label
    training.append(([bag, output_row]))
random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

#NB Classifier model

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# definition of the gauss
def gaussain_form(x, sig, mu):
    return 1/(np.sqrt(2*np.pi*sig**2)) * np.exp(-(1/(2*sig**2)) * (x - mu)**2)



class Naive_Bayes:
    def fit(self, X, y):
        samples, features = X.shape # X is a numpy n-d array first dim is no of samples
        # and the number of features is the no of cols
        # kind of obvious
        self._classes = np.unique(y) # finds the unique elements
        # y is basically a list of the class_nos for the data/samples from X
        # it is like [0,0,1,1,2,2,2,3,0,0,3,3,3,3,1,1]
        # so you just take out the unique elements to get the total no of classes
        # as in the above example - 4
        n_classes = len(self._classes) # gives us the number of classes
        # mean, variance and prior
        self._mean = np.zeros((n_classes, features), dtype=np.float64)
        self.var = np.zeros((n_classes, features), dtype=np.float64)
        self.prior = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[c == y] # take only for the class samples not all
            self._mean[c, :] = X_c.mean(axis=0) # for a row at a time for all cols
            self.var[c, :] = X_c.var(axis=0) # for one row - 'c' at a time and for all cols
            self.prior[c] = X_c.shape[0] / float(samples) #total prob of a thing (c)


    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        # here we find the argmax to get the labels
        # we tryin to find the posteriors here
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self.prior[idx])
            class_conditional = np.sum(np.log(self.pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]


    def pdf(self, class_idx, x): # pdf has been set like a bell curve of Gaussian Dist
        mean = self._mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-(x-mean)**2 / (2 * var))
        denom = np.sqrt(2* np.pi * var)
        return numerator/denom

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

print(np.array(train_y)[:,0].shape)
print(np.array(train_x)[:,:11].shape)

model = Naive_Bayes()
mod = model.fit(np.array(train_x), np.array(train_y)[:,0])
print('Fin')
#----------------------------------------------------------------------------------------------

lemmer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean(sentence):
    sent_words = nltk.word_tokenize(sentence)
    sent_words = [lemmer.lemmatize(word) for word in sent_words]
    return sent_words

def bag_of_words(sentence):
    sent_word = clean(sentence)
    bag = [0]*len(words)
    for w in sent_word:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    #res.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in res:
        return_list.append({'intent':classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_repsonse(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
message = ''

print('Running....\n')
while (message!= 'bye now' and message!= 'terminate the code' and message!= 'Bye now'):
    message = input('')
    ints = predict_class(message)
    res = get_repsonse(ints, intents)
    print(res)

