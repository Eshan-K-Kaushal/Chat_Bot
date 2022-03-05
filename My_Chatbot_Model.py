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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

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

# neural net

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(68, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax')) # we use softmax for categorical data

sgd = SGD(lr = 0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss = 'categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
mod = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose= 1)
model.save('chatbot_data.h5', mod)
print('Fin')