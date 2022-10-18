# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:09:23 2022

@author: filon
"""

import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq

# import dataset
path = '1661-0.txt'
text = open(path, encoding=("utf8")).read().lower()

# split dataset into each word in order, excluding special characters
tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)

unique_words = np.unique(words)
unique_word_index = dict((c,i) for i, c in enumerate(unique_words))

word_length = 5
prev_words = []
next_words = []
for i in range(len(words) - word_length):
    prev_words.append(words[i:i + word_length])
    next_words.append(words[i + word_length])
    
X = np.zeros((len(prev_words), word_length, len(unique_words)), dtype = bool)
Y = np.zeros((len(next_words), len(unique_words)), dtype = bool)
for i, each_words in enumerate(prev_words):
    for j, each_word in enumerate(each_words):
        X[i, j, unique_word_index[each_word]] = 1
    Y[i, unique_word_index[next_words[i]]] = 1
    
# Building the Recurrent Neural Network
model = Sequential()
model.add(LSTM(128, input_shape=(word_length, len(unique_words))))
model.add(Dense(len(unique_words)))
model.add(Activation("softmax"))

# Training the model
optimizer = RMSprop(lr = 0.01)
model.compile(loss = "categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
history = model.fit(X, Y, validation_split=0.5, batch_size = 128, epochs = 20, shuffle = True).history

# Saving the model
model.save('keras_next_word_model.h5')
pickle.dump(history, open("history.p", "wb"))
model = load_model('keras_next_word_model.h5')
history = pickle.load(open("history.p", "rb"))

# Evaluating the Model
plt.figure(0)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.figure(1)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

