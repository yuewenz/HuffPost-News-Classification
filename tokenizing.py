# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 11:39:06 2021

@author: Daisy
"""
import os
os.chdir(r"C:\Users\Daisy\Downloads\GT Coursework\Spring Semester 2021\CS7641 Machine Learning\Final Project\Data Processing and Tokenizing")

import numpy as np
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.initializers import Constant
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

## confusion matrix plot function
def plot_confusion_matrix(true_label, pred_label, ticklabels):     
    fig, ax = plt.subplots(figsize=(20,20))
    cm = confusion_matrix(true_label, pred_label)
    sns.heatmap(cm, annot = True, cbar = False, 
                fmt = '1d', cmap = 'Blues', ax = ax,
                xticklabels = ticklabels, yticklabels = ticklabels)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual News Class')
    plt.show()

## read data
train = pd.read_csv("news_classification_train.csv")
validation = pd.read_csv("news_classification_test.csv")

ticklabels = sorted(train["category"].unique().tolist())

### tr_y is the label for training data, val_y_true is the true label for validation data
tr_y = train["labels"]
val_y_true = validation["labels"]

### Method 1: Use CountVectorizer Method
count_vectorizer = CountVectorizer(max_features = 10000,
                                   ngram_range=(1,2))

## case 1: use headlines and short descriptions as feature for classification
tr_count = count_vectorizer.fit_transform(train["head_short"])
val_count = count_vectorizer.fit_transform(validation["head_short"])
###-------------------add your machine learning model below-----------------------------


## case 2: use headlines as feature for classification
tr_head_count = count_vectorizer.fit_transform(train["headline_new"])
val_head_count = count_vectorizer.fit_transform(validation["headline_new"])


## case 3: use short descriptions as feature for classification
tr_desc_count = count_vectorizer.fit_transform(train["short_description_new"])
val_desc_count = count_vectorizer.fit_transform(validation["short_description_new"])


### Method 2: Use TF-IDF Method
## case 1: use headlines and short descriptions as feature for classification
tfidf_vectorizer = TfidfVectorizer(max_features = 10000,
                                   ngram_range = (1,2))
tr_tfidf = tfidf_vectorizer.fit_transform(train["head_short"])
val_tfidf = tfidf_vectorizer.fit_transform(validation["head_short"])
###------------------add your machine learning model below-----------------------------


## case 2: use headlines as feature for classification
tr_head_tfidf = tfidf_vectorizer.fit_transform(train["headline_new"])
val_head_tfidf = tfidf_vectorizer.fit_transform(validation["headline_new"])


## case 3: use short descriptions as feature for classification
tr_desc_tfidf = tfidf_vectorizer.fit_transform(train["short_description_new"])
val_desc_tfidf = tfidf_vectorizer.fit_transform(validation["short_description_new"])


### Method 3: Use Word Embedding Method
vocab_size = 25000
max_length = 50
trunc_type ='post'
padding_type ='post'
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(train["head_short"])

word_index = tokenizer.word_index

tr_embed = tokenizer.texts_to_sequences(train["head_short"])
tr_embed = pad_sequences(tr_embed,maxlen= max_length,
                         padding=padding_type, 
                         truncating=trunc_type)
tr_y = np.asarray(train["labels"])
tr_y = pd.get_dummies(tr_y)

val_embed = tokenizer.texts_to_sequences(validation["head_short"])
val_embed = pad_sequences(val_embed,
                          maxlen = max_length,
                          padding = padding_type, 
                          truncating=trunc_type)
val_y = np.asarray(validation["labels"])
val_y = pd.get_dummies(val_y)

## create embedding matrix
### !wget http://nlp.stanford.edu/data/glove.6B.zip 
### !unzip -q glove.6B.zip 
glove_file =  'glove.6B.100d.txt'

num_tokens = len(word_index.items()) + 1
embedding_dim = 100
hits = 0
misses = 0

embeddings_index = {}
with open(glove_file, encoding = "utf-8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

embedding_layer = Embedding(len(word_index)+1,embedding_dim,
                            embeddings_initializer = Constant(embedding_matrix),
                            input_length = max_length,
                            trainable = False)
###------------------------add your deep learning model below------------------------



