import pandas as pd 
import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Embedding
from sklearn.utils import shuffle
import seaborn as sns

import re, string
from keras.models import Sequential
from keras.layers import LSTM, Dense

dataset = pd.read_csv('news_classification_train.csv')
dataset.drop(['authors', 'link', 'date'], axis = 1, inplace = True)

df = dataset.copy()
df.drop_duplicates(keep='last', inplace=True) 
df.duplicated(subset=['short_description_new','headline_new']).sum()
df.drop_duplicates(subset=['short_description_new','headline_new'],keep='last',inplace=True)

df.loc[df['headline_new'] == "", 'headline'] = np.nan
df.dropna(subset=['headline_new'], inplace=True)
df.loc[df['short_description_new'] == "", 'short_description_new'] = np.nan
df.dropna(subset=['short_description_new'], inplace=True)

df = shuffle(df)
df.reset_index(inplace=True, drop=True)

df['desc'] = df['headline_new'].astype(str)+"-"+df['short_description_new']
df.drop(columns =['headline_new','short_description_new'],axis = 1, inplace=True)
df.astype(str)

test = pd.read_csv('news_classification_test.csv')
test.drop(['authors', 'link', 'date'], axis = 1, inplace = True)

df_test = test.copy()
df_test.drop_duplicates(keep='last', inplace=True) 
df_test.duplicated(subset=['short_description_new','headline_new']).sum()
df_test.drop_duplicates(subset=['short_description_new','headline_new'],keep='last',inplace=True)

df_test.loc[df_test['headline_new'] == "", 'headline'] = np.nan
df_test.dropna(subset=['headline_new'], inplace=True)
df_test.loc[df_test['short_description_new'] == "", 'short_description_new'] = np.nan
df_test.dropna(subset=['short_description_new'], inplace=True)

df_test = shuffle(df_test)
df_test.reset_index(inplace=True, drop=True)

df_test['desc'] = df_test['headline_new'].astype(str)+"-"+df_test['short_description_new']
df_test.drop(columns =['headline_new','short_description_new'],axis = 1, inplace=True)
df_test.astype(str)

X_train = df['desc'] #train_data
y_train = df['category'] #train_label
X_val = df_test['desc'] #test_data
y_val = df_test['category'] #test_label

X_val, X_test , y_val, y_test= train_test_split(X_val,y_val, test_size=0.5, random_state=42)

vocab_size = 25000
max_length = 50
trunc_type ='post'
padding_type ='post'
oov_tok = "<OOV>"

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index

X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train,maxlen= max_length,padding=padding_type, truncating=trunc_type)
y_train = np.asarray(y_train)
y_train = pd.get_dummies(y_train)

X_val = tokenizer.texts_to_sequences(X_val)
X_val = pad_sequences(X_val,maxlen= max_length,padding=padding_type, truncating=trunc_type)
y_val = np.asarray(y_val)
y_val = pd.get_dummies(y_val)

train_set = np.array(X_train)
val_set = np.array(X_val)

train_label = np.array(y_train)
val_label = np.array(y_val)


y_test = pd.get_dummies(y_test)
y_test = np.asarray(y_test)
y_test = np.argmax(y_test,axis=1)   #this would be our ground truth label while testing

print(train_set.shape)
print(train_label.shape)


print(val_set.shape)
print(val_label.shape)

path_to_glove_file =  'glove.6B.100d.txt'
num_tokens = len(tokenizer.word_index.items()) + 2
embedding_dim = 100
hits = 0
misses = 0


embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))


# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))


tf.keras.backend.clear_session()
embed_size = 100
model = keras.models.Sequential([
                                 
        Embedding(num_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        mask_zero=True,input_shape=[None],trainable=False),
        keras.layers.Bidirectional(keras.layers.LSTM(256, dropout = 0.4)),
        keras.layers.Dense(28, activation="softmax")
            
        ])

model.summary()

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
history = model.fit( train_set,train_label,
                     batch_size = 32,
                     steps_per_epoch=len(X_train) // 32, 
                     validation_data = (val_set , val_label),
                     validation_steps = len(val_set)//32, epochs=20)

classes = dataset['category'].value_counts().index

def prediction(inference_data):
    X = tokenizer.texts_to_sequences(inference_data)
    X = pad_sequences(X,maxlen= max_length,padding=padding_type, truncating=trunc_type)
    pred = model.predict(X)
    pred_value = tf.argmax(pred,axis =1).numpy()                
    return pred_value

y_pred = prediction(X_test)
print(classification_report(np.asarray(y_test),np.asarray(y_pred)))


