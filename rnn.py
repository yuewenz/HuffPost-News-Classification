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
data = pd.read_json("News_Category_Dataset_v2.json", lines=True)[['category','headline','authors','short_description']]
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
y_test = np.argmax(y_test,axis=1) 

print(train_set.shape)
print(train_label.shape)


print(val_set.shape)
print(val_label.shape)

data.category[data.category=='THE WORLDPOST'] = 'WORLDPOST'
data.category[(data.category=='GREEN')] = 'ENVIRONMENT'
data.category[data.category=='CULTURE & ARTS'] = 'ARTS'
data.category[data.category=='COMEDY'] = 'ENTERTAINMENT'
data.category[data.category=='STYLE'] = 'STYLE & BEAUTY'
data.category[data.category=='ARTS'] = 'ARTS & CULTURE'
data.category[data.category=='COLLEGE'] = 'EDUCATION'
data.category[(data.category=='SCIENCE')|(data.category=='TECH')] = 'SCIENCE & TECH'
data.category[data.category=='WEDDINGS'] = 'GOOD NEWS'
data.category[data.category=='TASTE'] = 'FOOD & DRINK'
data.category[(data.category=='PARENTS') | (data.category=='FIFTY')] = 'PARENTING'
data.category[data.category=='WORLD NEWS'] = 'WORLDPOST'
print('After merging, there are', len(data["category"].unique()), 'news categories')

DIM = 100
embeddings = {}

with open('glove.6B.100d.txt','r') as file:
    line = file.readline().split()
    while not line == []: 
        word = line[0]
        vec = np.array(line[1:]).astype(np.float32)
        embeddings[word] = vec
        line = file.readline().split()
        
def embed(word):
    if word in embeddings.keys():
        return embeddings[word]
    else:
        return np.zeros(100)

_ = data[['authors','category']].groupby('authors').count().sort_values('category',ascending =False).index[:49]

top_50_authors = {}
i_50 = np.identity(50).astype(np.float32)
for i,author in enumerate(_):
    top_50_authors[author] = i_50[i,:]

def get_author_encoding(row):
    author = row['authors']
    if author in top_50_authors.keys():
        return top_50_authors[author]
    else:
        return i_50[-1,:]

CATEGORIES = {}
categories = data['category'].unique()
for i,c in enumerate(categories):
    zeros = np.zeros(len(categories))
    zeros[i] = 1
    CATEGORIES[c] = zeros

cat_vec2txt = {}
for i in list(CATEGORIES.keys()):
    cat_vec2txt[CATEGORIES[i].argmax()] = i

labels = data['category'].values
categories = set(labels)
train_idxs = np.array([])
test_idxs = np.array([])
train_distribution = []
for c in categories:
    subset = np.where(labels==c)[0]
    np.random.shuffle(subset)
    q = subset.shape[0]*8//10
    train_idxs = np.hstack((train_idxs,subset[:q]))
    test_idxs = np.hstack((test_idxs,subset[q:]))
    train_distribution.append(q)


train_data = data.iloc[train_idxs]
test_data = data.iloc[test_idxs]

MAX_WORDS = 50

regex = re.compile('[^a-zA-Z ]')
def get_text_encoding(row):
    text = regex.sub('',row['headline'] + ' ' + row['short_description']).lower().split()
    
    word_matrix = np.zeros((32,101))
    for i in range(32):
        if i<len(text):
            word_matrix[i] = np.append(embed(text[i]),0)
        else:
            v = np.zeros(101)
            v[-1] = 1
            word_matrix[i] = v
    return word_matrix

def get_input_matrix(df, idx):
    row = df.iloc[idx]
    
    word_matrix = get_text_encoding(row)
    author = get_author_encoding(row)
    author_matrix = np.zeros((word_matrix.shape[0],author.shape[0])) + author
    matrix = np.hstack((author_matrix,word_matrix))
    
    cat_vec = CATEGORIES[row['category']]
    return matrix,cat_vec

def generate_data(df, batch_size, shuffle = True):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    i = 0
    while True:
        image_batch = []
        category_batch = []
        for b in range(batch_size):
            if i == len(df):
                if shuffle:
                    df = df.sample(frac=1).reset_index(drop=True)
                i = 0
            image, category = get_input_matrix(df, i)
            image_batch.append(image)
            category_batch.append(category)
            i += 1

        yield np.array(image_batch), np.array(category_batch)

BATCH_SIZE = 32
EPOCHS = 20

model = Sequential()
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, input_shape=(50, 151)))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(28, activation='softmax'))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

history = model.fit_generator(
    generate_data(train_data, BATCH_SIZE),
    steps_per_epoch=6125,
    epochs=EPOCHS,
    validation_data = generate_data(test_data, BATCH_SIZE, shuffle=False),
    validation_steps = 1255
    )

def predict_gen(df, BATCH_SIZE):
    gen = generate_data(df, BATCH_SIZE, shuffle=False)
    global y_actual
    y_actual = []
    while True:
        x,y = next(gen)
        y_actual = y_actual + list(y)
        yield(x)
        
y_pred = model.predict(predict_gen(test_data, BATCH_SIZE), steps=1250)
y_actual = np.array(y_actual)
y_pred = y_pred.argmax(1)
y_actual = y_actual.argmax(1)[:len(y_pred)]

def plot_confusion_matrix(true_label, pred_label, ticklabels):
    fig, ax = plt.subplots(figsize=(20, 20))
    cm = confusion_matrix(true_label, pred_label)
    sns.heatmap(cm, annot=True, cbar=False,
                fmt='1d', cmap='Blues', ax=ax,
                xticklabels=ticklabels, yticklabels=ticklabels)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual News Class')
    plt.show()
    
ticklabels = ['ARTS & CULTURE',
 'BLACK VOICES',
 'BUSINESS',
 'CRIME',
 'DIVORCE',
 'EDUCATION',
 'ENTERTAINMENT',
 'ENVIRONMENT',
 'FOOD & DRINK',
 'GOOD NEWS',
 'HEALTHY LIVING',
 'HOME & LIVING',
 'IMPACT',
 'LATINO VOICES',
 'MEDIA',
 'MONEY',
 'PARENTING',
 'POLITICS',
 'QUEER VOICES',
 'RELIGION',
 'SCIENCE & TECH',
 'SPORTS',
 'STYLE & BEAUTY',
 'TRAVEL',
 'WEIRD NEWS',
 'WELLNESS',
 'WOMEN',
 'WORLDPOST']

plot_confusion_matrix(y_actual, y_pred ,ticklabels)


