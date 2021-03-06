# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 11:39:06 2021

@author: Daisy
"""
import os

os.chdir(r"E:\Project\help\task\CS7641\data")

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
    fig, ax = plt.subplots(figsize=(20, 20))
    cm = confusion_matrix(true_label, pred_label)
    sns.heatmap(cm, annot=True, cbar=False,
                fmt='1d', cmap='Blues', ax=ax,
                xticklabels=ticklabels, yticklabels=ticklabels)
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
print("Method 1: Use CountVectorizer Method")
count_vectorizer = CountVectorizer(max_features=10000,
                                   ngram_range=(1, 2))

## case 1: use headlines and short descriptions as feature for classification
tr_count = count_vectorizer.fit_transform(train["head_short"])
val_count = count_vectorizer.transform(validation["head_short"])

###-------------------add your machine learning model below-----------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
svm_clf = Pipeline(( ("scaler", StandardScaler(with_mean=False)),
                     ("linear_svc", LinearSVC(C=1, loss="hinge")) ,))
svm_clf.fit(tr_count,tr_y)
y_pred= svm_clf.predict(val_count)
plot_confusion_matrix(val_y_true, y_pred, ticklabels)
# print("Accuracy:", (val_y_true ==y_pred).sum() / (len(y_pred)),
#       "Linear SVM,use headlines and short descriptions as feature")
from sklearn.naive_bayes import GaussianNB  # ???????????????
tr_count = tr_count.toarray()
val_count = val_count.toarray()
gnb = GaussianNB()  # ??????
pred = gnb.fit(tr_count, tr_y)  # ??????
y_pred = pred.predict(val_count)  # ??????
plot_confusion_matrix(val_y_true, y_pred, ticklabels)
# print("Accuracy:", (val_y_true == y_pred).sum() / len(y_pred),
#       "Gaussian_naive_bayes,use headlines and short descriptions as feature")
del tr_count
del val_count
## case 2: use headlines as feature for classification
count_vectorizer = CountVectorizer(max_features=10000,
                                   ngram_range=(1, 2))
tr_head_count = count_vectorizer.fit_transform(train["headline_new"].values.astype('U'))
val_head_count = count_vectorizer.transform(validation["headline_new"].values.astype('U'))
svm_clf = Pipeline(( ("scaler", StandardScaler(with_mean=False)),
                     ("linear_svc", LinearSVC(C=1, loss="hinge")) ,))
svm_clf.fit(tr_head_count,tr_y)
y_pred= svm_clf.predict(val_head_count)
plot_confusion_matrix(val_y_true, y_pred, ticklabels)
# print("Accuracy:", (val_y_true ==y_pred).sum() / (len(y_pred)),
#       "Linear SVM,use headlines as feature for classification")
gnb = GaussianNB()  # ??????
tr_head_count = tr_head_count.toarray()
val_head_count = val_head_count.toarray()
pred = gnb.fit(tr_head_count, tr_y)  # ??????
y_pred = pred.predict(val_head_count)  # ??????
plot_confusion_matrix(val_y_true, y_pred, ticklabels)
# print("Accuracy:", (val_y_true == y_pred).sum() / len(y_pred),
#       "Gaussian_naive_bayes,use headlines as feature for classification")
del tr_head_count
del val_head_count
## case 3: use short descriptions as feature for classification
count_vectorizer = CountVectorizer(max_features=10000,
                                   ngram_range=(1, 2))
tr_desc_count = count_vectorizer.fit_transform(train["short_description_new"].values.astype('U'))
val_desc_count = count_vectorizer.transform(validation["short_description_new"].values.astype('U'))
svm_clf = Pipeline(( ("scaler", StandardScaler(with_mean=False)),
                     ("linear_svc", LinearSVC(C=1, loss="hinge")) ,))
svm_clf.fit(tr_desc_count,tr_y)
y_pred= svm_clf.predict(val_desc_count)
plot_confusion_matrix(val_y_true, y_pred, ticklabels)
# print("Accuracy:", (val_y_true ==y_pred).sum() / (len(y_pred)),
#       "Linear SVM,use short descriptions as feature for classification")

gnb = GaussianNB()  # ??????
tr_desc_count = tr_desc_count.toarray()
val_desc_count = val_desc_count.toarray()
pred = gnb.fit(tr_desc_count, tr_y)  # ??????
y_pred = pred.predict(val_desc_count)  # ??????
plot_confusion_matrix(val_y_true, y_pred, ticklabels)
# print("Accuracy:", (val_y_true == y_pred).sum() / len(y_pred),
#       "Gaussian_naive_bayes,use short descriptions as feature for classification")
del tr_desc_count
del val_desc_count
### Method 2: Use TF-IDF Method
print("Method 2: Use TF-IDF Method")
## case 1: use headlines and short descriptions as feature for classification
tfidf_vectorizer = TfidfVectorizer(max_features=10000,
                                   ngram_range=(1, 2))
tr_tfidf = tfidf_vectorizer.fit_transform(train["head_short"])
val_tfidf = tfidf_vectorizer.transform(validation["head_short"])
###------------------add your machine learning model below-----------------------------
svm_clf = Pipeline(( ("scaler", StandardScaler(with_mean=False)),
                     ("linear_svc", LinearSVC(C=1, loss="hinge")) ,))

svm_clf.fit(tr_tfidf,tr_y)
y_pred= svm_clf.predict(val_tfidf)
plot_confusion_matrix(val_y_true, y_pred, ticklabels)
# print("Accuracy:", (val_y_true ==y_pred).sum() / (len(y_pred)),
#       "Linear SVM,use headlines and short descriptions as feature")
gnb=GaussianNB() #??????
# tr_count = tr_count.toarray()
tr_tfidf = tr_tfidf.toarray()
val_tfidf = val_tfidf.toarray()
pred=gnb.fit(tr_tfidf,tr_y) #??????
y_pred=pred.predict(val_tfidf) #??????
plot_confusion_matrix(val_y_true, y_pred, ticklabels)
# print("Accuracy:", (val_y_true ==y_pred).sum() / (len(y_pred)),
#       "Gaussian_naive_bayes,use headlines and short descriptions as feature")
del tr_tfidf
del val_tfidf
## case 2: use headlines as feature for classification
tfidf_vectorizer = TfidfVectorizer(max_features=10000,
                                   ngram_range=(1, 2))
tr_head_tfidf = tfidf_vectorizer.fit_transform(train["headline_new"].values.astype('U'))
val_head_tfidf = tfidf_vectorizer.transform(validation["headline_new"].values.astype('U'))
# ---
svm_clf.fit(tr_head_tfidf,tr_y)
y_pred= svm_clf.predict(val_head_tfidf)
plot_confusion_matrix(val_y_true, y_pred, ticklabels)
# print("Accuracy:", (val_y_true ==y_pred).sum() / (len(y_pred)),
#       "Linear SVM,use headlines as feature for classification")
gnb=GaussianNB() #??????
# tr_count = tr_count.toarray()
tr_head_tfidf = tr_head_tfidf.toarray()
val_head_tfidf = val_head_tfidf.toarray()
pred=gnb.fit(tr_head_tfidf,tr_y) #??????
y_pred=pred.predict(val_head_tfidf) #??????
plot_confusion_matrix(val_y_true, y_pred, ticklabels)
# print("Accuracy:", (val_y_true ==y_pred).sum() / (len(y_pred)),
#       "Gaussian_naive_bayes,use headlines as feature for classification")
del tr_head_tfidf
del val_head_tfidf

## case 3: use short descriptions as feature for classification
tfidf_vectorizer = TfidfVectorizer(max_features=10000,
                                   ngram_range=(1, 2))
tr_desc_tfidf = tfidf_vectorizer.fit_transform(train["short_description_new"].values.astype('U'))
val_desc_tfidf = tfidf_vectorizer.transform(validation["short_description_new"].values.astype('U'))

# ---
svm_clf.fit(tr_desc_tfidf,tr_y)
y_pred= svm_clf.predict(val_desc_tfidf)
plot_confusion_matrix(val_y_true, y_pred, ticklabels)
# print("Accuracy:", (val_y_true ==y_pred).sum() / (len(y_pred)),
#       "Linear SVM,use short descriptions as feature for classification")
gnb=GaussianNB() #??????
# tr_count = tr_count.toarray()
tr_desc_tfidf = tr_desc_tfidf.toarray()
val_desc_tfidf = val_desc_tfidf.toarray()
pred=gnb.fit(tr_desc_tfidf,tr_y) #??????
y_pred=pred.predict(val_desc_tfidf) #??????
plot_confusion_matrix(val_y_true, y_pred, ticklabels)
# print("Accuracy:", (val_y_true ==y_pred).sum() / (len(y_pred)),
#       "Gaussian_naive_bayes,use headlines as feature for classification")
del tr_desc_tfidf
del val_desc_tfidf

