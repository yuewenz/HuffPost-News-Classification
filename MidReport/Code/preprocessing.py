# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 10:02:39 2021

@author: Daisy
"""
import os
os.chdir(r"C:\Users\Daisy\Downloads\GT Coursework\Spring Semester 2021\CS7641 Machine Learning\Final Project\Data Processing and Tokenizing")

import re
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# read data
df = pd.read_json("News_Category_Dataset_v2.json", lines = True)
## delete rows if either headlines or short descriptions are empty
df = df[(df["headline"].astype(bool))&(df["short_description"].astype(bool))]

print("There are {} rows and {} columns in the news data:".format(df.shape[0], df.shape[1]))

df.head(5)

df.describe()

# data exploration and visualization
## plot number of rows in each news category
count_df = pd.DataFrame(df['category'].value_counts()).reset_index()
print('There are', len(count_df), 'news categories')

sns.set_style('darkgrid')
plt.figure(figsize=(10, 12))
count_df_plot = sns.barplot(data=count_df, y='index', x='category', palette='Dark2')
for p in count_df_plot.patches:
    count_df_plot.annotate(int(p.get_width()), xy=(p.get_width(), p.get_y()+p.get_height()/2),
                           xytext=(5, 0), textcoords='offset points', ha="left", va="center")

plt.title('Number of News in Each Category', loc='left', fontsize=20)
plt.xlabel("")
plt.ylabel("")
plt.show()

## merge categories with similar meanings into one
df.category[df.category=='THE WORLDPOST'] = 'WORLDPOST'
df.category[(df.category=='GREEN')] = 'ENVIRONMENT'
df.category[df.category=='CULTURE & ARTS'] = 'ARTS'
df.category[df.category=='COMEDY'] = 'ENTERTAINMENT'
df.category[df.category=='STYLE'] = 'STYLE & BEAUTY'
df.category[df.category=='ARTS'] = 'ARTS & CULTURE'
df.category[df.category=='COLLEGE'] = 'EDUCATION'
df.category[(df.category=='SCIENCE')|(df.category=='TECH')] = 'SCIENCE & TECH'
df.category[df.category=='WEDDINGS'] = 'GOOD NEWS'
df.category[df.category=='TASTE'] = 'FOOD & DRINK'
df.category[(df.category=='PARENTS') | (df.category=='FIFTY')] = 'PARENTING'
df.category[df.category=='WORLD NEWS'] = 'WORLDPOST'
print('After merging, there are', len(df["category"].unique()), 'news categories')

## plot number of average words in headlines in each news category
df["headline_len"] = df.apply(lambda row:len(word_tokenize(row["headline"])), axis = 1)
headline_words_df = df.groupby(["category"])["headline_len"].mean().reset_index()
headline_words_df = headline_words_df.sort_values("headline_len", ascending=False)

sns.set_style('darkgrid')
plt.figure(figsize=(10, 12))
headline_df_plot = sns.barplot(data=headline_words_df, y='category', x='headline_len', palette='Dark2')
for p in headline_df_plot.patches:
    headline_df_plot.annotate("%.2f" %p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),
                           xytext=(5, 0), textcoords='offset points', ha="left", va="center")
plt.title('Average Number of Words in Headlines in Each Category', loc='left', fontsize=20)
plt.xlabel("")
plt.ylabel("")
plt.show()

## plot number of average words in short descriptions in each news category
df["description_len"] = df.apply(lambda row:len(word_tokenize(row["short_description"])), axis = 1)
description_words_df = df.groupby(["category"])["description_len"].mean().reset_index()
description_words_df = description_words_df.sort_values("description_len", ascending = False)

sns.set_style('darkgrid')
plt.figure(figsize=(10, 12))
description_df_plot = sns.barplot(data=description_words_df, y='category', x='description_len', palette='Dark2')
for p in description_df_plot.patches:
    description_df_plot.annotate("%.2f" %p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),
                           xytext=(5, 0), textcoords='offset points', ha="left", va="center")
plt.title('Average Number of Words in Short Descriptions in Each Category', loc='left', fontsize=20)
plt.xlabel("")
plt.ylabel("")
plt.show()

# data preprocessing
## remove non-alphabetic and non-numeric words
def remove_non_alpha_num(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'([^\s\w]|_)+', '',text)
    return text

## remove stopwords
sw_nltk = stopwords.words("english")
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in sw_nltk:
            final_text.append(i.strip())
    return " ".join(final_text)

## lemmatization 
stemmer = WordNetLemmatizer()
def lemmatization(text):
    text = text.split()
    final_text = [stemmer.lemmatize(word) for word in text]
    final_text = " ".join(final_text)
    return final_text

## apply functions to both headlines and short descriptions    
def text_preprocessing(df, column):
    column_new = column + "_new"
    df[column_new] = df[column].str.lower()
    df[column_new] = df[column_new].apply(remove_non_alpha_num)
    df[column_new] = df[column_new].apply(remove_stopwords)
    df[column_new] = df[column_new].apply(lemmatization)
    return df[column_new]

for i in ["headline","short_description"]:
    text_preprocessing(df, i)

## add one column combining headline and descriptions
df = df[(df["headline_new"].notnull())&(df["short_description_new"].notnull())]
df["head_short"] = df["headline_new"] + " " + df["short_description_new"]

# data preparation for model training     
## label encoding 
encoder = LabelEncoder()
df["labels"] = encoder.fit_transform(df["category"])

## split train and test set
train, validation = train_test_split(df, 
                                     test_size = 0.2, 
                                     stratify = df["labels"],
                                     random_state = 42)
 
train.to_csv("news_classification_train.csv", index = False)
validation.to_csv("news_classification_test.csv", index = False)















