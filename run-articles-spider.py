"""
Created on 10/2/2021
Author: Thu Pham
Description: Assignment 1

"""

##############################
# Import Packages
##############################
import nltk
nltk.download('punkt')

import scrapy  # object-oriented framework for crawling and scraping
import os  # operating system commands
import pandas as pd
import numpy as np
import json
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import Counter
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

##############################
# Directory Operations
##############################

# make directory for storing complete html code for web page
page_dirname = 'wikipages'
if not os.path.exists(page_dirname):
	os.makedirs(page_dirname)

# function for walking and printing directory structure
def list_all(current_directory):
    for root, dirs, files in os.walk(current_directory):
        level = root.replace(current_directory, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

# initial directory should have this form (except for items beginning with .):
#    TOP-LEVEL-DIRECTORY-FOR-SCRAPY-WORK
#        RUN-SCAPY-JOB-NAME.py
#        scrapy.cfg
#        DIRECTORY-FOR-SCRAPY
#            __init__.py
#            items.py
#            pipelines.py
#            settings.py
#            spiders
#                __init__.py
#                FIRST-SCRAPY-SPIDER.py
#                SECOND-SCRAPY-SPIDER.py

# examine the directory structure
current_directory = os.getcwd()
list_all(current_directory)

# list the avaliable spiders
print('\nScrapy spider names:\n')
os.system('scrapy list')

##############################
# Run Scrapy Crawler
##############################

# decide upon the desired format for exporting output: 
# such as csv, JSON, XML, or jl for JSON lines

# run the scraper exporting results as csv, json, jl, or xml
jl_file = 'items.jl'
#if os.path.exists(jl_file):
#        os.remove(jl_file)
#os.system('scrapy crawl articles-spider -o ' + jl_file)
#print('scrapy crawler has finished running. ' + jl_file + ' has been created.\n')

output_file = 'items.csv'
#if os.path.exists(output_file):
#        os.remove(output_file)
#os.system('scrapy crawl articles-spider -o ' + output_file)
#print('scrapy crawler has finished running. ' + output_file + ' has been created.\n')

# count the number of documents that were gathered
corpus = pd.read_csv(output_file)
corpus_len = len(corpus)
print(f' {corpus_len} documents were created.')
print(corpus.head())
print(corpus.dtypes)

# only keep documents that are 300 characters long or more
corpus['len'] = 0
for i in range(0, corpus_len):
        corpus['text'].iloc[i] = str(corpus['text'].iloc[i])
        corpus['len'].iloc[i] = len(corpus['text'].iloc[i])
corpus = corpus[corpus['len'] >= 300]
print(corpus)

# only keep documents that are in English
corpus['lang'] = ''
corpus_len = len(corpus)
for i in range(0, corpus_len):
        try:
                language = detect(corpus['text'].iloc[i])
        except:
                language = 'error'
        corpus['lang'].iloc[i] = language
corpus = corpus[corpus['lang']=='en']
print(corpus)

##############################
# Preprocess Text Documents
##############################

stopwords = set(stopwords.words('english'))
tokens = []
master_tokens = []
labels = []

body = corpus['text'].tolist()

# clean text
corpus_len = len(corpus)
corpus_len = 300
for i in range(0, corpus_len):
        # split into words
        words = word_tokenize(corpus['text'].iloc[i])
        # convert to lower case
        words = [w.lower() for w in words]
        # remove punctuation
        words = [w for w in words if w.isalpha()]
        # remove short words
        words = [w for w in words if len(w) > 2]
        # remove stop words
        words = [w for w in words if w not in stopwords]
        print(i)
        
        tokens.append(words)
        
        # add words to master list
        for w in words:
                if w not in master_tokens:
                        master_tokens.append(w)

        # add url to label list
        labels.append(corpus['url'].iloc[i])

body_clean = []
for i in range(0, corpus_len):
        body_clean.append(" ".join(tokens[i]))

##############################
# Vectorize Text
##############################

# Approach 1: Analyst Judgement
vector1 = []
for i in range(0, corpus_len):
        count = Counter(tokens[i])
        vector1.append(dict(count))
vector1 = pd.DataFrame.from_dict(vector1)
vector1 = vector1.fillna(0)
print(vector1)
        
# Approach 2: TF-IDF
vectorizer = TfidfVectorizer()
vector2_csr = vectorizer.fit_transform(body_clean)
vector2 = pd.DataFrame(vector2_csr.toarray(), columns=vectorizer.get_feature_names())
print(vector2)

# Approach 3: Doc2Vec
tagged_data = [TaggedDocument(doc, [i]) for i, doc in enumerate(body_clean)]
model = Doc2Vec(tagged_data, vector_size=1000)

vector3 = pd.DataFrame()
for i in range(0, corpus_len):
        vector = pd.DataFrame(model.infer_vector(tokens[i])).transpose()
        vector3 = vector3.append(vector)

vector3 = vector3.reset_index()
del vector3['index']
print(vector3)

##############################
# Compare Vectors
##############################
n = 15

v1 = vector1.sum(axis=0)
v1 = v1.sort_values(ascending=False)
sns.barplot(x=v1[:n], y=v1[:n].index).set_title('Top 10 Words in Vector 1 (Counts of Words)')
plt.show()

v2 = vector2.sum(axis=0)
v2 = v2.sort_values(ascending=False)
sns.barplot(x=v1[:n], y=v1[:n].index).set_title('Top 10 Words in Vector 2 (TF-IDF)')
plt.show()

##############################
# Random Forest Classifier
##############################
