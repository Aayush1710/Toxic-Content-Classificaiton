# coding: utf-8
# #import required packages
import pandas as pd 
import numpy as np

#stats
from scipy.misc import imread
from scipy import sparse
import scipy.stats as ss
import matplotlib.pyplot as plt

#nlp
import string
import re    #for regex

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score


#Looking at  the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
subm = pd.read_csv('sample_submission.csv')
print(train.head())
print('Example of toxic comment')
print(train['comment_text'][0])

#Labels to predict and None label for unknown.
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
print('After replacing few empty comments')
print(train.head())
print(len(train), len(test))

lens = train.comment_text.str.len()
print('Variations in comment lengths')
print(lens.mean(), lens.std(), lens.max())
print('Histogram to show variation in comment lengths')
plt.hist(lens)
plt.savefig('HistCommentLength.png')

print("Check for missing values in Train dataset")
null_check=train.isnull().sum()
print(null_check)
print("Check for missing values in Test dataset")
null_check=test.isnull().sum()
print(null_check)
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)


# class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# # re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
# re_tok = re.compile('([' + string.punctuation + '“”¨«»®´·º½¾¿¡§£₤‘’])')
# def tokenize(s): return re_tok.sub(r' \1 ', s).split()
#
# n = train.shape[0]
# vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
#                min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
#                smooth_idf=1, sublinear_tf=1 )
# trn_term_doc = vec.fit_transform(train[COMMENT])
# test_term_doc = vec.transform(test[COMMENT])