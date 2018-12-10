# coding: utf-8
# #import required packages
import pandas as pd 
import numpy as np

#stats
from scipy.misc import imread
from scipy import sparse
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")

#nlp
import string
import nltk
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from wordcloud import WordCloud
stopwords = nltk.corpus.stopwords.words('english')
from PIL import Image
# The wordcloud of Toxic Comments
plt.figure(figsize=(16,13))
import re    #for regex

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score

#Looking at  the data
train = pd.read_csv('train.csv')
df=pd.DataFrame(train)
test = pd.read_csv('test.csv')
subm = pd.read_csv('sample_submission.csv')
print(train.head())
print('Example of toxic comment')
print(train['comment_text'][0])

def corrank(X):
        import itertools
        df = pd.DataFrame([[(i,j),X.corr().loc[i,j]] for i,j in list(itertools.combinations(X.corr(), 2))],columns=['pairs','corr'])
        print(df.sort_values(by='corr',ascending=False))

print('Correlation among labels')
print(corrank(df))

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

colors_list = ["brownish green", "pine green", "ugly purple",
               "blood", "deep blue", "brown", "azure"]

palette= sns.xkcd_palette(colors_list)

x=train.iloc[:,2:].sum()

plt.figure(figsize=(9,6))
ax= sns.barplot(x.index, x.values,palette=palette)
plt.title("Class")
plt.ylabel('Occurrences', fontsize=12)
plt.xlabel('Type ')
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 10, label,
            ha='center', va='bottom')
print('Number of occurrences vs Number of labels')
plt.show()

rate_punctuation = 0.7
rate_capital = 0.7
rowsums=train.iloc[:,2:].sum(axis=1)
train['clean']=(rowsums==0)
train['clean'].sum()
def odd_comment(comment):
    punctuation_count=0
    capital_letter_count=0
    total_letter_count=0
    for token in comment:
        if token in list(string.punctuation):
            punctuation_count+=1
        capital_letter_count+=sum(1 for c in token if c.isupper())
        total_letter_count+=len(token)
    return((punctuation_count/len(comment))>=rate_punctuation or
           (capital_letter_count/total_letter_count)>rate_capital)

odd=train[COMMENT].apply(odd_comment)
odd_ones = odd[odd==True]
odd_comments = train.loc[list(odd_ones.index)]
odd_comments[odd_comments.clean == False].count()/len(odd_comments)
toxic=train[train.toxic==1]['comment_text'].values
severe_toxic=train[train.severe_toxic==1]['comment_text'].values
obscene=train[train.obscene==1]['comment_text'].values
threat=train[train.threat==1]['comment_text'].values
insult=train[train.insult==1]['comment_text'].values
identity_hate=train[train.identity_hate==1]['comment_text'].values

mask=np.array(Image.open('twitter_mask.png'))
mask=mask[:,:,1]
wc = WordCloud(background_color="black", max_words=500,mask=mask
             , stopwords=stopwords, max_font_size= 60)
wc.generate(" ".join(toxic))
plt.title("Twitter Wordlcloud Toxic Comments", fontsize=30)
# plt.imshow(wc.recolor( colormap= 'Pastel1_r' , random_state=17), alpha=0.98)
plt.imshow(wc.recolor( colormap= 'Set1' , random_state=1), alpha=0.98)
plt.axis('off')
plt.savefig('twitter_wc.png')

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
re_tok = re.compile('([' + string.punctuation + '“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])

temp_df=train.iloc[:,2:-1]

corr=temp_df.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot=True, cmap="YlGnBu")
plt.savefig('corr_heatmap_class_labels.png')
