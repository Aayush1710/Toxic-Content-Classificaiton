import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('Toxic_Comments_Classification/data/train.csv')
train = train.fillna(' ')
X = train.iloc[:,:2]
y = train.iloc[:,2:]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=.7)
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(sublinear_tf=True, analyzer='word', token_pattern=r'\w{1,}',
    stop_words='english', ngram_range=(1, 1), max_features=10000)

word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

train_features = train_word_features
test_features = test_word_features

scores = []
for class_name in class_names:
    classifier = LogisticRegression(C=0.2)
    train_target = train[class_name]

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, scoring='roc_auc', cv=3))
    scores.append(cv_score)
    print('CV score for class {1} is {0}'.format(cv_score, class_name))

    classifier.fit(train_features, train_target)

print('Total Cross Validation score is {0}'.format(np.mean(scores)))
