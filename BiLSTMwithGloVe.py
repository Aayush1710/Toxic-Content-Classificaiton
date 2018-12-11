import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing import text, sequence
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

#Importing data and Glove file (Select your path to run)
path = os.getcwd() + "/Data/"
data = pd.read_csv(path + "train.csv")
Glove_Embeddings = f'input/glove.6B.50d.txt'

# Maximum unique feature selection
max_features = 20000

# Maximum length of words in comments to consider
max_len = 150

# Single Word Vector Size - needs to be corresponding to the glove embedding file
embed_size = 50

# Selecting comments and id
X = data.iloc[:,:2]
# Selecting corresponding comment labels
y = data.iloc[:,2:]

# Testing - Training Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


# Data-Preprocessing - Filling empty entries
sentences_train = X_train["comment_text"].fillna("abcder111").values
sentences_test = X_test["comment_text"].fillna("abcder111").values


# Word Tokenization features ( Commment to word indices )
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(sentences_train))
tokenized_train = tokenizer.texts_to_sequences(sentences_train)
tokenized_test = tokenizer.texts_to_sequences(sentences_test)
X_tokenized_train = sequence.pad_sequences(tokenized_train, maxlen=max_len)
X_tokenized_test = sequence.pad_sequences(tokenized_test, maxlen=max_len)



# Glove vectors to dictionary indexing and generating embedding matrix
def word_vector_coef(word,*arr):
    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(word_vector_coef(*i.strip().split()) for i in open(Glove_Embeddings))

all_embeddings = np.stack(embeddings_index.values())
embeddings_mean,embeddings_std = all_embeddings.mean(), all_embeddings.std()

# Random Initialization for unknown words
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(embeddings_mean, embeddings_std, (nb_words, embed_size))

for word, i in word_index.items():
    if i >= max_features:
        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector



# Bidiectional LSTM Model
def create_model():
    inp = Input(shape=(max_len, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(LSTM(60, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model


model = create_model()
batch_size = 32
epochs = 4


# Saving best model weights
weight_file="Weights/LSTM_best_weights.hdf5"

#checkpointing best model performance
checkpoint = ModelCheckpoint(weight_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

callback = [checkpoint, early]

history = model.fit(X_tokenized_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks= callback)
model.load_weights(weight_file)

#Calculating predictions
y_score = model.predict(X_tokenized_test)


#Model Testing-Training Accuracy Plot

import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training_acc', 'validation-acc'], loc='upper left')
plt.show()


#RoC Scores Plot

from sklearn.metrics import roc_curve, auc
from itertools import cycle
from scipy import interp

fpr = dict()
tpr = dict()
roc_auc = dict()
Y_test = y_test.values
n_classes = y_train.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i],y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute macro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(),y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Scores')
plt.legend(loc="lower right")
plt.show()