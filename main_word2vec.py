import numpy as np
import pandas as pd

pos_train_data = pd.read_csv('train_pos.tsv',sep = '\t')
neg_train_data = pd.read_csv('train_neg.tsv',sep = '\t')
pos_test_data = pd.read_csv('test_pos.tsv',sep = '\t')
neg_test_data = pd.read_csv('test_neg.tsv',sep = '\t')

os_train_data = pos_train_data[['Text','Sentiment']]
neg_train_data = neg_train_data[['Text','Sentiment']]
pos_test_data = pos_test_data[['Text','Sentiment']]
neg_test_data = neg_test_data[['Text','Sentiment']]

data_train = pd.concat([pos_train_data,neg_train_data],ignore_index = True)
data_train = data_train.sample(frac=1).reset_index(drop=True)
data_train.head()

len(data_train)

data_test = pd.concat([pos_test_data,neg_test_data],ignore_index = True)
data_test = data_test.sample(frac=1).reset_index(drop=True)
data_test.head()

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

stop_words = set(stopwords.words('english'))
table = str.maketrans('', '', punctuation)

def textclean(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if not word in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


reviews = []

for index,row in data_train.iterrows():
    text = (row['Text'].lower())    
    reviews.append(textclean(text))
reviews[0]


import gensim
from gensim.models import Word2Vec

n_dim = 100

w2v_model = Word2Vec(reviews,min_count=5,size=n_dim)

w2v_model.wv['nice']

import itertools
linked_reviews = list(itertools.chain.from_iterable(reviews))

vocab_freq = dict()

linked_reviews[1]


for word in linked_reviews:
    if word not in vocab_freq:
        vocab_freq[word] = 1
    else:
        vocab_freq[word] += 1



vocab_freq

import operator

sorted_vocab_freq = list(reversed(sorted(vocab_freq.items(), key=operator.itemgetter(1))))

len(sorted_vocab_freq)



review_lengths = pd.DataFrame([len(review) for review in reviews])
review_lengths.columns = ['Len']

review_lengths


#Removal of outliers using Tukey's Method
first_q = review_lengths.Len.quantile([0.25])[0.25]
third_q = review_lengths.Len.quantile([0.75])[0.75]

upper_threshold = third_q + 1.5*(third_q-first_q)
lower_threshold = first_q - 1.5*(third_q-first_q)

upper_threshold,lower_threshold


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform(reviews)
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))


tfidf['try']


def create_word_vector(l,size):
    vector = np.zeros(size).reshape((1,size))
    count = 0.
    for word in l:
        try:
            vector += w2v_model[word].reshape((1, size)) * tfidf[word]
            count+=1
        except KeyError:
            continue
            
    if count!=0:
        vector /= count
    return vector


X_train = []
y_train = []

for i in range(len(data_train)):
    converted_review = create_word_vector(reviews[i],n_dim)
    X_train.append(converted_review)
    y_train.append(data_train['Sentiment'][i])


from sklearn.preprocessing import scale

X_train = np.concatenate(X_train)
X_train = scale(X_train)
y_train = np.array(y_train)


X_train.shape

data_test = pd.concat([pos_test_data,neg_test_data],ignore_index = True)
data_test = data_test.sample(frac=0.3).reset_index(drop=True)

validation_reviews = []

for index,row in data_test.iterrows():
    text = (row['Text'].lower())
    validation_reviews.append(textclean(text))
    
X_val = []
y_val = []

for i in range(len(data_test)):
    converted_review = create_word_vector(validation_reviews[i],n_dim)
    X_val.append(converted_review)
    y_val.append(data_test['Sentiment'][i])
        
X_val = np.concatenate(X_val)
X_val = scale(X_val)
y_val = np.array(y_val)


from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation

model = Sequential()

model.add(Dense(64,activation = 'relu',input_shape=X_train[0].shape))
model.add(Dropout(0.2))
model.add(Dense(32,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation = 'sigmoid'))


model.summary()



model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train, y_train,validation_data = (X_val,y_val), epochs=15, batch_size=32, verbose=2)