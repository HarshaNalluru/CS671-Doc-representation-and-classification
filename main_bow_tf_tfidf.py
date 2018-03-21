import pandas as pd 
from pandas import DataFrame, read_csv
import os
import csv 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Embedding
from keras.utils import np_utils, to_categorical
from keras.layers import LSTM



def accuracy(y_test, predicted):
	j = 0
	correct = 0
	for i in y_test:
		if i == predicted[j]:
			# print("True")
			correct += 1
		j += 1
	print("Accuracy = " + str(1.0*correct/len(predicted)))


index = []
text = []
rating = []
inpath = "aclImdb/train/"
outpath = "./"
name = "imdb_tr.csv"
i =  0 

print("Hello")
for filename in os.listdir(inpath+"pos"):
	data = open(inpath+"pos/"+filename, 'r').read()
	index.append(i)
	text.append(data)
	rating.append("1")
	i = i + 1
	# if i > 2000:
	# 	break

for filename in os.listdir(inpath+"neg"):
	data = open(inpath+"neg/"+filename, 'r').read()
	index.append(i)
	text.append(data)
	rating.append("0")
	i = i + 1
	# if i > 4000:
	# 	break



print("Loaded data")
Dataset = list(zip(index,text,rating))

np.random.shuffle(Dataset)

# index = {0, 1, 2 , ..................}
# labels = {0, 1}
df = pd.DataFrame(data = Dataset, columns=['index', 'document', 'label'])
df.to_csv(outpath+name, index=False, header=True)


data = pd.read_csv(name,header=0)
X = data['document']
Y = data['label']


X_train, y_train = X[:int(.8*Y.shape[0])], Y[:int(.8*Y.shape[0])]
X_test, y_test = X[int(.8*Y.shape[0]):], Y[int(.8*Y.shape[0]):]

################################################################################################
######################################    BOW    ###############################################
################################################################################################

## MultinomialNB  ##############################################################################

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
clf = MultinomialNB().fit(X_train_counts, y_train)

X_test_counts = count_vect.transform(X_test)
predicted = clf.predict(X_test_counts)
print(" BOW MultinomialNB")
accuracy(y_test, predicted)

## LogisticRegression ##########################################################################

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
clf = LogisticRegression().fit(X_train_counts, y_train)

X_test_counts = count_vect.transform(X_test)
predicted = clf.predict(X_test_counts)
print(" BOW LogisticRegression")
accuracy(y_test, predicted)

## Support Vector Machines #####################################################################

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
clf = svm.SVC().fit(X_train_counts, y_train)

X_test_counts = count_vect.transform(X_test)
predicted = clf.predict(X_test_counts)
print(" BOW SVM")
accuracy(y_test, predicted)


################################################################################################
######################################    TF   #################################################
################################################################################################

## MultinomialNB  ##############################################################################

normalized_tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = normalized_tf_transformer.transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tf, y_train)

X_test_counts = count_vect.transform(X_test)
X_new_tfidf = normalized_tf_transformer.transform(X_test_counts)
predicted = clf.predict(X_new_tfidf)
print(" TF MultinomialNB")
accuracy(y_test, predicted)

## LogisticRegression ##########################################################################

normalized_tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = normalized_tf_transformer.transform(X_train_counts)
clf = LogisticRegression().fit(X_train_tf, y_train)

X_test_counts = count_vect.transform(X_test)
X_new_tfidf = normalized_tf_transformer.transform(X_test_counts)
predicted = clf.predict(X_new_tfidf)
print(" TF LogisticRegression")
accuracy(y_test, predicted)

## Support Vector Machines #####################################################################

normalized_tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = normalized_tf_transformer.transform(X_train_counts)
clf = svm.SVC().fit(X_train_tf, y_train)

X_test_counts = count_vect.transform(X_test)
X_new_tfidf = normalized_tf_transformer.transform(X_test_counts)
predicted = clf.predict(X_new_tfidf)
print(" TF SVM")
accuracy(y_test, predicted)



################################################################################################
######################################   TF-IDF  ###############################################
################################################################################################

## MultinomialNB  ##############################################################################

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

X_test_counts = count_vect.transform(X_test)
X_new_tfidf = tfidf_transformer.transform(X_test_counts)
predicted = clf.predict(X_new_tfidf)
print(" TF-IDF MultinomialNB")
accuracy(y_test, predicted)

## LogisticRegression ##########################################################################

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = LogisticRegression().fit(X_train_tfidf, y_train)

X_test_counts = count_vect.transform(X_test)
X_new_tfidf = tfidf_transformer.transform(X_test_counts)
predicted = clf.predict(X_new_tfidf)
print(" TF-IDF LogisticRegression")
accuracy(y_test, predicted)

## Support Vector Machines #####################################################################

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = svm.SVC().fit(X_train_tfidf, y_train)

X_test_counts = count_vect.transform(X_test)
X_new_tfidf = tfidf_transformer.transform(X_test_counts)
predicted = clf.predict(X_new_tfidf)
print(" TF-IDF SVM")
accuracy(y_test, predicted)


################################################################################################
######################################   END  ##################################################
################################################################################################



################################################################################################

# print(type(np.array(X_train)))
# print(y_test)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
model = Sequential()
model.add(Dense(100, activation="relu", kernel_initializer="uniform", input_dim=100))
model.add(Dense(50, activation="relu", kernel_initializer="uniform"))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
model.fit(np.array(X_train), np.array(y_train), epochs=50, batch_size=128)

print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(np.array(X_test), np.array(y_test),
	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))


################################################################################################

model = Sequential()
model.add(Embedding(max_features, output_dim=2))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(np.array(X_train), np.array(y_train), epochs=50, batch_size=128)

print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(np.array(X_test), np.array(y_test),
	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))
