import numpy as np
import pandas as pd
import re,os
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Embedding
from keras.utils import np_utils, to_categorical
from keras.layers import LSTM

def accuracy(Y_test, predicted):
	j = 0
	correct = 0
	for i in Y_test:
		if i == predicted[j]:
			# print("True")
			correct += 1
		j += 1
	print("Accuracy = "+ str(1.0*correct/len(predicted)))

inpath = "aclImdb/train/"

index = []
text = []
rating = []

print("Hello")
pattern = re.compile(r"\b\w\w*\b")
i = 0
X = []
for filename in os.listdir(inpath+"pos"):
	data = open(inpath+"pos/"+filename, 'r').read()
	index.append(i)
	text.append(data)
	rating.append("1")

	matches = pattern.findall(data)
	# for match in matches:
	# 	print(match)
	# print(matches)
	X.append(matches)
	i = i + 1
	if i > 50:
		break

for filename in os.listdir(inpath+"neg"):
	data = open(inpath+"neg/"+filename, 'r').read()
	index.append(i)
	text.append(data)
	rating.append("0")

	matches = pattern.findall(data)
	# for match in matches:
	# 	print(match)
	X.append(matches)
	i = i + 1
	if i > 100:
		break

print("Loaded data")

y = rating[:]
X, y = np.array(X), np.array(y)

Z = zip(X,y)
np.random.shuffle(Z)

data_train = Z[:int(0.8*len(Z))]

data_test = Z[int(0.8*len(Z)):]


d2v_reviews = []
for i in range(len(Z)):
	d2v_reviews.append(TaggedDocument(words=Z[i][0], tags=['REV_'+str(i)]))

# print(d2v_reviews[25])

vec_size = 100
d2v_model = Doc2Vec(d2v_reviews,size=vec_size)

# print(d2v_model.docvecs['REV_3'])

# print(len(d2v_model.docvecs))


X_train = []
y_train = []

for i in range(len(data_train)):
	X_train.append(d2v_model.docvecs['REV_'+str(i)])
	y_train.append(Z[i][1])


X_test = []
y_test = []
for i in range(len(data_test)):
	X_test.append(d2v_model.docvecs['REV_'+str(i + int(0.8*len(Z)))])
	y_test.append(Z[i + int(0.8*len(Z))][1])



################################################################################################
######################################   Avg Doc2Vec  ########################################
################################################################################################


## BernoulliNB  ##############################################################################

clf = BernoulliNB().fit(X_train,y_train)

predicted = clf.predict(X_test)
print("Avg Doc2Vec - BernoulliNB")
accuracy(y_test, predicted)

## GaussianNB  ##############################################################################

clf = GaussianNB().fit(X_train,y_train)

predicted = clf.predict(X_test)
print("Avg Doc2Vec - GaussianNB")
accuracy(y_test, predicted)

## Support Vector Machines #####################################################################

clf = SVC()
clf = clf.fit(X_train,y_train)

predicted = clf.predict(X_test)
print("Avg Doc2Vec - SVM")
accuracy(y_test, predicted)

## LogisticRegression ##########################################################################

clf = LogisticRegression().fit(X_train,y_train)

predicted = clf.predict(X_test)
print(" Avg Doc2Vec - LogisticRegression")
accuracy(y_test, predicted)

## LSTM ########################################################################################

model = Sequential()
model.add(Embedding(input_dim = len(X_train[0]), output_dim=2, input_length=None ))
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
print(" Avg Doc2Vec LSTM")
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))



## Feed Norward Neural Network #################################################################


# print(type(np.array(X_train)))
# print(y_test)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
model = Sequential()
model.add(Dense(100, activation="relu", kernel_initializer="uniform", input_dim=len(X_train[0])))
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
print(" Avg Doc2Vec feed forward neural network")
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))


################################################################################################
######################################   END  ##################################################
################################################################################################
################################################################################################






