import numpy as np
import pandas as pd
import re,os
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils


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
# pattern = re.compile(r"\b\w\w*\b")
pattern = re.compile(r'([A-Z][^\.!?]*[\.!?])')
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
	if i > 200:
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
	if i > 400:
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
	for j in range(len(Z[i][0])):
		d2v_reviews.append(TaggedDocument(words=Z[i][0][j], tags=['REV_'+str(i)+ '_' + str(j)]))

# print(d2v_reviews[25])

vec_size = 100
d2v_model = Doc2Vec(d2v_reviews,size=vec_size)

# print(d2v_model.docvecs['REV_3_2'])

# print(len(d2v_model.docvecs))


X_train = []
y_train = []

for i in range(len(data_train)):
	curr_vec = np.zeros((vec_size,), dtype=np.float)
	for j in range(len(Z[i][0])):
		curr_vec += d2v_model.docvecs['REV_'+str(i)+ '_' + str(j)]

	X_train.append(curr_vec)
	y_train.append(Z[i][1])
	# if i == 5:
	# 	print(curr_vec)

clf = SVC()
clf = clf.fit(X_train,y_train)

X_test = []
y_test = []
for i in range(len(data_test)):
	curr_vec = np.zeros((vec_size,), dtype=np.float)
	for j in range(len(Z[i + int(0.8*len(Z))][0])):
		curr_vec += d2v_model.docvecs['REV_'+str(i + int(0.8*len(Z)))+ '_' + str(j)]
	X_test.append(curr_vec)
	y_test.append(Z[i + int(0.8*len(Z))][1])
	# if i == 5:
	# 	print(X_test[i])

predicted = clf.predict(X_test)
accuracy(y_test, predicted)




# clf = MultinomialNB().fit(X_train,y_train)

# predicted = clf.predict(X_test)
# accuracy(y_test, predicted)





clf = GaussianNB().fit(X_train,y_train)

predicted = clf.predict(X_test)
accuracy(y_test, predicted)




clf = BernoulliNB().fit(X_train,y_train)

predicted = clf.predict(X_test)
accuracy(y_test, predicted)


################################################################################################


model = Sequential()
model.add(Dense(100, input_dim=100, init="uniform",
	activation="relu"))
model.add(Dense(50, init="uniform", activation="relu"))
model.add(Dense(2))
model.add(Activation("softmax"))

sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
model.fit(X_train, y_train, nb_epoch=50, batch_size=128)

predicted = model.predict(X_test, y_test)
accuracy(y_test, predicted)


print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(X_test, y_test,
	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))