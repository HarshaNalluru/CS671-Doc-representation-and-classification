import numpy as np
import pandas as pd
import re,os
import itertools
import operator
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
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
		# print(j)
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
	#	 print(match)
	# print(matches)
	X.append(matches)
	i = i + 1
	# if i > 200:
	# 	break

for filename in os.listdir(inpath+"neg"):
	data = open(inpath+"neg/"+filename, 'r').read()
	index.append(i)
	text.append(data)
	rating.append("0")

	matches = pattern.findall(data)
	# for match in matches:
	#	 print(match)
	X.append(matches)
	i = i + 1
	# if i > 400:
	# 	break

print("Loaded data")


y = rating[:]
X, y = np.array(X), np.array(y)


Z = zip(X,y)
np.random.shuffle(Z)

reviews = X[:]


n_dim = 150

w2v_model = Word2Vec(reviews,min_count=3,size=n_dim)


linked_reviews = list(itertools.chain.from_iterable(reviews))

vocab_freq = dict()


for word in linked_reviews:
	if word not in vocab_freq:
		vocab_freq[word] = 1
	else:
		vocab_freq[word] += 1

# print(len(vocab_freq))

sorted_vocab_freq = list(reversed(sorted(vocab_freq.items(), key=operator.itemgetter(1))))


vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform(reviews)
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

def create_word_vector(l,size):
	vector = np.zeros(size).reshape((1,size))
	count = 0.
	for word in l:
		try:
			vector += w2v_model[word].reshape((1, size))
			count+=1
		except KeyError:
			continue

	if count!=0:
		vector /= count
	return vector


###############################################################################################
#####################################   Word2Vec - Mean  ########################################
###############################################################################################


X_train = []
y_train = []

data_train = Z[:int(0.8*len(reviews))]

data_test = Z[int(0.8*len(reviews)):]

for i in range(len(data_train)):
	converted_review = create_word_vector(Z[i][0],n_dim)
	# if i==1:
	# 	# print(converted_review[0][0])
	X_train.append(converted_review[0])
	y_train.append(Z[i][1])


# print(y_train)
X_test = []
y_test = []

for i in range(len(data_test)):
	converted_review = create_word_vector(Z[i+int(0.8*len(reviews))][0],n_dim)
	X_test.append(converted_review[0])
	y_test.append(Z[i+int(0.8*len(reviews))][1])

# print(y_test)
## BernoulliNB  ##############################################################################

clf = BernoulliNB().fit(X_train,y_train)

predicted = clf.predict(X_test)
print("Word2Vec - Mean BernoulliNB")
accuracy(y_test, predicted)

## GaussianNB  ##############################################################################

clf = GaussianNB().fit(X_train,y_train)

predicted = clf.predict(X_test)
print("Word2Vec - Mean GaussianNB")
accuracy(y_test, predicted)

## Support Vector Machines #####################################################################

clf = SVC().fit(X_train,y_train)

predicted = clf.predict(X_test)
print("Word2Vec - Mean SVM")
accuracy(y_test, predicted)

## LogisticRegression ##########################################################################

clf = LogisticRegression().fit(X_train, y_train)

predicted = clf.predict(X_test)
print("Word2Vec - Mean LogisticRegression")
accuracy(y_test, predicted)


## LSTM ##############################################################################################

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
print("Word2Vec - Mean LSTM")
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
print("Word2Vec - Mean feed forward neural network")
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))



###############################################################################################
#####################################   END  ##################################################
###############################################################################################

