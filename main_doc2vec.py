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
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

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
	# if i > 200:
	# 	break

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
	# if i > 400:
	# 	break

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

clf = SVC()
clf = clf.fit(X_train,y_train)

X_test = []
y_test = []
for i in range(len(data_test)):
	X_test.append(d2v_model.docvecs['REV_'+str(i + int(0.8*len(Z)))])
	y_test.append(Z[i + int(0.8*len(Z))][1])

predicted = clf.predict(X_test)
accuracy(y_test, predicted)




clf = GaussianNB().fit(X_train,y_train)

predicted = clf.predict(X_test)
accuracy(y_test, predicted)

clf = BernoulliNB().fit(X_train,y_train)

predicted = clf.predict(X_test)
accuracy(y_test, predicted)


################################################################################################


# model = Sequential()
# model.add(LSTM(4, input_shape=(1, look_back)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)


# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
# # invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
# # calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))

