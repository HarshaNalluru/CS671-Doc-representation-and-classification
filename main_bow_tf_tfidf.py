import pandas as pd 
from pandas import DataFrame, read_csv
import os
import csv 
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

def accuracy(Y_test, predicted):
	j = 0
	correct = 0
	for i in Y_test:
		if i == predicted[j]:
			# print("True")
			correct += 1
		j += 1
	print("Accuracy = ", 1.0*correct/len(predicted))


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
	if i > 2000:
		break

for filename in os.listdir(inpath+"neg"):
	data = open(inpath+"neg/"+filename, 'r').read()
	index.append(i)
	text.append(data)
	rating.append("0")
	i = i + 1
	if i > 4000:
		break



print("Loaded data")
Dataset = list(zip(index,text,rating))

np.random.shuffle(Dataset)

# index = {0, 1, 2 , ..................}
# labels = {0, 1}
df = pd.DataFrame(data = Dataset, columns=['index', 'document', 'label'])
df.to_csv(outpath+name, index=False, header=True)


data = pd.read_csv(name,header=0, encoding = 'ISO-8859-1')
X = data['document']
Y = data['label']


X_train, Y_train = X[:int(.8*Y.shape[0])], Y[:int(.8*Y.shape[0])]
X_test, Y_test = X[int(.8*Y.shape[0]):], Y[int(.8*Y.shape[0]):]

################################################################################################

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
clf = MultinomialNB().fit(X_train_counts, Y_train)

X_test_counts = count_vect.transform(X_test)
predicted = clf.predict(X_test_counts)
accuracy(Y_test, predicted)

################################################################################################

normalized_tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = normalized_tf_transformer.transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tf, Y_train)

X_test_counts = count_vect.transform(X_test)
X_new_tfidf = normalized_tf_transformer.transform(X_test_counts)
predicted = clf.predict(X_new_tfidf)
accuracy(Y_test, predicted)

################################################################################################

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, Y_train)

X_test_counts = count_vect.transform(X_test)
X_new_tfidf = tfidf_transformer.transform(X_test_counts)
predicted = clf.predict(X_new_tfidf)
accuracy(Y_test, predicted)

# X_new_tfidf[0] = (X_new_tfidf[0] + X_new_tfidf[1])/2
# predicted = clf.predict(X_new_tfidf[0])

# print(predicted)
# j = 0
# for i in Y_test:
# 	if j > 1:
# 		break
# 	print(i)
# 	j += 1


################################################################################################

# print(X_train_counts.shape)
# for i in range(0,10):
# 	print (X_train[i])
# for i in range(0,10):
# 	print (X_train_counts[i])
# print(predicted)
# print(Y_test)