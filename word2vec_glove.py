import numpy as np

# with open("glove.6B.50d.txt", "rb") as lines:
#	 w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
#			for line in lines}


import gensim
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from collections import Counter, defaultdict
from sklearn.cross_validation import StratifiedShuffleSplit
# let X be a list of tokenized texts (i.e. list of lists of tokens)

# # tokenize it using default tokenizer
# from estnltk import Tokenizer
# tokenizer = Tokenizer()
# document = tokenizer.tokenize(text)

# # tokenized results
# print (document.word_texts)
# print (document.sentence_texts)
# print (document.paragraph_texts)
# print (document.text)


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
	if i > 20000:
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
	if i > 40000:
		break


# Dataset = list(zip(index,text,rating))

# np.random.shuffle(Dataset)


# print("Loaded data")


y = rating[:]
X, y = np.array(X), np.array(y)


# model = gensim.models.Word2Vec(X, min_count=1, size = 100)
# w2v = list(zip(model.wv.index2word, model.wv.syn0))

model = gensim.models.Word2Vec(X, size=100, window=5, min_count=5, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

class MeanEmbeddingVectorizer(object):
	 def __init__(self, word2vec):
		 self.word2vec = word2vec
		 # if a text is empty we should return a vector of zeros
		 # with the same dimensionality as all the other vectors
		 self.dim = len(word2vec)

	 def fit(self, X, y):
		 return self

	 def transform(self, X):
		 return np.array([
			 np.mean([self.word2vec[w] for w in words if w in self.word2vec]
					 or [np.zeros(self.dim)], axis=0)
			 for words in X
		 ])


class TfidfEmbeddingVectorizer(object):
	 def __init__(self, word2vec):
		 self.word2vec = word2vec
		 self.word2weight = None
		 self.dim = len(word2vec)

	 def fit(self, X, y):
		 tfidf = TfidfVectorizer(analyzer=lambda x: x)
		 tfidf.fit(X)
		 # if a word was never seen - it must be at least as infrequent
		 # as any of the known words - so the default idf is the max of 
		 # known idf's
		 max_idf = max(tfidf.idf_)
		 self.word2weight = defaultdict(
			 lambda: max_idf,
			 [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

		 return self

	 def transform(self, X):
		 return np.array([
				 np.mean([self.word2vec[w] * self.word2weight[w]
						  for w in words if w in self.word2vec] or
						 [np.zeros(self.dim)], axis=0)
				 for words in X
			 ])


from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

etree_w2v = Pipeline([
	 ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
	 ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_tfidf = Pipeline([
	 ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
	 ("extra trees", ExtraTreesClassifier(n_estimators=200))])

for train, test in StratifiedShuffleSplit(y, n_iter=2, test_size=0.2):
	X_train, X_test = X[train], X[test]
	y_train, y_test = y[train], y[test]
	print(accuracy_score(etree_w2v_tfidf.fit(X_train, y_train).predict(X_test), y_test))


# X = [['Berlin', 'London'],
# 	  ['cow', 'cat'],
# 	  ['pink', 'yellow']]
# y = ['capitals', 'animals', 'colors']
# etree_glove_big.fit(X, y)

# # never before seen words!!!
# test_X = [['dog', 'cat'], ['red'], ['Madrid']]

# print etree_glove_big.predict(test_X)



# # this works with python3
# import numpy as np
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import ExtraTreesClassifier


# class MeanEmbeddingVectorizer(object):
#	 def __init__(self, word2vec):
#		 self.word2vec = word2vec
#		 # this line is different from python2 version - no more itervalues
#		 self.dim = len(list(word2vec.values())[0])

#	 def fit(self, X, y):
#		 return self

#	 def transform(self, X):
#		 return np.array([
#			 np.mean([self.word2vec[w] for w in words if w in self.word2vec]
#					 or [np.zeros(self.dim)], axis=0)
#			 for words in X
#		 ])

# w2v = {
#	 'Berlin': [1, 1],
#	 'London': [1.01, 1.01],
#	 'Madrid': [1.02, 1.02],
#	 'cow':	[-1, -1],
#	 'cat':	[-1.01, -1.01],
#	 'dog':	[-1.02, -1.02],
#	 'pink':   [1, -1],
#	 'yellow': [1.01, -1.01],
#	 'red':	[1.02, -1.02]
# }


# model = Pipeline([
#	 ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
#	 ("extra trees", ExtraTreesClassifier(n_estimators=200))])

# X = [['Berlin', 'London'],
#	  ['cow', 'cat'],
#	  ['pink', 'yellow']]
# y = ['capitals', 'animals', 'colors']

# model.fit(X, y)

# # never before seen words!!!
# test_X = [['pink', 'red'], ['red'], ['Madrid']]

# print(model.predict(test_X))
# # prints ['animals' 'colors' 'capitals']