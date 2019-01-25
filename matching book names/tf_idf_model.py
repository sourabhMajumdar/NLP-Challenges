import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy as np 
import re
from tqdm import tqdm

class TF_IDF_MODEL(object):
	"""docstring for TF_IDF_MODEL"""
	def __init__(self):
		super(TF_IDF_MODEL, self).__init__()
		nltk.download('punkt')
		self.stemmer = nltk.stem.porter.PorterStemmer()
		self.remove_punctuation_map = dict((ord(char),None) for char in string.punctutation)

	def stem_tokens(self,tokens):
		return [self.stemmer.stem(item) for item in tokens]

	def normalize(self,text) :
		return self.stem_tokens(nltk.word_tokenize(text.lower().translate(self.remove_punctuation_map)))

	def cosine_sim(self,text1,text2) :
		tfidf = vectorizer.fit_transform([text1,text2])
		return ((tfidf*tfidf.T).A)[0,1]

	def  predict_index(self,candidate_list,context) :
		candidate_scores = list()
		for i in range(len(candidate_list)) :
			score = cosine_sim(candidate_list[i],context)
			candidate_scores.append(score)

		candidate_array = np.array(candidate_scores)
		index_tuple = np.unravel_index(candidate_array.argmax(),candidate_array.shape)
		index = index_tuple[0] + 1
		return index

file_handle = open('train.txt')
lines = file_handle.readlines()
file_handle.close()

model = TF_IDF_MODEL()

set_a = list()
set_b = list()

concerned_set = set_a

for line in lines :
	if '**' in line :
		concerned_set = set_b
		continue
	else:
		concerned_set.append(line.lower().strip())

for ele in set_a :
	index = model.predict_index(set_b,ele)
	print(index)