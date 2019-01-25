import tensorflow as tf 
import re 
import random
from collections import Counter


file_handle = open('corpus.txt',encoding='utf-8')
lines = file_handle.readlines()
file_handle.close()

paragraphs = list()

story = str()
for line in lines :
	line = line.lower().strip()
	if not line :
		if not story :
			continue
			story = str()
		else :
			paragraphs.append(story)
			story = str()
	else :
		story = ''.join([story,line])

count = 1
'''for paragraph in paragraphs :
	print('paragraph #{}'.format(count))
	count += 1
	print(paragraph)
	nhuj = input()'''

def tokenize(line) :
	line = line.lower()
	stop_words = ['a','an','the','(',')','-','.','st','nd','&',"'",':','...',') (','/','/...','))','(...',',','_'"'s"]
	result = [x.strip() for x in re.split('(\W+)?',line) if x.strip() and x.strip() not in stop_words]
	return result

tokenized_paragraphs = list()
word_set = set()
max_paragraph_size = 0
for paragraph in paragraphs :
	tokenized_paragraph = tokenize(paragraph)
	#print(tokenized_paragraph)
	#hud = input('paragraph #{}'.format(count))
	count += 1
	max_paragraph_size = max(max_paragraph_size,len(tokenized_paragraph))
	for tokens in tokenized_paragraph :
		word_set.add(tokens)
	tokenized_paragraphs.append(tokenized_paragraph)

word_set = sorted(word_set)
word_dic = dict()
idx_word = dict()
for i,word in enumerate(word_set) :
	word_dic[word] = i
	idx_word[i] = word

def vectorize(line,word_dic,max_paragraph_size) :

	lq = max(0,max_paragraph_size - len(line))
	numeric_line = [word_dic[w] if w in word_dic else 0 for w in line] + lq*[0]
	vec_line = np.array(numeric_line)
	return vec_line

masculine_words = ['he','him','his','himself']
feminine_words = ['she','her','herself','hers']

names = ['john','sherlock','mary','hugh','martha','charlie']

for name in names :
	masculine_vote = 0
	feminine_vote = 0
	masculine_word_count = 0
	feminine_word_count = 0
	for paragraph in tokenized_paragraphs :
		masculine_word_count = 0
		feminine_word_count = 0

		#print('name is {}'.format(name))
		#print(paragraph)

		if name in paragraph :
			word_list = Counter(paragraph)
			#print(word_list)
			#yu = input()

			for masculine_word in masculine_words :

				if masculine_word in word_list :
					masculine_word_count += word_list[masculine_word]

			for feminine_word in feminine_words :
				if feminine_word in word_list :
					feminine_word_count += word_list[feminine_word]

		if masculine_word_count > feminine_word_count :
			masculine_vote += 1
		elif feminine_word_count > masculine_word_count :
			feminine_vote += 1

	if masculine_vote > feminine_vote :
		print('male')
	elif feminine_vote > masculine_vote :
		print('female')
	else :
		print('could not determine')
	print('male votes : {}\tfemale votes : {}'.format(masculine_vote,feminine_vote))