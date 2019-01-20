import tensorflow as tf 
import re
import os
import random
import numpy as np 
from tqdm import tqdm 
from sklearn import metrics

class VerbPredictor(object):
	"""docstring for VerbPredictor"""
	def __init__(self,file_name,embedding_size=128,learning_rate=0.001,name='verb_predictor',model_dir='models/'):
		super(VerbPredictor, self).__init__()
		self.stop_words = ['a','an','the','(',')','-','.','st','nd','&',"'",':','...',') (','/','/...','))','(...',',','_']
		self.file_name = file_name
		self.embedding_size = embedding_size
		self.learning_rate = learning_rate
		self.name = name
		self.model_dir = model_dir

		if not os.path.exists(self.model_dir) :
			os.makedirs(self.model_dir)

		self.session = tf.Session()
		self.pre_process(file_name=self.file_name)
		self.build_graph()
		self.saver = tf.train.Saver(max_to_keep=50)
		self.summary_writer = tf.summary.FileWriter(self.model_dir,self.cross_entropy_sum.graph)

	def tokenize(self,line) :
		line = line.lower()
		result = [x.strip() for x in re.split('(\W+)?',line) if x.strip() and x.strip() not in self.stop_words]
		return result

	def pre_process(self,file_name='corpus.txt') :
		f_handle = open(file_name,encoding='utf-8')
		self.lines = f_handle.readlines()
		f_handle.close()

		self.tokenized_story = list()
		self.vocab_list = set()
		self.vocab_list.add('----')
		for line in self.lines :
			tokenized_line = self.tokenize(line)
			for word in tokenized_line :
				self.vocab_list.add(word)
			self.tokenized_story.extend(tokenized_line)

		self.word_idx = dict()
		self.vocab_size = len(self.vocab_list)
		for i, word in enumerate(self.vocab_list) :
			self.word_idx[word] = i

		self.candidates = ['am','are','were','was','is','been','being','be']
		self.candidates.sort()
		self.candidate_size = len(self.candidates)
		self.candidate_dic = dict()
		self.idx_candidate = dict()

		for i, c in enumerate(self.candidates) :
			self.candidate_dic[c] = i
			self.idx_candidate[i] = c

		self.train_set = list()
		self.max_sentence_size = 0

		for start_index in range(len(self.tokenized_story) - 9) :
			end_index = min(len(self.tokenized_story),start_index + 9)
			inspect_line = self.tokenized_story[start_index:end_index]
			for candidate in self.candidates :
				if candidate in inspect_line :
					index = inspect_line.index(candidate)
					initial_seq = inspect_line[0:index]
					final_seq = inspect_line[index + 1:len(inspect_line)]

					complete_seq = initial_seq + ['----'] + final_seq
					self.max_sentence_size = max(self.max_sentence_size,len(complete_seq))
					self.train_set.append([complete_seq,candidate])

		self.vec_training_set = list()

		for train_example in self.train_set :
			complete_seq = train_example[0]
			answer = train_example[1]

			lq_i = max(0,self.max_sentence_size - len(complete_seq))

			i_q = [self.word_idx[w] if w in self.word_idx else 0 for w in complete_seq] + lq_i*[0]

			a = self.candidate_dic[answer]

			self.vec_training_set.append([i_q,a])

		self.train_i = list()
		self.train_a = list()

		for vec_example in self.vec_training_set :
			vec_initial_seq = vec_example[0]
			vec_a = vec_example[1]

			self.train_i.append(np.array(vec_initial_seq))
			self.train_a.append(np.array(vec_a))

	def pre_process_test(self,file_name='corpus.txt') :
		f_handle = open(file_name,encoding='utf-8')
		self.test_lines = f_handle.readlines()
		f_handle.close()

		self.tokenized_test_story = list()
		for line in self.lines :
			tokenized_line = self.tokenize(line)
			self.tokenized_test_story.extend(tokenized_line)

		self.test_set = list()
		self.max_sentence_size = 0

		for start_index in range(len(self.tokenized_test_story) - 9) :
			end_index = min(len(self.tokenized_test_story),start_index + 9)
			inspect_line = self.tokenized_test_story[start_index:end_index]
			if '----' in inspect_line[4:] :
				index = inspect_line.index('----')
				start_index += index
				self.test_set.append([inspect_line])

		self.vec_test_set = list()

		for test_example in self.test_set :
			complete_seq = test_example[0]

			lq = max(0,self.max_sentence_size - len(complete_seq))

			i_q = [self.word_idx[w] if w in self.word_idx else 0 for w in complete_seq] + lq*[0]

			self.vec_test_set.append([i_q])

		self.test_i = list()

		for vec_example in self.vec_test_set :
			vec_initial_seq = vec_example[0]

			self.test_i.append(np.array(vec_initial_seq))

	def build_graph(self) :
		self.x_i = tf.placeholder(tf.int32,[None,self.max_sentence_size],name="initial_seq")
		self.y = tf.placeholder(tf.int32,[None],name="answers")

		self.A = tf.get_variable("word_embeddings",[self.vocab_size,self.embedding_size])

		self.x_i_emb = tf.nn.embedding_lookup(self.A,self.x_i)

		self.i_rnn_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_size)
		self.f_rnn_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_size)

		self.i_outputs, self.i_states = tf.nn.bidirectional_dynamic_rnn(self.i_rnn_cell,self.f_rnn_cell,self.x_i_emb,dtype=tf.float32)
		self.i_outputs_fw, self.i_outputs_bw = self.i_outputs

		self.combined_meaning = tf.concat([self.i_outputs_fw[:,-1,:],self.i_outputs_bw[:,-1,:]],axis=1)

		self.logit_outputs = tf.layers.dense(self.combined_meaning,self.candidate_size,activation=tf.nn.softmax)

		self.predict_op = tf.argmax(self.logit_outputs,1,name='prdict_op')

		self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logit_outputs,labels=self.y,name="cross_entropy_loss")

		self.cross_entropy_sum = tf.reduce_sum(self.cross_entropy)

		self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.cross_entropy_sum)

	def train(self,epochs=200) :
		self.init_op = tf.global_variables_initializer()

		self.session.run(self.init_op)

		batches = zip(range(0,len(self.train_i)-32,32),range(32,len(self.train_i),32))
		batches = [(start,end) for start, end in batches]

		for epoch in range(epochs) :
			np.random.shuffle(batches)
			train_batches = batches[:-1]
			val_batches = batches[-1:]
			avg_epoch_loss = 0
			epoch_loss = 0
			best_val_acc = 0
			for start, end in tqdm(train_batches) :
				train_i_batch = self.train_i[start:end]
				train_ans = self.train_a[start:end]

				epoch_loss = self.batch_fit(train_i_batch,train_ans)
				avg_epoch_loss += epoch_loss
			avg_epoch_loss /= len(train_batches)

			val_preds = list()
			actual_preds = list()


			for start, end in tqdm(val_batches) :
				val_batch = self.train_i[start:end]
				val_ans = self.train_a[start:end]

				preds = self.batch_predict(val_batch)
				val_preds.extend(preds)
				actual_preds.extend(val_ans)

			val_acc = metrics.accuracy_score(np.array(val_preds),np.array(actual_preds))
			print('\n\n\t VALIDATION ACCURACY \n\n\t {}'.format(val_acc))

			if val_acc > best_val_acc :
				best_val_acc = val_acc
				self.saver.save(self.session,os.path.join(self.model_dir,'model.ckpt'),global_step=epoch)

		train_preds =  self.batch_predict(self.train_i)
		train_accuracy = metrics.accuracy_score(np.array(train_preds),self.train_a)

		print('\n\n\t TRAIN ACCURACY\n\n\t {}'.format(train_accuracy))


	def load_saved_model(self) :
		ckpt = tf.train.get_checkpoint_state(self.model_dir)
		if ckpt and ckpt.model_checkpoint_path :
			self.saver.restore(self.session,ckpt.model_checkpoint_path)
		else :
			print('No Checkpoint found')
	def test(self,file_name='corpus.txt') :
		self.pre_process_test(file_name='corpus.txt')
		self.load_saved_model()
		test_preds = self.batch_predict(self.test_i)

		for test_pred in test_preds :
			print(self.idx_candidate[test_pred])


	def batch_fit(self,train_batch,train_ans) :
		feed_dict = {
			self.x_i : train_batch,
			self.y : train_ans
		}
		train_metric, loss = self.session.run([self.train_op,self.cross_entropy_sum],feed_dict=feed_dict)

		return loss

	def batch_predict(self,train_batch) :
		feed_dict = {
			self.x_i : train_batch
		}
		preds = self.session.run(self.predict_op,feed_dict=feed_dict)
		return preds



if __name__ == '__main__' :
	verb_predictor = VerbPredictor(file_name='corpus.txt',embedding_size=128)
	verb_predictor.train(1)
	verb_predictor.test(file_name='corpus.txt')