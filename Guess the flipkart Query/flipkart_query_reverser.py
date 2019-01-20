import tensorflow as tf 
import re
import random
import numpy as np 
from tqdm import tqdm 
from sklearn import metrics 
import os

class QueryPredictor(object):
	"""docstring for QueryPredictor"""
	def __init__(self, file_name,embedding_size=128,learning_rate=0.001,name='query_predictor',model_dir='models/'):
		super(QueryPredictor, self).__init__()
		self.file_name = file_name
		self.embedding_size = embedding_size
		self.learning_rate = learning_rate
		self.name = name
		self.model_dir = model_dir
		if not os.path.exists(self.model_dir) :
			os.makedirs(self.model_dir)
		self.session = tf.Session()
		self.pre_process(file_name=self.file_name)
		self.build_variables()
		self.build_graph()
		self.saver = tf.train.Saver(max_to_keep=50)
		self.summary_writer = tf.summary.FileWriter(self.model_dir,self.cross_entropy_loss.graph)

	def tokenize(self,line) :

		line = line.lower()

		self.stop_words = ['a','an','the','(',')','-','.','st','nd','&',"'",':','...',') (','/','/...','))','(...',',']

		result = [x.strip() for x in re.split('(\W+)?',line) if x.strip() and x.strip() not in self.stop_words]

		return result

	def pre_process(self,file_name='training.txt') :
		f_handle = open(file_name)
		self.lines = f_handle.readlines()
		f_handle.close()

		self.candidates = set()
		for line in self.lines :
			sentences = line.split('\t')
			if len(sentences) > 1 :
				self.candidates.add(sentences[1].strip())

		self.candidate_dic = dict()
		self.candidates = sorted(self.candidates)
		for i,candidate in enumerate(self.candidates) :
			self.candidate_dic[candidate] = i

		training_set = list()
		self.max_sentence_size = 0
		self.word_idx = dict()
		words_set = set()

		for line in self.lines :
			sentences = line.split('\t')
			if len(sentences) > 1 :
				tokenized_line = self.tokenize(sentences[0])
				training_set.append([tokenized_line,self.candidate_dic[sentences[1].strip()]])
				self.max_sentence_size = max(self.max_sentence_size,len(tokenized_line))
				for word in tokenized_line :
					words_set.add(word)
		words_set = sorted(words_set)
		for i,word in enumerate(words_set) :
			self.word_idx[word] = i

		self.vocab_size = len(words_set)
		self.candidate_size = len(self.candidates)

		vec_training_set = list()
		for example in training_set :
			tokenized_line = example[0]
			candid_idx = example[1]
			lq = max(0,self.max_sentence_size-len(tokenized_line))
			q = [self.word_idx[w] if w in self.word_idx else 0 for w in tokenized_line] + [0]*lq
			vec_training_set.append([q,candid_idx])

		self.train_q = list()
		self.train_a = list()

		for vec_example in vec_training_set :
			vec_sentence = vec_example[0]
			vec_response = vec_example[1]
			self.train_q.append(np.array(vec_sentence))
			self.train_a.append(np.array(vec_response))

	def pre_process_test(self,file_name='testing.txt') :
		f_handle = open(file_name)
		self.test_lines = f_handle.readlines()
		f_handle.close()

		testing_set = list()

		for line in self.test_lines :
			sentences = line.split('\t')
			if len(sentences) > 1 :
				tokenized_line = self.tokenize(sentences[0])
				testing_set.append([tokenized_line,self.candidate_dic[sentences[1].strip()]])


		vec_testing_set = list()
		for example in testing_set :
			tokenized_line = example[0]
			candid_idx = example[1]
			lq = max(0,self.max_sentence_size-len(tokenized_line))
			q = [self.word_idx[w] if w in self.word_idx else 0 for w in tokenized_line] + lq*[0]
			vec_testing_set.append([q,candid_idx])

		self.test_q = list()
		self.test_a = list()

		for vec_example in vec_testing_set :
			vec_sentence = vec_example[0]
			vec_response = vec_example[1]
			self.test_q.append(np.array(vec_sentence))
			self.test_a.append(np.array(vec_response))

	def build_variables(self) :

		self.x = tf.placeholder(tf.int32,[None,self.max_sentence_size],name="input_seq")
		self.y = tf.placeholder(tf.int32,[None],name="answers")

		self.A = tf.get_variable("word_embeddings",[self.vocab_size,self.embedding_size])
		self.rnn_cell = tf.nn.rnn_cell.LSTMCell(self.embedding_size)

	def build_graph(self) :

		self.word_embeddings = tf.nn.embedding_lookup(self.A,self.x)
		self.outputs, self.states = tf.nn.dynamic_rnn(self.rnn_cell,self.word_embeddings,dtype=tf.float32)
		self.logit_outputs = tf.layers.dense(self.outputs[:,-1,:],self.candidate_size,activation=tf.nn.softmax)
		self.predict_op = tf.argmax(self.logit_outputs,1,name="predict_op")
		self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logit_outputs,labels=self.y,name="cross_entropy_liss")
		self.cross_entropy_loss = tf.reduce_sum(self.cross_entropy)
		self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy_loss)

	def train(self,epochs=1000) :
		self.init_op = tf.global_variables_initializer()

		batches = zip(range(0,len(self.train_q)-32,32),range(32,len(self.train_q),32))
		batches = [(start,end) for start,end in batches]
		self.session.run(self.init_op)
		avg_epoch_loss = 0
		epoch_loss = 0
		best_val_acc = 9999
		for epoch in range(epochs) :
			np.random.shuffle(batches)
			train_batches = batches[:-1]
			epoch_loss = 0
			avg_epoch_loss = 0
			for start, end in tqdm(train_batches) :
				train_batch = self.train_q[start:end]
				train_ans = self.train_a[start:end]

				_, loss = self.batch_fit(train_batch,train_ans)
				epoch_loss += loss
			avg_epoch_loss = epoch_loss / len(train_batches)

			val_preds = list()
			actual_preds = list()
			val_batches = batches[-1:]

			for start, end in tqdm(val_batches) :
				val_batch = self.train_q[start:end]
				val_ans = self.train_a[start:end]

				preds = self.batch_predict(val_batch)
				val_preds.extend(preds)
				actual_preds.extend(val_ans)

			val_acc = metrics.accuracy_score(np.array(val_preds),np.array(actual_preds))
			print('\n\n\t\t validation accuracy \n\t\t {}'.format(val_acc))
			#train_acc_summary = tf.summary.scalar(os.path.join(self.model_dir + '/','train_acc'), tf.constant((train_acc), dtype=tf.float32))
			val_acc_summary = tf.summary.scalar(os.path.join(self.model_dir + '/','val_acc'), tf.constant((val_acc), dtype=tf.float32))
			merged_summary = tf.summary.merge(
                    [val_acc_summary])
			summary_str = self.session.run(merged_summary)
			self.summary_writer.add_summary(summary_str, epoch)
			self.summary_writer.flush()
			if val_acc > best_val_acc :
				best_val_acc = val_acc
				self.saver.save(self.session, os.path.join(self.model_dir,'model.ckpt'), global_step=epoch)


		final_train_preds = self.batch_predict(self.train_q)
		final_accuracy = metrics.accuracy_score(np.array(final_train_preds),self.train_a)
		print('\n\n\t\tFINAL ACCURACY IS\n\n\t\t{}'.format(final_accuracy))

	def load_saved_model(self) :
		ckpt = tf.train.get_checkpoint_state(self.model_dir)
		if ckpt and ckpt.model_checkpoint_path :
			self.saver.restore(self.session,ckpt.model_checkpoint_path)
		else :
			print('No Checkpoint found')

	def test(self,file_name='testing.txt') :
		self.load_saved_model()
		self.pre_process_test(file_name=file_name)
		test_preds = self.batch_predict(self.test_q)
		test_accuracy = metrics.accuracy_score(np.array(test_preds),self.test_a)
		print('\n\n\t\tFINAL ACCURACY IS\n\n\t\t{}'.format(test_accuracy))

	def batch_fit(self,train_batch,train_ans) :

		feed_dict = { self.x : train_batch, self.y : train_ans}

		train_op_measure, loss = self.session.run([self.train_op,self.cross_entropy_loss],feed_dict=feed_dict)

		return train_op_measure, loss

	def batch_predict(self,test_batch) :

		predictions = self.session.run(self.predict_op,{self.x : test_batch})

		return predictions

if __name__ == '__main__' :
	query_predictor = QueryPredictor(file_name='training.txt',embedding_size=128)
	query_predictor.train(epochs=1000)
	query_predictor.test(file_name='training.txt')