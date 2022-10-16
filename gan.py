import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init
from math import ceil
import numpy as np
import sys
import torch.optim as optim
import pandas as pd
import re
import pickle
CUDA = torch.cuda.is_available()

# ===============================获取序列并编码===========================================
# 设置固定长度18
MAX_SEQ_LEN = 18
# 读取训练数据
data = pd.read_csv('PAO1db_data.csv', skiprows = 1, usecols = range(3), header=None, names=['ID','seq','len'])
# 维度(8230,3)
# 获取序列(字符型)
all_sequences = np.asarray(data['seq'])
# 维度(8230,1)
# 将序列进行整数编码
CHARACTER_DICT = {
	'A': 1, 'C': 2, 'E': 3, 'D': 4, 'F': 5, 'I': 6, 'H': 7,
	'K': 8, 'M': 9, 'L': 10, 'N': 11, 'Q': 12, 'P': 13, 'S': 14,
	'R': 15, 'T': 16, 'W': 17, 'V': 18, 'Y': 19, 'G': 20, 'O': 21, 'U': 22, 'Z': 23, 'X': 24}
INDEX_DICT = {
	1: 'A', 2: 'C', 3: 'E', 4: 'D', 5: 'F', 6: 'I', 7: 'H',
	8: 'K', 9: 'M', 10: 'L', 11: 'N', 12: 'Q', 13: 'P', 14: 'S',
	15: 'R', 16: 'T', 17: 'W', 18: 'V', 19: 'Y', 20: 'G', 21: 'O', 22: 'U', 23: 'Z', 24: 'X'}

def sequence_to_vector(sequence):
	default = np.asarray([25]*(MAX_SEQ_LEN))
	for i, character in enumerate(sequence[:MAX_SEQ_LEN]):
		default[i] = CHARACTER_DICT[character]
	return default.astype(int)
def vector_to_sequence(vector):
  
	return ''.join([INDEX_DICT.get(item, '0')  for item in vector])
all_data = []
#  获取序列(整数型)
for i in range(len(all_sequences)):
  all_data.append(sequence_to_vector(all_sequences[i]))   




# =====================创建model类===========================
# 生成器
class Generator(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=True, oracle_init=False):
		super(Generator, self).__init__()
		self.hidden_dim = hidden_dim
		self.embedding_dim = embedding_dim
		self.max_seq_len = max_seq_len
		self.vocab_size = vocab_size
		self.gpu = gpu
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.gru = nn.GRU(embedding_dim, hidden_dim)
		self.gru2out = nn.Linear(hidden_dim, vocab_size)

		# initialise oracle network with N(0,1)
		# otherwise variance of initialisation is very small => high NLL for data sampled from the same model
		if oracle_init:
			for p in self.parameters():
				nn.init.normal_(p, 0, 1)

	def init_hidden(self, batch_size=1):
		h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))

		if self.gpu:
			return h.cuda()
		else:
			return h

	def forward(self, inp, hidden):
		"""
		Embeds input and applies GRU one token at a time (seq_len = 1)
		"""
		# input dim                                             # batch_size
		emb = self.embeddings(inp)                              # batch_size x embedding_dim
		emb = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim
		out, hidden = self.gru(emb, hidden)                     # 1 x batch_size x hidden_dim (out)
		out = self.gru2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
		out = F.log_softmax(out, dim=1)
		return out, hidden

	def sample(self, num_samples, start_letter=0):
		"""
		Samples the network and returns num_samples samples of length max_seq_len.
		Outputs: samples, hidden
			- samples: num_samples x max_seq_length (a sampled sequence in each row)
		"""

		samples = torch.zeros(num_samples, self.max_seq_len).type(torch.LongTensor)

		h = self.init_hidden(num_samples)
		inp = autograd.Variable(torch.LongTensor([start_letter]*num_samples))

		if self.gpu:
			samples = samples.cuda()
			inp = inp.cuda()

		for i in range(self.max_seq_len):
			out, h = self.forward(inp, h)               # out: num_samples x vocab_size
			out = torch.multinomial(torch.exp(out), 1)  # num_samples x 1 (sampling from each row)
			samples[:, i] = out.view(-1).data

			inp = out.view(-1)

		return samples

	def batchNLLLoss(self, inp, target):
		"""
		Returns the NLL Loss for predicting target sequence.
		Inputs: inp, target
			- inp: batch_size x seq_len
			- target: batch_size x seq_len
			inp should be target with <s> (start letter) prepended
		"""

		loss_fn = nn.NLLLoss()
		batch_size, seq_len = inp.size()
		inp = inp.permute(1, 0)           # seq_len x batch_size
		target = target.permute(1, 0)     # seq_len x batch_size
		h = self.init_hidden(batch_size)

		loss = 0

		for i in range(seq_len):
			out, h = self.forward(inp[i], h)
			loss += loss_fn(out, target[i])

		return loss     # per batch

	def batchPGLoss(self, inp, target, reward):
		"""
		Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
		Inspired by the example in http://karpathy.github.io/2016/05/31/rl/
		Inputs: inp, target
			- inp: batch_size x seq_len
			- target: batch_size x seq_len
			- reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
					  sentence)
			inp should be target with <s> (start letter) prepended
		"""

		batch_size, seq_len = inp.size()
		inp = inp.permute(1, 0)          # seq_len x batch_size
		target = target.permute(1, 0)    # seq_len x batch_size
		h = self.init_hidden(batch_size)

		loss = 0
		for i in range(seq_len):
			out, h = self.forward(inp[i], h)
			# TODO: should h be detached from graph (.detach())?
			for j in range(batch_size):
				loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q

		return loss/batch_size
# 辨别器
class Discriminator(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=True, dropout=0.2):
		super(Discriminator, self).__init__()
		self.hidden_dim = hidden_dim
		self.embedding_dim = embedding_dim
		self.max_seq_len = max_seq_len
		self.gpu = gpu

		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
		self.gru2hidden = nn.Linear(2*2*hidden_dim, hidden_dim)
		self.dropout_linear = nn.Dropout(p=dropout)
		self.hidden2out = nn.Linear(hidden_dim, 1)

	def init_hidden(self, batch_size):
		h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))

		if self.gpu:
			return h.cuda()
		else:
			return h

	def forward(self, input, hidden):
		# input dim                                                # batch_size x seq_len
		emb = self.embeddings(input)                               # batch_size x seq_len x embedding_dim
		emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
		_, hidden = self.gru(emb, hidden)                          # 4 x batch_size x hidden_dim
		hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
		out = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))  # batch_size x 4*hidden_dim
		out = torch.tanh(out)
		out = self.dropout_linear(out)
		out = self.hidden2out(out)                                 # batch_size x 1
		out = torch.sigmoid(out)
		return out

	def batchClassify(self, inp):
		"""
		Classifies a batch of sequences.
		Inputs: inp
			- inp: batch_size x seq_len
		Returns: out
			- out: batch_size ([0,1] score)
		"""

		h = self.init_hidden(inp.size()[0])
		out = self.forward(inp, h)
		return out.view(-1)

	def batchBCELoss(self, inp, target):
		"""
		Returns Binary Cross Entropy Loss for discriminator.
		 Inputs: inp, target
			- inp: batch_size x seq_len
			- target: batch_size (binary 1/0)
		"""

		loss_fn = nn.BCELoss()
		h = self.init_hidden(inp.size()[0])
		out = self.forward(inp, h)
		return loss_fn(out, target)


# =======================数据集准备======================
def prepare_generator_batch(samples, start_letter=0, gpu=True):
	"""
	Takes samples (a batch) and returns
	Inputs: samples, start_letter, cuda
		- samples: batch_size x seq_len (Tensor with a sample in each row)
	Returns: inp, target
		- inp: batch_size x seq_len (same as target, but with start_letter prepended)
		- target: batch_size x seq_len (Variable same as samples)
	target:[18, 15, 13,  ..., 25, 25, 25]
	inp:[ 0, 18, 15,  ..., 25, 25, 25]
	"""
	batch_size, seq_len = samples.size()   # 16, 18
	inp = torch.zeros(batch_size, seq_len)   # (16,18)
	target = samples
	inp[:, 0] = start_letter
	inp[:, 1:] = target[:, :seq_len-1]
	inp = inp.type(torch.LongTensor)
	target = target.type(torch.LongTensor)
	if gpu:
		inp = inp.cuda()
		target = target.cuda()
	return inp, target
def prepare_discriminator_data(pos_samples, neg_samples, gpu=True):
	"""
	Takes positive (target) samples, negative (generator) samples and
	prepares inp and target data for discriminator.
	Inputs: pos_samples, neg_samples
		- pos_samples: pos_size x seq_len
		- neg_samples: neg_size x seq_len
	Returns: inp, target
		- inp: (pos_size + neg_size) x seq_len
		- target: pos_size + neg_size (boolean 1/0)
	"""
	inp = torch.cat((pos_samples, neg_samples), 0).type(torch.LongTensor)
	target = torch.ones(pos_samples.size()[0] + neg_samples.size()[0])
	target[pos_samples.size()[0]:] = 0

	# shuffle
	perm = torch.randperm(target.size()[0])
	target = target[perm]
	inp = inp[perm]

#    inp = Variable(inp)
#    target = Variable(target)

	if gpu:
		inp = inp.cuda()
		target = target.cuda()

	return inp, target
def batchwise_sample(gen, num_samples, batch_size):
	"""
	Sample num_samples samples batch_size samples at a time from gen.
	Does not require gpu since gen.sample() takes care of that.
	"""

	samples = []
	for i in range(int(ceil(num_samples/float(batch_size)))):
		samples.append(gen.sample(batch_size))

	return torch.cat(samples, 0)[:num_samples]
def batchwise_oracle_nll(gen, oracle, num_samples, batch_size, max_seq_len, start_letter=0, gpu=True):
	s = batchwise_sample(gen, num_samples, batch_size)
	oracle_nll = 0
	for i in range(0, num_samples, batch_size):
		inp, target = prepare_generator_batch(s[i:i+batch_size], start_letter, gpu)
		oracle_loss = oracle.batchNLLLoss(inp, target) / max_seq_len
		oracle_nll += oracle_loss.data.item()

	return oracle_nll/(num_samples/batch_size)
def train_generator_MLE(gen, gen_opt, oracle, real_data_samples, epochs):
	"""
	Max Likelihood Pretraining for the generator
	"""
	for epoch in range(epochs):
		print('epoch %d : ' % (epoch + 1), end='')
		sys.stdout.flush()
		total_loss = 0

		for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
			inp, target = prepare_generator_batch(real_data_samples[i:i + BATCH_SIZE], start_letter=START_LETTER,
														  gpu=CUDA)
			gen_opt.zero_grad()
			loss = gen.batchNLLLoss(inp, target)
			loss.backward()
			gen_opt.step()

			total_loss += loss.data.item()

			if (i / BATCH_SIZE) % ceil(
							ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
				print('.', end='')
				sys.stdout.flush()

		# each loss in a batch is loss per sample
		total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN

		# sample from generator and compute oracle NLL
		oracle_loss = batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                                  start_letter=START_LETTER, gpu=CUDA)
		loss_g.append(oracle_loss)
		print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))
        
        
def train_generator_PG(gen, gen_opt, oracle, dis, num_batches):
	"""
	The generator is trained using policy gradients, using the reward from the discriminator.
	Training is done for num_batches batches.
	"""

	for batch in range(num_batches):
		s = gen.sample(BATCH_SIZE*2)        # 64 works best
		inp, target = prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)
		rewards = dis.batchClassify(target)

		gen_opt.zero_grad()
		pg_loss = gen.batchPGLoss(inp, target, rewards)
		pg_loss.backward()
		gen_opt.step()
    # sample from generator and compute oracle NLL
	oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                                   start_letter=START_LETTER, gpu=CUDA)

	print(' oracle_sample_NLL = %.4f' % oracle_loss)    
        
    
def train_discriminator(discriminator, dis_opt, real_data_samples, generator, oracle, d_steps, epochs):
	"""
	Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
	Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
	"""
	# generating a small validation set before training (using oracle and generator)
	indice = random.sample(range(len(real_data_samples)), 100)
	indice = torch.tensor(indice)
	pos_val = real_data_samples[indice]
#    pos_val = real_data_samples(np.random.choice(len(real_data_samples), size=100, replace=False))
	neg_val = generator.sample(100)
	val_inp, val_target = prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

	for d_step in range(d_steps):
		s = batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
		dis_inp, dis_target = prepare_discriminator_data(real_data_samples, s, gpu=CUDA)
		for epoch in range(epochs):
			print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
			sys.stdout.flush()
			total_loss = 0
			total_acc = 0

			for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE): #2 * POS_NEG_SAMPLES because both pos
																#and neg samples included in dis_inp
				inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
				dis_opt.zero_grad()
				out = discriminator.batchClassify(inp)
				loss_fn = nn.BCELoss()
				loss = loss_fn(out, target)
				loss.backward()
				dis_opt.step()

				total_loss += loss.data.item()
				total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

				if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
						BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
					print('.', end='')
					sys.stdout.flush()

			total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
			total_acc /= float(2 * POS_NEG_SAMPLES)

			val_pred = discriminator.batchClassify(val_inp)
			print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
				total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/200.))

			loss_d.append(total_loss)

# ==================================设置参数===================================
#Fixed Params
VOCAB_SIZE = 26 #Starting Letter + 20 AA + Padding
MAX_SEQ_LEN = 18 #2000 kDa / 110 kDa = 18
START_LETTER = 0 
POS_NEG_SAMPLES = len(all_data) #Size of AVPDb dataset
torch.manual_seed(11)

#Variables
BATCH_SIZE = 16
ADV_TRAIN_EPOCHS = 100

#Generator Parameters 生成器的参数
MLE_TRAIN_EPOCHS = 50
GEN_EMBEDDING_DIM = 3
GEN_HIDDEN_DIM = 128
NUM_PG_BATCHES = 1
GEN_lr = 0.00005

#Discriminator Parameters  鉴别器的参数
DIS_EMBEDDING_DIM = 3            
DIS_HIDDEN_DIM = 128
D_STEPS = 30
D_EPOCHS = 10

# Adversarial Training Generator
ADV_D_EPOCHS = 5
ADV_D_STEPS = 1

# 保存模型
gen_model = 'gen_500_PAO1_402.pth'
dis_model = 'dis_500_PAO1_402.pth'


# ========================================主函数===============================================
if __name__ == '__main__':
    # 实例化模型
	oracle = Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA, oracle_init=True)
	gen = Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
	dis = Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)

	loss_g = []
	loss_d = []

	if CUDA:
		oracle = oracle.cuda()
		gen = gen.cuda()
		dis = dis.cuda()

		#Makes a dataset which follows oracle's distribution
		oracle_samples = torch.Tensor(all_data).type(torch.LongTensor)
		oracle_samples = oracle_samples.cuda()
	else:
		oracle_samples = torch.IntTensor(all_data).type(torch.LongTensor)

	# 1.  GENERATOR MLE TRAINING
	print('Starting Generator MLE Training...')
	gen_optimizer = optim.Adam(gen.parameters(), lr = GEN_lr)
	train_generator_MLE(gen, gen_optimizer, oracle, oracle_samples, MLE_TRAIN_EPOCHS)
	print('Finished Generator MLE Training...')


	# PRETRAIN DISCRIMINATOR
	print('\nStarting Discriminator Training...')
	dis_optimizer = optim.Adagrad(dis.parameters())
	train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, D_STEPS, D_EPOCHS)

	# ADVERSARIAL TRAINING
	print('\nStarting Adversarial Training...')
	for epoch in range(ADV_TRAIN_EPOCHS):
		print('\n--------\nEPOCH %d\n--------' % (epoch+1))
		# TRAIN GENERATOR
		print('\nAdversarial Training Generator : ', end='')
		sys.stdout.flush()
		train_generator_PG(gen, gen_optimizer, oracle, dis, NUM_PG_BATCHES)
		# TRAIN DISCRIMINATOR
		print('\nAdversarial Training Discriminator : ')
		train_discriminator(dis, dis_optimizer, oracle_samples, gen, oracle, ADV_D_STEPS, ADV_D_EPOCHS)
# 	with open('AVPDb_20000.txt', 'w') as f:
# 		for item in loss_d:
# 			f.write("%s\n" % item)
	torch.save(gen.state_dict(), './models/' + gen_model)
	torch.save(dis.state_dict(), './models/' + dis_model)
