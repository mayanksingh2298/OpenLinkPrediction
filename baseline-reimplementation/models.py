import heapq
import argparse
import logging
import os
import pickle
import pprint
import sys
import time
import numpy as np
import random
import torch
from tqdm import tqdm
import datetime
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from kb import kb
import utils
from dataset import Dataset
# import hsoftmax
# from hsoftmax import BalancedHSoftmax
class rotatELSTM(torch.nn.Module):
	"""
		stores embeddings for each e and r token
		uses lstm to compose them
		sends to rotatE for scoring the triple
	"""
	def __init__(self, etoken_count, rtoken_count, embedding_dim, num_lstm_layers = 1, gamma = 12, clamp_v=None, reg=2,
				 batch_norm=False, unit_reg=False, has_cuda=True, initial_token_embedding = None, entity_tokens = [], relation_tokens = [], lstm_dropout=0):
		device = torch.device("cuda" if torch.cuda.is_available() and has_cuda else "cpu")
		super(rotatELSTM, self).__init__()
		self.etoken_count = etoken_count
		self.rtoken_count = rtoken_count
		self.embedding_dim = embedding_dim
		self.Et_im = torch.nn.Embedding(self.etoken_count, self.embedding_dim)
		self.Et_re = torch.nn.Embedding(self.etoken_count, self.embedding_dim)
		self.Rt = torch.nn.Embedding(self.rtoken_count, self.embedding_dim)
		self.Et_im.to(device)
		self.Et_re.to(device)
		self.Rt.to(device)
		self.gamma = gamma
		self.epsilon = 2.0
		self.embedding_range = (self.gamma + self.epsilon) / self.embedding_dim
		if(not initial_token_embedding):
			torch.nn.init.uniform_(
				tensor=self.Et_re.weight.data, 
				a=-self.embedding_range, 
				b=self.embedding_range
			)
			torch.nn.init.uniform_(
				tensor=self.Et_im.weight.data, 
				a=-self.embedding_range, 
				b=self.embedding_range
			)
			torch.nn.init.uniform_(
				tensor=self.Rt.weight.data, 
				a=-self.embedding_range, 
				b=self.embedding_range
			)
		self.minimum_value = -self.embedding_dim * self.embedding_dim
		self.dropout = torch.nn.Dropout(p = lstm_dropout)

		self.lstm = torch.nn.LSTM(input_size = embedding_dim, hidden_size=embedding_dim, num_layers = num_lstm_layers, batch_first = True)
		self.lstm.to(device)

	def get_mention_embedding(self, data, flag, lengths):
		"""
			returns the embedding of the mention after composing with LSTM
			flag = 0 for entity
			flag = 1 for relation
			data has shape Batch x max seq len
			lengths has shape just batch and contains the actual lengths(besides padding) of each element of batch
		"""
		if flag == 0:
			real_token_embeddings = self.Et_re(data)
			img_token_embeddings = self.Et_im(data)
		else:
			token_embeddings = self.Rt(data)
			#Make phases of relations uniformly distributed in [-pi, pi]
			phase_relation = token_embeddings/(self.embedding_range/np.pi)
			real_token_embeddings = torch.cos(phase_relation)
			img_token_embeddings = torch.sin(phase_relation)
		
		indices = (lengths-1).reshape(-1,1,1).expand(len(data),1,self.embedding_dim).cuda() 

		real_token_embeddings = torch.nn.utils.rnn.pack_padded_sequence(real_token_embeddings, lengths, batch_first=True,enforce_sorted=False)
		real_lstm_embeddings,_ = self.lstm(real_token_embeddings)
		real_lstm_embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(real_lstm_embeddings, batch_first=True)
		real_lstm_embeddings = real_lstm_embeddings.gather(1,indices).squeeze(1)

		img_token_embeddings = torch.nn.utils.rnn.pack_padded_sequence(img_token_embeddings, lengths, batch_first=True,enforce_sorted=False)
		img_lstm_embeddings,_ = self.lstm(img_token_embeddings)
		img_lstm_embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(img_lstm_embeddings, batch_first=True)
		img_lstm_embeddings = img_lstm_embeddings.gather(1,indices).squeeze(1)
		


		# img_lstm_embeddings,_ = self.lstm(img_token_embeddings)
		# img_lstm_embeddings = img_lstm_embeddings[:,-1,:]
		real_lstm_embeddings = self.dropout(real_lstm_embeddings)
		img_lstm_embeddings = self.dropout(img_lstm_embeddings)


		return real_lstm_embeddings, img_lstm_embeddings

	def complex_score_e1_r_with_all_ementions(self,e1_r,e1_i,r_r,r_i,all_e2_r,all_e2_i, split = 1):
		"""
			#tail prediction
			e1_r,e1_i,r_r,r_i are tensors of shape batch x embed_dim
			all_e2 has shape # of entity mentions x embed dim
			split is used when dimension size is so big that it doesn't fit on gpu. So you split over the dimension to solve each splitted section
			returns a tensor of shape batch x # of entity mentions i.e. the score with each entity mention
		"""
		section = all_e2_i.shape[1]//split
		ans = torch.zeros((e1_r.shape[0],all_e2_r.shape[0])).cuda()
		for i in range(0,all_e2_i.shape[1],section):
			# re_score = re_head * re_relation - im_head * im_relation
   #          im_score = re_head * im_relation + im_head * re_relation
   #          re_score = re_score - re_tail
   #          im_score = im_score - im_tail

			re_score = e1_r[:,i:i+section] * r_r[:,i:i+section] - e1_i[:,i:i+section] * r_i[:,i:i+section]
			im_score = e1_r[:,i:i+section] * r_i[:,i:i+section] + e1_i[:,i:i+section] * r_r[:,i:i+section]

			re_score = re_score.unsqueeze(1).expand(-1,len(all_e2_r),-1) # batch x len of e mentions x embed dim
			im_score = im_score.unsqueeze(1).expand(-1,len(all_e2_r),-1) # batch x len of e mentions x embed dim

			all_e2_r_tmp = all_e2_r[:,i:i+section].unsqueeze(0).expand(len(e1_r),-1,-1) # batch x len of e mentions x embed dim
			all_e2_i_tmp = all_e2_i[:,i:i+section].unsqueeze(0).expand(len(e1_r),-1,-1) # batch x len of e mentions x embed dim

			re_score = re_score - all_e2_r_tmp
			im_score = im_score - all_e2_i_tmp
			score = torch.stack([re_score, im_score], dim = 0)
			score = score.norm(dim = 0)
			# score = torch.sqrt(re_score**2 + im_score**2)
			
			ans += score.sum(dim = -1) # batch x len of e mentions

		return self.gamma - ans

	def complex_score_e2_r_with_all_ementions(self,e2_r,e2_i,r_r,r_i,all_e1_r,all_e1_i, split=1):
		"""
			#head prediction
			e2_r,e2_i,r_r,r_i are tensors of shape batch x embed_dim
			all_e1 has shape # of entity mentions x embed dim
			split is used when dimension size is so big that it doesn't fit on gpu. So you split over the dimension to solve each splitted section
			returns a tensor of shape batch x # of entity mentions i.e. the score with each entity mention
		"""

		section = all_e1_i.shape[1]//split
		ans = torch.zeros((e2_r.shape[0],all_e1_r.shape[0])).cuda()
		for i in range(0,all_e1_i.shape[1],section):
			# re_score = re_relation * re_tail + im_relation * im_tail
   #          im_score = re_relation * im_tail - im_relation * re_tail
   #          re_score = re_score - re_head
   #          im_score = im_score - im_head

			re_score = e2_r[:,i:i+section] * r_r[:,i:i+section] + e2_i[:,i:i+section] * r_i[:,i:i+section]
			im_score = e2_i[:,i:i+section] * r_r[:,i:i+section] - e2_r[:,i:i+section] * r_i[:,i:i+section]

			re_score = re_score.unsqueeze(1).expand(-1,len(all_e1_r),-1) # batch x len of e mentions x embed dim
			im_score = im_score.unsqueeze(1).expand(-1,len(all_e1_r),-1) # batch x len of e mentions x embed dim

			all_e1_r_tmp = all_e1_r[:,i:i+section].unsqueeze(0).expand(len(e2_r),-1,-1) # batch x len of e mentions x embed dim
			all_e1_i_tmp = all_e1_i[:,i:i+section].unsqueeze(0).expand(len(e2_r),-1,-1) # batch x len of e mentions x embed dim

			re_score = re_score - all_e1_r_tmp
			im_score = im_score - all_e1_i_tmp
			score = torch.stack([re_score, im_score], dim = 0)
			score = score.norm(dim = 0)

			ans += score.sum(dim = -1) # batch x len of e mentions
		return self.gamma - ans


class complexLSTM(torch.nn.Module):
	"""
		stores embeddings for each e and r token
		uses lstm to compose them
		sends to complex for scoring the triple
	"""
	def __init__(self, etoken_count, rtoken_count, embedding_dim, num_lstm_layers = 1, clamp_v=None, reg=2,
				 batch_norm=False, unit_reg=False, has_cuda=True, initial_token_embedding = None, entity_tokens = [], relation_tokens = [], lstm_dropout=0):
		# etoken_count is assumed to be incremented by 1 to handle unk
		device = torch.device("cuda" if torch.cuda.is_available() and has_cuda else "cpu")

		super(complexLSTM, self).__init__()
		self.etoken_count = etoken_count
		self.rtoken_count = rtoken_count
		self.embedding_dim = embedding_dim
		self.Et_im = torch.nn.Embedding(self.etoken_count, self.embedding_dim)
		self.Rt_im = torch.nn.Embedding(self.rtoken_count, self.embedding_dim)
		self.Et_re = torch.nn.Embedding(self.etoken_count, self.embedding_dim)
		self.Rt_re = torch.nn.Embedding(self.rtoken_count, self.embedding_dim)

		if(initial_token_embedding):
			print("Initializing token embeddings from: {}".format(initial_token_embedding))
			embeddings_dict = {}
			lines = open(initial_token_embedding).readlines()
			for line in lines:
				values = line.split()
				word = values[0]
				vector = torch.tensor(np.asarray(values[1:], "float32"))
				embeddings_dict[word] = vector
			for i,token in enumerate(entity_tokens):
				if token in embeddings_dict:
					self.Et_im.weight.data[i] = embeddings_dict[token]
					self.Et_re.weight.data[i] = embeddings_dict[token]
			for i,token in enumerate(relation_tokens):
				if token in embeddings_dict:
					self.Rt_im.weight.data[i] = embeddings_dict[token]
					self.Rt_re.weight.data[i] = embeddings_dict[token]

		self.Et_im.to(device)
		self.Rt_im.to(device)
		self.Et_re.to(device)
		self.Rt_re.to(device)

		if(not initial_token_embedding):
			torch.nn.init.normal_(self.Et_re.weight.data, 0, 0.1)
			torch.nn.init.normal_(self.Et_im.weight.data, 0, 0.1)
			torch.nn.init.normal_(self.Rt_re.weight.data, 0, 0.1)
			torch.nn.init.normal_(self.Rt_im.weight.data, 0, 0.1)

		self.minimum_value = -self.embedding_dim * self.embedding_dim
		# not using these for now, will see later how these help
		self.clamp_v = clamp_v
		self.unit_reg = unit_reg
		self.reg = reg
		self.batch_norm = batch_norm

		self.dropout = torch.nn.Dropout(p = lstm_dropout)

		self.lstm = torch.nn.LSTM(input_size = embedding_dim, hidden_size=embedding_dim, num_layers = num_lstm_layers, batch_first = True)
		self.lstm.to(device)

		self.entity_batchnorm = torch.nn.BatchNorm1d(embedding_dim, momentum=0.1, eps=1e-5)
		self.relation_batchnorm = torch.nn.BatchNorm1d(embedding_dim, momentum=0.1, eps=1e-5)
		torch.nn.init.uniform_(self.entity_batchnorm.weight)
		torch.nn.init.uniform_(self.relation_batchnorm.weight)
		self.entity_batchnorm.to(device)
		self.relation_batchnorm.to(device)

	def get_mention_embedding(self, data, flag, lengths):
		"""
			returns the embedding of the mention after composing with LSTM
			flag = 0 for entity
			flag = 1 for relation
			data has shape Batch x max seq len
			lengths has shape just batch and contains the actual lengths(besides padding) of each element of batch
		"""
		if flag == 0:
			real_token_embeddings = self.Et_re(data)
			img_token_embeddings = self.Et_im(data)
		else:
			real_token_embeddings = self.Rt_re(data)
			img_token_embeddings = self.Rt_im(data)
		
		indices = (lengths-1).reshape(-1,1,1).expand(len(data),1,self.embedding_dim).cuda() 

		real_token_embeddings = torch.nn.utils.rnn.pack_padded_sequence(real_token_embeddings, lengths, batch_first=True,enforce_sorted=False)
		real_lstm_embeddings,_ = self.lstm(real_token_embeddings)
		real_lstm_embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(real_lstm_embeddings, batch_first=True)
		real_lstm_embeddings = real_lstm_embeddings.gather(1,indices).squeeze(1)

		img_token_embeddings = torch.nn.utils.rnn.pack_padded_sequence(img_token_embeddings, lengths, batch_first=True,enforce_sorted=False)
		img_lstm_embeddings,_ = self.lstm(img_token_embeddings)
		img_lstm_embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(img_lstm_embeddings, batch_first=True)
		img_lstm_embeddings = img_lstm_embeddings.gather(1,indices).squeeze(1)
		


		# img_lstm_embeddings,_ = self.lstm(img_token_embeddings)
		# img_lstm_embeddings = img_lstm_embeddings[:,-1,:]
		if flag == 0:
			norm_func = self.entity_batchnorm
		else:
			norm_func = self.relation_batchnorm

		real_lstm_embeddings = norm_func(real_lstm_embeddings)
		img_lstm_embeddings = norm_func(img_lstm_embeddings)


		real_lstm_embeddings = self.dropout(real_lstm_embeddings)
		img_lstm_embeddings = self.dropout(img_lstm_embeddings)


		return real_lstm_embeddings, img_lstm_embeddings

	def complex_score_e1_r_with_all_ementions(self,e1_r,e1_i,r_r,r_i,all_e2_r,all_e2_i, split = 1):
		"""
			#tail prediction
			e1_r,e1_i,r_r,r_i are tensors of shape batch x embed_dim
			all_e2 has shape # of entity mentions x embed dim
			split is used when dimension size is so big that it doesn't fit on gpu. So you split over the dimension to solve each splitted section
			returns a tensor of shape batch x # of entity mentions i.e. the score with each entity mention
		"""
	# result = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
		 # = (s_re * r_re - s_im * r_im) * o_re + (s_im * r_re + s_re * r_im) * o_im
		if split>1:
			raise NotImplementedError("splits not implemented")
		section = all_e2_i.shape[1]//split
		ans = torch.zeros((e1_r.shape[0],all_e2_r.shape[0])).cuda()
		for i in range(0,all_e2_i.shape[1],section):
			tmp1 = e1_r[:,i:i+section] * r_r[:,i:i+section] - e1_i[:,i:i+section] * r_i[:,i:i+section]
			tmp2 = e1_i[:,i:i+section] * r_r[:,i:i+section] + e1_r[:,i:i+section] * r_i[:,i:i+section]

			ans_2 = tmp1.mm(all_e2_r.transpose(0,1)) + tmp2.mm(all_e2_i.transpose(0,1))

			# tmp1 = tmp1.unsqueeze(1).expand(-1,len(all_e2_r),-1) # batch x len of e mentions x embed dim
			# tmp2 = tmp2.unsqueeze(1).expand(-1,len(all_e2_r),-1) # batch x len of e mentions x embed dim

			# all_e2_r_tmp = all_e2_r[:,i:i+section].unsqueeze(0).expand(len(e1_r),-1,-1) # batch x len of e mentions x embed dim
			# all_e2_i_tmp = all_e2_i[:,i:i+section].unsqueeze(0).expand(len(e1_r),-1,-1) # batch x len of e mentions x embed dim

			# ans += (tmp1 * all_e2_r_tmp + tmp2 * all_e2_i_tmp).sum(dim = -1) # batch x len of e mentions

		# return ans
		return ans_2


	def complex_score_e2_r_with_all_ementions(self,e2_r,e2_i,r_r,r_i,all_e1_r,all_e1_i, split=1):
		"""
			#head prediction
			e2_r,e2_i,r_r,r_i are tensors of shape batch x embed_dim
			all_e1 has shape # of entity mentions x embed dim
			split is used when dimension size is so big that it doesn't fit on gpu. So you split over the dimension to solve each splitted section
			returns a tensor of shape batch x # of entity mentions i.e. the score with each entity mention
		"""
	# result = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
		if split>1:
			raise NotImplementedError("splits not implemented")
		section = all_e1_i.shape[1]//split
		ans = torch.zeros((e2_r.shape[0],all_e1_r.shape[0])).cuda()
		for i in range(0,all_e1_i.shape[1],section):

			tmp1 = e2_r[:,i:i+section] * r_r[:,i:i+section] + e2_i[:,i:i+section] * r_i[:,i:i+section]
			tmp2 = e2_i[:,i:i+section] * r_r[:,i:i+section] - e2_r[:,i:i+section] * r_i[:,i:i+section]

			ans_2 = tmp1.mm(all_e1_r.transpose(0,1)) + tmp2.mm(all_e1_i.transpose(0,1))


			# tmp1 = tmp1.unsqueeze(1).expand(-1,len(all_e1_r),-1) # batch x len of e mentions x embed dim
			# tmp2 = tmp2.unsqueeze(1).expand(-1,len(all_e1_r),-1) # batch x len of e mentions x embed dim

			# all_e1_r_tmp = all_e1_r[:,i:i+section].unsqueeze(0).expand(len(e2_r),-1,-1) # batch x len of e mentions x embed dim
			# all_e1_i_tmp = all_e1_i[:,i:i+section].unsqueeze(0).expand(len(e2_r),-1,-1) # batch x len of e mentions x embed dim

			# ans += (tmp1 * all_e1_r_tmp + tmp2 * all_e1_i_tmp).sum(dim = -1) # batch x len of e mentions
		return ans_2

	def regularizer(self, s, r, o, reg_val=0):
		raise NotImplemented

	def forward():
		# think of how to do this while writing code for training
		pass

class complexLSTM_2(torch.nn.Module):
	"""
		Speciality: has 2 lstms - one for entity, one for relation
		stores embeddings for each e and r token
		uses lstm to compose them
		sends to complex for scoring the triple
	"""

	def __init__(self, etoken_count, rtoken_count, embedding_dim, num_lstm_layers = 1, clamp_v=None, reg=2,
				 batch_norm=False, unit_reg=False, has_cuda=True, initial_token_embedding = None, entity_tokens = [], relation_tokens = [], lstm_dropout=0):
		# etoken_count is assumed to be incremented by 1 to handle unk
		device = torch.device("cuda" if torch.cuda.is_available() and has_cuda else "cpu")

		super(complexLSTM_2, self).__init__()
		self.etoken_count = etoken_count
		self.rtoken_count = rtoken_count
		self.embedding_dim = embedding_dim
		# self.Et_im = torch.nn.Embedding(self.etoken_count, self.embedding_dim, padding_idx=0)
		# self.Rt_im = torch.nn.Embedding(self.rtoken_count, self.embedding_dim, padding_idx=0)
		# self.Et_re = torch.nn.Embedding(self.etoken_count, self.embedding_dim, padding_idx=0)
		# self.Rt_re = torch.nn.Embedding(self.rtoken_count, self.embedding_dim, padding_idx=0)
		self.Et = torch.nn.Embedding(self.etoken_count, self.embedding_dim, padding_idx=0)
		self.Rt = torch.nn.Embedding(self.rtoken_count, self.embedding_dim, padding_idx=0)

		if(initial_token_embedding):
			print("Initializing token embeddings from: {}".format(initial_token_embedding))
			embeddings_dict = {}
			lines = open(initial_token_embedding).readlines()
			for line in lines:
				values = line.split()
				word = values[0]
				vector = torch.tensor(np.asarray(values[1:], "float32"))
				embeddings_dict[word] = vector
			for i,token in enumerate(entity_tokens):
				if token in embeddings_dict:
					# self.Et_im.weight.data[i] = embeddings_dict[token]
					# self.Et_re.weight.data[i] = embeddings_dict[token]
					self.Et.weight.data[i] = embeddings_dict[token]
			for i,token in enumerate(relation_tokens):
				if token in embeddings_dict:
					# self.Rt_im.weight.data[i] = embeddings_dict[token]
					# self.Rt_re.weight.data[i] = embeddings_dict[token]
					self.Rt.weight.data[i] = embeddings_dict[token]

		# self.Et_im.to(device)
		# self.Rt_im.to(device)
		# self.Et_re.to(device)
		# self.Rt_re.to(device)
		self.Et.to(device)
		self.Rt.to(device)

		if(not initial_token_embedding):
			# torch.nn.init.normal_(self.Et_re.weight.data, 0, 0.1)
			# torch.nn.init.normal_(self.Et_im.weight.data, 0, 0.1)
			# torch.nn.init.normal_(self.Rt_re.weight.data, 0, 0.1)
			# torch.nn.init.normal_(self.Rt_im.weight.data, 0, 0.1)
			torch.nn.init.normal_(self.Et.weight.data, 0, 0.1)
			torch.nn.init.normal_(self.Rt.weight.data, 0, 0.1)

		self.minimum_value = -self.embedding_dim * self.embedding_dim
		# not using these for now, will see later how these help
		self.clamp_v = clamp_v
		self.unit_reg = unit_reg
		self.reg = reg
		self.batch_norm = batch_norm

		self.dropout = torch.nn.Dropout(p = lstm_dropout)

		self.lstm_e = torch.nn.LSTM(input_size = embedding_dim, hidden_size=embedding_dim, num_layers = num_lstm_layers, batch_first = True)
		self.lstm_r = torch.nn.LSTM(input_size = embedding_dim, hidden_size=embedding_dim, num_layers = num_lstm_layers, batch_first = True)
		
		self.lstm_e.to(device)
		self.lstm_r.to(device)

		self.entity_batchnorm = torch.nn.BatchNorm1d(embedding_dim, momentum=0.1, eps=1e-5)
		self.relation_batchnorm = torch.nn.BatchNorm1d(embedding_dim, momentum=0.1, eps=1e-5)
		torch.nn.init.uniform_(self.entity_batchnorm.weight)
		torch.nn.init.uniform_(self.relation_batchnorm.weight)
		self.entity_batchnorm.to(device)
		self.relation_batchnorm.to(device)

	def get_mention_embedding(self, data, flag, lengths):
		"""
			returns the embedding of the mention after composing with LSTM
			flag = 0 for entity
			flag = 1 for relation
			data has shape Batch x max seq len
			lengths has shape just batch and contains the actual lengths(besides padding) of each element of batch
		"""
		if flag == 0:
			# real_token_embeddings = self.Et_re(data)
			# img_token_embeddings = self.Et_im(data)
			token_embeddings = self.Et(data)
			lstm_func = self.lstm_e
		else:
			# real_token_embeddings = self.Rt_re(data)
			# img_token_embeddings = self.Rt_im(data)
			token_embeddings = self.Rt(data)
			lstm_func = self.lstm_r

		
		indices = (lengths-1).reshape(-1,1,1).expand(len(data),1,self.embedding_dim).cuda() 

		# real_token_embeddings = torch.nn.utils.rnn.pack_padded_sequence(real_token_embeddings, lengths, batch_first=True,enforce_sorted=False)
		# real_lstm_embeddings,_ = lstm_func(real_token_embeddings)
		# real_lstm_embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(real_lstm_embeddings, batch_first=True)
		# real_lstm_embeddings = real_lstm_embeddings.gather(1,indices).squeeze(1)

		# img_token_embeddings = torch.nn.utils.rnn.pack_padded_sequence(img_token_embeddings, lengths, batch_first=True,enforce_sorted=False)
		# img_lstm_embeddings,_ = lstm_func(img_token_embeddings)
		# img_lstm_embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(img_lstm_embeddings, batch_first=True)
		# img_lstm_embeddings = img_lstm_embeddings.gather(1,indices).squeeze(1)
		token_embeddings = torch.nn.utils.rnn.pack_padded_sequence(token_embeddings, lengths, batch_first=True,enforce_sorted=False)
		lstm_embeddings,_ = lstm_func(token_embeddings)
		lstm_embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_embeddings, batch_first=True)
		lstm_embeddings = lstm_embeddings.gather(1,indices).squeeze(1)
		
		if flag == 0:
			norm_func = self.entity_batchnorm
		else:
			norm_func = self.relation_batchnorm

		# real_lstm_embeddings = norm_func(real_lstm_embeddings)
		# img_lstm_embeddings = norm_func(img_lstm_embeddings)
		lstm_embeddings = norm_func(lstm_embeddings)

		# real_lstm_embeddings = self.dropout(real_lstm_embeddings)
		# img_lstm_embeddings = self.dropout(img_lstm_embeddings)
		lstm_embeddings = self.dropout(lstm_embeddings)
		real_lstm_embeddings, img_lstm_embeddings = lstm_embeddings.chunk(2, dim=-1)

		return real_lstm_embeddings, img_lstm_embeddings

	def complex_score_e1_r_with_all_ementions(self,e1_r,e1_i,r_r,r_i,all_e2_r,all_e2_i, split = 1):
		"""
			#tail prediction
			e1_r,e1_i,r_r,r_i are tensors of shape batch x embed_dim
			all_e2 has shape # of entity mentions x embed dim
			split is used when dimension size is so big that it doesn't fit on gpu. So you split over the dimension to solve each splitted section
			returns a tensor of shape batch x # of entity mentions i.e. the score with each entity mention
		"""
	# result = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
		 # = (s_re * r_re - s_im * r_im) * o_re + (s_im * r_re + s_re * r_im) * o_im
		if split>1:
			raise NotImplementedError("splits not implemented")
		section = all_e2_i.shape[1]//split
		ans = torch.zeros((e1_r.shape[0],all_e2_r.shape[0])).cuda()
		for i in range(0,all_e2_i.shape[1],section):
			tmp1 = e1_r[:,i:i+section] * r_r[:,i:i+section] - e1_i[:,i:i+section] * r_i[:,i:i+section]
			tmp2 = e1_i[:,i:i+section] * r_r[:,i:i+section] + e1_r[:,i:i+section] * r_i[:,i:i+section]

			ans_2 = tmp1.mm(all_e2_r.transpose(0,1)) + tmp2.mm(all_e2_i.transpose(0,1))
			# tmp1 = tmp1.unsqueeze(1).expand(-1,len(all_e2_r),-1) # batch x len of e mentions x embed dim
			# tmp2 = tmp2.unsqueeze(1).expand(-1,len(all_e2_r),-1) # batch x len of e mentions x embed dim

			# all_e2_r_tmp = all_e2_r[:,i:i+section].unsqueeze(0).expand(len(e1_r),-1,-1) # batch x len of e mentions x embed dim
			# all_e2_i_tmp = all_e2_i[:,i:i+section].unsqueeze(0).expand(len(e1_r),-1,-1) # batch x len of e mentions x embed dim

			# ans += (tmp1 * all_e2_r_tmp + tmp2 * all_e2_i_tmp).sum(dim = -1) # batch x len of e mentions
		return ans_2

	def complex_score_e2_r_with_all_ementions(self,e2_r,e2_i,r_r,r_i,all_e1_r,all_e1_i, split=1):
		"""
			#head prediction
			e2_r,e2_i,r_r,r_i are tensors of shape batch x embed_dim
			all_e1 has shape # of entity mentions x embed dim
			split is used when dimension size is so big that it doesn't fit on gpu. So you split over the dimension to solve each splitted section
			returns a tensor of shape batch x # of entity mentions i.e. the score with each entity mention
		"""
	# result = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
		if split>1:
			raise NotImplementedError("splits not implemented")
		section = all_e1_i.shape[1]//split
		ans = torch.zeros((e2_r.shape[0],all_e1_r.shape[0])).cuda()
		for i in range(0,all_e1_i.shape[1],section):

			tmp1 = e2_r[:,i:i+section] * r_r[:,i:i+section] + e2_i[:,i:i+section] * r_i[:,i:i+section]
			tmp2 = e2_i[:,i:i+section] * r_r[:,i:i+section] - e2_r[:,i:i+section] * r_i[:,i:i+section]

			ans_2 = tmp1.mm(all_e1_r.transpose(0,1)) + tmp2.mm(all_e1_i.transpose(0,1))
			# tmp1 = tmp1.unsqueeze(1).expand(-1,len(all_e1_r),-1) # batch x len of e mentions x embed dim
			# tmp2 = tmp2.unsqueeze(1).expand(-1,len(all_e1_r),-1) # batch x len of e mentions x embed dim

			# all_e1_r_tmp = all_e1_r[:,i:i+section].unsqueeze(0).expand(len(e2_r),-1,-1) # batch x len of e mentions x embed dim
			# all_e1_i_tmp = all_e1_i[:,i:i+section].unsqueeze(0).expand(len(e2_r),-1,-1) # batch x len of e mentions x embed dim

			# ans += (tmp1 * all_e1_r_tmp + tmp2 * all_e1_i_tmp).sum(dim = -1) # batch x len of e mentions
		return ans_2

	def regularizer(self, s, r, o, reg_val=0):
		raise NotImplemented

	def forward():
		# think of how to do this while writing code for training
		pass

class complexLSTM_3(torch.nn.Module):
	"""
		Speciality: has 2 lstms - one for entity, one for relation
		stores embeddings for each e and r token
		uses lstm to compose them
		sends to complex for scoring the triple
	"""

	def __init__(self, etoken_count, rtoken_count, embedding_dim, num_lstm_layers = 1, clamp_v=None, reg=2,
				 batch_norm=False, unit_reg=False, has_cuda=True, initial_token_embedding = None, entity_tokens = [], relation_tokens = [], lstm_dropout=0):
		# etoken_count is assumed to be incremented by 1 to handle unk
		device = torch.device("cuda" if torch.cuda.is_available() and has_cuda else "cpu")

		super(complexLSTM_3, self).__init__()
		self.etoken_count = etoken_count
		self.rtoken_count = rtoken_count
		self.embedding_dim = embedding_dim
		# self.Et_im = torch.nn.Embedding(self.etoken_count, self.embedding_dim, padding_idx=0)
		# self.Rt_im = torch.nn.Embedding(self.rtoken_count, self.embedding_dim, padding_idx=0)
		# self.Et_re = torch.nn.Embedding(self.etoken_count, self.embedding_dim, padding_idx=0)
		# self.Rt_re = torch.nn.Embedding(self.rtoken_count, self.embedding_dim, padding_idx=0)
		self.Etail = torch.nn.Embedding(self.etoken_count, self.embedding_dim, padding_idx=0)
		self.Ehead = torch.nn.Embedding(self.etoken_count, self.embedding_dim, padding_idx=0)
		self.Rt = torch.nn.Embedding(self.rtoken_count, self.embedding_dim, padding_idx=0)


		# self.Et_im.to(device)
		# self.Rt_im.to(device)
		# self.Et_re.to(device)
		# self.Rt_re.to(device)
		self.Etail.to(device)
		self.Ehead.to(device)
		self.Rt.to(device)

		if(not initial_token_embedding):
			# torch.nn.init.normal_(self.Et_re.weight.data, 0, 0.1)
			# torch.nn.init.normal_(self.Et_im.weight.data, 0, 0.1)
			# torch.nn.init.normal_(self.Rt_re.weight.data, 0, 0.1)
			# torch.nn.init.normal_(self.Rt_im.weight.data, 0, 0.1)
			torch.nn.init.normal_(self.Etail.weight.data, 0, 0.1)
			torch.nn.init.normal_(self.Ehead.weight.data, 0, 0.1)
			torch.nn.init.normal_(self.Rt.weight.data, 0, 0.1)

		self.minimum_value = -self.embedding_dim * self.embedding_dim
		# not using these for now, will see later how these help
		self.clamp_v = clamp_v
		self.unit_reg = unit_reg
		self.reg = reg
		self.batch_norm = batch_norm

		self.dropout = torch.nn.Dropout(p = lstm_dropout)

		self.lstm_ehead = torch.nn.LSTM(input_size = embedding_dim, hidden_size=embedding_dim, num_layers = num_lstm_layers, batch_first = True)
		self.lstm_etail = torch.nn.LSTM(input_size = embedding_dim, hidden_size=embedding_dim, num_layers = num_lstm_layers, batch_first = True)
		self.lstm_r = torch.nn.LSTM(input_size = embedding_dim, hidden_size=embedding_dim, num_layers = num_lstm_layers, batch_first = True)
		
		self.lstm_ehead.to(device)
		self.lstm_etail.to(device)
		self.lstm_r.to(device)

		self.ehead_batchnorm = torch.nn.BatchNorm1d(embedding_dim, momentum=0.1, eps=1e-5)
		self.etail_batchnorm = torch.nn.BatchNorm1d(embedding_dim, momentum=0.1, eps=1e-5)
		self.relation_batchnorm = torch.nn.BatchNorm1d(embedding_dim, momentum=0.1, eps=1e-5)
		torch.nn.init.uniform_(self.ehead_batchnorm.weight)
		torch.nn.init.uniform_(self.etail_batchnorm.weight)
		torch.nn.init.uniform_(self.relation_batchnorm.weight)
		self.ehead_batchnorm.to(device)
		self.etail_batchnorm.to(device)
		self.relation_batchnorm.to(device)

	def get_mention_embedding(self, data, flag, lengths):
		"""
			returns the embedding of the mention after composing with LSTM
			flag = 0 for entity head
			flag = 1 for relation
			flag = 2 for entity tail
			data has shape Batch x max seq len
			lengths has shape just batch and contains the actual lengths(besides padding) of each element of batch
		"""
		if flag == 0:
			# real_token_embeddings = self.Et_re(data)
			# img_token_embeddings = self.Et_im(data)
			token_embeddings = self.Ehead(data)
			lstm_func = self.lstm_ehead
			norm_func = self.ehead_batchnorm
		elif flag == 1:
			# real_token_embeddings = self.Rt_re(data)
			# img_token_embeddings = self.Rt_im(data)
			token_embeddings = self.Rt(data)
			lstm_func = self.lstm_r
			norm_func = self.relation_batchnorm
		else:
			token_embeddings = self.Etail(data)
			lstm_func = self.lstm_etail
			norm_func = self.etail_batchnorm
		
		indices = (lengths-1).reshape(-1,1,1).expand(len(data),1,self.embedding_dim).cuda() 

		token_embeddings = torch.nn.utils.rnn.pack_padded_sequence(token_embeddings, lengths, batch_first=True,enforce_sorted=False)
		lstm_embeddings,_ = lstm_func(token_embeddings)
		lstm_embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_embeddings, batch_first=True)
		lstm_embeddings = lstm_embeddings.gather(1,indices).squeeze(1)
		lstm_embeddings = norm_func(lstm_embeddings)
		lstm_embeddings = self.dropout(lstm_embeddings)
		real_lstm_embeddings, img_lstm_embeddings = lstm_embeddings.chunk(2, dim=-1)

		return real_lstm_embeddings, img_lstm_embeddings

	def complex_score_e1_r_with_all_ementions(self,e1_r,e1_i,r_r,r_i,all_e2_r,all_e2_i, split = 1):
		"""
			#tail prediction
			e1_r,e1_i,r_r,r_i are tensors of shape batch x embed_dim
			all_e2 has shape # of entity mentions x embed dim
			split is used when dimension size is so big that it doesn't fit on gpu. So you split over the dimension to solve each splitted section
			returns a tensor of shape batch x # of entity mentions i.e. the score with each entity mention
		"""
	# result = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
		 # = (s_re * r_re - s_im * r_im) * o_re + (s_im * r_re + s_re * r_im) * o_im
		if split>1:
			raise NotImplementedError("splits not implemented")
		section = all_e2_i.shape[1]//split
		ans = torch.zeros((e1_r.shape[0],all_e2_r.shape[0])).cuda()
		for i in range(0,all_e2_i.shape[1],section):
			tmp1 = e1_r[:,i:i+section] * r_r[:,i:i+section] - e1_i[:,i:i+section] * r_i[:,i:i+section]
			tmp2 = e1_i[:,i:i+section] * r_r[:,i:i+section] + e1_r[:,i:i+section] * r_i[:,i:i+section]

			ans_2 = tmp1.mm(all_e2_r.transpose(0,1)) + tmp2.mm(all_e2_i.transpose(0,1))
			# tmp1 = tmp1.unsqueeze(1).expand(-1,len(all_e2_r),-1) # batch x len of e mentions x embed dim
			# tmp2 = tmp2.unsqueeze(1).expand(-1,len(all_e2_r),-1) # batch x len of e mentions x embed dim

			# all_e2_r_tmp = all_e2_r[:,i:i+section].unsqueeze(0).expand(len(e1_r),-1,-1) # batch x len of e mentions x embed dim
			# all_e2_i_tmp = all_e2_i[:,i:i+section].unsqueeze(0).expand(len(e1_r),-1,-1) # batch x len of e mentions x embed dim

			# ans += (tmp1 * all_e2_r_tmp + tmp2 * all_e2_i_tmp).sum(dim = -1) # batch x len of e mentions
		return ans_2

	def complex_score_e2_r_with_all_ementions(self,e2_r,e2_i,r_r,r_i,all_e1_r,all_e1_i, split=1):
		"""
			#head prediction
			e2_r,e2_i,r_r,r_i are tensors of shape batch x embed_dim
			all_e1 has shape # of entity mentions x embed dim
			split is used when dimension size is so big that it doesn't fit on gpu. So you split over the dimension to solve each splitted section
			returns a tensor of shape batch x # of entity mentions i.e. the score with each entity mention
		"""
	# result = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
		if split>1:
			raise NotImplementedError("splits not implemented")
		section = all_e1_i.shape[1]//split
		ans = torch.zeros((e2_r.shape[0],all_e1_r.shape[0])).cuda()
		for i in range(0,all_e1_i.shape[1],section):

			tmp1 = e2_r[:,i:i+section] * r_r[:,i:i+section] + e2_i[:,i:i+section] * r_i[:,i:i+section]
			tmp2 = e2_i[:,i:i+section] * r_r[:,i:i+section] - e2_r[:,i:i+section] * r_i[:,i:i+section]

			ans_2 = tmp1.mm(all_e1_r.transpose(0,1)) + tmp2.mm(all_e1_i.transpose(0,1))
			# tmp1 = tmp1.unsqueeze(1).expand(-1,len(all_e1_r),-1) # batch x len of e mentions x embed dim
			# tmp2 = tmp2.unsqueeze(1).expand(-1,len(all_e1_r),-1) # batch x len of e mentions x embed dim

			# all_e1_r_tmp = all_e1_r[:,i:i+section].unsqueeze(0).expand(len(e2_r),-1,-1) # batch x len of e mentions x embed dim
			# all_e1_i_tmp = all_e1_i[:,i:i+section].unsqueeze(0).expand(len(e2_r),-1,-1) # batch x len of e mentions x embed dim

			# ans += (tmp1 * all_e1_r_tmp + tmp2 * all_e1_i_tmp).sum(dim = -1) # batch x len of e mentions
		return ans_2

	def regularizer(self, s, r, o, reg_val=0):
		raise NotImplemented

	def forward():
		# think of how to do this while writing code for training
		pass

class complexLSTM_2_all_e(torch.nn.Module):
	"""
		Speciality: has 2 lstms - one for entity, one for relation
					has a separate embedding for each entity when present as a target
	"""

	def __init__(self, etoken_count, rtoken_count, entity_count, embedding_dim, num_lstm_layers = 1, clamp_v=None, reg=2,
				 batch_norm=False, unit_reg=False, has_cuda=True, initial_token_embedding = None, entity_tokens = [], relation_tokens = [], lstm_dropout=0):
		# etoken_count is assumed to be incremented by 1 to handle unk
		device = torch.device("cuda" if torch.cuda.is_available() and has_cuda else "cpu")

		super(complexLSTM_2_all_e, self).__init__()
		self.etoken_count = etoken_count
		self.rtoken_count = rtoken_count
		self.entity_count = entity_count

		self.embedding_dim = embedding_dim
		self.Et = torch.nn.Embedding(self.etoken_count, self.embedding_dim, padding_idx=0)
		self.Rt = torch.nn.Embedding(self.rtoken_count, self.embedding_dim, padding_idx=0)
		self.E_atomic = torch.nn.Embedding(self.entity_count, self.embedding_dim, sparse=False)

		self.Et.to(device)
		self.Rt.to(device)
		self.E_atomic.to(device)


		if(not initial_token_embedding):
			torch.nn.init.normal_(self.Et.weight.data, 0, 0.1)
			torch.nn.init.normal_(self.Rt.weight.data, 0, 0.1)
			torch.nn.init.normal_(self.E_atomic.weight.data, 0, 0.1)


		self.minimum_value = -self.embedding_dim * self.embedding_dim
		# not using these for now, will see later how these help
		self.clamp_v = clamp_v
		self.unit_reg = unit_reg
		self.reg = reg
		self.batch_norm = batch_norm

		self.dropout = torch.nn.Dropout(p = lstm_dropout)

		self.lstm_e = torch.nn.LSTM(input_size = embedding_dim, hidden_size=embedding_dim, num_layers = num_lstm_layers, batch_first = True)
		self.lstm_r = torch.nn.LSTM(input_size = embedding_dim, hidden_size=embedding_dim, num_layers = num_lstm_layers, batch_first = True)
		
		self.lstm_e.to(device)
		self.lstm_r.to(device)

		self.entity_batchnorm = torch.nn.BatchNorm1d(embedding_dim, momentum=0.1, eps=1e-5)
		self.relation_batchnorm = torch.nn.BatchNorm1d(embedding_dim, momentum=0.1, eps=1e-5)
		torch.nn.init.uniform_(self.entity_batchnorm.weight)
		torch.nn.init.uniform_(self.relation_batchnorm.weight)
		self.entity_batchnorm.to(device)
		self.relation_batchnorm.to(device)

	def get_atomic_entity_embeddings(self, indices):
		embeddings = self.E_atomic(indices)
		real_embed, img_embed = embeddings.chunk(2, dim = -1)
		return real_embed, img_embed

	def get_mention_embedding(self, data, flag, lengths):
		"""
			returns the embedding of the mention after composing with LSTM
			flag = 0 for entity
			flag = 1 for relation
			data has shape Batch x max seq len
			lengths has shape just batch and contains the actual lengths(besides padding) of each element of batch
		"""
		if flag == 0:
			# real_token_embeddings = self.Et_re(data)
			# img_token_embeddings = self.Et_im(data)
			token_embeddings = self.Et(data)
			lstm_func = self.lstm_e
		else:
			# real_token_embeddings = self.Rt_re(data)
			# img_token_embeddings = self.Rt_im(data)
			token_embeddings = self.Rt(data)
			lstm_func = self.lstm_r

		
		indices = (lengths-1).reshape(-1,1,1).expand(len(data),1,self.embedding_dim).cuda() 

		# real_token_embeddings = torch.nn.utils.rnn.pack_padded_sequence(real_token_embeddings, lengths, batch_first=True,enforce_sorted=False)
		# real_lstm_embeddings,_ = lstm_func(real_token_embeddings)
		# real_lstm_embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(real_lstm_embeddings, batch_first=True)
		# real_lstm_embeddings = real_lstm_embeddings.gather(1,indices).squeeze(1)

		# img_token_embeddings = torch.nn.utils.rnn.pack_padded_sequence(img_token_embeddings, lengths, batch_first=True,enforce_sorted=False)
		# img_lstm_embeddings,_ = lstm_func(img_token_embeddings)
		# img_lstm_embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(img_lstm_embeddings, batch_first=True)
		# img_lstm_embeddings = img_lstm_embeddings.gather(1,indices).squeeze(1)
		token_embeddings = torch.nn.utils.rnn.pack_padded_sequence(token_embeddings, lengths, batch_first=True,enforce_sorted=False)
		lstm_embeddings,_ = lstm_func(token_embeddings)
		lstm_embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_embeddings, batch_first=True)
		lstm_embeddings = lstm_embeddings.gather(1,indices).squeeze(1)
		
		if flag == 0:
			norm_func = self.entity_batchnorm
		else:
			norm_func = self.relation_batchnorm

		# real_lstm_embeddings = norm_func(real_lstm_embeddings)
		# img_lstm_embeddings = norm_func(img_lstm_embeddings)
		lstm_embeddings = norm_func(lstm_embeddings)

		# real_lstm_embeddings = self.dropout(real_lstm_embeddings)
		# img_lstm_embeddings = self.dropout(img_lstm_embeddings)
		lstm_embeddings = self.dropout(lstm_embeddings)
		real_lstm_embeddings, img_lstm_embeddings = lstm_embeddings.chunk(2, dim=-1)

		return real_lstm_embeddings, img_lstm_embeddings

	def complex_score_e1_r_with_all_ementions(self,e1_r,e1_i,r_r,r_i,all_e2_r,all_e2_i, split = 1):
		"""
			#tail prediction
			e1_r,e1_i,r_r,r_i are tensors of shape batch x embed_dim
			all_e2 has shape # of entity mentions x embed dim
			split is used when dimension size is so big that it doesn't fit on gpu. So you split over the dimension to solve each splitted section
			returns a tensor of shape batch x # of entity mentions i.e. the score with each entity mention
		"""
	# result = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
		 # = (s_re * r_re - s_im * r_im) * o_re + (s_im * r_re + s_re * r_im) * o_im
		if split>1:
			raise NotImplementedError("splits not implemented")
		section = all_e2_i.shape[1]//split
		ans = torch.zeros((e1_r.shape[0],all_e2_r.shape[0])).cuda()
		for i in range(0,all_e2_i.shape[1],section):
			tmp1 = e1_r[:,i:i+section] * r_r[:,i:i+section] - e1_i[:,i:i+section] * r_i[:,i:i+section]
			tmp2 = e1_i[:,i:i+section] * r_r[:,i:i+section] + e1_r[:,i:i+section] * r_i[:,i:i+section]

			ans_2 = tmp1.mm(all_e2_r.transpose(0,1)) + tmp2.mm(all_e2_i.transpose(0,1))
			# tmp1 = tmp1.unsqueeze(1).expand(-1,len(all_e2_r),-1) # batch x len of e mentions x embed dim
			# tmp2 = tmp2.unsqueeze(1).expand(-1,len(all_e2_r),-1) # batch x len of e mentions x embed dim

			# all_e2_r_tmp = all_e2_r[:,i:i+section].unsqueeze(0).expand(len(e1_r),-1,-1) # batch x len of e mentions x embed dim
			# all_e2_i_tmp = all_e2_i[:,i:i+section].unsqueeze(0).expand(len(e1_r),-1,-1) # batch x len of e mentions x embed dim

			# ans += (tmp1 * all_e2_r_tmp + tmp2 * all_e2_i_tmp).sum(dim = -1) # batch x len of e mentions
		return ans_2

	def complex_score_e2_r_with_all_ementions(self,e2_r,e2_i,r_r,r_i,all_e1_r,all_e1_i, split=1):
		"""
			#head prediction
			e2_r,e2_i,r_r,r_i are tensors of shape batch x embed_dim
			all_e1 has shape # of entity mentions x embed dim
			split is used when dimension size is so big that it doesn't fit on gpu. So you split over the dimension to solve each splitted section
			returns a tensor of shape batch x # of entity mentions i.e. the score with each entity mention
		"""
	# result = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
		if split>1:
			raise NotImplementedError("splits not implemented")
		section = all_e1_i.shape[1]//split
		ans = torch.zeros((e2_r.shape[0],all_e1_r.shape[0])).cuda()
		for i in range(0,all_e1_i.shape[1],section):

			tmp1 = e2_r[:,i:i+section] * r_r[:,i:i+section] + e2_i[:,i:i+section] * r_i[:,i:i+section]
			tmp2 = e2_i[:,i:i+section] * r_r[:,i:i+section] - e2_r[:,i:i+section] * r_i[:,i:i+section]

			ans_2 = tmp1.mm(all_e1_r.transpose(0,1)) + tmp2.mm(all_e1_i.transpose(0,1))
			# tmp1 = tmp1.unsqueeze(1).expand(-1,len(all_e1_r),-1) # batch x len of e mentions x embed dim
			# tmp2 = tmp2.unsqueeze(1).expand(-1,len(all_e1_r),-1) # batch x len of e mentions x embed dim

			# all_e1_r_tmp = all_e1_r[:,i:i+section].unsqueeze(0).expand(len(e2_r),-1,-1) # batch x len of e mentions x embed dim
			# all_e1_i_tmp = all_e1_i[:,i:i+section].unsqueeze(0).expand(len(e2_r),-1,-1) # batch x len of e mentions x embed dim

			# ans += (tmp1 * all_e1_r_tmp + tmp2 * all_e1_i_tmp).sum(dim = -1) # batch x len of e mentions
		return ans_2

	def regularizer(self, s, r, o, reg_val=0):
		raise NotImplemented

	def forward():
		# think of how to do this while writing code for training
		pass


class PureComplex(torch.nn.Module):
	"""
		Speciality: This is a pure complex model without an lstm and atomic entity and relation embeddings
	"""

	def __init__(self, entity_count, relation_count, embedding_dim, num_lstm_layers = 1, clamp_v=None, reg=2,
				 batch_norm=False, unit_reg=False, has_cuda=True, initial_token_embedding = None, entity_tokens = [], relation_tokens = [], lstm_dropout=0):
		# etoken_count is assumed to be incremented by 1 to handle unk
		device = torch.device("cuda" if torch.cuda.is_available() and has_cuda else "cpu")

		super(PureComplex, self).__init__()
		self.entity_count = entity_count
		self.relation_count = relation_count


		self.embedding_dim = embedding_dim
		self.E_atomic = torch.nn.Embedding(self.entity_count, self.embedding_dim, sparse=True)
		self.R_atomic = torch.nn.Embedding(self.relation_count, self.embedding_dim, sparse=True)

		# self.E_atomic.to(device)
		# self.R_atomic.to(device)


		if(not initial_token_embedding):
			torch.nn.init.normal_(self.E_atomic.weight.data, 0, 0.1)
			torch.nn.init.normal_(self.R_atomic.weight.data, 0, 0.1)


		self.minimum_value = -self.embedding_dim * self.embedding_dim
		# not using these for now, will see later how these help
		self.clamp_v = clamp_v
		self.unit_reg = unit_reg
		self.reg = reg
		self.batch_norm = batch_norm

		# self.dropout = torch.nn.Dropout(p = lstm_dropout)

		# self.lstm_e = torch.nn.LSTM(input_size = embedding_dim, hidden_size=embedding_dim, num_layers = num_lstm_layers, batch_first = True)
		# self.lstm_r = torch.nn.LSTM(input_size = embedding_dim, hidden_size=embedding_dim, num_layers = num_lstm_layers, batch_first = True)
		
		# self.lstm_e.to(device)
		# self.lstm_r.to(device)

		# self.entity_batchnorm = torch.nn.BatchNorm1d(embedding_dim, momentum=0.1, eps=1e-5)
		# self.relation_batchnorm = torch.nn.BatchNorm1d(embedding_dim, momentum=0.1, eps=1e-5)
		# torch.nn.init.uniform_(self.entity_batchnorm.weight)
		# torch.nn.init.uniform_(self.relation_batchnorm.weight)
		# self.entity_batchnorm.to(device)
		# self.relation_batchnorm.to(device)

	def get_atomic_embeddings(self, indices, flag):
		"""
			flag = 0 for entity; 1 for relation
		"""
		if flag==0:
			embeddings = self.E_atomic(indices)
		else:
			embeddings = self.R_atomic(indices)
		real_embed, img_embed = embeddings.chunk(2, dim = -1)
		return real_embed, img_embed
		
	def complex_score_e1_r_with_all_ementions(self,e1_r,e1_i,r_r,r_i,all_e2_r,all_e2_i, split = 1):
		"""
			#tail prediction
			e1_r,e1_i,r_r,r_i are tensors of shape batch x embed_dim
			all_e2 has shape # of entity mentions x embed dim
			split is used when dimension size is so big that it doesn't fit on gpu. So you split over the dimension to solve each splitted section
			returns a tensor of shape batch x # of entity mentions i.e. the score with each entity mention
		"""
	# result = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
		 # = (s_re * r_re - s_im * r_im) * o_re + (s_im * r_re + s_re * r_im) * o_im
		if split>1:
			raise NotImplementedError("splits not implemented")
		section = all_e2_i.shape[1]//split
		ans = torch.zeros((e1_r.shape[0],all_e2_r.shape[0])).cuda()
		for i in range(0,all_e2_i.shape[1],section):
			tmp1 = e1_r[:,i:i+section] * r_r[:,i:i+section] - e1_i[:,i:i+section] * r_i[:,i:i+section]
			tmp2 = e1_i[:,i:i+section] * r_r[:,i:i+section] + e1_r[:,i:i+section] * r_i[:,i:i+section]

			ans_2 = tmp1.mm(all_e2_r.transpose(0,1)) + tmp2.mm(all_e2_i.transpose(0,1))
			# tmp1 = tmp1.unsqueeze(1).expand(-1,len(all_e2_r),-1) # batch x len of e mentions x embed dim
			# tmp2 = tmp2.unsqueeze(1).expand(-1,len(all_e2_r),-1) # batch x len of e mentions x embed dim

			# all_e2_r_tmp = all_e2_r[:,i:i+section].unsqueeze(0).expand(len(e1_r),-1,-1) # batch x len of e mentions x embed dim
			# all_e2_i_tmp = all_e2_i[:,i:i+section].unsqueeze(0).expand(len(e1_r),-1,-1) # batch x len of e mentions x embed dim

			# ans += (tmp1 * all_e2_r_tmp + tmp2 * all_e2_i_tmp).sum(dim = -1) # batch x len of e mentions
		return ans_2

	def complex_score_e2_r_with_all_ementions(self,e2_r,e2_i,r_r,r_i,all_e1_r,all_e1_i, split=1):
		"""
			#head prediction
			e2_r,e2_i,r_r,r_i are tensors of shape batch x embed_dim
			all_e1 has shape # of entity mentions x embed dim
			split is used when dimension size is so big that it doesn't fit on gpu. So you split over the dimension to solve each splitted section
			returns a tensor of shape batch x # of entity mentions i.e. the score with each entity mention
		"""
	# result = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
		if split>1:
			raise NotImplementedError("splits not implemented")
		section = all_e1_i.shape[1]//split
		ans = torch.zeros((e2_r.shape[0],all_e1_r.shape[0])).cuda()
		for i in range(0,all_e1_i.shape[1],section):

			tmp1 = e2_r[:,i:i+section] * r_r[:,i:i+section] + e2_i[:,i:i+section] * r_i[:,i:i+section]
			tmp2 = e2_i[:,i:i+section] * r_r[:,i:i+section] - e2_r[:,i:i+section] * r_i[:,i:i+section]

			ans_2 = tmp1.mm(all_e1_r.transpose(0,1)) + tmp2.mm(all_e1_i.transpose(0,1))
			# tmp1 = tmp1.unsqueeze(1).expand(-1,len(all_e1_r),-1) # batch x len of e mentions x embed dim
			# tmp2 = tmp2.unsqueeze(1).expand(-1,len(all_e1_r),-1) # batch x len of e mentions x embed dim

			# all_e1_r_tmp = all_e1_r[:,i:i+section].unsqueeze(0).expand(len(e2_r),-1,-1) # batch x len of e mentions x embed dim
			# all_e1_i_tmp = all_e1_i[:,i:i+section].unsqueeze(0).expand(len(e2_r),-1,-1) # batch x len of e mentions x embed dim

			# ans += (tmp1 * all_e1_r_tmp + tmp2 * all_e1_i_tmp).sum(dim = -1) # batch x len of e mentions
		return ans_2

	def regularizer(self, s, r, o, reg_val=0):
		raise NotImplemented

	def forward():
		# think of how to do this while writing code for training
		pass


class complexLSTM_2_all_e_hsoftmax(torch.nn.Module):
	"""
		Speciality: has 2 lstms - one for entity, one for relation
					has a separate embedding for each entity when present as a target
					implements hoffmax hier softmax
	"""

	def __init__(self, etoken_count, rtoken_count, entity_count, embedding_dim, entity_freq, num_lstm_layers = 1, clamp_v=None, reg=2,
				 batch_norm=False, unit_reg=False, has_cuda=True, initial_token_embedding = None, entity_tokens = [], relation_tokens = [], lstm_dropout=0):
		super(complexLSTM_2_all_e_hsoftmax, self).__init__()
		# etoken_count is assumed to be incremented by 1 to handle unk
		# entity_freq - dictionary with key as entity id and val as frequency of this entity
		# entity_count = 100
		self.logsigmoid = torch.nn.LogSigmoid()
		for i in range(entity_count):
			entity_freq[i] = i
		device = torch.device("cuda" if torch.cuda.is_available() and has_cuda else "cpu")
		self.device = device
		self.etoken_count = etoken_count
		self.rtoken_count = rtoken_count
		self.entity_count = entity_count

		self.embedding_dim = embedding_dim
		self.Et = torch.nn.Embedding(self.etoken_count, self.embedding_dim, padding_idx=0)
		self.Rt = torch.nn.Embedding(self.rtoken_count, self.embedding_dim, padding_idx=0)
		self.E_atomic = torch.nn.Embedding(self.entity_count, self.embedding_dim, padding_idx=entity_count-1, sparse=True)
		self.E_atomic.to(device)

		self.Et.to(device)
		self.Rt.to(device)


		if(not initial_token_embedding):
			torch.nn.init.normal_(self.Et.weight.data, 0, 0.1)
			torch.nn.init.normal_(self.Rt.weight.data, 0, 0.1)
			torch.nn.init.normal_(self.E_atomic.weight.data, 0, 0.1)


		self.minimum_value = -self.embedding_dim * self.embedding_dim
		# not using these for now, will see later how these help
		self.clamp_v = clamp_v
		self.unit_reg = unit_reg
		self.reg = reg
		self.batch_norm = batch_norm

		self.dropout = torch.nn.Dropout(p = lstm_dropout)

		self.lstm_e = torch.nn.LSTM(input_size = embedding_dim, hidden_size=embedding_dim, num_layers = num_lstm_layers, batch_first = True)
		self.lstm_r = torch.nn.LSTM(input_size = embedding_dim, hidden_size=embedding_dim, num_layers = num_lstm_layers, batch_first = True)
		
		self.lstm_e.to(device)
		self.lstm_r.to(device)

		self.entity_batchnorm = torch.nn.BatchNorm1d(embedding_dim, momentum=0.1, eps=1e-5)
		self.relation_batchnorm = torch.nn.BatchNorm1d(embedding_dim, momentum=0.1, eps=1e-5)
		torch.nn.init.uniform_(self.entity_batchnorm.weight)
		torch.nn.init.uniform_(self.relation_batchnorm.weight)
		self.entity_batchnorm.to(device)
		self.relation_batchnorm.to(device)
		# self.hsoftmax = pickle.load(open("/home/mayank/olpbench/hsoftmax/random_depth-10.pkl",'rb'))
		# self.hsoftmax = pickle.load(open("/home/mayank/olpbench/hsoftmax/atomic-entities_bknn_depth-10.pkl",'rb'))
		self.hsoftmax = pickle.load(open("/home/mayank/olpbench/hsoftmax/maxfreq-relation_depth-10.pkl",'rb'))

	def get_all_tree_nodes_debug(self, inputs, batch_shared_entities, labels):
		"""
			example:
				inputs:
					[head_inputs, tail_inputs]:
						head_inputs[0] is num_samples_for_head x 1 -> contains r
						head_inputs[1] is num_samples_for_head x 1 -> contains e2
						tail_inputs[0] is num_samples_for_tail x 1 -> contains e1
						tail_inputs[1] is num_samples_for_tail x 1 -> contains r

				batch_shared_entities = [10,231,23,23,1] (shape n)
				labels = [  [1,0,0,0,0]
							[0,1,0,0,0]
							[0,0,1,0,0]
							[0,0,0,1,0]
							[0,0,1,0,1]
							[0,0,0,1,1]  Batch x n
				]


			batch_shared_entities : tensor of shape (n)
			labels: binary matrix of shape m x n. tells which is the correct label for mth row (index into batch_shared_entities)

			returns:
				for queries with multiple labels, returns length of common prefix of all labels
				head:  [12,12,3,...]
				tail:  [12,12,3,...]  # can be of different lengths because different queries have multiple labels for head and tail
		"""
		def common_prefix_lengths(a):
			"""
				returns an int -> length of common prefix in all strings in lis
			"""
			size = len(a) 
			if (size == 0): 
				return "" 
			if (size == 1): 
				return a[0] 
			a.sort() 
			end = min(len(a[0]), len(a[size - 1])) 
			i = 0
			while (i < end and 
				   a[0][i] == a[size - 1][i]): 
				i += 1
			pre = a[0][0: i] 
			return len(pre) 

		head = []
		tail = []

		head_input0      = []
		head_input1      = []
		head_mask   = []
		head_node_indices = []
		tail_input0      = []
		tail_input1      = []
		tail_mask   = []
		tail_node_indices = []

		label_counter = 0
		for mode,model_inputs in zip(["head","tail"],inputs):
			if model_inputs==None:
				continue
			if mode=="head":
				tree_nodes_indices = head_node_indices
				mask = head_mask
				input0 = head_input0
				input1 = head_input1
				debug_list = head
			else:
				tree_nodes_indices = tail_node_indices
				mask = tail_mask
				input0 = tail_input0
				input1 = tail_input1
				debug_list = tail
			for counter in range(len(model_inputs[0])):
				labels_indices = labels[label_counter].nonzero().squeeze()
				empty_flag = labels_indices.nelement()==0
				if empty_flag:
					tree_nodes_indices.append(self.hsoftmax.default_node_indices)
					mask.append(self.hsoftmax.default_mask)
					input0.append([model_inputs[0][counter].item()])
					input1.append([model_inputs[1][counter].item()])
				else:
					if labels_indices.nelement()==1:
						label_index = labels_indices.item()
						correct_labels = [batch_shared_entities[label_index]]
					else:
						labels_indices = labels_indices.tolist()[:10] # taking 10 elements just in case kuch zyada hi aa gaya
						correct_labels = batch_shared_entities[labels_indices]
					if len(correct_labels)>1:
						masks = []
						for correct_label in correct_labels:
							tmp = ""
							correct_label = correct_label.item()
							for x in self.hsoftmax.mask_for_e[correct_label]:
								if x==0:
									tmp+="0"
								elif x==1:
									tmp+="1"
								elif x==-1:
									tmp+="2"
								else:
									raise Exception('error!')
							masks.append(tmp)            
						debug_list.append(common_prefix_lengths(masks))

					for correct_label in correct_labels:
						correct_label = correct_label.item()
						tree_nodes_indices.append(self.hsoftmax.node_indices_for_e[correct_label])
						mask.append(self.hsoftmax.mask_for_e[correct_label])            
						input0.append([model_inputs[0][counter].item()]) #this weird list is there to match expectations in training loop
						input1.append([model_inputs[1][counter].item()])
				label_counter+=1

		return head, tail



	def get_all_tree_nodes(self, inputs, batch_shared_entities, labels):
		"""
			example:
				inputs:
					[head_inputs, tail_inputs]:
						head_inputs[0] is num_samples_for_head x 1 -> contains r
						head_inputs[1] is num_samples_for_head x 1 -> contains e2
						tail_inputs[0] is num_samples_for_tail x 1 -> contains e1
						tail_inputs[1] is num_samples_for_tail x 1 -> contains r

				batch_shared_entities = [10,231,23,23,1] (shape n)
				labels = [  [1,0,0,0,0]
							[0,1,0,0,0]
							[0,0,1,0,0]
							[0,0,0,1,0]
							[0,0,1,0,1]
							[0,0,0,1,1]  Batch x n
				]


			batch_shared_entities : tensor of shape (n)
			labels: binary matrix of shape m x n. tells which is the correct label for mth row (index into batch_shared_entities)

			returns:
				head_inputs     = as present in the input but some elements duplicated because of multiple labels for them
				tree_nodes_head = [real, imaginary] embedding for the tree nodes in the path of correct entity. 
									shape m x max_tree_depth x D
				head_mask       = tensor of 1, -1 or 0. tells whether left, right or padding elements. shape m x max_tree_depth
				
				tail_inputs
				tree_nodes_tail
				tail_mask
		"""
		head_input0      = []
		head_input1      = []
		head_mask   = []
		head_node_indices = []
		tail_input0      = []
		tail_input1      = []
		tail_mask   = []
		tail_node_indices = []

		label_counter = 0
		for mode,model_inputs in zip(["head","tail"],inputs):
			if model_inputs==None:
				continue
			if mode=="head":
				tree_nodes_indices = head_node_indices
				mask = head_mask
				input0 = head_input0
				input1 = head_input1
			else:
				tree_nodes_indices = tail_node_indices
				mask = tail_mask
				input0 = tail_input0
				input1 = tail_input1
			for counter in range(len(model_inputs[0])):
				labels_indices = labels[label_counter].nonzero().squeeze()
				empty_flag = labels_indices.nelement()==0
				if empty_flag:
					tree_nodes_indices.append(self.hsoftmax.default_node_indices)
					mask.append(self.hsoftmax.default_mask)
					input0.append([model_inputs[0][counter].item()])
					input1.append([model_inputs[1][counter].item()])
				else:
					if labels_indices.nelement()==1:
						label_index = labels_indices.item()
						correct_labels = [batch_shared_entities[label_index]]
					else:
						labels_indices = labels_indices.tolist()[:10] # taking 10 elements just in case kuch zyada hi aa gaya
						correct_labels = batch_shared_entities[labels_indices]
					for correct_label in correct_labels:
						correct_label = correct_label.item()
						tree_nodes_indices.append(self.hsoftmax.node_indices_for_e[correct_label])
						mask.append(self.hsoftmax.mask_for_e[correct_label])            
						input0.append([model_inputs[0][counter].item()]) #this weird list is there to match expectations in training loop
						input1.append([model_inputs[1][counter].item()])
				label_counter+=1
		head_mask = torch.tensor(head_mask, device = self.device)
		head_inputs = None
		if(len(head_input0)>0):
			head_inputs = [torch.tensor(head_input0,dtype=torch.int32), torch.tensor(head_input1,dtype=torch.int32)]
		tail_mask = torch.tensor(tail_mask, device = self.device)
		tail_inputs = None
		if(len(tail_input0)>0):
			tail_inputs = [torch.tensor(tail_input0,dtype=torch.int32), torch.tensor(tail_input1,dtype=torch.int32)]

		head_node_indices = torch.tensor(head_node_indices, device=self.device)
		head_nodes = [None,None]
		head_nodes[0], head_nodes[1] = self.E_atomic(head_node_indices).chunk(2,dim=-1)
		tail_node_indices = torch.tensor(tail_node_indices, device=self.device)
		tail_nodes = [None,None]
		tail_nodes[0], tail_nodes[1] = self.E_atomic(tail_node_indices).chunk(2,dim=-1)

		return head_inputs, head_nodes, head_mask, tail_inputs, tail_nodes, tail_mask	

	def get_tree_nodes(self, batch_shared_entities, labels, num_samples_for_head):
		"""
			example:
				batch_shared_entities = [10,231,23,23,1] (shape n)
				labels = [  [1,0,0,0,0]
							[0,1,0,0,0]
							[0,0,1,0,0]
							[0,0,0,1,0]
							[0,0,0,0,1]
							[0,0,0,0,1]  Batch x n
				]


			batch_shared_entities : tensor of shape (n)
			labels: binary matrix of shape m x n. tells which is the correct label for mth row (index into batch_shared_entities)
			num_samples_for_head: these many out of m are for head

			returns:
				tree_nodes_head = [real, imaginary] embedding for the tree nodes in the path of correct entity. shape m x max_tree_depth x D
				
				head_mask       = tensor of 1, -1 or 0. tells whether left, right or padding elements. shape m x max_tree_depth
				

				tree_nodes_tail
				tail_mask
		"""

		tree_nodes_indices = []
		mask = []
		maxlen = 0
		for i in range(labels.shape[0]):
			labels_indices = labels[i].nonzero().squeeze()
			empty_flag = labels_indices.nelement()==0

			# tree_nodes_indices.append([])
			# mask.append([])

			if empty_flag:
				tree_nodes_indices.append(self.hsoftmax.default_node_indices)
				mask.append(self.hsoftmax.default_mask)
			else:
				# import pdb
				# pdb.set_trace()
				if labels_indices.nelement()==1:
					label_index = labels_indices.item()
					correct_label = batch_shared_entities[label_index].item()
				else:
					choice = random.randint(0,len(labels_indices)-1)
					label_index = labels_indices[choice].item()
					correct_label = batch_shared_entities[label_index].item()
				tree_nodes_indices.append(self.hsoftmax.node_indices_for_e[correct_label])
				mask.append(self.hsoftmax.mask_for_e[correct_label])

		mask = torch.tensor(mask, device=self.device)
		tree_nodes_indices = torch.tensor(tree_nodes_indices, device=self.device)
		tree_nodes = [None,None]
		tree_nodes[0], tree_nodes[1] = self.E_atomic(tree_nodes_indices).chunk(2,dim=-1)


		tree_nodes_head = [tree_nodes[0][:num_samples_for_head],tree_nodes[1][:num_samples_for_head]]
		tree_nodes_tail = [tree_nodes[0][num_samples_for_head:],tree_nodes[1][num_samples_for_head:]]
		head_mask = mask[:num_samples_for_head]
		tail_mask = mask[num_samples_for_head:]

		return tree_nodes_head, head_mask, tree_nodes_tail, tail_mask

	def test_query(self, this_e1_real, this_e1_img, this_r_real, this_r_img, this_e2_real, this_e2_img, mode, cut_mask, cut_nodes_real, cut_nodes_img):
		if mode=="tail":
			scores = self.complex_score_e1_r_with_given_ementions(this_e1_real.unsqueeze(0).expand(len(cut_mask),-1),this_e1_img.unsqueeze(0).expand(len(cut_mask),-1),this_r_real.unsqueeze(0).expand(len(cut_mask),-1),this_r_img.unsqueeze(0).expand(len(cut_mask),-1),cut_nodes_real,cut_nodes_img) # shape -> #node_at_cut x cut_depth
		else:
			scores = self.complex_score_e2_r_with_given_ementions(this_e2_real.unsqueeze(0).expand(len(cut_mask),-1),this_e2_img.unsqueeze(0).expand(len(cut_mask),-1),this_r_real.unsqueeze(0).expand(len(cut_mask),-1),this_r_img.unsqueeze(0).expand(len(cut_mask),-1),cut_nodes_real,cut_nodes_img) # shape -> #node_at_cut x cut_depth
		scores = self.logsigmoid(scores*cut_mask).sum(dim=-1) # shape -> #nodes_at_cut
		# import pdb
		# pdb.set_trace()
		best_cut_node = self.hsoftmax.nodes_at_cut[scores.argmax().item()]
		e_to_check    = self.hsoftmax.e_for_cut_nodes[best_cut_node]
		# return e_to_check


		nodes_indices = []
		mask          = []
		for i in e_to_check:
			# pakka +1 ayega? nai ayega
			nodes_indices.append(self.hsoftmax.node_indices_for_e[i][self.hsoftmax.cut_depth:])
			mask.append(self.hsoftmax.mask_for_e[i][self.hsoftmax.cut_depth:])
		mask          = torch.tensor(mask, device=self.device)
		nodes_indices = torch.tensor(nodes_indices, device=self.device)
		nodes_real, nodes_img = self.E_atomic(nodes_indices).chunk(2,dim=-1)
		if mode=="tail":
			scores = self.complex_score_e1_r_with_given_ementions(this_e1_real.unsqueeze(0).expand(len(mask),-1),this_e1_img.unsqueeze(0).expand(len(mask),-1),this_r_real.unsqueeze(0).expand(len(mask),-1),this_r_img.unsqueeze(0).expand(len(mask),-1),nodes_real,nodes_img) # shape -> #node_at_cut x remaining depth
		else:
			scores = self.complex_score_e2_r_with_given_ementions(this_e2_real.unsqueeze(0).expand(len(mask),-1),this_e2_img.unsqueeze(0).expand(len(mask),-1),this_r_real.unsqueeze(0).expand(len(mask),-1),this_r_img.unsqueeze(0).expand(len(mask),-1),nodes_real,nodes_img) # shape -> #node_at_cut x remaining depth
		scores = self.logsigmoid(scores*mask).sum(dim=-1) # shape -> #nodes_at_cut

		simi             = torch.ones(self.entity_count, device=self.device) * -99999
		simi[e_to_check] = scores
		return simi

	def test_query_debug(self, this_e1_real, this_e1_img, this_r_real, this_r_img, this_e2_real, this_e2_img, mode, cut_mask, cut_nodes_real, cut_nodes_img, gold_entities = []):
		if mode=="tail":
			scores = self.complex_score_e1_r_with_given_ementions(this_e1_real.unsqueeze(0).expand(len(cut_mask),-1),this_e1_img.unsqueeze(0).expand(len(cut_mask),-1),this_r_real.unsqueeze(0).expand(len(cut_mask),-1),this_r_img.unsqueeze(0).expand(len(cut_mask),-1),cut_nodes_real,cut_nodes_img) # shape -> #node_at_cut x cut_depth
		else:
			scores = self.complex_score_e2_r_with_given_ementions(this_e2_real.unsqueeze(0).expand(len(cut_mask),-1),this_e2_img.unsqueeze(0).expand(len(cut_mask),-1),this_r_real.unsqueeze(0).expand(len(cut_mask),-1),this_r_img.unsqueeze(0).expand(len(cut_mask),-1),cut_nodes_real,cut_nodes_img) # shape -> #node_at_cut x cut_depth
		scores = self.logsigmoid(scores*cut_mask).sum(dim=-1) # shape -> #nodes_at_cut
		# import pdb
		# pdb.set_trace()
		best_cut_node = self.hsoftmax.nodes_at_cut[scores.argmax().item()]
		e_to_check    = self.hsoftmax.e_for_cut_nodes[best_cut_node]
		# return e_to_check


		nodes_indices = []
		mask          = []
		for i in e_to_check:
			# pakka +1 ayega? nai ayega
			nodes_indices.append(self.hsoftmax.node_indices_for_e[i])
			mask.append(self.hsoftmax.mask_for_e[i])

		for i in gold_entities:
			nodes_indices.append(self.hsoftmax.node_indices_for_e[i])
			mask.append(self.hsoftmax.mask_for_e[i])


		mask          = torch.tensor(mask, device=self.device)
		nodes_indices = torch.tensor(nodes_indices, device=self.device)
		nodes_real, nodes_img = self.E_atomic(nodes_indices).chunk(2,dim=-1)
		if mode=="tail":
			scores = self.complex_score_e1_r_with_given_ementions(this_e1_real.unsqueeze(0).expand(len(mask),-1),this_e1_img.unsqueeze(0).expand(len(mask),-1),this_r_real.unsqueeze(0).expand(len(mask),-1),this_r_img.unsqueeze(0).expand(len(mask),-1),nodes_real,nodes_img) # shape -> #node_at_cut x remaining depth
		else:
			scores = self.complex_score_e2_r_with_given_ementions(this_e2_real.unsqueeze(0).expand(len(mask),-1),this_e2_img.unsqueeze(0).expand(len(mask),-1),this_r_real.unsqueeze(0).expand(len(mask),-1),this_r_img.unsqueeze(0).expand(len(mask),-1),nodes_real,nodes_img) # shape -> #node_at_cut x remaining depth
		scores = self.logsigmoid(scores*mask).sum(dim=-1) # shape -> #nodes_at_cut


		if scores.argmax().item()<len(e_to_check):
			return torch.tensor(2.).cuda()
		else:
			return torch.tensor(1.).cuda()


		# simi             = torch.ones(self.entity_count, device=self.device) * -99999
		# simi[e_to_check] = scores
		# return simi

	# def tree_traversal_test_e1_r(self,e1_r,e1_i,r_r,r_i, root, current_score, scores):
	# 	if(root==None):
	# 		return
	# 	if(root.token != None):
	# 		scores[root.token] = current_score
	# 		return
	# 	tmp1 = e1_r * r_r - e1_i * r_i
	# 	tmp2 = e1_i * r_r + e1_r * r_i
	# 	# this_score = tmp1.mm(all_e2_r.transpose(0,1)) + tmp2.mm(all_e2_i.transpose(0,1))
	# 	this_real, this_img = self.E_atomic(torch.tensor(root.id,device=self.device)).chunk(2, dim=-1)
	# 	this_score = torch.log(torch.sigmoid(tmp1.dot(this_real) + tmp2.dot(this_img)))
	# 	self.tree_traversal_test_e1_r(e1_r,e1_i,r_r,r_i, root.left, current_score + this_score, scores)
	# 	self.tree_traversal_test_e1_r(e1_r,e1_i,r_r,r_i, root.right, current_score - this_score, scores)

	# def get_loss(self,e_r,e_i,r_r,r_i,target,mode):
	# 	"""
	# 		mode = head or tail
	# 		if head:
	# 			e_r = tail real embedding of shape dimension
	# 		if tail:
	# 			e_r = head real embedding of shape dimension
	# 		target = entity id of target entity
	# 	"""
	# 	path_to_word = self.Hsoftmax_codes[target]
	# 	loss = torch.zeros(1, requires_grad=True, dtype=torch.float, device = self.device)
	# 	root = self.Hsoftmax_root
	# 	if mode=="head":
	# 		complex_function = self.complex_score_e2_r_with_all_ementions
	# 	else:
	# 		complex_function = self.complex_score_e1_r_with_all_ementions
	# 	for i in path_to_word:
	# 		# import pdb
	# 		# pdb.set_trace()
	# 		complex_score = complex_function(e_r.unsqueeze(0),e_i.unsqueeze(0),r_r.unsqueeze(0),r_i.unsqueeze(0),root.vec_real.unsqueeze(0),root.vec_img.unsqueeze(0))
	# 		# complex_score = complex_function(e_r.unsqueeze(0),e_i.unsqueeze(0),r_r.unsqueeze(0),r_i.unsqueeze(0),root.vec_real.unsqueeze(0),root.vec_img.unsqueeze(0))
	# 		if(i=='0'):
	# 			# loss = loss + torch.log(torch.sigmoid(complex_score))
	# 			# loss = loss +  torch.log(torch.sigmoid(torch.dot(root.vec, h)))
	# 			root = root.left
	# 		else:
	# 			complex_score *= -1
	# 			# loss = loss +  torch.log(torch.sigmoid(-1*torch.dot(root.vec, h)))
	# 			root = root.right
	# 		loss = loss + torch.log(torch.sigmoid(complex_score))

	# 	loss = loss*-1
	# 	return loss.squeeze()

	def get_atomic_entity_embeddings(self, indices):
		embeddings = self.E_atomic(indices)
		real_embed, img_embed = embeddings.chunk(2, dim = -1)
		return real_embed, img_embed

	def get_mention_embedding(self, data, flag, lengths):
		"""
			returns the embedding of the mention after composing with LSTM
			flag = 0 for entity
			flag = 1 for relation
			data has shape Batch x max seq len
			lengths has shape just batch and contains the actual lengths(besides padding) of each element of batch
		"""
		if flag == 0:
			# real_token_embeddings = self.Et_re(data)
			# img_token_embeddings = self.Et_im(data)
			token_embeddings = self.Et(data)
			lstm_func = self.lstm_e
		else:
			# real_token_embeddings = self.Rt_re(data)
			# img_token_embeddings = self.Rt_im(data)
			token_embeddings = self.Rt(data)
			lstm_func = self.lstm_r

		
		indices = (lengths-1).reshape(-1,1,1).expand(len(data),1,self.embedding_dim).cuda() 

		# real_token_embeddings = torch.nn.utils.rnn.pack_padded_sequence(real_token_embeddings, lengths, batch_first=True,enforce_sorted=False)
		# real_lstm_embeddings,_ = lstm_func(real_token_embeddings)
		# real_lstm_embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(real_lstm_embeddings, batch_first=True)
		# real_lstm_embeddings = real_lstm_embeddings.gather(1,indices).squeeze(1)

		# img_token_embeddings = torch.nn.utils.rnn.pack_padded_sequence(img_token_embeddings, lengths, batch_first=True,enforce_sorted=False)
		# img_lstm_embeddings,_ = lstm_func(img_token_embeddings)
		# img_lstm_embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(img_lstm_embeddings, batch_first=True)
		# img_lstm_embeddings = img_lstm_embeddings.gather(1,indices).squeeze(1)
		token_embeddings = torch.nn.utils.rnn.pack_padded_sequence(token_embeddings, lengths, batch_first=True,enforce_sorted=False)
		lstm_embeddings,_ = lstm_func(token_embeddings)
		lstm_embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_embeddings, batch_first=True)
		lstm_embeddings = lstm_embeddings.gather(1,indices).squeeze(1)
		
		if flag == 0:
			norm_func = self.entity_batchnorm
		else:
			norm_func = self.relation_batchnorm

		# real_lstm_embeddings = norm_func(real_lstm_embeddings)
		# img_lstm_embeddings = norm_func(img_lstm_embeddings)
		lstm_embeddings = norm_func(lstm_embeddings)

		# real_lstm_embeddings = self.dropout(real_lstm_embeddings)
		# img_lstm_embeddings = self.dropout(img_lstm_embeddings)
		lstm_embeddings = self.dropout(lstm_embeddings)
		real_lstm_embeddings, img_lstm_embeddings = lstm_embeddings.chunk(2, dim=-1)

		return real_lstm_embeddings, img_lstm_embeddings

	def complex_score_e1_r_with_all_ementions(self,e1_r,e1_i,r_r,r_i,all_e2_r,all_e2_i, split = 1):
		"""
			#tail prediction
			e1_r,e1_i,r_r,r_i are tensors of shape batch x embed_dim
			all_e2 has shape # of entity mentions x embed dim
			split is used when dimension size is so big that it doesn't fit on gpu. So you split over the dimension to solve each splitted section
			returns a tensor of shape batch x # of entity mentions i.e. the score with each entity mention
		"""
		 # = (s_re * r_re - s_im * r_im) * o_re + (s_im * r_re + s_re * r_im) * o_im
		if split>1:
			raise NotImplementedError("splits not implemented")
		section = all_e2_i.shape[1]//split
		ans = torch.zeros((e1_r.shape[0],all_e2_r.shape[0])).cuda()
		for i in range(0,all_e2_i.shape[1],section):
			tmp1 = e1_r[:,i:i+section] * r_r[:,i:i+section] - e1_i[:,i:i+section] * r_i[:,i:i+section]
			tmp2 = e1_i[:,i:i+section] * r_r[:,i:i+section] + e1_r[:,i:i+section] * r_i[:,i:i+section]

			ans_2 = tmp1.mm(all_e2_r.transpose(0,1)) + tmp2.mm(all_e2_i.transpose(0,1))
			# tmp1 = tmp1.unsqueeze(1).expand(-1,len(all_e2_r),-1) # batch x len of e mentions x embed dim
			# tmp2 = tmp2.unsqueeze(1).expand(-1,len(all_e2_r),-1) # batch x len of e mentions x embed dim

			# all_e2_r_tmp = all_e2_r[:,i:i+section].unsqueeze(0).expand(len(e1_r),-1,-1) # batch x len of e mentions x embed dim
			# all_e2_i_tmp = all_e2_i[:,i:i+section].unsqueeze(0).expand(len(e1_r),-1,-1) # batch x len of e mentions x embed dim

			# ans += (tmp1 * all_e2_r_tmp + tmp2 * all_e2_i_tmp).sum(dim = -1) # batch x len of e mentions
		return ans_2

	def complex_score_e1_r_with_given_ementions(self,e1_r,e1_i,r_r,r_i,given_e2_r,given_e2_i):
		"""
			#tail prediction
			e1_r,e1_i,r_r,r_i are tensors of shape batch x embed_dim
			given_e2 has shape batch x max_tree_depth x embed dim
			returns a tensor of shape batch x max_tree_depth i.e. the score (e1,r) with each of the max_tree_depth e2 elements in its batch
		"""
		 # = (s_re * r_re - s_im * r_im) * o_re + (s_im * r_re + s_re * r_im) * o_im
		# if split>1:
		# 	raise NotImplementedError("splits not implemented")
		# section = all_e2_i.shape[1]//split
		# ans = torch.zeros((e1_r.shape[0],all_e2_r.shape[0])).cuda()
		tmp1 = e1_r * r_r - e1_i * r_i
		tmp2 = e1_i * r_r + e1_r * r_i
		ans = torch.bmm(given_e2_r, tmp1.unsqueeze(2)).squeeze(2) + torch.bmm(given_e2_i, tmp2.unsqueeze(2)).squeeze(2)
		return ans
		# for i in range(0,all_e2_i.shape[1],section):
		# 	tmp1 = e1_r[:,i:i+section] * r_r[:,i:i+section] - e1_i[:,i:i+section] * r_i[:,i:i+section]
		# 	tmp2 = e1_i[:,i:i+section] * r_r[:,i:i+section] + e1_r[:,i:i+section] * r_i[:,i:i+section]

		# 	ans_2 = tmp1.mm(all_e2_r.transpose(0,1)) + tmp2.mm(all_e2_i.transpose(0,1))
		# return ans_2

	def complex_score_e2_r_with_all_ementions(self,e2_r,e2_i,r_r,r_i,all_e1_r,all_e1_i, split=1):
		"""
			#head prediction
			e2_r,e2_i,r_r,r_i are tensors of shape batch x embed_dim
			all_e1 has shape # of entity mentions x embed dim
			split is used when dimension size is so big that it doesn't fit on gpu. So you split over the dimension to solve each splitted section
			returns a tensor of shape batch x # of entity mentions i.e. the score with each entity mention
		"""
	# result = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
		if split>1:
			raise NotImplementedError("splits not implemented")
		section = all_e1_i.shape[1]//split
		ans = torch.zeros((e2_r.shape[0],all_e1_r.shape[0])).cuda()
		for i in range(0,all_e1_i.shape[1],section):

			tmp1 = e2_r[:,i:i+section] * r_r[:,i:i+section] + e2_i[:,i:i+section] * r_i[:,i:i+section]
			tmp2 = e2_i[:,i:i+section] * r_r[:,i:i+section] - e2_r[:,i:i+section] * r_i[:,i:i+section]

			ans_2 = tmp1.mm(all_e1_r.transpose(0,1)) + tmp2.mm(all_e1_i.transpose(0,1))
			# tmp1 = tmp1.unsqueeze(1).expand(-1,len(all_e1_r),-1) # batch x len of e mentions x embed dim
			# tmp2 = tmp2.unsqueeze(1).expand(-1,len(all_e1_r),-1) # batch x len of e mentions x embed dim

			# all_e1_r_tmp = all_e1_r[:,i:i+section].unsqueeze(0).expand(len(e2_r),-1,-1) # batch x len of e mentions x embed dim
			# all_e1_i_tmp = all_e1_i[:,i:i+section].unsqueeze(0).expand(len(e2_r),-1,-1) # batch x len of e mentions x embed dim

			# ans += (tmp1 * all_e1_r_tmp + tmp2 * all_e1_i_tmp).sum(dim = -1) # batch x len of e mentions
		return ans_2

	def complex_score_e2_r_with_given_ementions(self,e2_r,e2_i,r_r,r_i,given_e1_r,given_e1_i):
		"""
			#head prediction
			e2_r,e2_i,r_r,r_i are tensors of shape batch x embed_dim
			given_e1 has shape batch x max_tree_depth x embed dim
			returns a tensor of shape batch x max_tree_depth i.e. the score (e2,r) with each of the max_tree_depth e1 elements in its batch
		"""
	# result = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
		tmp1 = e2_r * r_r + e2_i * r_i
		tmp2 = e2_i * r_r - e2_r * r_i
		ans = torch.bmm(given_e1_r, tmp1.unsqueeze(2)).squeeze(2) + torch.bmm(given_e1_i, tmp2.unsqueeze(2)).squeeze(2)
		return ans
		# for i in range(0,all_e1_i.shape[1],section):

		# 	tmp1 = e2_r[:,i:i+section] * r_r[:,i:i+section] + e2_i[:,i:i+section] * r_i[:,i:i+section]
		# 	tmp2 = e2_i[:,i:i+section] * r_r[:,i:i+section] - e2_r[:,i:i+section] * r_i[:,i:i+section]

		# 	ans_2 = tmp1.mm(all_e1_r.transpose(0,1)) + tmp2.mm(all_e1_i.transpose(0,1))
		# return ans_2

	def regularizer(self, s, r, o, reg_val=0):
		raise NotImplemented

	def forward():
		# think of how to do this while writing code for training
		pass

class complexLSTM_2_all_e_PLT(torch.nn.Module):
	"""
		Speciality: has 2 lstms - one for entity, one for relation
					has a separate embedding for each entity when present as a target
					implements Probabilistic label tree, where at each node, the embedding tells whether the label is below that node or not
	"""

	def __init__(self, etoken_count, rtoken_count, entity_count, embedding_dim, entity_freq, num_lstm_layers = 1, clamp_v=None, reg=2,
				 batch_norm=False, unit_reg=False, has_cuda=True, initial_token_embedding = None, entity_tokens = [], relation_tokens = [], lstm_dropout=0):
		super(complexLSTM_2_all_e_PLT, self).__init__()
		# etoken_count is assumed to be incremented by 1 to handle unk
		# entity_freq - dictionary with key as entity id and val as frequency of this entity
		# entity_count = 100
		self.logsigmoid = torch.nn.LogSigmoid()
		for i in range(entity_count):
			entity_freq[i] = i
		device = torch.device("cuda" if torch.cuda.is_available() and has_cuda else "cpu")
		self.device = device
		self.etoken_count = etoken_count
		self.rtoken_count = rtoken_count
		self.entity_count = entity_count

		self.embedding_dim = embedding_dim
		
		self.Et = torch.nn.Embedding(self.etoken_count, self.embedding_dim, padding_idx=0)
		self.Rt = torch.nn.Embedding(self.rtoken_count, self.embedding_dim, padding_idx=0)
		self.E_atomic = torch.nn.Embedding(2*self.entity_count, self.embedding_dim, padding_idx=2*entity_count-1, sparse=True)
		self.E_atomic.to(device)

		self.Et.to(device)
		self.Rt.to(device)


		if(not initial_token_embedding):
			torch.nn.init.normal_(self.Et.weight.data, 0, 0.1)
			torch.nn.init.normal_(self.Rt.weight.data, 0, 0.1)
			torch.nn.init.normal_(self.E_atomic.weight.data, 0, 0.1)


		self.minimum_value = -self.embedding_dim * self.embedding_dim
		# not using these for now, will see later how these help
		self.clamp_v = clamp_v
		self.unit_reg = unit_reg
		self.reg = reg
		self.batch_norm = batch_norm

		self.dropout = torch.nn.Dropout(p = lstm_dropout)

		self.lstm_e = torch.nn.LSTM(input_size = embedding_dim, hidden_size=embedding_dim, num_layers = num_lstm_layers, batch_first = True)
		self.lstm_r = torch.nn.LSTM(input_size = embedding_dim, hidden_size=embedding_dim, num_layers = num_lstm_layers, batch_first = True)
		
		self.lstm_e.to(device)
		self.lstm_r.to(device)

		self.entity_batchnorm = torch.nn.BatchNorm1d(embedding_dim, momentum=0.1, eps=1e-5)
		self.relation_batchnorm = torch.nn.BatchNorm1d(embedding_dim, momentum=0.1, eps=1e-5)
		torch.nn.init.uniform_(self.entity_batchnorm.weight)
		torch.nn.init.uniform_(self.relation_batchnorm.weight)
		self.entity_batchnorm.to(device)
		self.relation_batchnorm.to(device)
		# self.hsoftmax = pickle.load(open("/home/mayank/olpbench/hsoftmax/random_depth-10.pkl",'rb'))
		# self.hsoftmax = pickle.load(open("/home/mayank/olpbench/hsoftmax/atomic-entities_bknn_depth-10.pkl",'rb'))
		self.hsoftmax = pickle.load(open("/home/mayank/olpbench/PLT/PLT-maxfreq-relation_depth-10.pkl",'rb'))
		self.nodes_above_cut = set()
		for node in tqdm(self.hsoftmax.nodes_at_cut, desc="negative nodes (1/4):"):
			lis = self.hsoftmax.node_indices_for_t[node]
			for x in lis:
				self.nodes_above_cut.add(x)

		#step 2: get all nodes for each cut node
		self.nodes_below_cut = {} #{<cut node index>:<set of all nodes below this>,...}
		for node in tqdm(self.hsoftmax.nodes_at_cut, desc="negative nodes (2/4):"):
			self.nodes_below_cut[node] = set()
			entities_for_cut = self.hsoftmax.e_for_cut_nodes[node]
			for entity in entities_for_cut:
				lis = self.hsoftmax.node_indices_for_e[entity][self.hsoftmax.cut_depth+1:]
				for x in lis:
					self.nodes_below_cut[node].add(x)
		
	def get_negative_nodes(self,entities):
		"""
			entities: [112,12,312,443,...]
			returns: tree_nodes_indices_neg, mask_neg type: list of list [[],[],...] (not tenorized but padded)
		"""
		#step 3: for each entity add step1 and step2 nodes minus its gold nodes
		SAMPLE = 50
		maxlen = -1
		tree_nodes_indices_neg = []
		mask_neg               = []
		for entity in entities:
			if entity == -1: # null entity
				tree_nodes_indices_neg.append([])
				mask_neg.append([])
			else:
				cut_node = self.hsoftmax.node_indices_for_e[entity][self.hsoftmax.cut_depth]
				s1 = set(random.sample(self.nodes_above_cut,SAMPLE) + random.sample(self.nodes_below_cut[cut_node],SAMPLE))

				tree_nodes_indices_neg.append(list(s1.difference(set(self.hsoftmax.node_indices_for_e[entity]).union(set([2*self.hsoftmax.entity_count-1])))))
				# tree_nodes_indices_neg.append(list(s1.difference(set(self.hsoftmax.node_indices_for_e[entity]))))

				mask_neg.append([1] * len(tree_nodes_indices_neg[-1]))
			maxlen = max(maxlen,len(mask_neg[-1]))

		#step 4: pad it!
		for entity in range(len(entities)):
			for i in range(len(mask_neg[entity]),maxlen):
				mask_neg[entity].append(0)
				tree_nodes_indices_neg[entity].append(2*self.hsoftmax.entity_count-1)

		return tree_nodes_indices_neg, mask_neg

	def get_tree_nodes(self, batch_shared_entities, labels, num_samples_for_head):
		"""
			example:
				batch_shared_entities = [10,231,23,23,1] (shape n)
				labels = [  [1,0,0,0,0]
							[0,1,0,0,0]
							[0,0,1,0,0]
							[0,0,0,1,0]
							[0,0,0,0,1]
							[0,0,0,0,1]  Batch x n
				]


			batch_shared_entities : tensor of shape (n)
			labels: binary matrix of shape m x n. tells which is the correct label for mth row (index into batch_shared_entities)
			num_samples_for_head: these many out of m are for head

			returns:
				tree_nodes_head = [real, imaginary] embedding for the tree nodes in the path of correct entity. shape m x max_tree_depth x D
				head_mask       = tensor of 1, 0. padding elements. shape m x max_tree_depth
				tree_nodes_tail
				tail_mask

				tree_nodes_head_neg = [real, imaginary] embedding for the tree nodes which are negative samples. shape m x max_tree_depth x D
				head_mask_neg       = tensor of 1, 0. padding elements. shape m x max_tree_depth
				tree_nodes_tail_neg
				tail_mask_neg
		"""

		tree_nodes_indices = []
		mask = []
		tree_nodes_indices_neg = []
		mask_neg = []
		maxlen = 0
		entities = []
		for i in range(labels.shape[0]):
			labels_indices = labels[i].nonzero().squeeze()
			empty_flag = labels_indices.nelement()==0

			# tree_nodes_indices.append([])
			# mask.append([])

			if empty_flag:
				tree_nodes_indices.append(self.hsoftmax.default_node_indices)
				mask.append(self.hsoftmax.default_mask)
				# tree_nodes_indices_neg.append(self.hsoftmax.default_node_indices)
				# mask_neg.append(self.hsoftmax.default_mask)
				entities.append(-1)
			else:
				# import pdb
				# pdb.set_trace()
				if labels_indices.nelement()==1:
					label_index = labels_indices.item()
					correct_label = batch_shared_entities[label_index].item()
				else:
					choice = random.randint(0,len(labels_indices)-1)
					label_index = labels_indices[choice].item()
					correct_label = batch_shared_entities[label_index].item()
				tree_nodes_indices.append(self.hsoftmax.node_indices_for_e[correct_label])
				mask.append(self.hsoftmax.mask_for_e[correct_label])
				entities.append(correct_label)
				# tree_nodes_indices_neg.append(self.hsoftmax.neg_node_indices_for_e[correct_label])
				# mask_neg.append(self.hsoftmax.neg_mask_for_e[correct_label])
		tree_nodes_indices_neg, mask_neg = self.get_negative_nodes(entities)
		mask = torch.tensor(mask, device=self.device)
		tree_nodes_indices = torch.tensor(tree_nodes_indices, device=self.device)
		mask_neg = torch.tensor(mask_neg, device=self.device)
		tree_nodes_indices_neg = torch.tensor(tree_nodes_indices_neg, device=self.device)
		# tree_nodes_indices_neg = tree_nodes_indices_neg[:,:1]
		# mask_neg = mask_neg[:,:1]
		# import pdb
		# pdb.set_trace()
		tree_nodes = [None,None]
		tree_nodes[0], tree_nodes[1] = self.E_atomic(tree_nodes_indices).chunk(2,dim=-1)
		tree_nodes_neg = [None,None]
		tree_nodes_neg[0], tree_nodes_neg[1] = self.E_atomic(tree_nodes_indices_neg).chunk(2,dim=-1)

		tree_nodes_head = [tree_nodes[0][:num_samples_for_head],tree_nodes[1][:num_samples_for_head]]
		tree_nodes_tail = [tree_nodes[0][num_samples_for_head:],tree_nodes[1][num_samples_for_head:]]
		head_mask = mask[:num_samples_for_head]
		tail_mask = mask[num_samples_for_head:]
		tree_nodes_head_neg = [tree_nodes_neg[0][:num_samples_for_head],tree_nodes_neg[1][:num_samples_for_head]]
		tree_nodes_tail_neg = [tree_nodes_neg[0][num_samples_for_head:],tree_nodes_neg[1][num_samples_for_head:]]
		head_mask_neg = mask_neg[:num_samples_for_head]
		tail_mask_neg = mask_neg[num_samples_for_head:]

		return tree_nodes_head, head_mask, tree_nodes_tail, tail_mask, tree_nodes_head_neg, head_mask_neg, tree_nodes_tail_neg, tail_mask_neg

	def test_query(self, this_e1_real, this_e1_img, this_r_real, this_r_img, this_e2_real, this_e2_img, mode, cut_mask, cut_nodes_real, cut_nodes_img):
		if mode=="tail":
			scores = self.complex_score_e1_r_with_given_ementions(this_e1_real.unsqueeze(0).expand(len(cut_mask),-1),this_e1_img.unsqueeze(0).expand(len(cut_mask),-1),this_r_real.unsqueeze(0).expand(len(cut_mask),-1),this_r_img.unsqueeze(0).expand(len(cut_mask),-1),cut_nodes_real,cut_nodes_img) # shape -> #node_at_cut x cut_depth
		else:
			scores = self.complex_score_e2_r_with_given_ementions(this_e2_real.unsqueeze(0).expand(len(cut_mask),-1),this_e2_img.unsqueeze(0).expand(len(cut_mask),-1),this_r_real.unsqueeze(0).expand(len(cut_mask),-1),this_r_img.unsqueeze(0).expand(len(cut_mask),-1),cut_nodes_real,cut_nodes_img) # shape -> #node_at_cut x cut_depth
		scores = self.logsigmoid(scores*cut_mask).sum(dim=-1) # shape -> #nodes_at_cut
		# import pdb
		# pdb.set_trace()
		best_cut_node = self.hsoftmax.nodes_at_cut[scores.argmax().item()]
		e_to_check    = self.hsoftmax.e_for_cut_nodes[best_cut_node]
		# return e_to_check


		nodes_indices = []
		mask          = []
		for i in e_to_check:
			# pakka +1 ayega? nai ayega
			nodes_indices.append(self.hsoftmax.node_indices_for_e[i][self.hsoftmax.cut_depth+1:])
			mask.append(self.hsoftmax.mask_for_e[i][self.hsoftmax.cut_depth+1:])
		mask          = torch.tensor(mask, device=self.device)
		nodes_indices = torch.tensor(nodes_indices, device=self.device)
		nodes_real, nodes_img = self.E_atomic(nodes_indices).chunk(2,dim=-1)
		if mode=="tail":
			scores = self.complex_score_e1_r_with_given_ementions(this_e1_real.unsqueeze(0).expand(len(mask),-1),this_e1_img.unsqueeze(0).expand(len(mask),-1),this_r_real.unsqueeze(0).expand(len(mask),-1),this_r_img.unsqueeze(0).expand(len(mask),-1),nodes_real,nodes_img) # shape -> #node_at_cut x remaining depth
		else:
			scores = self.complex_score_e2_r_with_given_ementions(this_e2_real.unsqueeze(0).expand(len(mask),-1),this_e2_img.unsqueeze(0).expand(len(mask),-1),this_r_real.unsqueeze(0).expand(len(mask),-1),this_r_img.unsqueeze(0).expand(len(mask),-1),nodes_real,nodes_img) # shape -> #node_at_cut x remaining depth
		scores = self.logsigmoid(scores*mask).sum(dim=-1) # shape -> #nodes_at_cut

		simi             = torch.ones(self.entity_count, device=self.device) * -99999
		simi[e_to_check] = scores
		return simi

	def test_query_debug(self, this_e1_real, this_e1_img, this_r_real, this_r_img, this_e2_real, this_e2_img, mode, cut_mask, cut_nodes_real, cut_nodes_img, gold_entities = []):
		if mode=="tail":
			scores = self.complex_score_e1_r_with_given_ementions(this_e1_real.unsqueeze(0).expand(len(cut_mask),-1),this_e1_img.unsqueeze(0).expand(len(cut_mask),-1),this_r_real.unsqueeze(0).expand(len(cut_mask),-1),this_r_img.unsqueeze(0).expand(len(cut_mask),-1),cut_nodes_real,cut_nodes_img) # shape -> #node_at_cut x cut_depth
		else:
			scores = self.complex_score_e2_r_with_given_ementions(this_e2_real.unsqueeze(0).expand(len(cut_mask),-1),this_e2_img.unsqueeze(0).expand(len(cut_mask),-1),this_r_real.unsqueeze(0).expand(len(cut_mask),-1),this_r_img.unsqueeze(0).expand(len(cut_mask),-1),cut_nodes_real,cut_nodes_img) # shape -> #node_at_cut x cut_depth
		scores = self.logsigmoid(scores*cut_mask).sum(dim=-1) # shape -> #nodes_at_cut
		# import pdb
		# pdb.set_trace()
		best_cut_node = self.hsoftmax.nodes_at_cut[scores.argmax().item()]
		e_to_check    = self.hsoftmax.e_for_cut_nodes[best_cut_node]
		# return e_to_check


		nodes_indices = []
		mask          = []
		for i in e_to_check:
			# pakka +1 ayega? nai ayega
			nodes_indices.append(self.hsoftmax.node_indices_for_e[i])
			mask.append(self.hsoftmax.mask_for_e[i])

		for i in gold_entities:
			nodes_indices.append(self.hsoftmax.node_indices_for_e[i])
			mask.append(self.hsoftmax.mask_for_e[i])


		mask          = torch.tensor(mask, device=self.device)
		nodes_indices = torch.tensor(nodes_indices, device=self.device)
		nodes_real, nodes_img = self.E_atomic(nodes_indices).chunk(2,dim=-1)
		if mode=="tail":
			scores = self.complex_score_e1_r_with_given_ementions(this_e1_real.unsqueeze(0).expand(len(mask),-1),this_e1_img.unsqueeze(0).expand(len(mask),-1),this_r_real.unsqueeze(0).expand(len(mask),-1),this_r_img.unsqueeze(0).expand(len(mask),-1),nodes_real,nodes_img) # shape -> #node_at_cut x remaining depth
		else:
			scores = self.complex_score_e2_r_with_given_ementions(this_e2_real.unsqueeze(0).expand(len(mask),-1),this_e2_img.unsqueeze(0).expand(len(mask),-1),this_r_real.unsqueeze(0).expand(len(mask),-1),this_r_img.unsqueeze(0).expand(len(mask),-1),nodes_real,nodes_img) # shape -> #node_at_cut x remaining depth
		scores = self.logsigmoid(scores*mask).sum(dim=-1) # shape -> #nodes_at_cut


		if scores.argmax().item()<len(e_to_check):
			return torch.tensor(2.).cuda()
		else:
			return torch.tensor(1.).cuda()


		# simi             = torch.ones(self.entity_count, device=self.device) * -99999
		# simi[e_to_check] = scores
		# return simi

	# def tree_traversal_test_e1_r(self,e1_r,e1_i,r_r,r_i, root, current_score, scores):
	# 	if(root==None):
	# 		return
	# 	if(root.token != None):
	# 		scores[root.token] = current_score
	# 		return
	# 	tmp1 = e1_r * r_r - e1_i * r_i
	# 	tmp2 = e1_i * r_r + e1_r * r_i
	# 	# this_score = tmp1.mm(all_e2_r.transpose(0,1)) + tmp2.mm(all_e2_i.transpose(0,1))
	# 	this_real, this_img = self.E_atomic(torch.tensor(root.id,device=self.device)).chunk(2, dim=-1)
	# 	this_score = torch.log(torch.sigmoid(tmp1.dot(this_real) + tmp2.dot(this_img)))
	# 	self.tree_traversal_test_e1_r(e1_r,e1_i,r_r,r_i, root.left, current_score + this_score, scores)
	# 	self.tree_traversal_test_e1_r(e1_r,e1_i,r_r,r_i, root.right, current_score - this_score, scores)

	# def get_loss(self,e_r,e_i,r_r,r_i,target,mode):
	# 	"""
	# 		mode = head or tail
	# 		if head:
	# 			e_r = tail real embedding of shape dimension
	# 		if tail:
	# 			e_r = head real embedding of shape dimension
	# 		target = entity id of target entity
	# 	"""
	# 	path_to_word = self.Hsoftmax_codes[target]
	# 	loss = torch.zeros(1, requires_grad=True, dtype=torch.float, device = self.device)
	# 	root = self.Hsoftmax_root
	# 	if mode=="head":
	# 		complex_function = self.complex_score_e2_r_with_all_ementions
	# 	else:
	# 		complex_function = self.complex_score_e1_r_with_all_ementions
	# 	for i in path_to_word:
	# 		# import pdb
	# 		# pdb.set_trace()
	# 		complex_score = complex_function(e_r.unsqueeze(0),e_i.unsqueeze(0),r_r.unsqueeze(0),r_i.unsqueeze(0),root.vec_real.unsqueeze(0),root.vec_img.unsqueeze(0))
	# 		# complex_score = complex_function(e_r.unsqueeze(0),e_i.unsqueeze(0),r_r.unsqueeze(0),r_i.unsqueeze(0),root.vec_real.unsqueeze(0),root.vec_img.unsqueeze(0))
	# 		if(i=='0'):
	# 			# loss = loss + torch.log(torch.sigmoid(complex_score))
	# 			# loss = loss +  torch.log(torch.sigmoid(torch.dot(root.vec, h)))
	# 			root = root.left
	# 		else:
	# 			complex_score *= -1
	# 			# loss = loss +  torch.log(torch.sigmoid(-1*torch.dot(root.vec, h)))
	# 			root = root.right
	# 		loss = loss + torch.log(torch.sigmoid(complex_score))

	# 	loss = loss*-1
	# 	return loss.squeeze()

	def get_atomic_entity_embeddings(self, indices):
		embeddings = self.E_atomic(indices)
		real_embed, img_embed = embeddings.chunk(2, dim = -1)
		return real_embed, img_embed

	def get_mention_embedding(self, data, flag, lengths):
		"""
			returns the embedding of the mention after composing with LSTM
			flag = 0 for entity
			flag = 1 for relation
			data has shape Batch x max seq len
			lengths has shape just batch and contains the actual lengths(besides padding) of each element of batch
		"""
		if flag == 0:
			# real_token_embeddings = self.Et_re(data)
			# img_token_embeddings = self.Et_im(data)
			token_embeddings = self.Et(data)
			lstm_func = self.lstm_e
		else:
			# real_token_embeddings = self.Rt_re(data)
			# img_token_embeddings = self.Rt_im(data)
			token_embeddings = self.Rt(data)
			lstm_func = self.lstm_r

		
		indices = (lengths-1).reshape(-1,1,1).expand(len(data),1,self.embedding_dim).cuda() 

		# real_token_embeddings = torch.nn.utils.rnn.pack_padded_sequence(real_token_embeddings, lengths, batch_first=True,enforce_sorted=False)
		# real_lstm_embeddings,_ = lstm_func(real_token_embeddings)
		# real_lstm_embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(real_lstm_embeddings, batch_first=True)
		# real_lstm_embeddings = real_lstm_embeddings.gather(1,indices).squeeze(1)

		# img_token_embeddings = torch.nn.utils.rnn.pack_padded_sequence(img_token_embeddings, lengths, batch_first=True,enforce_sorted=False)
		# img_lstm_embeddings,_ = lstm_func(img_token_embeddings)
		# img_lstm_embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(img_lstm_embeddings, batch_first=True)
		# img_lstm_embeddings = img_lstm_embeddings.gather(1,indices).squeeze(1)
		token_embeddings = torch.nn.utils.rnn.pack_padded_sequence(token_embeddings, lengths, batch_first=True,enforce_sorted=False)
		lstm_embeddings,_ = lstm_func(token_embeddings)
		lstm_embeddings, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_embeddings, batch_first=True)
		lstm_embeddings = lstm_embeddings.gather(1,indices).squeeze(1)
		
		if flag == 0:
			norm_func = self.entity_batchnorm
		else:
			norm_func = self.relation_batchnorm

		# real_lstm_embeddings = norm_func(real_lstm_embeddings)
		# img_lstm_embeddings = norm_func(img_lstm_embeddings)
		lstm_embeddings = norm_func(lstm_embeddings)

		# real_lstm_embeddings = self.dropout(real_lstm_embeddings)
		# img_lstm_embeddings = self.dropout(img_lstm_embeddings)
		lstm_embeddings = self.dropout(lstm_embeddings)
		real_lstm_embeddings, img_lstm_embeddings = lstm_embeddings.chunk(2, dim=-1)

		return real_lstm_embeddings, img_lstm_embeddings

	def complex_score_e1_r_with_all_ementions(self,e1_r,e1_i,r_r,r_i,all_e2_r,all_e2_i, split = 1):
		"""
			#tail prediction
			e1_r,e1_i,r_r,r_i are tensors of shape batch x embed_dim
			all_e2 has shape # of entity mentions x embed dim
			split is used when dimension size is so big that it doesn't fit on gpu. So you split over the dimension to solve each splitted section
			returns a tensor of shape batch x # of entity mentions i.e. the score with each entity mention
		"""
		 # = (s_re * r_re - s_im * r_im) * o_re + (s_im * r_re + s_re * r_im) * o_im
		if split>1:
			raise NotImplementedError("splits not implemented")
		section = all_e2_i.shape[1]//split
		ans = torch.zeros((e1_r.shape[0],all_e2_r.shape[0])).cuda()
		for i in range(0,all_e2_i.shape[1],section):
			tmp1 = e1_r[:,i:i+section] * r_r[:,i:i+section] - e1_i[:,i:i+section] * r_i[:,i:i+section]
			tmp2 = e1_i[:,i:i+section] * r_r[:,i:i+section] + e1_r[:,i:i+section] * r_i[:,i:i+section]

			ans_2 = tmp1.mm(all_e2_r.transpose(0,1)) + tmp2.mm(all_e2_i.transpose(0,1))
			# tmp1 = tmp1.unsqueeze(1).expand(-1,len(all_e2_r),-1) # batch x len of e mentions x embed dim
			# tmp2 = tmp2.unsqueeze(1).expand(-1,len(all_e2_r),-1) # batch x len of e mentions x embed dim

			# all_e2_r_tmp = all_e2_r[:,i:i+section].unsqueeze(0).expand(len(e1_r),-1,-1) # batch x len of e mentions x embed dim
			# all_e2_i_tmp = all_e2_i[:,i:i+section].unsqueeze(0).expand(len(e1_r),-1,-1) # batch x len of e mentions x embed dim

			# ans += (tmp1 * all_e2_r_tmp + tmp2 * all_e2_i_tmp).sum(dim = -1) # batch x len of e mentions
		return ans_2

	def complex_score_e1_r_with_given_ementions(self,e1_r,e1_i,r_r,r_i,given_e2_r,given_e2_i):
		"""
			#tail prediction
			e1_r,e1_i,r_r,r_i are tensors of shape batch x embed_dim
			given_e2 has shape batch x max_tree_depth x embed dim
			returns a tensor of shape batch x max_tree_depth i.e. the score (e1,r) with each of the max_tree_depth e2 elements in its batch
		"""
		 # = (s_re * r_re - s_im * r_im) * o_re + (s_im * r_re + s_re * r_im) * o_im
		# if split>1:
		# 	raise NotImplementedError("splits not implemented")
		# section = all_e2_i.shape[1]//split
		# ans = torch.zeros((e1_r.shape[0],all_e2_r.shape[0])).cuda()
		tmp1 = e1_r * r_r - e1_i * r_i
		tmp2 = e1_i * r_r + e1_r * r_i
		ans = torch.bmm(given_e2_r, tmp1.unsqueeze(2)).squeeze(2) + torch.bmm(given_e2_i, tmp2.unsqueeze(2)).squeeze(2)
		return ans
		# for i in range(0,all_e2_i.shape[1],section):
		# 	tmp1 = e1_r[:,i:i+section] * r_r[:,i:i+section] - e1_i[:,i:i+section] * r_i[:,i:i+section]
		# 	tmp2 = e1_i[:,i:i+section] * r_r[:,i:i+section] + e1_r[:,i:i+section] * r_i[:,i:i+section]

		# 	ans_2 = tmp1.mm(all_e2_r.transpose(0,1)) + tmp2.mm(all_e2_i.transpose(0,1))
		# return ans_2

	def complex_score_e2_r_with_all_ementions(self,e2_r,e2_i,r_r,r_i,all_e1_r,all_e1_i, split=1):
		"""
			#head prediction
			e2_r,e2_i,r_r,r_i are tensors of shape batch x embed_dim
			all_e1 has shape # of entity mentions x embed dim
			split is used when dimension size is so big that it doesn't fit on gpu. So you split over the dimension to solve each splitted section
			returns a tensor of shape batch x # of entity mentions i.e. the score with each entity mention
		"""
	# result = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
		if split>1:
			raise NotImplementedError("splits not implemented")
		section = all_e1_i.shape[1]//split
		ans = torch.zeros((e2_r.shape[0],all_e1_r.shape[0])).cuda()
		for i in range(0,all_e1_i.shape[1],section):

			tmp1 = e2_r[:,i:i+section] * r_r[:,i:i+section] + e2_i[:,i:i+section] * r_i[:,i:i+section]
			tmp2 = e2_i[:,i:i+section] * r_r[:,i:i+section] - e2_r[:,i:i+section] * r_i[:,i:i+section]

			ans_2 = tmp1.mm(all_e1_r.transpose(0,1)) + tmp2.mm(all_e1_i.transpose(0,1))
			# tmp1 = tmp1.unsqueeze(1).expand(-1,len(all_e1_r),-1) # batch x len of e mentions x embed dim
			# tmp2 = tmp2.unsqueeze(1).expand(-1,len(all_e1_r),-1) # batch x len of e mentions x embed dim

			# all_e1_r_tmp = all_e1_r[:,i:i+section].unsqueeze(0).expand(len(e2_r),-1,-1) # batch x len of e mentions x embed dim
			# all_e1_i_tmp = all_e1_i[:,i:i+section].unsqueeze(0).expand(len(e2_r),-1,-1) # batch x len of e mentions x embed dim

			# ans += (tmp1 * all_e1_r_tmp + tmp2 * all_e1_i_tmp).sum(dim = -1) # batch x len of e mentions
		return ans_2

	def complex_score_e2_r_with_given_ementions(self,e2_r,e2_i,r_r,r_i,given_e1_r,given_e1_i):
		"""
			#head prediction
			e2_r,e2_i,r_r,r_i are tensors of shape batch x embed_dim
			given_e1 has shape batch x max_tree_depth x embed dim
			returns a tensor of shape batch x max_tree_depth i.e. the score (e2,r) with each of the max_tree_depth e1 elements in its batch
		"""
	# result = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
		tmp1 = e2_r * r_r + e2_i * r_i
		tmp2 = e2_i * r_r - e2_r * r_i
		ans = torch.bmm(given_e1_r, tmp1.unsqueeze(2)).squeeze(2) + torch.bmm(given_e1_i, tmp2.unsqueeze(2)).squeeze(2)
		return ans
		# for i in range(0,all_e1_i.shape[1],section):

		# 	tmp1 = e2_r[:,i:i+section] * r_r[:,i:i+section] + e2_i[:,i:i+section] * r_i[:,i:i+section]
		# 	tmp2 = e2_i[:,i:i+section] * r_r[:,i:i+section] - e2_r[:,i:i+section] * r_i[:,i:i+section]

		# 	ans_2 = tmp1.mm(all_e1_r.transpose(0,1)) + tmp2.mm(all_e1_i.transpose(0,1))
		# return ans_2

	def regularizer(self, s, r, o, reg_val=0):
		raise NotImplemented

	def forward():
		# think of how to do this while writing code for training
		pass
