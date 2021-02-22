import sys
import os
sys.path.append(sys.path[0]+"/../")
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



# class Dataset(torch.utils.data.Dataset):
#   def __init__(self, triples):
#         self.e1 = triples[:,0].tolist()
#         self.r = triples[:,1].tolist()
#         self.e2 = triples[:,2].tolist()

#   def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.e1)

#   def __getitem__(self, index):
#         'Generates one sample of data'
#         return self.e1[index],self.r[index],self.e2[index]

class complexLSTM(torch.nn.Module):
	"""
		stores embeddings for each e and r token
		uses lstm to compose them
		sends to complex for scoring the triple
	"""
	def __init__(self, etoken_count, rtoken_count, embedding_dim, num_lstm_layers = 1, clamp_v=None, reg=2,
				 batch_norm=False, unit_reg=False, has_cuda=True, initial_token_embedding = None, entity_tokens = [], relation_tokens = []):
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
			torch.nn.init.normal_(self.Et_re.weight.data, 0, 0.05)
			torch.nn.init.normal_(self.Et_im.weight.data, 0, 0.05)
			torch.nn.init.normal_(self.Rt_re.weight.data, 0, 0.05)
			torch.nn.init.normal_(self.Rt_im.weight.data, 0, 0.05)

		self.minimum_value = -self.embedding_dim * self.embedding_dim
		# not using these for now, will see later how these help
		self.clamp_v = clamp_v
		self.unit_reg = unit_reg
		self.reg = reg
		self.batch_norm = batch_norm

		self.dropout = torch.nn.Dropout(p = args.lstm_dropout)

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
		section = all_e2_i.shape[1]//split
		ans = torch.zeros((e1_r.shape[0],all_e2_r.shape[0])).cuda()
		for i in range(0,all_e2_i.shape[1],section):
			tmp1 = e1_r[:,i:i+section] * r_r[:,i:i+section] - e1_i[:,i:i+section] * r_i[:,i:i+section]
			tmp2 = e1_i[:,i:i+section] * r_r[:,i:i+section] + e1_r[:,i:i+section] * r_i[:,i:i+section]

			tmp1 = tmp1.unsqueeze(1).expand(-1,len(all_e2_r),-1) # batch x len of e mentions x embed dim
			tmp2 = tmp2.unsqueeze(1).expand(-1,len(all_e2_r),-1) # batch x len of e mentions x embed dim

			all_e2_r_tmp = all_e2_r[:,i:i+section].unsqueeze(0).expand(len(e1_r),-1,-1) # batch x len of e mentions x embed dim
			all_e2_i_tmp = all_e2_i[:,i:i+section].unsqueeze(0).expand(len(e1_r),-1,-1) # batch x len of e mentions x embed dim

			ans += (tmp1 * all_e2_r_tmp + tmp2 * all_e2_i_tmp).sum(dim = -1) # batch x len of e mentions
		return ans

	def complex_score_e2_r_with_all_ementions(self,e2_r,e2_i,r_r,r_i,all_e1_r,all_e1_i, split=1):
		"""
			#head prediction
			e2_r,e2_i,r_r,r_i are tensors of shape batch x embed_dim
			all_e1 has shape # of entity mentions x embed dim
			split is used when dimension size is so big that it doesn't fit on gpu. So you split over the dimension to solve each splitted section
			returns a tensor of shape batch x # of entity mentions i.e. the score with each entity mention
		"""
	# result = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
		section = all_e1_i.shape[1]//split
		ans = torch.zeros((e2_r.shape[0],all_e1_r.shape[0])).cuda()
		for i in range(0,all_e1_i.shape[1],section):

			tmp1 = e2_r[:,i:i+section] * r_r[:,i:i+section] + e2_i[:,i:i+section] * r_i[:,i:i+section]
			tmp2 = e2_i[:,i:i+section] * r_r[:,i:i+section] - e2_r[:,i:i+section] * r_i[:,i:i+section]

			tmp1 = tmp1.unsqueeze(1).expand(-1,len(all_e1_r),-1) # batch x len of e mentions x embed dim
			tmp2 = tmp2.unsqueeze(1).expand(-1,len(all_e1_r),-1) # batch x len of e mentions x embed dim

			all_e1_r_tmp = all_e1_r[:,i:i+section].unsqueeze(0).expand(len(e2_r),-1,-1) # batch x len of e mentions x embed dim
			all_e1_i_tmp = all_e1_i[:,i:i+section].unsqueeze(0).expand(len(e2_r),-1,-1) # batch x len of e mentions x embed dim

			ans += (tmp1 * all_e1_r_tmp + tmp2 * all_e1_i_tmp).sum(dim = -1) # batch x len of e mentions
		return ans

	def regularizer(self, s, r, o, reg_val=0):
		raise NotImplemented

	def forward():
		# think of how to do this while writing code for training
		pass

def convert_string_to_indices(data, mapp, maxlen=15, PAD_MAPPING = 0, START_MAPPING = 1, END_MAPPING = 2, use_tqdm=False):
	"""
		data: list of strings like ["the boy is playing", "the kite is flying", ...]
		map: contains mapping from words to integer. len(mapp) represents UNK
		maxlen: pad with PAD_MAPPING to fit the maxlen
		This function also adds START_MAPPING and END_MAPPING 
		returns mapped_data and actual lengths tensors
	"""
	mapped_data = []
	lengths = []
	for line in tqdm(data, disable=not use_tqdm):
		#add start token
		mapped_line = [START_MAPPING]
		line = line.strip().split()
		for word in line:
			mapped_line.append(mapp.get(word,len(mapp)))
		#add end token
		mapped_line.append(END_MAPPING)
		lengths.append(len(mapped_line))
		assert(len(mapped_line) <= maxlen)
		for i in range(maxlen-len(mapped_line)):
			mapped_line.append(PAD_MAPPING)
		mapped_data.append(mapped_line)
	return torch.tensor(mapped_data),torch.tensor(lengths)


has_cuda = torch.cuda.is_available()
if not has_cuda:
	utils.colored_print("yellow", "CUDA is not available, using cpu")

def main(args):
	hits_1_triple = []
	hits_1_correct_answers = []
	hits_1_model_top10 = []

	nothits_50_triple = []
	nothits_50_correct_answers = []
	nothits_50_model_top10 = []


	# read token maps
	etokens, etoken_map = utils.get_tokens_map(os.path.join(args.data_dir,"mapped_to_ids","entity_token_id_map.txt"))
	rtokens, rtoken_map = utils.get_tokens_map(os.path.join(args.data_dir,"mapped_to_ids","relation_token_id_map.txt"))
	entity_mentions,em_map = utils.read_mentions(os.path.join(args.data_dir,"mapped_to_ids","entity_id_map.txt"))
	_,rm_map = utils.read_mentions(os.path.join(args.data_dir,"mapped_to_ids","relation_id_map.txt"))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	#train code (+1 for unk token)
	model = complexLSTM(len(etoken_map)+1,len(rtoken_map)+1,args.embedding_dim, initial_token_embedding =args.initial_token_embedding, entity_tokens = etokens, relation_tokens = rtokens)

	if(args.resume):
		print("Resuming from:",args.resume)
		checkpoint = torch.load(args.resume,map_location=lambda storage, loc: storage)
		model.load_state_dict(checkpoint['state_dict'])

	model.eval()

	# get embeddings for all entity mentions
	entity_mentions_tensor, entity_mentions_lengths = convert_string_to_indices(entity_mentions,etoken_map,maxlen=args.max_seq_length,use_tqdm=True)
	entity_mentions_tensor = entity_mentions_tensor.cuda()
	entity_mentions_lengths = entity_mentions_lengths.cuda()

	ementions_real_lis = []
	ementions_img_lis = []
	split = 100 #cant fit all in gpu together. hence split
	with torch.no_grad():
		for i in tqdm(range(0,len(entity_mentions_tensor),len(entity_mentions_tensor)//split)):
			data = entity_mentions_tensor[i:i+len(entity_mentions_tensor)//split,:]
			data_lengths = entity_mentions_lengths[i:i+len(entity_mentions_tensor)//split]
			ementions_real_lstm,ementions_img_lstm = model.get_mention_embedding(data,0,data_lengths)			

			ementions_real_lis.append(ementions_real_lstm.cpu())
			ementions_img_lis.append(ementions_img_lstm.cpu())
	del entity_mentions_tensor,ementions_real_lstm,ementions_img_lstm
	torch.cuda.empty_cache()
	ementions_real = torch.cat(ementions_real_lis).cuda()
	ementions_img = torch.cat(ementions_img_lis).cuda()
	########################################################################

	if "olpbench" in args.data_dir:
		# test_kb = kb(os.path.join(args.data_dir,"test_data_sophis.txt"), em_map = em_map, rm_map = rm_map)
		test_kb = kb(os.path.join(args.data_dir,"test_data.txt"), em_map = em_map, rm_map = rm_map)
	else:
		test_kb = kb(os.path.join(args.data_dir,"test.txt"), em_map = em_map, rm_map = rm_map)
	print("Loading all_known pickled data...(takes times since large)")
	all_known_e2 = {}
	all_known_e1 = {}
	all_known_e2,all_known_e1 = pickle.load(open(os.path.join(args.data_dir,"all_knowns_simple_linked.pkl"),"rb"))


	test_e1_tokens_tensor, test_e1_tokens_lengths = convert_string_to_indices(test_kb.triples[:,0], etoken_map,maxlen=args.max_seq_length)
	test_r_tokens_tensor, test_r_tokens_lengths = convert_string_to_indices(test_kb.triples[:,1], rtoken_map,maxlen=args.max_seq_length)
	test_e2_tokens_tensor, test_e2_tokens_lengths = convert_string_to_indices(test_kb.triples[:,2], etoken_map,maxlen=args.max_seq_length)
	
	# e2_tensor = convert_string_to_indices(test_kb.triples[:,2], etoken_map)
	indices = torch.Tensor(range(len(test_kb.triples))) #indices would be used to fetch alternative answers while evaluating
	test_data = TensorDataset(indices, test_e1_tokens_tensor, test_r_tokens_tensor, test_e2_tokens_tensor, test_e1_tokens_lengths, test_r_tokens_lengths, test_e2_tokens_lengths)
	test_sampler = SequentialSampler(test_data)
	test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
	split_dim_for_eval = 1
	if(args.embedding_dim>=512 and "olpbench" in args.data_dir):
		split_dim_for_eval = 4
	for index, test_e1_tokens, test_r_tokens, test_e2_tokens, test_e1_lengths, test_r_lengths, test_e2_lengths in tqdm(test_dataloader,desc="Test dataloader"):
		test_e1_tokens, test_e1_lengths = test_e1_tokens.to(device), test_e1_lengths.to(device)
		test_r_tokens, test_r_lengths = test_r_tokens.to(device), test_r_lengths.to(device)
		test_e2_tokens, test_e2_lengths = test_e2_tokens.to(device), test_e2_lengths.to(device)
		with torch.no_grad():
			e1_real_lstm, e1_img_lstm = model.get_mention_embedding(test_e1_tokens,0, test_e1_lengths)
			r_real_lstm, r_img_lstm = model.get_mention_embedding(test_r_tokens,1, test_r_lengths)	
			e2_real_lstm, e2_img_lstm = model.get_mention_embedding(test_e2_tokens,0, test_e2_lengths)


		for count in tqdm(range(index.shape[0]), desc="Evaluating"):
			this_e1_real = e1_real_lstm[count].unsqueeze(0)
			this_e1_img  = e1_img_lstm[count].unsqueeze(0)
			this_r_real  = r_real_lstm[count].unsqueeze(0)
			this_r_img   = r_img_lstm[count].unsqueeze(0)
			this_e2_real = e2_real_lstm[count].unsqueeze(0)
			this_e2_img  = e2_img_lstm[count].unsqueeze(0)
			
			# get known answers for filtered ranking
			ind = index[count]
			this_correct_mentions_e2 = test_kb.e2_all_answers[int(ind.item())]
			this_correct_mentions_e1 = test_kb.e1_all_answers[int(ind.item())] 

			all_correct_mentions_e2 = all_known_e2.get((em_map[test_kb.triples[int(ind.item())][0]],rm_map[test_kb.triples[int(ind.item())][1]]),[])
			all_correct_mentions_e1 = all_known_e1.get((em_map[test_kb.triples[int(ind.item())][2]],rm_map[test_kb.triples[int(ind.item())][1]]),[])
			if(args.head_or_tail=="tail"):
				simi = model.complex_score_e1_r_with_all_ementions(this_e1_real,this_e1_img,this_r_real,this_r_img,ementions_real,ementions_img,split=split_dim_for_eval).squeeze(0)
				best_score = simi[this_correct_mentions_e2].max()
				simi[all_correct_mentions_e2] = -20000000 # MOST NEGATIVE VALUE
				greatereq = simi.ge(best_score).float()
				equal = simi.eq(best_score).float()
				rank = greatereq.sum()+1+equal.sum()/2.0

			else:
				simi = model.complex_score_e2_r_with_all_ementions(this_e2_real,this_e2_img,this_r_real,this_r_img,ementions_real,ementions_img,split=split_dim_for_eval).squeeze(0)
				best_score = simi[this_correct_mentions_e1].max()
				simi[all_correct_mentions_e1] = -20000000 # MOST NEGATIVE VALUE
				greatereq = simi.ge(best_score).float()
				equal = simi.eq(best_score).float()
				rank = greatereq.sum()+1+equal.sum()/2.0

			if(rank<=1):
				#hits1
				hits_1_triple.append([test_kb.triples[int(ind.item())][0],test_kb.triples[int(ind.item())][1],test_kb.triples[int(ind.item())][2]])
				if(args.head_or_tail=="tail"):
					# hits_1_correct_answers.append(this_correct_mentions_e2)
					hits_1_correct_answers.append([entity_mentions[x] for x in this_correct_mentions_e2])
				else:
					hits_1_correct_answers.append([entity_mentions[x] for x in this_correct_mentions_e1])
				hits_1_model_top10.append([])
			elif(rank>50):
				#nothits50
				nothits_50_triple.append([test_kb.triples[int(ind.item())][0],test_kb.triples[int(ind.item())][1],test_kb.triples[int(ind.item())][2]])
				if(args.head_or_tail=="tail"):
					nothits_50_correct_answers.append([entity_mentions[x] for x in this_correct_mentions_e2])
				else:
					nothits_50_correct_answers.append([entity_mentions[x] for x in this_correct_mentions_e1])
				tmp = simi.sort()[1].tolist()[::-1][:10]
				nothits_50_model_top10.append([entity_mentions[x] for x in tmp])
	
	indices = list(range(len(hits_1_triple)))
	random.shuffle(indices)
	indices = indices[:args.sample]
	for ind in indices:
		print(ind,"|",hits_1_triple[ind],"|",hits_1_correct_answers[ind],"|",hits_1_model_top10[ind])
	print("---------------------------------------------------------------------------------------------")
	indices = list(range(len(nothits_50_triple)))
	random.shuffle(indices)
	indices = indices[:args.sample]
	for ind in indices:
		print(ind,"|",nothits_50_triple[ind],"|",nothits_50_correct_answers[ind],"|",nothits_50_model_top10[ind])



if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default=None, type=str, required=True)
	parser.add_argument("--output_dir",default=None,type=str,required=False,help="The output directory where the model checkpoints will be written.")
	parser.add_argument("--max_seq_length",default=15,type=int)
	parser.add_argument("--eval_batch_size",
						default=8,
						type=int,
						help="Total batch size for eval.")
	parser.add_argument("--embedding_dim",
						default=256,
						type=int,
						help="Dimension of embeddings for token.")
	parser.add_argument("--no_cuda",
						default=False,
						action='store_true',
						help="Whether not to use CUDA when available")
	parser.add_argument('--seed',
						type=int,
						default=42,
						help="random seed for initialization")
	parser.add_argument("--lstm_dropout",
						default=0,
						type=float)
	parser.add_argument('--sample',
						type=int,
						default=100,
						help="Number of points to sample in each")
	parser.add_argument('--resume',
						type=str,
						help="Path of already saved checkpoint")
	parser.add_argument('--initial_token_embedding',
						type=str,
						help="Path to intial glove embeddings")
	parser.add_argument('--head_or_tail',
						type=str,
						default="tail",
						help="Head entity evaluation or tail?")
	args = parser.parse_args()

	main(args)


