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
from models import rotatELSTM, complexLSTM

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

def get_simi(data1, data2, rtoken_map, model):
	"""
		data1 and data2 and lists of same size like ["playing","sitting",...]
		returns similarity between ith member of data1 with ith member of data2
	"""
	d1 = data1
	d2 = data2

	cosineSimi = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


	d1_tensor, d1_lengths = convert_string_to_indices(d1,rtoken_map,maxlen=args.max_seq_length)
	d2_tensor, d2_lengths = convert_string_to_indices(d2,rtoken_map,maxlen=args.max_seq_length)
	d1_tensor = d1_tensor.cuda()
	d1_lengths = d1_lengths.cuda()
	d2_tensor = d2_tensor.cuda()
	d2_lengths = d2_lengths.cuda()

	d1_real_lstm, d1_img_lstm = model.get_mention_embedding(d1_tensor,1,d1_lengths)
	d2_real_lstm, d2_img_lstm = model.get_mention_embedding(d2_tensor,1,d2_lengths)
	real_simi =  cosineSimi(d1_real_lstm,d2_real_lstm)
	img_simi = cosineSimi(d1_img_lstm,d2_img_lstm)
	ans = (real_simi + img_simi ) / 2 
	return ans

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default=None, type=str, required=True)
	# parser.add_argument("--output_dir",default=None,type=str,required=False,help="The output directory where the model checkpoints will be written.")
	parser.add_argument("--max_seq_length",default=15,type=int)
	parser.add_argument("--test_file",type=str)
	parser.add_argument("--evidence_file",type=str)
	parser.add_argument("--n_times",type=int,default=1)
	parser.add_argument("--lstm_dropout",default=0,type=float)
	parser.add_argument("--model",default="complex",type=str)
	parser.add_argument("--embedding_dim",
						default=256,
						type=int,
						help="Dimension of embeddings for token.")
	parser.add_argument('--resume',
						type=str,
						help="Path of already saved checkpoint")
	parser.add_argument('--resume_for_simi',
						type=str,
						help="Path of already saved checkpoint for calculating relation simi")
	parser.add_argument("--do_eval",
						default=False,
						action='store_true',
						help="Whether to run eval on the dev set.")
	parser.add_argument("--eval_batch_size",
						default=512,
						type=int,
						help="Total batch size for eval.")
	args = parser.parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



	# read token maps
	etokens, etoken_map = utils.get_tokens_map(os.path.join(args.data_dir,"mapped_to_ids","entity_token_id_map.txt"))
	rtokens, rtoken_map = utils.get_tokens_map(os.path.join(args.data_dir,"mapped_to_ids","relation_token_id_map.txt"))
	entity_mentions,em_map = utils.read_mentions(os.path.join(args.data_dir,"mapped_to_ids","entity_id_map.txt"))
	relation_mentions,rm_map = utils.read_mentions(os.path.join(args.data_dir,"mapped_to_ids","relation_id_map.txt"))

	if args.model=="complex":
		model = complexLSTM(len(etoken_map)+1,len(rtoken_map)+1,args.embedding_dim, initial_token_embedding = None, entity_tokens = etokens, relation_tokens = rtokens, lstm_dropout=args.lstm_dropout)
	elif args.model == "rotate":
		model = rotatELSTM(len(etoken_map)+1,len(rtoken_map)+1,args.embedding_dim, initial_token_embedding = None, entity_tokens = etokens, relation_tokens = rtokens, gamma = args.gamma_rotate, lstm_dropout=args.lstm_dropout)
	if args.resume:
		print("Resuming from:",args.resume)
		checkpoint = torch.load(args.resume)
		model.load_state_dict(checkpoint['state_dict'])
	model.eval()

	# if args.model=="complex":
	# 	model_simi = complexLSTM(len(etoken_map)+1,len(rtoken_map)+1,args.embedding_dim, initial_token_embedding = None, entity_tokens = etokens, relation_tokens = rtokens, lstm_dropout=args.lstm_dropout)
	# elif args.model == "rotate":
	# 	model_simi = rotatELSTM(len(etoken_map)+1,len(rtoken_map)+1,args.embedding_dim, initial_token_embedding = None, entity_tokens = etokens, relation_tokens = rtokens, gamma = args.gamma_rotate, lstm_dropout=args.lstm_dropout)
	# if args.resume_for_simi:
	# 	print("Resuming from:",args.resume_for_simi)
	# 	checkpoint = torch.load(args.resume_for_simi)
	# 	model_simi.load_state_dict(checkpoint['state_dict'])
	# model_simi.eval()

	d1 = ["nominated"]
	d2 = ["elected"]
	# random_d1 = []
	# random_d2 = []
	# for i in range(100):
	# 	random_d1.append(relation_mentions[random.randint(0,len(relation_mentions)-1)])
	# 	random_d2.append(relation_mentions[random.randint(0,len(relation_mentions)-1)])
	# print(get_simi(random_d1,random_d2,rtoken_map,model))
	print(get_simi(d1,d2,rtoken_map,model))
	import pdb
	pdb.set_trace()
	if(not args.do_eval):
		test_rels = kb(args.test_file, em_map = em_map, rm_map = rm_map).triples[:,1].reshape(-1,1)
		injected_rels = kb(args.evidence_file, em_map = em_map, rm_map = rm_map).triples[:,1].reshape(-1,args.n_times)
		assert test_rels.shape[0]==injected_rels.shape[0]
		simis = []
		for i in tqdm(range(test_rels.shape[0])):
		# for i in tqdm(range(100)):
			simi_with_all = get_simi(test_rels[i].tolist()*args.n_times,injected_rels[i].tolist(),rtoken_map,model_simi)
			avg_simi = sum(simi_with_all)/len(simi_with_all)
			simis.append(avg_simi.item())
		print("average simi of test r with all evidences:",sum(simis)/len(simis))
	else:
		#eval code
		metrics = {}
		metrics['mr'] = 0
		metrics['mrr'] = 0
		metrics['S_inf'] = 0
		metrics['S_1'] = 0
		metrics['S_10'] = 0
		metrics['S_50'] = 0
		metrics['hits1'] = 0
		metrics['hits10'] = 0
		metrics['hits50'] = 0

		metrics['mr_t'] = 0
		metrics['mrr_t'] = 0
		metrics['S_inf_t'] = 0
		metrics['S_1_t'] = 0
		metrics['S_10_t'] = 0
		metrics['S_50_t'] = 0
		metrics['hits1_t'] = 0
		metrics['hits10_t'] = 0
		metrics['hits50_t'] = 0

		metrics['mr_h'] = 0
		metrics['mrr_h'] = 0
		metrics['S_inf_h'] = 0
		metrics['S_1_h'] = 0
		metrics['S_10_h'] = 0
		metrics['S_50_h'] = 0
		metrics['hits1_h'] = 0
		metrics['hits10_h'] = 0
		metrics['hits50_h'] = 0


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
				# a = model.Et_im(entity_mentions_tensor[i:i+len(entity_mentions_tensor)//split,:])
				# b = model.Et_re(entity_mentions_tensor[i:i+len(entity_mentions_tensor)//split,:])
				
				# a_lstm,_ = model.lstm(a)
				# a_lstm = a_lstm[:,-1,:]

				
				# b_lstm,_ = model.lstm(b)
				# b_lstm = b_lstm[:,-1,:]

				ementions_real_lis.append(ementions_real_lstm.cpu())
				ementions_img_lis.append(ementions_img_lstm.cpu())
		del entity_mentions_tensor,ementions_real_lstm,ementions_img_lstm
		torch.cuda.empty_cache()
		ementions_real = torch.cat(ementions_real_lis).cuda()
		ementions_img = torch.cat(ementions_img_lis).cuda()
		########################################################################
		if "olpbench" in args.data_dir:
			test_kb = kb(os.path.join(args.data_dir,"test_data.txt"), em_map = em_map, rm_map = rm_map)
		else:
			test_kb = kb(os.path.join(args.data_dir,"test.txt"), em_map = em_map, rm_map = rm_map)

		print("Loading all_known pickled data...(takes times since large)")
		all_known_e2 = {}
		all_known_e1 = {}
		all_known_e2,all_known_e1 = pickle.load(open(os.path.join(args.data_dir,"all_knowns_thorough_linked.pkl"),"rb"))


		test_e1_tokens_tensor, test_e1_tokens_lengths = convert_string_to_indices(test_kb.triples[:,0], etoken_map,maxlen=args.max_seq_length)
		test_r_tokens_tensor, test_r_tokens_lengths = convert_string_to_indices(test_kb.triples[:,1], rtoken_map,maxlen=args.max_seq_length)
		test_e2_tokens_tensor, test_e2_tokens_lengths = convert_string_to_indices(test_kb.triples[:,2], etoken_map,maxlen=args.max_seq_length)
		
		# e2_tensor = convert_string_to_indices(test_kb.triples[:,2], etoken_map)
		indices = torch.Tensor(range(len(test_kb.triples))) #indices would be used to fetch alternative answers while evaluating
		test_data = TensorDataset(indices, test_e1_tokens_tensor, test_r_tokens_tensor, test_e2_tokens_tensor, test_e1_tokens_lengths, test_r_tokens_lengths, test_e2_tokens_lengths)
		test_sampler = SequentialSampler(test_data)
		test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
		split_dim_for_eval = 1
		if(args.embedding_dim>=256 and "olpbench" in args.data_dir and "rotat" in args.model):
			split_dim_for_eval = 4
		if(args.embedding_dim>=512 and "olpbench" in args.data_dir):
			split_dim_for_eval = 4
		if(args.embedding_dim>=512 and "olpbench" in args.data_dir and "rotat" in args.model):
			split_dim_for_eval = 6
		test_rels = kb(args.test_file, em_map = em_map, rm_map = rm_map).triples[:,1].reshape(-1,1)
		injected_rels = kb(args.evidence_file, em_map = em_map, rm_map = rm_map).triples[:,1].reshape(-1,args.n_times)
		assert test_rels.shape[0]==injected_rels.shape[0]
		for index, test_e1_tokens, test_r_tokens, test_e2_tokens, test_e1_lengths, test_r_lengths, test_e2_lengths in tqdm(test_dataloader,desc="Test dataloader"):
			print(metrics)
			test_e1_tokens, test_e1_lengths = test_e1_tokens.to(device), test_e1_lengths.to(device)
			test_r_tokens, test_r_lengths = test_r_tokens.to(device), test_r_lengths.to(device)
			test_e2_tokens, test_e2_lengths = test_e2_tokens.to(device), test_e2_lengths.to(device)
			with torch.no_grad():
				e1_real_lstm, e1_img_lstm = model.get_mention_embedding(test_e1_tokens,0, test_e1_lengths)
				r_real_lstm, r_img_lstm = model.get_mention_embedding(test_r_tokens,1, test_r_lengths)	
				e2_real_lstm, e2_img_lstm = model.get_mention_embedding(test_e2_tokens,0, test_e2_lengths)


			for count in tqdm(range(index.shape[0]), desc="Evaluating"):
				# breakpoint()
				this_e1_real = e1_real_lstm[count].unsqueeze(0)
				this_e1_img  = e1_img_lstm[count].unsqueeze(0)
				this_r_real  = r_real_lstm[count].unsqueeze(0)
				this_r_img   = r_img_lstm[count].unsqueeze(0)
				this_e2_real = e2_real_lstm[count].unsqueeze(0)
				this_e2_img  = e2_img_lstm[count].unsqueeze(0)
				simi_t = model.complex_score_e1_r_with_all_ementions(this_e1_real,this_e1_img,this_r_real,this_r_img,ementions_real,ementions_img,split=split_dim_for_eval).squeeze(0)
				simi_h = model.complex_score_e2_r_with_all_ementions(this_e2_real,this_e2_img,this_r_real,this_r_img,ementions_real,ementions_img,split=split_dim_for_eval).squeeze(0)
				# get known answers for filtered ranking
				ind = index[count]
				this_correct_mentions_e2 = test_kb.e2_all_answers[int(ind.item())]
				this_correct_mentions_e1 = test_kb.e1_all_answers[int(ind.item())] 

				all_correct_mentions_e2 = all_known_e2.get((em_map[test_kb.triples[int(ind.item())][0]],rm_map[test_kb.triples[int(ind.item())][1]]),[])
				all_correct_mentions_e1 = all_known_e1.get((em_map[test_kb.triples[int(ind.item())][2]],rm_map[test_kb.triples[int(ind.item())][1]]),[])
				simi_with_all = get_simi(test_rels[int(ind.item())].tolist()*args.n_times,injected_rels[int(ind.item())].tolist(),rtoken_map,model)
				avg_simi = (sum(simi_with_all)/len(simi_with_all)).item()
				# avg_simi = max(simi_with_all).item()
				
				# avg_simi = 0
				# del simi_with_all
				# compute metrics
				best_score = simi_t[this_correct_mentions_e2].max()
				simi_t[all_correct_mentions_e2] = -20000000 # MOST NEGATIVE VALUE
				greatereq = simi_t.ge(best_score).float()
				equal = simi_t.eq(best_score).float()
				rank = greatereq.sum()+1+equal.sum()/2.0

				metrics['mr_t'] += rank.item()
				metrics['mrr_t'] += 1.0/rank.item()
				metrics['hits1_t'] += rank.le(1).float().item()
				metrics['hits10_t'] += rank.le(10).float().item()
				metrics['hits50_t'] += rank.le(50).float().item()
				if(rank<=1):
					metrics['S_1_t'] += avg_simi
				if(rank<=10):
					metrics['S_10_t'] += avg_simi
				if(rank<=50):
					metrics['S_50_t'] += avg_simi
				metrics['S_inf_t'] += avg_simi

				best_score = simi_h[this_correct_mentions_e1].max()
				simi_h[all_correct_mentions_e1] = -20000000 # MOST NEGATIVE VALUE
				greatereq = simi_h.ge(best_score).float()
				equal = simi_h.eq(best_score).float()
				rank = greatereq.sum()+1+equal.sum()/2.0
				metrics['mr_h'] += rank.item()
				metrics['mrr_h'] += 1.0/rank.item()
				metrics['hits1_h'] += rank.le(1).float().item()
				metrics['hits10_h'] += rank.le(10).float().item()
				metrics['hits50_h'] += rank.le(50).float().item()
				if(rank<=1):
					metrics['S_1_h'] += avg_simi
				if(rank<=10):
					metrics['S_10_h'] += avg_simi
				if(rank<=50):
					metrics['S_50_h'] += avg_simi
				metrics['S_inf_h'] += avg_simi

				metrics['mr'] = (metrics['mr_h']+metrics['mr_t'])/2
				metrics['mrr'] = (metrics['mrr_h']+metrics['mrr_t'])/2
				metrics['hits1'] = (metrics['hits1_h']+metrics['hits1_t'])/2
				metrics['hits10'] = (metrics['hits10_h']+metrics['hits10_t'])/2
				metrics['hits50'] = (metrics['hits50_h']+metrics['hits50_t'])/2
				metrics['S_inf'] = (metrics['S_inf_h']+metrics['S_inf_t'])/2
				metrics['S_1'] = (metrics['S_1_h']+metrics['S_1_t'])/2
				metrics['S_10'] = (metrics['S_10_h']+metrics['S_10_t'])/2
				metrics['S_50'] = (metrics['S_50_h']+metrics['S_50_t'])/2


		metrics['S_1_t'] /= metrics['hits1_t']
		metrics['S_10_t'] /= metrics['hits10_t']
		metrics['S_50_t'] /= metrics['hits50_t']
		metrics['S_1_h'] /= metrics['hits1_h']
		metrics['S_10_h'] /= metrics['hits10_h']
		metrics['S_50_h'] /= metrics['hits50_h']
		metrics['S_1'] = (metrics['S_1_h']+metrics['S_1_t'])/2
		metrics['S_10'] = (metrics['S_10_h']+metrics['S_10_t'])/2
		metrics['S_50'] = (metrics['S_50_h']+metrics['S_50_t'])/2


		for key in metrics:
			if key not in ['S_1_t','S_10_t','S_50_t','S_1_h','S_10_h','S_50_h','S_1','S_10','S_50']:
				metrics[key] = metrics[key] / len(test_kb.triples)
		print(metrics)


