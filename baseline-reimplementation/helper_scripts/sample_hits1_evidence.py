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


has_cuda = torch.cuda.is_available()
if not has_cuda:
	utils.colored_print("yellow", "CUDA is not available, using cpu")

def main(args):
	hits_1_triple = []
	hits_1_correct_answers = []
	hits_1_model_top10 = []
	hits_1_evidence = []
	baseline_tail_hits1_indices = set([36, 91, 95, 101, 119, 158, 282, 397, 638, 728, 740, 763, 914, 959, 972, 992, 1184, 1478, 1669, 1686, 1732, 1795, 1796, 1822, 1826, 1845, 1924, 1939, 1943, 2055, 2178, 2317, 2319, 2325, 2482, 2513, 2589, 2627, 2674, 2736, 2862, 2985, 3049, 3311, 3327, 3491, 3660, 3728, 3817, 3818, 4111, 4263, 4387, 4437, 4438, 4452, 4525, 4591, 4670, 4856, 5114, 5159, 5318, 5587, 5851, 5857, 5893, 5925, 5942, 5990, 6056, 6079, 6119, 6172, 6195, 6211, 6228, 6262, 6267, 6460, 6491, 6509, 6584, 6676, 6699, 6862, 6982, 7057, 7078, 7084, 7221, 7597, 7733, 7837, 8045, 8278, 8326, 8380, 8433, 8453, 8479, 8534, 8540, 8742, 8813, 8860, 8906, 8930, 9234, 9333, 9500, 9535, 9589, 9663, 9803, 9809, 9866, 9999])
	baseline_correct = 0
	# nothits_50_triple = []
	# nothits_50_correct_answers = []
	# nothits_50_model_top10 = []

	injected_rels = kb(args.evidence_file, em_map = None, rm_map = None).triples[:,1].reshape(-1,args.n_times)

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
	model = complexLSTM(len(etoken_map)+1,len(rtoken_map)+1,args.embedding_dim, initial_token_embedding =args.initial_token_embedding, entity_tokens = etokens, relation_tokens = rtokens, lstm_dropout=0)

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

			if int(ind.item()) in baseline_tail_hits1_indices:
				if rank<=1:
					baseline_correct+=1
				continue
			if(rank<=1):
				#hits1
				hits_1_triple.append([test_kb.triples[int(ind.item())][0],test_kb.triples[int(ind.item())][1],test_kb.triples[int(ind.item())][2]])
				hits_1_evidence.append(injected_rels[int(ind.item())].tolist())
				if(args.head_or_tail=="tail"):
					# hits_1_correct_answers.append(this_correct_mentions_e2)
					hits_1_correct_answers.append([entity_mentions[x] for x in this_correct_mentions_e2])
				else:
					hits_1_correct_answers.append([entity_mentions[x] for x in this_correct_mentions_e1])
				hits_1_model_top10.append([])
			# elif(rank>50):
			# 	#nothits50
			# 	nothits_50_triple.append([test_kb.triples[int(ind.item())][0],test_kb.triples[int(ind.item())][1],test_kb.triples[int(ind.item())][2]])
			# 	if(args.head_or_tail=="tail"):
			# 		nothits_50_correct_answers.append([entity_mentions[x] for x in this_correct_mentions_e2])
			# 	else:
			# 		nothits_50_correct_answers.append([entity_mentions[x] for x in this_correct_mentions_e1])
			# 	tmp = simi.sort()[1].tolist()[::-1][:10]
			# 	nothits_50_model_top10.append([entity_mentions[x] for x in tmp])
	
	indices = list(range(len(hits_1_triple)))
	random.shuffle(indices)
	indices = indices[:args.sample]
	print(baseline_correct)
	for ind in indices:
		print(ind,"|",hits_1_triple[ind],"|",hits_1_correct_answers[ind],"|",hits_1_model_top10[ind],"|",hits_1_evidence[ind])
	# print("---------------------------------------------------------------------------------------------")
	# indices = list(range(len(nothits_50_triple)))
	# random.shuffle(indices)
	# indices = indices[:args.sample]
	# for ind in indices:
	# 	print(ind,"|",nothits_50_triple[ind],"|",nothits_50_correct_answers[ind],"|",nothits_50_model_top10[ind])



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
	parser.add_argument("--evidence_file",type=str)
	parser.add_argument("--n_times",type=int,default=1)
	args = parser.parse_args()

	main(args)


