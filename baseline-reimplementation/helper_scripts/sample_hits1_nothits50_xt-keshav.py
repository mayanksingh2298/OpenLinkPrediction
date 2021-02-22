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
import ast


has_cuda = torch.cuda.is_available()
if not has_cuda:
	utils.colored_print("yellow", "CUDA is not available, using cpu")

def main():
	data_dir = "data/olpbench"
	head_or_tail = "tail"
	sample = 100
	train_kb = kb(os.path.join(data_dir,"train_data_thorough.txt"), em_map = None, rm_map = None)
	freq_r = {}
	freq_r_e2 = {}
	for triple in train_kb.triples:
		e1 = triple[0].item()
		r  = triple[1].item()
		e2 = triple[2].item()
		if r not in freq_r:
			freq_r[r] = 0
		freq_r[r] += 1

		# if (r,e2) not in freq_r_e2:
		# 	freq_r_e2[(r,e2)] = 0
		# freq_r_e2[(r,e2)] += 1

		if r not in freq_r_e2:
			freq_r_e2[r] = {}
		if e2 not in freq_r_e2[r]:
			freq_r_e2[r][e2] = 0
		freq_r_e2[r][e2] += 1


	pred_file = "helper_scripts/keshav_xt-preds/validation_data_linked_mention.txt.tail_thorough_n2_e50_lr0.1.stage1"
	_,rm_map = utils.read_mentions(os.path.join(data_dir,"mapped_to_ids","relation_id_map.txt"))
	entity_mentions,em_map = utils.read_mentions(os.path.join(data_dir,"mapped_to_ids","entity_id_map.txt"))
	
	random.seed(42)
	np.random.seed(42)
	torch.manual_seed(42)

	print("Loading all_known pickled data...(takes times since large)")
	all_known_e2 = {}
	all_known_e1 = {}
	all_known_e2,all_known_e1 = pickle.load(open(os.path.join(data_dir,"all_knowns_thorough_linked.pkl"),"rb"))

	hits_1_triple = []
	hits_1_correct_answers = []
	hits_1_model_top10 = []

	nothits_50_triple = []
	nothits_50_correct_answers = []
	nothits_50_model_top10 = []
	

	ranks = []
	lines = open(pred_file).readlines()
	for line in tqdm(lines,desc="preds"):
		line = line.strip().split("\t")
		e1 = line[0]
		r  = line[1]
		e2 = line[2]

		this_correct_mentions_e2_raw = line[4].split("|||")
		this_correct_mentions_e2 = []
		for mention in this_correct_mentions_e2_raw:
			if mention in em_map:
				this_correct_mentions_e2.append(em_map[mention])
		this_correct_mentions_e1_raw = line[3].split("|||")
		this_correct_mentions_e1 = []
		for mention in this_correct_mentions_e1_raw:
			if mention in em_map:
				this_correct_mentions_e1.append(em_map[mention])

		all_correct_mentions_e2 = all_known_e2.get((em_map[e1],rm_map[r]),[])
		all_correct_mentions_e1 = all_known_e1.get((em_map[e2],rm_map[r]),[])		

		indices_scores = torch.tensor(ast.literal_eval(line[5]))
		topk_scores = indices_scores[:,1]
		indices = indices_scores[:,0].long()
		if(head_or_tail=="tail"):
			this_gold = this_correct_mentions_e2
			all_gold = all_correct_mentions_e2
		else:
			this_gold = this_correct_mentions_e1
			all_gold = all_correct_mentions_e1

		best_score = -2000000000
		for i,j in enumerate(indices):
			if j in this_gold:
				best_score = max(best_score,topk_scores[i].item())
				topk_scores[i] = -2000000000

		for i,j in enumerate(indices):
			if j in all_gold:
				topk_scores[i] = -2000000000
		greatereq = topk_scores.ge(best_score).float()
		equal = topk_scores.eq(best_score).float()
		rank = (greatereq.sum()+1+equal.sum()/2.0).item()
		if rank<=1:
			hits_1_triple.append([e1,r,e2])
			hits_1_correct_answers.append([entity_mentions[x] for x in this_gold])
			hits_1_model_top10.append([])
		elif rank>50:
			nothits_50_triple.append([e1,r,e2])
			nothits_50_correct_answers.append([entity_mentions[x] for x in this_gold])
			nothits_50_model_top10.append([entity_mentions[x.item()] for x in indices])			
		ranks.append(rank)

	result = {}
	result["hits1"] = 0
	result["hits10"] = 0
	result["hits50"] = 0
	for rank in ranks:
		if rank<=1:
			result["hits1"]+=1
		if rank<=10:
			result["hits10"]+=1
		if rank<=50:
			result["hits50"]+=1
	result["hits1"] /= len(lines)
	result["hits10"] /= len(lines)
	result["hits50"] /= len(lines)
	print(result)

# print format
# triple, correct answers, model predictions, r: freq, r_gold: freq, r_prediction: freq, r_max-e2: freq 
	indices = list(range(len(hits_1_triple)))
	random.shuffle(indices)
	indices = indices[:sample]
	for ind in indices:
		# ratio = " {} /{}".format(freq_r_e2.get((hits_1_triple[ind][1],hits_1_triple[ind][2]),0),freq_r.get(hits_1_triple[ind][1],0))
		freq_of_r = freq_r.get(hits_1_triple[ind][1],0)
		freq_of_r_gold = freq_r_e2.get(hits_1_triple[ind][1],{}).get(hits_1_triple[ind][2],0)
		freq_of_r_pred = "N/A"
		freq_of_r_maxe = max(list(freq_r_e2.get(hits_1_triple[ind][1],{0:0}).values()))
		print(hits_1_triple[ind],"|",hits_1_correct_answers[ind],"|",hits_1_model_top10[ind],"|",freq_of_r,"|",freq_of_r_gold,"|",freq_of_r_pred,"|",freq_of_r_maxe)
	print("---------------------------------------------------------------------------------------------")
	indices = list(range(len(nothits_50_triple)))
	random.shuffle(indices)
	indices = indices[:sample]
	for ind in indices:
		# ratio = " {} /{}".format(freq_r_e2.get((nothits_50_triple[ind][1],nothits_50_triple[ind][2]),0),freq_r.get(nothits_50_triple[ind][1],0))
		freq_of_r = freq_r.get(nothits_50_triple[ind][1],0)
		freq_of_r_gold = freq_r_e2.get(nothits_50_triple[ind][1],{}).get(nothits_50_triple[ind][2],0)
		freq_of_r_pred = freq_r_e2.get(nothits_50_triple[ind][1],{}).get(nothits_50_model_top10[ind][0],0)
		freq_of_r_maxe = max(list(freq_r_e2.get(nothits_50_triple[ind][1],{0:0}).values()))
		print(nothits_50_triple[ind],"|",nothits_50_correct_answers[ind],"|",nothits_50_model_top10[ind],"|",freq_of_r,"|",freq_of_r_gold,"|",freq_of_r_pred,"|",freq_of_r_maxe)



if __name__=="__main__":
	main()


