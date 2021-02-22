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
	entity_mentions,em_map = utils.read_mentions(os.path.join(data_dir,"mapped_to_ids","entity_id_map.txt"))
	relation_mentions,rm_map = utils.read_mentions(os.path.join(data_dir,"mapped_to_ids","relation_id_map.txt"))
	random.seed(42)
	np.random.seed(42)
	torch.manual_seed(42)
	test_kb = kb(os.path.join(data_dir,"test_data.txt"), em_map = em_map, rm_map = rm_map)
	xt_lines = open("helper_scripts/tmp/test_data_preds.txt.tail_thorough_f5_d300_e50.stage1",'r').readlines()
	cache_e = pickle.load(open("helper_scripts/tmp/top_1000_neighbors_val.pkl",'rb'))

	test_kb.triples = np.delete(test_kb.triples,4996,0)
	del test_kb.e1_all_answers[4996]
	del test_kb.e2_all_answers[4996]

	answers_t = []
	for line in tqdm(xt_lines, desc="nudging"):
		line = line.strip().split("\t")
		e1 = em_map[line[0]]
		e1_neighbors = cache_e.get(e1,[]) # [[12,12312],[211,2312],...]
		tmp_answers_t = ast.literal_eval(line[-1])
		for neighbor in e1_neighbors:
			for i in range(len(tmp_answers_t)):
				if neighbor[0]==tmp_answers_t[i][0]:
					tmp_answers_t[i][1] += neighbor[1]
		answers_t.append(tmp_answers_t)

	result = utils.get_metrics_using_topk(os.path.join(data_dir,"all_knowns_thorough_linked.pkl"),test_kb,answers_t,answers_t,em_map,rm_map)
	print(result)

if __name__=="__main__":
	main()


