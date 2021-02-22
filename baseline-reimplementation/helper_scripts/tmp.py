import argparse
import logging
import os
import pickle
import pprint
import sys
sys.path.append(sys.path[0]+"/../")
import time
import numpy as np
import random
import torch
torch.backends.cudnn.benchmark=True
from tqdm import tqdm
import datetime
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from kb import kb
import utils
from dataset import Dataset
from models import rotatELSTM, complexLSTM, complexLSTM_2

def main(args):
	# read token maps
	# etokens, etoken_map = utils.get_tokens_map(os.path.join(args.data_dir,"mapped_to_ids","entity_token_id_map.txt"))
	# rtokens, rtoken_map = utils.get_tokens_map(os.path.join(args.data_dir,"mapped_to_ids","relation_token_id_map.txt"))
	# entity_mentions,em_map = utils.read_mentions(os.path.join(args.data_dir,"mapped_to_ids","entity_id_map.txt"))
	# _,rm_map = utils.read_mentions(os.path.join(args.data_dir,"mapped_to_ids","relation_id_map.txt"))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	#train code (+1 for unk token)
	if "olpbench" in args.data_dir:
		train_kb = kb(os.path.join(args.data_dir,"train_data_{}.txt".format(args.train_data_type)), em_map = None, rm_map = None)
		# train_kb = kb(os.path.join(args.data_dir,"train_data_thorough_r_sorted.txt"), em_map = em_map, rm_map = rm_map)
		# train_kb = kb(os.path.join(args.data_dir,"test_data.txt"), em_map = None, rm_map = None)

	
	train_data = Dataset(train_kb.triples)
	train_sampler = RandomSampler(train_data,replacement=False)
	#train_sampler = SequentialSampler(train_data)
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

	# crossEntropyLoss = torch.nn.CrossEntropyLoss(reduction='mean')
	BCEloss = torch.nn.BCEWithLogitsLoss(reduction='sum')

	for epoch in tqdm(range(0,args.num_train_epochs),desc="epoch"):
		iteration = 0
		repeated_e2 = 0
		total_e2 = 0
		for train_e1_batch, train_r_batch, train_e2_batch in tqdm(train_dataloader,desc="Train dataloader"):
			# do something
			freq = {}
			for i in (train_e2_batch):
				if i not in freq:
					freq[i] = 0
				freq[i] += 1
			for key in freq:
				if freq[key]>1:
					repeated_e2 += 1
				total_e2 += 1
			# import pdb
			# pdb.set_trace()
	print(repeated_e2,total_e2)

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default=None, type=str, required=True)
	parser.add_argument("--train_batch_size",
						default=32,
						type=int,
						help="Total batch size for training.")
	parser.add_argument("--num_train_epochs",
						default=3,
						type=int,
						help="Total number of training epochs to perform.")
	parser.add_argument("--no_cuda",
						default=False,
						action='store_true',
						help="Whether not to use CUDA when available")
	parser.add_argument('--seed',
						type=int,
						default=42,
						help="random seed for initialization")
	parser.add_argument("--debug", default=False,action='store_true',)
	parser.add_argument('--train_data_type',type=str,default="thorough")


	args = parser.parse_args()
	
	print(args)
	main(args)


