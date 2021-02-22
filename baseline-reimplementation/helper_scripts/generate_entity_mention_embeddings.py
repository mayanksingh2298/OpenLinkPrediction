import sys
import os
sys.path.append(sys.path[0]+"/../")
import utils
import argparse
import logging
import os
import pickle
import pprint
import time
import numpy as np
import random
import torch
from tqdm import tqdm
import datetime
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import pickle

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default=None, type=str, required=True)
	parser.add_argument("--max_seq_length",default=128,type=int)
	parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=False,
					help="Bert pre-trained model selected in the list: bert-base-uncased, "
					 "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
	parser.add_argument("--eval_batch_size",
						default=8,
						type=int,
						help="Total batch size for eval.")
	parser.add_argument('--seed',
						type=int,
						default=42,
						help="random seed for initialization")
	parser.add_argument("--output_dir", default=None, type=str, required=True)
	parser.add_argument('--resume',
						type=str,
						help="Path of already saved checkpoint")
	args = parser.parse_args()
	# Loading entity mentions
	entity_mentions = []
	lines = open(os.path.join(args.data_dir,"mapped_to_ids","entity_id_map.txt"),'r').readlines()
	print("Reading entity mentions...")
	for line in tqdm(lines[1:]):
		line = line.strip().split("\t")
		entity_mentions.append(line[0])

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
	model = BertModel.from_pretrained(args.bert_model,
				cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1),
			)
	model.to(device)

	if(args.resume):
		print("Resuming from:",args.resume)
		checkpoint = torch.load(args.resume)
		model.load_state_dict(checkpoint['state_dict'])

	print("Getting e2 bert features...")
	model.eval()
	all_input_ids, all_input_masks, all_segment_ids = utils.e2_to_bertfeatures(entity_mentions,tokenizer,args.max_seq_length, use_tqdm=True)
	entity_mentions_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids)
	entity_mentions_sampler = SequentialSampler(entity_mentions_data)
	entity_mentions_dataloader = DataLoader(entity_mentions_data, sampler=entity_mentions_sampler, batch_size=args.eval_batch_size)
	print("Getting embeddings for all entity mentions...")
	lis = []
	for input_ids, input_mask, segment_ids in tqdm(entity_mentions_dataloader):
		input_ids = input_ids.to(device)
		input_mask = input_mask.to(device)
		segment_ids = segment_ids.to(device)	
		with torch.no_grad():
			# _, e2_embeddings_batch = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, output_all_encoded_layers=False)
			out, _ = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, output_all_encoded_layers=False)
			e2_embeddings_batch = out[:,0,:]
		lis.append(e2_embeddings_batch.cpu())
	e2_embeddings = torch.cat(lis)
	try:
		torch.save(e2_embeddings,args.output_dir)
	except:
		import pdb
		pdb.set_trace()