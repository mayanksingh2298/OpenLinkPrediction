import pickle
import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import sys
import csv
import subprocess
import shutil

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import BertModel
# from pytorch_pretrained_bert.optimization import BertAdam
# from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
color_map = {"green":32, "black":0, "blue":34, "red":31, 'yellow':33}


def get_token_indices_from_mention_indices(data, mapp, maxlen=15, PAD_MAPPING = 0, START_MAPPING = 1, END_MAPPING = 2, use_tqdm=False):
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
			if(len(mapped_line)==maxlen-1):
				break
		#add end token
		mapped_line.append(END_MAPPING)
		lengths.append(len(mapped_line))
		assert(len(mapped_line) <= maxlen)
		for i in range(maxlen-len(mapped_line)):
			mapped_line.append(PAD_MAPPING)
		mapped_data.append(mapped_line)
	return mapped_data,lengths

def get_metrics_using_topk(all_known_path,test_kb,answers_t,answers_h,em_map,rm_map):
	"""
		answers: [[(e1_id,e1_score),...],...]
					ith elements contains a list of (e_id,e_score) for k answers for the ith test triple
	"""
	all_known_e2 = {}
	all_known_e1 = {}
	print("Loading all knowns from {}. Might take time...".format(all_known_path))
	if(all_known_path):
		all_known_e2,all_known_e1 = pickle.load(open(all_known_path,"rb"))

	metrics = {}
	metrics['mr'] = 0
	metrics['mrr'] = 0
	metrics['hits1'] = 0
	metrics['hits10'] = 0
	metrics['hits50'] = 0
	metrics['mr_t'] = 0
	metrics['mrr_t'] = 0
	metrics['hits1_t'] = 0
	metrics['hits10_t'] = 0
	metrics['hits50_t'] = 0
	metrics['mr_h'] = 0
	metrics['mrr_h'] = 0
	metrics['hits1_h'] = 0
	metrics['hits10_h'] = 0
	metrics['hits50_h'] = 0

	def get_rank(this_answers,this_correct_mentions,all_correct_mentions):
		best_score = -9999999999999999
		for answer_ind in range(len(this_answers)):
			answer = this_answers[answer_ind]
			if answer[0] in this_correct_mentions:
				best_score = max(best_score,answer[1])
				this_answers[answer_ind][1] = -999999999
		for answer_ind in range(len(this_answers)):
			answer = this_answers[answer_ind]
			if answer[0] in all_correct_mentions:
				this_answers[answer_ind][1] = -999999999
		if best_score==-9999999999999999:
			rank = 9999999999999999
		else:
			greatereq = 0
			equal = 0
			for answer_ind in range(len(this_answers)):
				answer = this_answers[answer_ind]
				if answer[1] >= best_score:
					greatereq += 1
				if answer[1] == best_score:
					equal += 1
			rank = 1 + greatereq + equal/2.0
		return rank

	for ind, triple in tqdm(list(enumerate(test_kb.triples)), desc="Test dataloader"):
		e1 = triple[0].item()
		r  = triple[1].item()
		e2 = triple[2].item()

		# tail evaluation
		this_answers = answers_t[ind] # [(e_id,e_score),...]
		for answer_ind in range(len(this_answers)):
			 this_answers[answer_ind] = list(this_answers[answer_ind])
		this_correct_mentions = test_kb.e2_all_answers[int(ind)]
		all_correct_mentions = all_known_e2.get((em_map[test_kb.triples[int(ind)][0]],rm_map[test_kb.triples[int(ind)][1]]),[])
		rank = get_rank(this_answers,this_correct_mentions,all_correct_mentions)
		metrics['mr_t'] += rank
		metrics['mrr_t'] += 1.0/rank
		if rank<=1:
			metrics['hits1_t'] += 1.0
		if rank<=10:
			metrics['hits10_t'] += 1.0
		if rank<=1000:
			metrics['hits50_t'] += 1.0		


		# head evaluation
		this_answers = answers_h[ind] # [(e_id,e_score),...]
		for answer_ind in range(len(this_answers)):
			 this_answers[answer_ind] = list(this_answers[answer_ind])
		this_correct_mentions = test_kb.e1_all_answers[int(ind)]
		all_correct_mentions = all_known_e1.get((em_map[test_kb.triples[int(ind)][2]],rm_map[test_kb.triples[int(ind)][1]]),[])
		rank = get_rank(this_answers,this_correct_mentions,all_correct_mentions)
		metrics['mr_h'] += rank
		metrics['mrr_h'] += 1.0/rank
		if rank<=1:
			metrics['hits1_h'] += 1.0
		if rank<=10:
			metrics['hits10_h'] += 1.0
		if rank<=1000:
			metrics['hits50_h'] += 1.0

		metrics['mr'] = (metrics['mr_h']+metrics['mr_t'])/2
		metrics['mrr'] = (metrics['mrr_h']+metrics['mrr_t'])/2
		metrics['hits1'] = (metrics['hits1_h']+metrics['hits1_t'])/2
		metrics['hits10'] = (metrics['hits10_h']+metrics['hits10_t'])/2
		metrics['hits50'] = (metrics['hits50_h']+metrics['hits50_t'])/2

	for key in metrics:
		metrics[key] = metrics[key] / len(test_kb.triples)
	return metrics



# get entity/relation tokens map given path to a file which lists all entity/relation tokens
# first line is header
# use the len(map) for unk(basically the last token)
def get_tokens_map(path):
	"""
		<PAD>: 0
		<INIT>: 1
		<END>: 2
		...
	"""
	lines = open(path,'r').readlines()
	token_map = {"<PAD>":0,"<INIT>":1,"<END>":2}
	tokens = ["<PAD>","<INIT>","<END>"]
	for line in tqdm(lines[1:],desc="Reading file for token map"):
		line = line.strip().split("\t")
		tokens.append(line[0])
		token_map[line[0]] = len(token_map)
	return tokens,token_map

# read entity or relation mentions from a file containing them 
# 1st line is header
def read_mentions(path):
	mapp = {}
	mentions = []
	lines = open(path,'r').readlines()
	for line in tqdm(lines[1:]): 
		line = line.strip().split("\t")
		mentions.append(line[0])
		mapp[line[0]] = len(mapp)
	return mentions,mapp


def colored_print(color, message):
	"""
	Simple utility to print in color
	:param color: The name of color from color_map
	:param message: The message to print in color
	:return: None
	"""
	print('\033[1;%dm%s\033[0;0m' % (color_map[color], message))

def duplicate_stdout(filename):
	"""
	This function is used to duplicate and redires stdout into a file. This enables a permanent record of the log on
	disk\n
	:param filename: The filename to which stdout should be duplicated
	:return: None
	"""
	print("duplicating")
	sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')
	tee = subprocess.Popen(["tee", filename], stdin=subprocess.PIPE)
	os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
	os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()

def e1r_to_bertfeatures(examples_e1, examples_r, tokenizer, max_seq_length):
	# - [CLS] e1 [SEP] r [SEP]
	# . . .
	# format of examples_e1:
	#	[
	#  	    "e1",
	#		 ...
	#	]
	# features = []
	all_input_ids = []
	all_input_masks = []
	all_segment_ids = []
	assert len(examples_e1)==len(examples_r)
	for i in range(len(examples_e1)):
		e1_tokens = tokenizer.tokenize(examples_e1[i])
		r_tokens = tokenizer.tokenize(examples_r[i])

		# Modifies `e1_tokens` and `r_tokens` in
		# place so that the total length is less than the
		# specified length.  Account for [CLS], [SEP], [SEP] with
		# "- 3"
		_truncate_seq_pair(e1_tokens, r_tokens, max_seq_length - 3)

		segment_ids = [0] * (len(e1_tokens) + 2) + [1] * (len(r_tokens) + 1)
		tokens = ["[CLS]"] + e1_tokens + ["[SEP]"] + r_tokens + ["[SEP]"] 
		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1] * len(input_ids)
		# Zero-pad up to the sequence length.
		padding = [0] * (max_seq_length - len(input_ids))
		input_ids += padding
		input_mask += padding
		segment_ids += padding

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length
		# features.append((tokens, input_ids, input_mask, segment_ids))
		all_input_ids.append(input_ids)
		all_input_masks.append(input_mask)
		all_segment_ids.append(segment_ids)
	all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
	all_input_masks = torch.tensor(all_input_masks, dtype=torch.long)
	all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
	return all_input_ids,all_input_masks,all_segment_ids

def e1rMASK_to_bertfeatures(examples_e1, examples_r, tokenizer, max_seq_length):
	# - [CLS] e1 [SEP] r [SEP] [MASK] [SEP]
	# . . .
	# format of examples_e1:
	#	[
	#  	    "e1",
	#		 ...
	#	]
	# features = []
	all_input_ids = []
	all_input_masks = []
	all_segment_ids = []
	mask_token_indices = []
	assert len(examples_e1)==len(examples_r)
	for i in range(len(examples_e1)):
		e1_tokens = tokenizer.tokenize(examples_e1[i])
		r_tokens = tokenizer.tokenize(examples_r[i])

		# Modifies `e1_tokens` and `r_tokens` in
		# place so that the total length is less than the
		# specified length.  Account for [CLS], [SEP], [SEP] with
		# "- 3"
		_truncate_seq_pair(e1_tokens, r_tokens, max_seq_length - 5)

		segment_ids = [0] * (len(e1_tokens) + 2) + [1] * (len(r_tokens) + 3)
		tokens = ["[CLS]"] + e1_tokens + ["[SEP]"] + r_tokens + ["[SEP]","[MASK]","[SEP]"] 
		mask_token_indices.append(len(tokens)-2)
		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1] * len(input_ids)
		# Zero-pad up to the sequence length.
		padding = [0] * (max_seq_length - len(input_ids))
		input_ids += padding
		input_mask += padding
		segment_ids += padding
		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length
		# features.append((tokens, input_ids, input_mask, segment_ids))
		all_input_ids.append(input_ids)
		all_input_masks.append(input_mask)
		all_segment_ids.append(segment_ids)
	all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
	all_input_masks = torch.tensor(all_input_masks, dtype=torch.long)
	all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
	mask_token_indices = torch.tensor(mask_token_indices, dtype=torch.long)
	return all_input_ids,all_input_masks,all_segment_ids,mask_token_indices

def e2_to_bertfeatures(examples, tokenizer, max_seq_length, use_tqdm = False):
	# - [CLS] e2 [SEP]
	# . . .
	# format of examples:
	#	[
	#  	    "e2"
	#		,...
	#	]
	# features = []
	all_input_ids = []
	all_input_masks = []
	all_segment_ids = []
	for example in tqdm(examples,disable=not use_tqdm):
		e2_tokens = tokenizer.tokenize(example)

		# Modifies `e2_tokens` in
		# place so that the total length is less than the
		# specified length.  Account for [CLS], [SEP], [SEP] with
		# "- 3"
		_truncate_seq_pair(e2_tokens, [], max_seq_length - 2)

		segment_ids = [0] * (len(e2_tokens) + 2) 
		tokens = ["[CLS]"] + e2_tokens + ["[SEP]"] 
		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1] * len(input_ids)
		# Zero-pad up to the sequence length.
		padding = [0] * (max_seq_length - len(input_ids))
		input_ids += padding
		input_mask += padding
		segment_ids += padding

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length
		# features.append((tokens, input_ids, input_mask, segment_ids))
		all_input_ids.append(input_ids)
		all_input_masks.append(input_mask)
		all_segment_ids.append(segment_ids)
	all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
	all_input_masks = torch.tensor(all_input_masks, dtype=torch.long)
	all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
	return all_input_ids,all_input_masks,all_segment_ids

def save_checkpoint(state,path):
	try:
		torch.save(state,path)
	except Exception as E:
		colored_print("red", "unable to save model")
		print(E)

if __name__=="__main__":
	"""
		Use this section to practice
	"""
	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
	e1 = ["this is a good game","la la la"]
	all_input_ids, all_input_masks, all_segment_ids = e1r_to_bertfeatures(e1,e1,tokenizer,10)
		
	


