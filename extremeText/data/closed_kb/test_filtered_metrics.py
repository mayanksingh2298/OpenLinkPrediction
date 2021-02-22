import numpy as np
import argparse
import os
from tqdm import trange
import torch
import copy
import pickle

def read_map(path):
	lis = []
	mapp = {}
	lines = open(path).readlines()
	for line in lines[:-1]: # dont take the last oov
		line = line.strip().split("\t")
		lis.append(line[1])
		mapp[line[1]] = len(mapp)
	return lis,mapp

def get_xt_preds(path):
	lines = open(path).readlines()
	data = []
	for line in lines:
		line = line.strip().split()
		labels = []
		scores = []
		for i in range(len(line)):
			if i%2==0:
				assert '__label__' in line[i]
				labels.append(int(line[i][9:]))	
			else:
				scores.append(float(line[i]))
		data.append([labels,scores])
	return data

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dataset', help="Name of the dataset as in data folder", required=True)
	parser.add_argument('-m', '--mode', help="test|valid|train", required=True)

	args = parser.parse_args()
	entities, entity_map = read_map(os.path.join(args.dataset,"entities.dict"))
	relations, relation_map = read_map(os.path.join(args.dataset,"relations.dict"))
	if args.mode not in ['test','train','valid']:
		raise Exception("mode should be in ['test','train','valid']. Aborting!")
	# load test data
	mapped_test_facts = []
	lines = open(os.path.join(args.dataset,args.mode+".txt")).readlines()
	for line in lines:
		line = line.strip().split("\t")
		e1 = entity_map.get(line[0], len(entity_map))
		r  = relation_map.get(line[1], len(relation_map))
		e2 = entity_map.get(line[2], len(entity_map))
		mapped_test_facts.append((e1,r,e2))

	# load head and tail predictions
	xt_preds_head = get_xt_preds(os.path.join(args.dataset,args.mode+".head.preds.txt"))
	xt_preds_tail = get_xt_preds(os.path.join(args.dataset,args.mode+".tail.preds.txt"))

	# load knowns data here
	all_knowns_head = pickle.load(open(os.path.join(args.dataset,"all_knowns_head.pkl"),'rb'))
	all_knowns_tail = pickle.load(open(os.path.join(args.dataset,"all_knowns_tail.pkl"),'rb'))

	if args.mode == "train":
		pkl_outpath = os.path.join(args.dataset,"train_data.pkl")
	elif args.mode == "test":
		pkl_outpath = os.path.join(args.dataset,"test_data.pkl")
	elif args.mode == "valid":
		pkl_outpath = os.path.join(args.dataset,"validation_data.pkl")
	pickled_data = {}
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

	for i in trange(len(mapped_test_facts)):
		e1 = mapped_test_facts[i][0]
		r  = mapped_test_facts[i][1]
		e2 = mapped_test_facts[i][2]

		key = (e1,r,e2)
		pickled_data[key] = {'head-batch':{}, 'tail-batch':{}}



		knowns_tail = all_knowns_tail[(e1,r)]
		scores_tail = torch.tensor([-999999.]*(len(entities)+1))
		labels,scores = xt_preds_tail[i]		
		pickled_data[key]['tail-batch']['confidence'] = np.array(scores[:11])
		pickled_data[key]['tail-batch']['index']      = np.array(labels[:11])
		pickled_data[key]['tail-batch']['bias']       = list(knowns_tail)
		scores_tail[labels] = torch.tensor(scores)
		expected_score = copy.deepcopy(scores_tail[e2])
		scores_tail[list(knowns_tail)] = -999999
		greater = scores_tail.gt(expected_score).float()
		equal = scores_tail.eq(expected_score).float()
		rank = greater.sum()+1+equal.sum()/2.0
		metrics['mr_t'] += rank.item()
		metrics['mrr_t'] += (1.0/rank).item()
		metrics['hits1_t'] += (rank.le(1).float()).item()
		metrics['hits10_t'] += (rank.le(10).float()).item()
		metrics['hits50_t'] += (rank.le(50).float()).item()
		pickled_data[key]['tail-batch']['score'] = {
				'MRR':(1/rank).item(),
				'MR':rank.item(),
				'HITS@1':float((rank<=1).item()),
				'HITS@3':float((rank<=3).item()),
				'HITS@10':float((rank<=10).item())
			}

		knowns_head = all_knowns_head[(e2,r)]
		scores_head = torch.tensor([-999999.]*(len(entities)+1))
		labels,scores = xt_preds_head[i]		
		pickled_data[key]['head-batch']['confidence'] = np.array(scores[:11])
		pickled_data[key]['head-batch']['index']      = np.array(labels[:11])
		pickled_data[key]['head-batch']['bias']       = list(knowns_head)
		scores_head[labels] = torch.tensor(scores)
		expected_score = copy.deepcopy(scores_head[e1])
		scores_head[list(knowns_head)] = -999999
		greater = scores_head.gt(expected_score).float()
		equal = scores_head.eq(expected_score).float()
		rank = greater.sum()+1+equal.sum()/2.0
		metrics['mr_h'] += (rank).item()
		metrics['mrr_h'] += (1.0/rank).item()
		metrics['hits1_h'] += (rank.le(1).float()).item()
		metrics['hits10_h'] += (rank.le(10).float()).item()
		metrics['hits50_h'] += (rank.le(50).float()).item()
		pickled_data[key]['head-batch']['score'] = {
				'MRR':(1/rank).item(),
				'MR':rank.item(),
				'HITS@1':float((rank<=1).item()),
				'HITS@3':float((rank<=3).item()),
				'HITS@10':float((rank<=10).item())
			}

		metrics['mr'] = (metrics['mr_h']+metrics['mr_t'])/2
		metrics['mrr'] = (metrics['mrr_h']+metrics['mrr_t'])/2
		metrics['hits1'] = (metrics['hits1_h']+metrics['hits1_t'])/2
		metrics['hits10'] = (metrics['hits10_h']+metrics['hits10_t'])/2
		metrics['hits50'] = (metrics['hits50_h']+metrics['hits50_t'])/2

	for key in metrics:
		metrics[key] = (metrics[key] / len(mapped_test_facts))
	print(metrics)
	pickle.dump(pickled_data,open(pkl_outpath,'wb'))




