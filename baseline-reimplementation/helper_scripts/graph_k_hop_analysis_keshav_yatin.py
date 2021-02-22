from multiprocessing import Manager
import sys
import os
sys.path.append(sys.path[0]+"/../")
import argparse
import logging
import os
from collections import defaultdict, Counter
import pickle
import pprint
import multiprocessing as mp
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

from collections import deque


has_cuda = torch.cuda.is_available()
if not has_cuda:
	utils.colored_print("yellow", "CUDA is not available, using cpu")

def bfs_2hop(e1,k,graph):
	"""
		This function returns the top k 2 hop neighbors of e1
		graph: {
					"e1":{
							"e2":#edges(e1,e2)
					}
				} 

		returns: Counter({
					"e2": #count,...
				})
	"""
	init_neighbors = graph.get(e1,{})
	final_e2 = defaultdict(int)

	for entity,freq in init_neighbors.items():
		final_e2[entity] += freq # adding the 1 hop neighbors

		for e2,freq_2 in graph.get(entity,{}).items():
			final_e2[e2] += freq*freq_2 # adding 2 hop neighbors
	
	final_e2[e1] = 0
	final_e2 = Counter(final_e2)
	return final_e2.most_common(k)

	# keys = list(final_e2.keys())
	# sort(keys, key = lambda x: final_e2[x], reverse=True)
	# return keys[:k]

def convert_to_list(e_counter, total_e_size):
	to_return = [0]*len(total_e_size)
	for key,val in e_counter:
		to_return[key] = val
	return torch.tensor(to_return)

def bfs(e1,k,graph):
	"""
		starting from e1, does a bfs until k hop using graph adj_list
		returns: all 2 hop e2s for this e1 
	"""
	init_neighbors = graph.get(e1,[])
	# for neighbor in init_neighbors:
		# neighbor.append(0)
	# 	if neighbor[0]==e2:
	# 		return True

	queue = deque(init_neighbors)
	queue.append("LEVEL") # to demarkate levels in bfs
	# queue has elements: [[e,r],...,"LEVEL",...]	
	current_level = 1
	while queue!=[]:
		popped = queue.popleft()
		if popped=="LEVEL":
			current_level += 1
			return list(queue)
			queue.append("LEVEL")
		else:	
			# current_level = popped[2]
			# if len(queue)%1000==0:
			# 	print(current_level,"queue:",len(queue))
			if current_level<k:
				neighbors = graph.get(popped,[])
				queue.extend(neighbors)
			# for neighbor in tqdm(neighbors):
			# 	# neighbor.append(current_level+1)
			# 	if neighbor[0]==e2:
			# 		return True
			# 	queue.append(neighbor)

	return []

def custom_intersection(s1,s2):
	"""
		s1 has elements {(x1,y1),...}
		s2 has elements {(x2,y2),...}

		returns a new set which is the intersection in some sense but only considers xi during intersecting
	"""
	final_set = set()
	for (x1,y1) in s1:
		for (x2,y2) in s2:
			if x1==x2:
				final_set.add((y1,x2,y2))
	return final_set	

def forw_back_check(e1,e2,graph_forward,graph_backward,rels_for_e1_e2):
	"""
		starting from e1, does a bfs until k hop using graph adj_list
		returns: True if there is some intersection in graph_forward[e1] and graph_backward[e2]
	"""
	intersection = graph_forward.get(e1,set()).intersection(graph_backward.get(e2,set()))
	# intersection = custom_intersection(graph_forward.get(e1,set()), graph_backward.get(e2,set()))
	if intersection:
		final_proof = []
		for e_ in intersection:
			final_proof.append((rels_for_e1_e2.get((e1,e_),[]),e_,rels_for_e1_e2.get((e_,e2),[])))
		return True,[e1,e2,final_proof]
	return False,None


def func(cache_e, test_kb, pid, size, em_map, rm_map, graph, K):
	for ind,triple in enumerate(test_kb.triples[pid*size:(pid+1)*size]):
		if ind%100==0:
			print("child:",pid,"done:",ind)
		e1 = em_map[triple[0].item()]
		r  = rm_map[triple[1].item()]
		e2 = em_map[triple[2].item()]

		if e1 not in cache_e:
			e1_neighbours = bfs_2hop(e1,K,graph)
			cache_e[e1] = e1_neighbours
		if e2 not in cache_e:
			e2_neighbours = bfs_2hop(e2,K,graph)
			cache_e[e2] = e2_neighbours
		# else:
		# 	e1_neighbours = cache_e[e1]

def main():
	K = 1000
	data_dir = "data/olpbench"
	entity_mentions,em_map = utils.read_mentions(os.path.join(data_dir,"mapped_to_ids","entity_id_map.txt"))
	relation_mentions,rm_map = utils.read_mentions(os.path.join(data_dir,"mapped_to_ids","relation_id_map.txt"))
	random.seed(42)
	np.random.seed(42)
	torch.manual_seed(42)
	test_kb = kb(os.path.join(data_dir,"validation_data_linked.txt"), em_map = em_map, rm_map = rm_map)
	cache_e = pickle.load(open("helper_scripts/tmp/top_1000_neighbors_val.pkl",'rb'))
	if (0):
		graph = pickle.load(open("helper_scripts/tmp/graph_thorough_no_r_count_paths.pkl",'rb'))
		if(0):
			train_kb = kb(os.path.join(data_dir,"train_data_thorough.txt"), em_map = None, rm_map = None)
			graph = {}

			f = open("helper_scripts/tmp/graph_thorough_no_r_count_paths.pkl",'wb')
			for triple in tqdm(train_kb.triples):
				e1 = em_map[triple[0].item()]
				r  = rm_map[triple[1].item()]
				e2 = em_map[triple[2].item()]

				if e1 not in graph:
					graph[e1] = defaultdict(int)
				graph[e1][e2] += 1

				if e2 not in graph:
					graph[e2] = defaultdict(int)
				graph[e2][e1] += 1

			pickle.dump(graph,f)
			f.close()
			exit()
		manager = Manager()
		cache_e = manager.dict()
		process_array = []
		for i in range(10):
			p = mp.Process(target = func, args = (cache_e,test_kb,i,1000,em_map,rm_map,graph,K))
			p.start()
			process_array.append(p)
		for p in process_array:
			p.join()
		cache_e_final = {}
		for key in cache_e:
			cache_e_final[key] = cache_e[key]
		f = open("helper_scripts/tmp/top_10000_neighbors_val.pkl",'wb')
		pickle.dump(cache_e_final,f)
		f.close()
		exit()
	answers_t = []
	answers_h = []
	for ind,triple in enumerate(test_kb.triples):
		e1 = em_map[triple[0].item()]
		r  = rm_map[triple[1].item()]
		e2 = em_map[triple[2].item()]
		answers_t.append(cache_e[e1])
		answers_h.append(cache_e[e2])
	
	metrics = utils.get_metrics_using_topk(os.path.join(data_dir,"all_knowns_thorough_linked.pkl"),test_kb,answers_t,answers_h,em_map,rm_map)
	print(metrics)



if __name__=="__main__":
	main()


