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

from collections import deque


has_cuda = torch.cuda.is_available()
if not has_cuda:
	utils.colored_print("yellow", "CUDA is not available, using cpu")

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


def main():
	K = 2
	data_dir = "data/olpbench"
	entity_mentions,em_map = utils.read_mentions(os.path.join(data_dir,"mapped_to_ids","entity_id_map.txt"))
	relation_mentions,rm_map = utils.read_mentions(os.path.join(data_dir,"mapped_to_ids","relation_id_map.txt"))

	graph = pickle.load(open("helper_scripts/tmp/graph_thorough_no_r.pkl",'rb'))
	if(0):
		train_kb = kb(os.path.join(data_dir,"train_data_thorough.txt"), em_map = None, rm_map = None)
		# train_kb = kb(os.path.join(data_dir,"test_data.txt"), em_map = None, rm_map = None)
		# train_kb = kb(os.path.join(data_dir,"validation_data_linked.txt"), em_map = None, rm_map = None)

		
		graph = {}
		graph_forward = {}
		graph_backward = {}
		rels_for_e1_e2 = {}

		f = open("helper_scripts/tmp/graph_thorough_no_r.pkl",'wb')
		for triple in tqdm(train_kb.triples):
			e1 = em_map[triple[0].item()]
			r  = rm_map[triple[1].item()]
			e2 = em_map[triple[2].item()]

			if e1 not in graph:
				graph[e1] = []
			graph[e1].append(e2)

			if e2 not in graph:
				graph[e2] = []
			graph[e2].append(e1)

		pickle.dump(graph,f)
		f.close()
		exit()
			# ----------------------------------------------------
			# if (e1,e2) not in rels_for_e1_e2:
			# 	rels_for_e1_e2[(e1,e2)] = []
			# rels_for_e1_e2[(e1,e2)].append(r)

			# if e1 not in graph_forward:
			# 	graph_forward[e1] = set()
			# # graph_forward[e1].add((e2,r))
			# graph_forward[e1].add(e2)


			# if e2 not in graph_backward:
			# 	graph_backward[e2] = set()
			# # graph_backward[e2].add((e1,r))
			# graph_backward[e2].add(e1)





	random.seed(42)
	np.random.seed(42)
	torch.manual_seed(42)

	test_kb = kb(os.path.join(data_dir,"test_data.txt"), em_map = None, rm_map = None)
	keshav_xt_lines = open("helper_scripts/tmp/test_data.txt.tail.xt",'r').readlines()
	f 				= open("helper_scripts/tmp/test_data.txt.tail.xt.all-2-hop-neighbors",'w')

	for i in range(len(keshav_xt_lines)):
		keshav_xt_lines[i] = keshav_xt_lines[i][keshav_xt_lines[i].index(" ")+1:].strip()

	two_hop_data = []
	count = 0 
	for ind,triple in tqdm(enumerate(test_kb.triples),desc="test triples"):
		e1 = em_map[triple[0].item()]
		r  = rm_map[triple[1].item()]
		e2 = em_map[triple[2].item()]
		e1_neighbours = bfs(e1,K,graph)[:10000]
		if e2 in e1_neighbours:
			count += 1
		neighbour_string = ""
		for neighbor in e1_neighbours:
			neighbour_string += "__label__"+str(neighbor)+" "
		to_write = neighbour_string + keshav_xt_lines[ind]
		f.write(to_write+"\n")
		f.flush()
			# print(to_write,file = f)
	print(count)
	f.close()

if __name__=="__main__":
	main()


