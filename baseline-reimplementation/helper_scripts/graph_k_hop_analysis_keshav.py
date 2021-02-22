import copy
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
				}).most_common(K)
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


def func(test_numerator,test_denominator,test_sub_lines,cache_e, test_kb, pid, size, em_map, rm_map, graph, K):
	"""
		test_numerator  : shared dict, test_numerator[pid] represents local number of unfiltered hits1 test elements (for evaluation)
		test_denominator: shared dict, test_denominator[pid] represents local number of test elements (for evaluation;denominator)
		test_sub_lines: shared dict, used to store those test lines where e2 is present in e1 neighbors
		cache_e: shared cache to store entity neighbors
	"""
	test_numerator[pid]  = 0
	test_denominator[pid] = 0
	XT_PATH = "../extremeText-s2/"
	keshav_xt_lines = open("helper_scripts/tmp/test_data.txt.tail.xt",'r').readlines()
	# for i in range(len(keshav_xt_lines)):
	# 	keshav_xt_lines[i] = keshav_xt_lines[i][keshav_xt_lines[i].index(" ")+1:].strip()
	
	ind = 0
	for triple,xt_line in zip(test_kb.triples[pid*size:(pid+1)*size], keshav_xt_lines[pid*size:(pid+1)*size]):
		if ind%20==0:
			if(test_denominator[pid]!=0):
				print("***************************************************************************************************************************************child:",pid,"done:",ind,"partial result:",test_numerator[pid]/test_denominator[pid])
			else:
				print("***************************************************************************************************************************************child:",pid,"done:",ind)
		e1 = em_map[triple[0].item()]
		r  = rm_map[triple[1].item()]
		e2 = em_map[triple[2].item()]

		if e1 not in cache_e:
			e1_neighbours = bfs_2hop(e1,K,graph)
			# cache_e[e1] = e1_neighbours
		else:
			e1_neighbours = cache_e[e1]

		#convert to simple list
		e1_final_neighbours = []
		flag = False
		for neighbor in e1_neighbours:
			if neighbor[0]==e2:
				flag=True
			e1_final_neighbours.append(neighbor[0])

		if flag:
			test_sub_lines[pid].append(xt_line)
			neighbour_string = ""
			for neighbor in e1_final_neighbours:
				neighbour_string += "__label__"+str(neighbor)+" "
			to_write = neighbour_string + xt_line[xt_line.index(" ")+1:].strip()
			#complete eval
			#step 1: write to_write to a file and run xt on it
			os.system("rm helper_scripts/xt_tmp_folder/"+str(pid)+"/*")
			f = open("helper_scripts/xt_tmp_folder/"+str(pid)+"/xt_input.txt",'w')
			f.write(to_write)
			f.close()
			os.system(XT_PATH+"extremetext get-prob "+XT_PATH+"models/tail_thorough_f5_d300_e50.bin helper_scripts/xt_tmp_folder/"+str(pid)+"/xt_input.txt 0 temp.txt 1 > helper_scripts/xt_tmp_folder/"+str(pid)+"/xt_output.txt")

			#step 2: Read the result file written by xt and calc bool value if in hits 1 or not
			#step 3: add this bool result and 1 in global variables (both steps mixed)
			target_entity = e2
			f = open("helper_scripts/xt_tmp_folder/"+str(pid)+"/xt_output.txt")
			line = f.readline().strip().split()
			best_score = -999999999
			best_index = -1
			for i in range(0,len(line),2):
				score = float(line[i+1])
				index = int(line[i][9:])
				if score>best_score:
					best_score = score
					best_index = index
			predicted_entity = best_index
			f.close()
			if predicted_entity==target_entity:
				test_numerator[pid] += 1
			test_denominator[pid] += 1

		# if e2 not in cache_e:
		# 	e2_neighbours = bfs_2hop(e2,K,graph)
		# 	cache_e[e2] = e2_neighbours
		ind += 1

def main():
	K = 1000000000000000000
	data_dir = "data/olpbench"
	entity_mentions,em_map = utils.read_mentions(os.path.join(data_dir,"mapped_to_ids","entity_id_map.txt"))
	relation_mentions,rm_map = utils.read_mentions(os.path.join(data_dir,"mapped_to_ids","relation_id_map.txt"))
	random.seed(42)
	np.random.seed(42)
	torch.manual_seed(42)
	NPROCS = 50
	LOCAL_SIZE = 10000//NPROCS
	# test_kb = kb(os.path.join(data_dir,"validation_data_linked.txt"), em_map = em_map, rm_map = rm_map)
	test_kb = kb(os.path.join(data_dir,"test_data.txt"), em_map = em_map, rm_map = rm_map)
	if (1):
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
		test_numerator = manager.dict()
		test_denominator = manager.dict()
		test_sub_lines = manager.dict()
		for i in range(NPROCS):
			test_sub_lines[i] = manager.list()
		process_array = []
		for i in range(NPROCS):
			os.system("mkdir helper_scripts/xt_tmp_folder/"+str(i))
			p = mp.Process(target = func, args = (test_numerator,test_denominator,test_sub_lines,cache_e,test_kb,i,LOCAL_SIZE,em_map,rm_map,graph,K))
			p.start()
			process_array.append(p)
		for p in process_array:
			p.join()

		num = 0
		den = 0
		final_test_lines = []
		for key in test_numerator:
			num += test_numerator[key]
			den += test_denominator[key]
			final_test_lines.extend(test_sub_lines[key])
		print(num/den)
		print(num/10000)
		f = open("helper_scripts/tmp/test_data-full_neighbors-subset_2hop.txt.tail.xt",'w')
		for line in final_test_lines:
			f.write(line)
		f.close()




if __name__=="__main__":
	main()




