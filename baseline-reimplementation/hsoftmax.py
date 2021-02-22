import heapq
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
from tqdm import tqdm, trange
import datetime
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from kb import kb
import utils
from dataset import Dataset
from balanced_kmeans import kmeans_equal

device = "cpu"
def convert_mention_to_token_indices(data,mapp,length_mapp):
	"""
		data: [id0, id1, ...]
		mapp: [[tokenids for id0], [tokenids for id1], ...]
		length_mapp: [length_for_id0, length_for_id1, ...]

		returns: tensor for mapped_token_indices, tensor for actual lengths
	"""
	mapped_data = []
	lengths = []
	for index in data:
		mapped_data.append(mapp[index])
		lengths.append(length_mapp[index])

	return torch.tensor(mapped_data), torch.tensor(lengths)

class Hsoftmax_node:
	def __init__(self, identity, token, entities):
		"""
			for the entity nodes(leaves), token is entity id
			for the internal nodes, it is a new index with which we'll get its embedding, token is None
			lis_entities stores the list of target entities below this node
		"""
		self.id    = identity
		self.token = token
		# self.freq = freq
		self.left = None
		self.right = None
		self.lis_entities = entities
		
	def __lt__(self, other):
		return self.freq < other.freq
	
	def __gt__(self, other):
		return self.freq > other.freq
	
	def __eq__(self, other):
		if(other == None):
			return False
		if(not isinstance(other, Node)):
			return False
		return self.freq == other.freq

class HuffmanHSoftmax:
	def __init__(self, entity_freq):
		"""
			left is +1 mask
			right is -1 mask
		"""
		self.entity_freq = entity_freq
		self.entity_count = len(entity_freq)
		# self.em_map = em_map
		self.max_tree_depth = None
		# the 2 most important variables
		self.node_indices_for_e = {}
		self.mask_for_e = {}


		# hsoftmax
		global_ct = 0
		heap = []
		self.root = None
		print("Start building tree...")
		for key in entity_freq:
			node = Hsoftmax_node(None, key, entity_freq[key])
			heapq.heappush(heap, node)
		# merge nodes
		pbar = tqdm(total=self.entity_count)
		while(len(heap)>1):
			node1 = heapq.heappop(heap)
			node2 = heapq.heappop(heap)
			
			merged = Hsoftmax_node(global_ct,None, node1.freq + node2.freq)
			global_ct+=1
			merged.left = node1
			merged.right = node2
			heapq.heappush(heap, merged)
			pbar.update(1)
		assert global_ct==self.entity_count-1
		# make codes
		root = heapq.heappop(heap)
		self.root = root
		self.make_codes_helper(root, [], [])
		self.max_tree_depth = self.get_tree_depth(self.root,0)
		self.pad_data()
		self.default_node_indices = [self.entity_count-1]*(self.max_tree_depth-1)
		self.default_mask = [0]*(self.max_tree_depth-1)
		print("Hsoftmax tree built!")


	def make_codes_helper(self, root, current_mask, current_path):
		if(root==None):
			return
		if(root.token != None):
			self.mask_for_e[root.token] = current_mask
			self.node_indices_for_e[root.token] = current_path
			return
		self.make_codes_helper(root.left, current_mask + [1], current_path + [root.id])
		self.make_codes_helper(root.right, current_mask + [-1], current_path + [root.id])

	def get_tree_depth(self,root,curr_depth):
		if(root==None):
			return curr_depth
		if(root.token!=None):
			return curr_depth
		return max(self.get_tree_depth(root.left,curr_depth+1),self.get_tree_depth(root.right,curr_depth+1))

	def pad_data(self):
		for key in self.mask_for_e:
			assert len(self.mask_for_e[key]) == len(self.node_indices_for_e[key]) 
			assert len(self.mask_for_e[key]) <= self.max_tree_depth-1
			for i in range(len(self.mask_for_e[key]),self.max_tree_depth-1):
				self.mask_for_e[key].append(0)
				self.node_indices_for_e[key].append(self.entity_count-1)

class BalancedHSoftmax:
	def __init__(self, num_entities, cut_depth):
		"""
			left is +1 mask
			right is -1 mask
			THE TREE IS SUPPOSED TO BE BALANCED! t referers to nodes at cut. e refers to entity target at leaves
			
			cut_depth                 : at what depth has this tree been cut
			nodes_at_cut              : list of nodes at cut point
			e_for_cut_nodes     : {<cut node id>:<list of entities whose ancestor is this cut node>,...} 

			node_indices_for_e
			mask_for_e 			: store complete indices and mask
			node_indices_for_t
			mask_for_t 			: store indices and mask for the cut_nodes.  
		"""
		# self.em_map = em_map
		self.entity_count = num_entities
		self.max_tree_depth = None
		self.cut_depth = cut_depth
		self.nodes_at_cut = []
		self.e_for_cut_nodes = {}

		self.node_indices_for_e = {}
		self.mask_for_e = {}
		self.node_indices_for_t = {}
		self.mask_for_t = {}

	def split_in_two(self,entities):
		"""
			entities  : is a list of size n which contains index to be used in self.embedding
			# data    : has shape n x dim
			returns : entities1, entities2 lists with half size
		"""
		if len(entities) == 1:
			# already a single element
			raise Exception("Already a single element")
		if len(entities) == 2:
			# only 2 elements
			# return data[0].unsqueeze(0), data[1].unsqueeze(0)
			return [entities[0]], [entities[1]]

		if len(entities) == 3:
			# only 3 elements
			return [entities[0], entities[1]], [entities[2]]

		if len(entities)%2==0:
			# already even
			data = self.embedding(torch.tensor(entities,device=device))
			if len(entities)>=self.entity_count//4:
				data = data[:,:10]
			num_clusters = 2
			cluster_size = data.shape[0]//num_clusters
			choices, centers = kmeans_equal(data.unsqueeze(0), num_clusters=num_clusters, cluster_size=cluster_size)
			choice = choices[0]
			# data1 = data[(choice==0).nonzero().squeeze()]
			# data2 = data[(choice==1).nonzero().squeeze()]
			data1, data2 = [], []
			for i in range(len(entities)):
				if choice[i]==0:
					data1.append(entities[i])
				else:
					data2.append(entities[i])
			return data1, data2
		else:
			data = self.embedding(torch.tensor(entities,device=device))
			if len(entities)>=self.entity_count//4:
				data = data[:,:10]
			num_clusters = 2
			cluster_size = (data.shape[0]-1)//num_clusters
			choices, centers = kmeans_equal(data[:-1].unsqueeze(0), num_clusters=num_clusters, cluster_size=cluster_size)
			choice  = choices[0]
			centers = centers[0]
			# data1 = data[:-1][(choice==0).nonzero().squeeze()]
			# data2 = data[:-1][(choice==1).nonzero().squeeze()]
			data1, data2 = [], []
			for i in range(len(entities)-1):
				if choice[i]==0:
					data1.append(entities[i])
				else:
					data2.append(entities[i])
			if torch.dist(data[-1],centers[0]) < torch.dist(data[-1],centers[1]):
				data1.append(entities[-1])
			else:
				data2.append(entities[-1])
			return data1, data2

	def load_embedding(self):
		# return torch.nn.Embedding(self.entity_count,512)

		# Step 1 get train data
		# DATA_DIR = "/home/mayank/olpbench"
		DATA_DIR = "/home/yatin/mayank/olpbench"
		etokens, etoken_map = utils.get_tokens_map(os.path.join(DATA_DIR,"mapped_to_ids","entity_token_id_map.txt"))
		rtokens, rtoken_map = utils.get_tokens_map(os.path.join(DATA_DIR,"mapped_to_ids","relation_token_id_map.txt"))
		entity_mentions,em_map = utils.read_mentions(os.path.join(DATA_DIR,"mapped_to_ids","entity_id_map.txt"))
		relation_mentions,rm_map = utils.read_mentions(os.path.join(DATA_DIR,"mapped_to_ids","relation_id_map.txt"))
		train_kb = kb(os.path.join(DATA_DIR,"train_data_thorough.txt"), em_map = em_map, rm_map = rm_map)

		# Step 2 get those 2 helper things for relation
		relation_token_indices, relation_lengths = utils.get_token_indices_from_mention_indices(relation_mentions, rtoken_map, maxlen=10, use_tqdm=True)

		# Step 3 for each entity get the top frequent relation
		freq_e = {}
		for triple in tqdm(train_kb.triples):
			# e1 = triple[0].item()
			# r  = triple[1].item()
			# e2 = triple[2].item()
			e1 = em_map[triple[0].item()]
			r  = rm_map[triple[1].item()]
			e2 = em_map[triple[2].item()]
			if e1 not in freq_e:
				freq_e[e1] = {}
			if r not in freq_e[e1]:
				freq_e[e1][r] = 0
			freq_e[e1][r] += 1
			if e2 not in freq_e:
				freq_e[e2] = {}
			if r not in freq_e[e2]:
				freq_e[e2][r] = 0
			freq_e[e2][r] += 1
		for key in tqdm(freq_e):
			freq_e[key] = max(freq_e[key], key = freq_e[key].get)


		# Step 4 get the embedding for that relation and save it in torch.nn.embedding against that entity
		from models import complexLSTM_2_all_e
		model = complexLSTM_2_all_e(196007,39303, 2473409, 512, lstm_dropout=0.1)
		print("Resuming...")
		# checkpoint = torch.load("/home/mayank/olpbench/models/author_data_2lstm_thorough_all-e/checkpoint_epoch_43",map_location="cpu")
		checkpoint = torch.load("/home/yatin/mayank/olpbench/models/checkpoint_epoch_43",map_location="cpu")
		model.load_state_dict(checkpoint['state_dict'])
		embedding = torch.nn.Embedding(self.entity_count,512)
		model.eval()
		model.to("cuda")
		for entity in trange(2473409, desc="Creating entity tensors finally!"):
			# import pdb
			# pdb.set_trace()
			if entity not in freq_e:
				embedding.weight.data[entity] = torch.zeros(512)
			else:
				r_mention_tensor, r_lengths = convert_mention_to_token_indices([freq_e[entity]], relation_token_indices, relation_lengths)
				r_mention_tensor, r_lengths = r_mention_tensor.cuda(), r_lengths.cuda()
				r_real_lstm, r_img_lstm     = model.get_mention_embedding(r_mention_tensor,1,r_lengths)
				r_real_lstm = r_real_lstm[0]
				r_img_lstm  = r_img_lstm[0]
				embedding.weight.data[entity] = torch.cat([r_real_lstm, r_img_lstm]).cpu()
		# import pdb
		# pdb.set_trace()
		return embedding



		# from models import complexLSTM_2_all_e
		# model = complexLSTM_2_all_e(196007,39303, 2473409, 512, lstm_dropout=0.1)
		# print("Resuming...")
		# checkpoint = torch.load("/home/yatin/mayank/olpbench/models/checkpoint_epoch_43",map_location="cpu")
		# # checkpoint = torch.load("/home/mayank/olpbench/models/author_data_2lstm_thorough_all-e/checkpoint_epoch_43",map_location="cpu")
		# model.load_state_dict(checkpoint['state_dict'])
		# return model.E_atomic





	def create_pickled_tree(self):
		self.embedding = self.load_embedding()
		self.embedding.to(device)
		self.global_ct = 0
		self.pbar = tqdm(total=self.entity_count)
		self.root      = Hsoftmax_node(self.global_ct, None, list(range(self.entity_count)))
		self.global_ct += 1
		self.pbar.update(1)
		self.create_pickled_tree_helper(self.root,[],[])
		# pad
		# create other helper variables
		self.max_tree_depth = self.get_tree_depth(self.root,0)
		self.pad_data_and_other_vars()
		self.embedding = None
		self.pbar = None
		self.default_node_indices = [self.entity_count-1]*(self.max_tree_depth)
		self.default_mask = [0]*(self.max_tree_depth)
		# pickle.dump(self,open("/home/mayank/olpbench/PLT/dummy.pkl",'wb'))

		pickle.dump(self,open("/home/mayank/olpbench/hsoftmax/maxfreq-relation_depth-10.pkl",'wb'))
		# pickle.dump(self,open("/home/yatin/mayank/olpbench/models/maxfreq-relation_depth-10.pkl",'wb'))
		# pickle.dump(self,open("/home/yatin/mayank/olpbench/models/atomic-entities_bknn_depth-10.pkl",'wb'))
		# pickle.dump(self,open("/home/mayank/olpbench/hsoftmax/atomic-entities_bknn_depth-10.pkl",'wb'))
		# pickle.dump(self,open("/home/mayank/olpbench/hsoftmax/random_depth-10.pkl",'wb'))


	def create_pickled_tree_helper(self,root,current_path,current_mask):
		if root is None:
			return
		if len(root.lis_entities)==1:
			# child reached
			assert root.id is None and root.token == root.lis_entities[0]			
			self.mask_for_e[root.token] = current_mask
			self.node_indices_for_e[root.token] = current_path

			return
		data1, data2 = self.split_in_two(root.lis_entities)
		# mid = len(root.lis_entities)//2
		# data1, data2 = root.lis_entities[:mid], root.lis_entities[mid:]

		if len(data1)!=1:
			node_left = Hsoftmax_node(self.global_ct, None, data1)
			self.global_ct += 1
			self.pbar.update(1)
		else:
			node_left = Hsoftmax_node(None, data1[0], data1)
		if len(data2)!=1:
			node_right = Hsoftmax_node(self.global_ct, None, data2)
			self.global_ct += 1
			self.pbar.update(1)
		else:
			node_right = Hsoftmax_node(None, data2[0], data2)
		root.left  = node_left
		root.right = node_right
		self.create_pickled_tree_helper(node_left,current_path + [root.id], current_mask + [1])
		self.create_pickled_tree_helper(node_right,current_path + [root.id], current_mask + [-1])
		


	def get_tree_depth(self,root,curr_depth):
		if(root==None):
			return curr_depth
		if(root.token!=None):
			return curr_depth
		return max(self.get_tree_depth(root.left,curr_depth+1),self.get_tree_depth(root.right,curr_depth+1))

	def pad_data_and_other_vars(self):
		self.node_indices_for_t = {}
		self.mask_for_t = {}
		nodes_at_cut = set()
		for key in self.mask_for_e:
			assert len(self.mask_for_e[key]) == len(self.node_indices_for_e[key]) 
			assert len(self.mask_for_e[key]) <= self.max_tree_depth
			for i in range(len(self.mask_for_e[key]),self.max_tree_depth):
				self.mask_for_e[key].append(0)
				self.node_indices_for_e[key].append(self.entity_count-1)
			cut_node = self.node_indices_for_e[key][self.cut_depth]
			nodes_at_cut.add(cut_node)
			if cut_node not in self.e_for_cut_nodes:
				self.e_for_cut_nodes[cut_node] = []
			self.e_for_cut_nodes[cut_node].append(key)
			if cut_node not in self.mask_for_t:
				self.mask_for_t[cut_node] = self.mask_for_e[key][:self.cut_depth]
				self.node_indices_for_t[cut_node] = self.node_indices_for_e[key][:self.cut_depth]

		nodes_at_cut = list(nodes_at_cut)
		nodes_at_cut.sort()
		self.nodes_at_cut = nodes_at_cut

class BalancedPLT:
	def __init__(self, num_entities, cut_depth):
		"""
			left is +1 mask
			right is -1 mask
			THE TREE IS SUPPOSED TO BE BALANCED! t referers to nodes at cut. e refers to entity target at leaves
			
			cut_depth                 : at what depth has this tree been cut
			nodes_at_cut              : list of nodes at cut point
			e_for_cut_nodes     : {<cut node id>:<list of entities whose ancestor is this cut node>,...} 

			*****
			neg_node_indices_for_e
			neg_mask_for_e
			*****

			node_indices_for_e
			mask_for_e 			: store complete indices and mask
			node_indices_for_t
			mask_for_t 			: store indices and mask for the cut_nodes.  
		"""
		# self.em_map = em_map
		self.entity_count = num_entities
		self.max_tree_depth = None
		self.cut_depth = cut_depth
		self.nodes_at_cut = []
		self.e_for_cut_nodes = {}

		self.node_indices_for_e = {}
		self.mask_for_e = {}
		self.neg_node_indices_for_e = {}
		self.neg_mask_for_e = {}
		self.node_indices_for_t = {}
		self.mask_for_t = {}

	def split_in_two(self,entities):
		"""
			entities  : is a list of size n which contains index to be used in self.embedding
			# data    : has shape n x dim
			returns : entities1, entities2 lists with half size
		"""
		if len(entities) == 1:
			# already a single element
			raise Exception("Already a single element")
		if len(entities) == 2:
			# only 2 elements
			# return data[0].unsqueeze(0), data[1].unsqueeze(0)
			return [entities[0]], [entities[1]]

		if len(entities) == 3:
			# only 3 elements
			return [entities[0], entities[1]], [entities[2]]

		if len(entities)%2==0:
			# already even
			data = self.embedding(torch.tensor(entities,device=device))
			if len(entities)>=self.entity_count//4:
				data = data[:,:10]
			num_clusters = 2
			cluster_size = data.shape[0]//num_clusters
			choices, centers = kmeans_equal(data.unsqueeze(0), num_clusters=num_clusters, cluster_size=cluster_size)
			choice = choices[0]
			# data1 = data[(choice==0).nonzero().squeeze()]
			# data2 = data[(choice==1).nonzero().squeeze()]
			data1, data2 = [], []
			for i in range(len(entities)):
				if choice[i]==0:
					data1.append(entities[i])
				else:
					data2.append(entities[i])
			return data1, data2
		else:
			data = self.embedding(torch.tensor(entities,device=device))
			if len(entities)>=self.entity_count//4:
				data = data[:,:10]
			num_clusters = 2
			cluster_size = (data.shape[0]-1)//num_clusters
			choices, centers = kmeans_equal(data[:-1].unsqueeze(0), num_clusters=num_clusters, cluster_size=cluster_size)
			choice  = choices[0]
			centers = centers[0]
			# data1 = data[:-1][(choice==0).nonzero().squeeze()]
			# data2 = data[:-1][(choice==1).nonzero().squeeze()]
			data1, data2 = [], []
			for i in range(len(entities)-1):
				if choice[i]==0:
					data1.append(entities[i])
				else:
					data2.append(entities[i])
			if torch.dist(data[-1],centers[0]) < torch.dist(data[-1],centers[1]):
				data1.append(entities[-1])
			else:
				data2.append(entities[-1])
			return data1, data2

	def load_embedding(self):
		# return torch.nn.Embedding(self.entity_count,512)

		# Step 1 get train data
		# DATA_DIR = "/home/mayank/olpbench"
		DATA_DIR = "/home/yatin/mayank/olpbench"
		etokens, etoken_map = utils.get_tokens_map(os.path.join(DATA_DIR,"mapped_to_ids","entity_token_id_map.txt"))
		rtokens, rtoken_map = utils.get_tokens_map(os.path.join(DATA_DIR,"mapped_to_ids","relation_token_id_map.txt"))
		entity_mentions,em_map = utils.read_mentions(os.path.join(DATA_DIR,"mapped_to_ids","entity_id_map.txt"))
		relation_mentions,rm_map = utils.read_mentions(os.path.join(DATA_DIR,"mapped_to_ids","relation_id_map.txt"))
		train_kb = kb(os.path.join(DATA_DIR,"train_data_thorough.txt"), em_map = em_map, rm_map = rm_map)

		# Step 2 get those 2 helper things for relation
		relation_token_indices, relation_lengths = utils.get_token_indices_from_mention_indices(relation_mentions, rtoken_map, maxlen=10, use_tqdm=True)

		# Step 3 for each entity get the top frequent relation
		freq_e = {}
		for triple in tqdm(train_kb.triples):
			# e1 = triple[0].item()
			# r  = triple[1].item()
			# e2 = triple[2].item()
			e1 = em_map[triple[0].item()]
			r  = rm_map[triple[1].item()]
			e2 = em_map[triple[2].item()]
			if e1 not in freq_e:
				freq_e[e1] = {}
			if r not in freq_e[e1]:
				freq_e[e1][r] = 0
			freq_e[e1][r] += 1
			if e2 not in freq_e:
				freq_e[e2] = {}
			if r not in freq_e[e2]:
				freq_e[e2][r] = 0
			freq_e[e2][r] += 1
		for key in tqdm(freq_e):
			freq_e[key] = max(freq_e[key], key = freq_e[key].get)


		# Step 4 get the embedding for that relation and save it in torch.nn.embedding against that entity
		from models import complexLSTM_2_all_e
		model = complexLSTM_2_all_e(196007,39303, 2473409, 512, lstm_dropout=0.1)
		print("Resuming...")
		# checkpoint = torch.load("/home/mayank/olpbench/models/author_data_2lstm_thorough_all-e/checkpoint_epoch_43",map_location="cpu")
		checkpoint = torch.load("/home/yatin/mayank/olpbench/models/checkpoint_epoch_43",map_location="cpu")
		model.load_state_dict(checkpoint['state_dict'])
		embedding = torch.nn.Embedding(self.entity_count,512)
		model.eval()
		model.to("cuda")
		for entity in trange(2473409, desc="Creating entity tensors finally!"):
			# import pdb
			# pdb.set_trace()
			if entity not in freq_e:
				embedding.weight.data[entity] = torch.zeros(512)
			else:
				r_mention_tensor, r_lengths = convert_mention_to_token_indices([freq_e[entity]], relation_token_indices, relation_lengths)
				r_mention_tensor, r_lengths = r_mention_tensor.cuda(), r_lengths.cuda()
				r_real_lstm, r_img_lstm     = model.get_mention_embedding(r_mention_tensor,1,r_lengths)
				r_real_lstm = r_real_lstm[0]
				r_img_lstm  = r_img_lstm[0]
				embedding.weight.data[entity] = torch.cat([r_real_lstm, r_img_lstm]).cpu()
		# import pdb
		# pdb.set_trace()
		return embedding



		# from models import complexLSTM_2_all_e
		# model = complexLSTM_2_all_e(196007,39303, 2473409, 512, lstm_dropout=0.1)
		# print("Resuming...")
		# checkpoint = torch.load("/home/yatin/mayank/olpbench/models/checkpoint_epoch_43",map_location="cpu")
		# # checkpoint = torch.load("/home/mayank/olpbench/models/author_data_2lstm_thorough_all-e/checkpoint_epoch_43",map_location="cpu")
		# model.load_state_dict(checkpoint['state_dict'])
		# return model.E_atomic





	def create_pickled_tree(self):
		# self.embedding = self.load_embedding()
		# self.embedding.to(device)
		self.global_ct = 0
		self.pbar = tqdm(total=2*self.entity_count)
		self.root      = Hsoftmax_node(self.global_ct, None, list(range(self.entity_count)))
		self.global_ct += 1
		self.pbar.update(1)
		self.create_pickled_tree_helper(self.root,[self.root.id],[1],[],[])
		# pad
		# create other helper variables
		self.max_tree_depth = self.get_tree_depth(self.root,0)
		print("Tree created! Now padding and creating other variables...")
		self.pad_data_and_other_vars()
		self.create_negative_nodes()
		self.embedding = None
		self.pbar = None
		self.default_node_indices = [2*self.entity_count-1]*(self.max_tree_depth)
		self.default_mask = [0]*(self.max_tree_depth)
		# pickle.dump(self,open("/home/mayank/olpbench/PLT/dummy.pkl",'wb'))
		pickle.dump(self,open("/home/yatin/mayank/olpbench/models/dummy.pkl",'wb'))

		# pickle.dump(self,open("/home/yatin/mayank/olpbench/models/PLT-maxfreq-relation_depth-10.pkl",'wb'))
		
		# pickle.dump(self,open("/home/mayank/olpbench/hsoftmax/maxfreq-relation_depth-10.pkl",'wb'))
		# pickle.dump(self,open("/home/yatin/mayank/olpbench/models/maxfreq-relation_depth-10.pkl",'wb'))
		# pickle.dump(self,open("/home/yatin/mayank/olpbench/models/atomic-entities_bknn_depth-10.pkl",'wb'))
		# pickle.dump(self,open("/home/mayank/olpbench/hsoftmax/atomic-entities_bknn_depth-10.pkl",'wb'))
		# pickle.dump(self,open("/home/mayank/olpbench/hsoftmax/random_depth-10.pkl",'wb'))


	def create_pickled_tree_helper(self,root,current_path,current_mask,neg_current_path,neg_current_mask):
		if root is None:
			return
		if len(root.lis_entities)==1:
			# child reached
			# assert root.token == root.lis_entities[0]			
			root.token = root.lis_entities[0]
			self.mask_for_e[root.token] = current_mask
			self.node_indices_for_e[root.token] = current_path

			self.neg_mask_for_e[root.token] = neg_current_mask
			self.neg_node_indices_for_e[root.token] = neg_current_path

			return
		# data1, data2 = self.split_in_two(root.lis_entities)
		mid = len(root.lis_entities)//2
		data1, data2 = root.lis_entities[:mid], root.lis_entities[mid:]

		# if len(data1)!=1:
		node_left = Hsoftmax_node(self.global_ct, None, data1)
		self.global_ct += 1
		self.pbar.update(1)
		# else:
		# 	node_left = Hsoftmax_node(None, data1[0], data1)
		# if len(data2)!=1:
		node_right = Hsoftmax_node(self.global_ct, None, data2)
		self.global_ct += 1
		self.pbar.update(1)
		# else:
			# node_right = Hsoftmax_node(None, data2[0], data2)
		root.left  = node_left
		root.right = node_right
		self.create_pickled_tree_helper(node_left, current_path + [node_left.id],  current_mask + [1], neg_current_path + [node_right.id], neg_current_mask + [1])
		self.create_pickled_tree_helper(node_right,current_path + [node_right.id], current_mask + [1], neg_current_path + [node_left.id],  neg_current_mask + [1])
		


	def get_tree_depth(self,root,curr_depth):
		if(root==None):
			return curr_depth
		# if(root.token!=None):
		# 	return curr_depth
		return max(self.get_tree_depth(root.left,curr_depth+1),self.get_tree_depth(root.right,curr_depth+1))

	def pad_data_and_other_vars(self):
		self.node_indices_for_t = {}
		self.mask_for_t = {}
		nodes_at_cut = set()
		for key in tqdm(self.mask_for_e, desc="Creating other variables:"):
			assert len(self.mask_for_e[key])     == len(self.node_indices_for_e[key]) 
			assert len(self.neg_mask_for_e[key]) == len(self.neg_node_indices_for_e[key]) 
			assert len(self.mask_for_e[key]) <= self.max_tree_depth
			for i in range(len(self.mask_for_e[key]),self.max_tree_depth):
				self.mask_for_e[key].append(0)
				self.node_indices_for_e[key].append(2*self.entity_count-1)
			for i in range(len(self.neg_mask_for_e[key]),self.max_tree_depth):
				self.neg_mask_for_e[key].append(0)
				self.neg_node_indices_for_e[key].append(2*self.entity_count-1)
			cut_node = self.node_indices_for_e[key][self.cut_depth]
			nodes_at_cut.add(cut_node)
			if cut_node not in self.e_for_cut_nodes:
				self.e_for_cut_nodes[cut_node] = []
			self.e_for_cut_nodes[cut_node].append(key)
			if cut_node not in self.mask_for_t:
				self.mask_for_t[cut_node] = self.mask_for_e[key][:self.cut_depth+1]
				self.node_indices_for_t[cut_node] = self.node_indices_for_e[key][:self.cut_depth+1]

		nodes_at_cut = list(nodes_at_cut)
		nodes_at_cut.sort()
		self.nodes_at_cut = nodes_at_cut
	def create_negative_nodes(self):
		#step 1: get all nodes above and including cut nodes
		nodes_above_cut = set()
		for node in tqdm(self.nodes_at_cut, desc="negative nodes (1/4):"):
			lis = self.node_indices_for_t[node]
			for x in lis:
				nodes_above_cut.add(x)

		#step 2: get all nodes for each cut node
		nodes_below_cut = {} #{<cut node index>:<set of all nodes below this>,...}
		for node in tqdm(self.nodes_at_cut, desc="negative nodes (2/4):"):
			nodes_below_cut[node] = set()
			entities_for_cut = self.e_for_cut_nodes[node]
			for entity in entities_for_cut:
				lis = self.node_indices_for_e[entity][self.cut_depth+1:]
				for x in lis:
					nodes_below_cut[node].add(x)

		#step 3: for each entity add step1 and step2 nodes minus its gold nodes
		maxlen = -1
		for entity in tqdm(range(self.entity_count), desc="negative nodes (3/4):"):
			cut_node = self.node_indices_for_e[entity][self.cut_depth]
			self.neg_node_indices_for_e[entity] = list(nodes_above_cut.union(nodes_below_cut[cut_node]).difference(set(self.node_indices_for_e[entity])).difference(set([2*self.entity_count-1])))
			self.neg_mask_for_e[entity] = [1] * len(self.neg_node_indices_for_e[entity])
			maxlen = max(maxlen,len(self.neg_mask_for_e[entity]))

		#step 4: pad it!
		for entity in tqdm(range(self.entity_count), desc="negative nodes (4/4):"):
			for i in range(len(self.neg_mask_for_e[entity]),maxlen):
				self.neg_mask_for_e[entity].append(0)
				self.neg_node_indices_for_e[entity].append(2*self.entity_count-1)
# use this in a python interactive shell
# from hsoftmax import BalancedHSoftmax
# b1 = BalancedHSoftmax(2473409,10)
# b1.create_pickled_tree()

# from hsoftmax import BalancedPLT
# b1 = BalancedPLT(2473409,10)
# b1.create_pickled_tree()



# b1 = BalancedHSoftmax(10001,2)
# b1.load_embedding()
# N = 11
# b1.embedding = torch.nn.Embedding(N,2)
# # data = torch.tensor([[1,2],[1,2],[2,1],[2,1.]])
# data = torch.rand(N, 3, device=device)
# data[:5,:]+=1
# data[5:,:]-=1
# b1.embedding.weight.data = data
# data1, data2 = b1.split_in_two(list(range(N)))


# save to disk
# b1.embedding = None
# b1.pbar = None
# pickle.dump(b1,open("/home/mayank/olpbench/hsoftmax/random_depth-10.pkl",'wb'))
# # pickle.dump(b1,open("/home/mayank/olpbench/hsoftmax/atomic-entities_bknn_depth-10.pkl",'wb'))

# import pdb
# pdb.set_trace()