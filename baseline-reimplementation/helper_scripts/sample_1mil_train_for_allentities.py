"""
	This gets the entities in test and val sets.
	Then gets random entities from train set such that the final_entities count is 300000

	Then gets 5 facts(if possible) from train_thorough for each of these entities.
"""


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
import utils
import random

def get_entites(triples):
	"""returns a set of entities"""
	e_set = set()
	for triple in triples:
		e1 = triple[0].item()
		r  = triple[1].item()
		e2 = triple[2].item()
			
		e_set.add(e1)
		e_set.add(e2)

	return e_set

data_dir = "data/olpbench"

train_kb = kb(os.path.join(data_dir,"train_data_thorough.txt"), em_map = None, rm_map = None)
test_kb = kb(os.path.join(data_dir,"test_data.txt"), em_map = None, rm_map = None)
val_kb = kb(os.path.join(data_dir,"validation_data_linked.txt"), em_map = None, rm_map = None)

train_entities = list(get_entites(train_kb.triples))
random.shuffle(train_entities)

test_entities = get_entites(test_kb.triples)
val_entities = get_entites(val_kb.triples)

final_entities = test_entities.union(val_entities)
print("Expanding final entities set...")

i = 0
while len(final_entities)<300000:
	final_entities.add(train_entities[i])
	i+=1	

print("Loaded final entities set")


entity_train_facts = {}
for i,triple in tqdm(enumerate(train_kb.triples),desc="getting facts for each entity"):
	e1 = triple[0].item()
	r  = triple[1].item()
	e2 = triple[2].item()

	if e1 not in entity_train_facts:
		entity_train_facts[e1] = []
	entity_train_facts[e1].append(i)

	if e2 not in entity_train_facts:
		entity_train_facts[e2] = []
	entity_train_facts[e2].append(i)

final_indices = set()
for ent in tqdm(final_entities):
    val = entity_train_facts.get(ent,[])[:5]
    for i in val:
        final_indices.add(i)

to_write = open("train_data_thorough_1mil.txt",'w')
for ind in final_indices:
	e1 = train_kb.triples[ind][0].item()
	r  = train_kb.triples[ind][1].item()
	e2 = train_kb.triples[ind][2].item()
	e1_alt = train_kb.e1_all_answers[ind]
	e2_alt = train_kb.e2_all_answers[ind]
	print("{}\t{}\t{}\t{}\t{}".format(e1,r,e2,"|||".join(e1_alt),"|||".join(e2_alt)),file=to_write)



