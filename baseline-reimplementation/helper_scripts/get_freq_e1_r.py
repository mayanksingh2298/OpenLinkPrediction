"""
	This script counts for each test triple freq of (e1,r) in train. Similarly for (e2,r) of test
"""
import sys
sys.path.append(sys.path[0]+"/../")
from kb import kb
from tqdm import trange
import torch

test_triples = kb("data/olpbench/test_data.txt", em_map = None, rm_map = None).triples

train_triples = kb("data/olpbench/train_data_thorough.txt", em_map = None, rm_map = None).triples

dic = {}

# for i in range(test_triples.shape[0]):
# 	dic[(test_triples[i][0],test_triples[i][2])] = 0

for i in trange(train_triples.shape[0]):
	e1 = train_triples[i][0].item()
	r  = train_triples[i][1].item()
	e2 = train_triples[i][2].item()

	if (e1,r) not in dic:
		dic[(e1,r)] = 0
	dic[(e1,r)] += 1

	if (r,e2) not in dic:
		dic[(r,e2)] = 0
	dic[(r,e2)] += 1

	if e1 not in dic:
		dic[e1] = 0
	dic[e1] += 1

	if e2 not in dic:
		dic[e2] = 0
	dic[e2] += 1


head_test = []
tail_test = []
head_test_single = []
tail_test_single = []

for i in trange(test_triples.shape[0]):
	e1 = test_triples[i][0].item()
	r  = test_triples[i][1].item()
	e2 = test_triples[i][2].item()	
	head_test.append(dic.get((r,e2),0))
	tail_test.append(dic.get((e1,r),0))

	head_test_single.append(dic.get(e1,0))
	tail_test_single.append(dic.get(e2,0))


head_test = torch.tensor(head_test)
tail_test = torch.tensor(tail_test)
head_test_single = torch.tensor(head_test_single)
tail_test_single = torch.tensor(tail_test_single)

import pdb
pdb.set_trace()