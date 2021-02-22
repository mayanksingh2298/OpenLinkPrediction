"""
	This script counts e1,e2 kitni baare aaye train me for each e1, e2 in test
"""
import sys
sys.path.append(sys.path[0]+"/../")
from kb import kb
from tqdm import trange
test_triples = kb("data/olpbench/test_data.txt", em_map = None, rm_map = None).triples

train_triples = kb("data/olpbench/train_data_simple.txt", em_map = None, rm_map = None).triples

dic = {}
# for i in range(test_triples.shape[0]):
# 	dic[(test_triples[i][0],test_triples[i][2])] = 0

for i in trange(train_triples.shape[0]):
	pair = (train_triples[i][0],train_triples[i][2])
	if pair not in dic:
		dic[pair] = 0
	dic[pair]+=1

	#test only
	# if pair in dic:
	# 	dic[pair]+=1
val = list(dic.values())
print(sum(val)/len(val))
# breakpoint()
