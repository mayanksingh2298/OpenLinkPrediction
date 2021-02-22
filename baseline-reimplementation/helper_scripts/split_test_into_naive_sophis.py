"""
	This script counts e1,e2 kitni baare aaye train me for each e1, e2 in test
"""
import sys
sys.path.append(sys.path[0]+"/../")
from kb import kb
from tqdm import trange
test_kb = kb("data/olpbench/validation_data_linked.txt", em_map = None, rm_map = None)
test_triples = test_kb.triples
test_head_answers = test_kb.e1_all_answers
test_tail_answers = test_kb.e2_all_answers
naive_test = set()
train_triples = kb("data/olpbench/train_data_simple.txt", em_map = None, rm_map = None).triples
# train_triples = kb("data/olpbench/delta_simple_thorough.txt", em_map = None, rm_map = None).triples

train = set()
# for i in range(test_triples.shape[0]):
# 	dic[(test_triples[i][0],test_triples[i][2])] = 0

for i in trange(train_triples.shape[0]):
	pair = (train_triples[i][0],train_triples[i][2])
	# if pair not in dic:
	# 	dic[pair] = 0
	# dic[pair]+=1
	train.add(pair)

	#test only
	# if pair in dic:
	# 	dic[pair]+=1
ct = 0
for i in trange(test_triples.shape[0]):
	all_answers = test_tail_answers[i]
	for e2 in all_answers:
		pair = (test_triples[i][0],e2)
		pair_rev = (e2,test_triples[i][0])
		# if pair in train:
		if pair in train or pair_rev in train:
			ct+=1
			naive_test.add(i)
			break
print("#Naive test:",ct)

f1 = open("data/olpbench/validation_data_linked_naive.txt",'w')
f2 = open("data/olpbench/validation_data_linked_sophis.txt",'w')
for i in trange(test_triples.shape[0]):
	to_write = "{}\t{}\t{}\t{}\t{}".format(test_triples[i][0].item(),test_triples[i][1].item(),test_triples[i][2].item(),"|||".join(test_head_answers[i]),"|||".join(test_tail_answers[i]))
	if i in naive_test:
		print(to_write,file=f1)
	else:
		print(to_write,file=f2)
f1.close()
f2.close()


