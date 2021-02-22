import os
import pickle
import ipdb
import sys
from tqdm import tqdm
import faiss

'''
f = open('olpbench/train_data_thorough.txt','r')
e1_dict = {}
lines = f.readlines()
for line in tqdm(lines):
    line = line.strip('\n')
    e1, r, e2, _, _ = line.split('\t')
    key = r+' '+e2
    if key not in e1_dict:
        e1_dict[key] = [e1]
    else:
        e1_dict[key].append(e1)
pickle.dump(e1_dict, open('olpbench/tail_thorough_rfacts_e1dict', 'wb'))
'''

index_vecs = pickle.load(open('olpbench/train_data_thorough.txt.tail.xt.vecs.pkl','rb'))
# index = faiss.IndexFlatL2(300)
# index.add(index_vecs)

query_vecs = pickle.load(open('olpbench/test_data.txt.tail.xt.vecs.pkl','rb'))

index_f = open('olpbench/train_data_thorough.txt','r')
query_f = open('olpbench/test_data.txt.tail_thorough_f5_d300_e50.stage1','r')

out_f = open('olpbench/test_data.txt.tail_thorough_f5_d300_e50.stage1.ret','w')

def read_mentions(path):
    mapp = {}
    mentions = []
    lines = open(path,'r').readlines()
    for line in tqdm(lines[1:]): 
        line = line.strip().split("\t")
        mentions.append(line[0])
        mapp[line[0]] = len(mapp)
    return mentions,mapp

relation_mentions, rm_map = read_mentions('olpbench/mapped_to_ids/entity_id_map.txt')

e2_dict = {}
print('Reading index file...')
for line_idx, line in tqdm(enumerate(index_f)):
    e1, r, e2, _, _ = line.split('\t')
    e2_idx = rm_map[e2]
    if e2_idx in e2_dict:
        e2_dict[e2_idx] = e2_dict[e2_idx].append((e1, r, line_idx))
    else:
        e2_dict[e2_idx] = [(e1, r, line_idx)]

print('Reading query file...')
for line_idx, line in tqdm(enumerate(query_f)):
    e1, r, e2, _, _, sample_e2s = line.split('\t')
    query_vec = query_vecs[line_idx]
    for se2 in sample_e2s:
        sample_vecs = []
        for (se1, sr, line_idx) in e2_dict[se2]:
            if se1 == e1 and sr == r and se2 == rm_map[e2]: 
                # can happen only for train
                continue
            sample_vecs.append(index_vecs[line_idx])
