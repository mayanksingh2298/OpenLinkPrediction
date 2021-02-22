import pickle
import ipdb
import os
import argparse
from tqdm import tqdm
import regex as re

parser = argparse.ArgumentParser()
parser.add_argument('--inp_fp')
parser.add_argument('--type') # head, tail
parser.add_argument('--num_frequent', type=int)
args = parser.parse_args()

inp_fp = args.inp_fp
x = open(inp_fp,'r').readlines()

out = open(inp_fp+'.'+args.type+'.xt','w')
out_parallel = open(inp_fp+'.'+args.type+'.xtp','w')

def read_mentions(path):
    mapp = {}
    mentions = []
    lines = open(path,'r').readlines()
    for line in tqdm(lines[1:]):
        line = line.strip().split("\t")
        mentions.append(line[0])
        mapp[line[0]] = len(mapp)
    return mentions,mapp

entityList, entityD = read_mentions('olpbench/mapped_to_ids/entity_id_map.txt')
entityList = dict([(i,el) for i,el in enumerate(entityList)]) # conver to dictionary for faster search operation

if args.num_frequent:
    print('Loading Frequent entities list...')
    frequentD = pickle.load(open('olpbench/r-freq_top100_thorough_'+args.type+'.pkl','rb'))    

line_num = 0
hits1 = 0
for xi in tqdm(x):
    xi = xi.strip('\n')
    fields = xi.split('\t')
    e1, r, e2 = fields[:3]

    if args.type == 'tail':
        # e2_join = '_'.join(e2.split())
        example = '__label__'+str(entityD[e2])+' '+e1+' '+r
    elif args.type == 'head':
        # e1_join = '_'.join(e1.split())
        example = '__label__'+str(entityD[e1])+' '+r+' '+e2 

    if args.num_frequent:
        frequent_str = ''
        if r in frequentD:
            for fi, freq_ent in enumerate(frequentD[r]):
                # if fi == 0:
                #     hits1 += entityD[freq_ent[1]] == e2
                if freq_ent[1] not in entityList:
                    continue
                if fi > args.num_frequent-1:
                    break
                frequent_str += entityList[freq_ent[1]]+' , '
        example = example + ' , ' + frequent_str.strip(' , ')

    out.write(example+'\n')
    example_parallel = '__label__lineparallel_'+str(line_num)+' '+example
    out_parallel.write(example_parallel+'\n')

    line_num += 1

# print("HITS1: ",hits1)
out.close()
out_parallel.close()
