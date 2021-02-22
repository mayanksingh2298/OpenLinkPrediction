"""
    e2e
    yago train tail
    python xt_input.py --inp yago/train.txt --out yago/train --type tail --relation2text_file yago/relation2text.txt --entity2text_file yago/entity2text.txt --entity_id_file yago/entities.dict

    fb15k237 train tail
    python xt_input.py --inp fb15k237/train.txt --out fb15k237/train --type tail --relation2text_file fb15k237/relation2text.txt --entity2text_file fb15k237/entity2text.txt --entity_id_file fb15k237/entities.dict
"""

import pickle
import ipdb
import os
import argparse
from tqdm import tqdm
import regex as re
parser = argparse.ArgumentParser()
parser.add_argument('--inp_fp')
parser.add_argument('--out_fp')
parser.add_argument('--type') # head, tail
parser.add_argument('--relation2text_file')
parser.add_argument('--entity2text_file')
parser.add_argument('--entity_id_file')
args = parser.parse_args()

def read_ids(path):
    dic = {}
    ct  = 0
    lines = open(path).readlines()
    for line in lines:
        line = line.strip().split('\t')[1]
        dic[line] = ct
        ct+=1
    return dic

def read_dict(file_path):
    outputD = dict()
    all_values = set()
    with open(file_path, 'r') as fin:
        for line in fin:
            key, value = line.strip().split('\t')
            i = 0
            orig_value = value
            while value in all_values:
                i += 1
                value = orig_value + " " + str(i)
            all_values.add(value)
            outputD[key] = value
    return outputD


if __name__ == '__main__':
    inp_fp = args.inp_fp
    lines = open(inp_fp,'r').readlines()
    out = open(args.out_fp+'.'+args.type+'.xt','w')
    # out_parallel = open(inp_fp+'.'+args.type+'.xtp','w')

    entity_int_map    = read_ids(args.entity_id_file)
    entity_text_map   = read_dict(args.entity2text_file)
    relation_text_map = read_dict(args.relation2text_file)
    

    # line_num = 0
    # hits1 = 0
    for line in tqdm(lines):
        line = line.strip('\n').split("\t")
        e1, r, e2 = line

        if args.type == 'tail':
            # e2_join = '_'.join(e2.split())
            example = '__label__'+str(entity_int_map.get(e2,len(entity_int_map)-1))+' '+entity_text_map[e1]+' '+relation_text_map[r]
        elif args.type == 'head':
            # e1_join = '_'.join(e1.split())
            example = '__label__'+str(entity_int_map.get(e1,len(entity_int_map)-1))+' '+relation_text_map[r]+' '+entity_text_map[e2] 

        out.write(example+'\n')
        # example_parallel = '__label__lineparallel_'+str(line_num)+' '+example
        # out_parallel.write(example_parallel+'\n')

        # line_num += 1

    # print("HITS1: ",hits1)
    out.close()
    # out_parallel.close()
