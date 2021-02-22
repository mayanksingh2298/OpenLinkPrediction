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


has_cuda = torch.cuda.is_available()
if not has_cuda:
    utils.colored_print("yellow", "CUDA is not available, using cpu")

def main():
    data_dir = "../olpbench"
    freq_r_tail = {}
    freq_r_head = {}
    entity_mentions,em_map = utils.read_mentions(os.path.join(data_dir,"mapped_to_ids","entity_id_map.txt"))
    _,rm_map = utils.read_mentions(os.path.join(data_dir,"mapped_to_ids","relation_id_map.txt"))

    # train_kb = kb(os.path.join(data_dir,"test_data.txt"), em_map = None, rm_map = None)
    train_kb = kb(os.path.join(data_dir,"train_data_thorough.txt"), em_map = None, rm_map = None)
    for triple in tqdm(train_kb.triples, desc="getting r freq"):
        e1 = triple[0].item()
        r  = triple[1].item()
        e2 = triple[2].item()
        if r not in freq_r_tail:
            freq_r_tail[r] = {}
        if em_map[e2] not in freq_r_tail[r]:
            freq_r_tail[r][em_map[e2]] = 0
        freq_r_tail[r][em_map[e2]] += 1

        if r not in freq_r_head:
            freq_r_head[r] = {}
        if em_map[e1] not in freq_r_head[r]:
            freq_r_head[r][em_map[e1]] = 0
        freq_r_head[r][em_map[e1]] += 1

    f = open("../olpbench/r-freq_top100_thorough_head.pkl","wb")
    final_data = {}
    for r in freq_r_head:
        final_list = list(zip(list(freq_r_head[r].values()),list(freq_r_head[r].keys())))
        final_list.sort(reverse=True)
        final_list = final_list[:100]
        final_data[r] = final_list
    pickle.dump(final_data,f)
    f.close()

    f = open("../olpbench/r-freq_top100_thorough_tail.pkl","wb")
    final_data = {}
    for r in freq_r_tail:
        final_list = list(zip(list(freq_r_tail[r].values()),list(freq_r_tail[r].keys())))
        final_list.sort(reverse=True)
        final_list = final_list[:100]
        final_data[r] = final_list
    pickle.dump(final_data,f)
    f.close()

if __name__=="__main__":
    main()


