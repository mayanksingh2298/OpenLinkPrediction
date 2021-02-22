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
    data_dir = "data/olpbench"
    freq_r_tail = {}
    freq_r_head = {}
    entity_mentions,em_map = utils.read_mentions(os.path.join(data_dir,"mapped_to_ids","entity_id_map.txt"))
    _,rm_map = utils.read_mentions(os.path.join(data_dir,"mapped_to_ids","relation_id_map.txt"))

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


    test_kb = kb(os.path.join(data_dir,"test_data.txt"), em_map = em_map, rm_map = rm_map)
    answers_t = []
    answers_h = []

    for triple in test_kb.triples:
        r = triple[1].item()
        val = freq_r_tail.get(r,{})
        this_answer = []
        for key in val:
            this_answer.append([key,val[key]])
        answers_t.append(this_answer)

        val = freq_r_head.get(r,{})
        this_answer = []
        for key in val:
            this_answer.append([key,val[key]])
        answers_h.append(this_answer)
    metrics = utils.get_metrics_using_topk(os.path.join(data_dir,"all_knowns_thorough_linked.pkl"),test_kb,answers_t,answers_h,em_map,rm_map)
    print(metrics)
    
    # random.seed(42)
    # np.random.seed(42)
    # torch.manual_seed(42)

    # print("Loading all_known pickled data...(takes times since large)")
    # all_known_e2 = {}
    # all_known_e1 = {}
    # all_known_e2,all_known_e1 = pickle.load(open(os.path.join(data_dir,"all_knowns_simple_linked.pkl"),"rb"))

    # metrics = {}
    # metrics['mr'] = 0
    # metrics['mrr'] = 0
    # metrics['hits1'] = 0
    # metrics['hits10'] = 0
    # metrics['hits50'] = 0
    # metrics['mr_t'] = 0
    # metrics['mrr_t'] = 0
    # metrics['hits1_t'] = 0
    # metrics['hits10_t'] = 0
    # metrics['hits50_t'] = 0
    # metrics['mr_h'] = 0
    # metrics['mrr_h'] = 0
    # metrics['hits1_h'] = 0
    # metrics['hits10_h'] = 0
    # metrics['hits50_h'] = 0

    # def convert_to_list(freq_dict,r):
    #     to_return = [0]*len(entity_mentions)
    #     val = freq_dict.get(r,{})
    #     for key in val:
    #         to_return[key] = val[key]
    #     return torch.tensor(to_return)

    # for ind, triple in tqdm(enumerate(test_kb.triples), desc="Test dataloader"):
    #     e1 = triple[0].item()
    #     r  = triple[1].item()
    #     e2 = triple[2].item()

    #     simi_t = convert_to_list(freq_r_tail,r)
    #     simi_h = convert_to_list(freq_r_head,r)

    #     this_correct_mentions_e2 = test_kb.e2_all_answers[int(ind)]
    #     this_correct_mentions_e1 = test_kb.e1_all_answers[int(ind)] 

    #     all_correct_mentions_e2 = all_known_e2.get((em_map[test_kb.triples[int(ind)][0]],rm_map[test_kb.triples[int(ind)][1]]),[])
    #     all_correct_mentions_e1 = all_known_e1.get((em_map[test_kb.triples[int(ind)][2]],rm_map[test_kb.triples[int(ind)][1]]),[])
        
    #     # compute metrics
    #     best_score = simi_t[this_correct_mentions_e2].max()
    #     if best_score==0:
    #         rank = torch.tensor(99999.)
    #     else:
    #         simi_t[all_correct_mentions_e2] = -20000000 # MOST NEGATIVE VALUE
    #         greatereq = simi_t.ge(best_score).float()
    #         equal = simi_t.eq(best_score).float()
    #         rank = greatereq.sum()+1+equal.sum()/2.0
    #     metrics['mr_t'] += rank
    #     metrics['mrr_t'] += 1.0/rank
    #     metrics['hits1_t'] += rank.le(1).float()
    #     metrics['hits10_t'] += rank.le(10).float()
    #     metrics['hits50_t'] += rank.le(50).float()

    #     best_score = simi_h[this_correct_mentions_e1].max()
    #     if best_score == 0:
    #         rank = torch.tensor(99999.)
    #     else:
    #         simi_h[all_correct_mentions_e1] = -20000000 # MOST NEGATIVE VALUE
    #         greatereq = simi_h.ge(best_score).float()
    #         equal = simi_h.eq(best_score).float()
    #         rank = greatereq.sum()+1+equal.sum()/2.0
    #     metrics['mr_h'] += rank
    #     metrics['mrr_h'] += 1.0/rank
    #     metrics['hits1_h'] += rank.le(1).float()
    #     metrics['hits10_h'] += rank.le(10).float()
    #     metrics['hits50_h'] += rank.le(50).float()

    #     metrics['mr'] = (metrics['mr_h']+metrics['mr_t'])/2
    #     metrics['mrr'] = (metrics['mrr_h']+metrics['mrr_t'])/2
    #     metrics['hits1'] = (metrics['hits1_h']+metrics['hits1_t'])/2
    #     metrics['hits10'] = (metrics['hits10_h']+metrics['hits10_t'])/2
    #     metrics['hits50'] = (metrics['hits50_h']+metrics['hits50_t'])/2



    # for key in metrics:
    #     metrics[key] = metrics[key] / len(test_kb.triples)
    # print(metrics)

    



if __name__=="__main__":
    main()


