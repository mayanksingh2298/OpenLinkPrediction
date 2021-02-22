import os
import ipdb
import sys
import pickle
from tqdm import tqdm
import regex as re

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--inp') 
parser.add_argument('--model')
parser.add_argument('--type')
parser.add_argument('--parallel', action='store_true') 

args = parser.parse_args()

def read_mentions(path):
	mapp = {}
	mentions = []
	lines = open(path,'r').readlines()
	for line in tqdm(lines[1:]): 
		line = line.strip().split("\t")
		mentions.append(line[0])
		mapp[line[0]] = len(mapp)
	return mentions,mapp

entitiesList, entitiesD = read_mentions('olpbench/mapped_to_ids/entity_id_map.txt')

def parse_prediction_line(line):
    input_line, pred_line = line.split('</s>')
    input_line = input_line.strip('\n')
    try:
        input_line_no = int(input_line.split()[0].split('__label__lineparallel_')[1])
    except:
        input_line_no = -1
    pred_line = pred_line.strip('\n')

    predicted_labels = []
    for pred in pred_line.split():
        if '__label__' in pred:
            label = int(pred.split('__label__')[1])
        else:
            prob = float(pred)            
            predicted_labels.append([label, prob])
    
    return input_line_no, predicted_labels

if args.type == 'head' or args.type == 'tail':
    dump_f = open(args.inp+'.'+args.type+'_'+args.model+'.stage1','w')

    skipped, hits1 = 0, 0
    input_lines = open(args.inp,'r').readlines()
    predicted_lines = open(args.inp+'.'+args.type+'_'+args.model+'.preds','r').readlines()

    for predicted_line in predicted_lines:
        input_line_no, predicted_labels = parse_prediction_line(predicted_line)
        
        if input_line_no == -1: # either because line is not parse correctly or some labels are missing
            skipped += 1
            print('Examples skipped = ',skipped)
            continue        
        input_line = input_lines[input_line_no].strip('\n')
        if entitiesList[predicted_labels[0][0]] == input_line.split('\t')[2]:
            hits1 += 1
        dump_line = input_line+'\t'+str(predicted_labels)
        dump_f.write(dump_line+'\n')

    print('HITS1: ', hits1)
    dump_f.close()
else:
    print('Error! Code not yet written for combined')
    sys.exit(1)
