import os
import pickle
import ipdb
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--inp') # olpbench/train_data_simple.txt
parser.add_argument('--out') # olpbench/train_data_simple.txt.facts
args = parser.parse_args()

inpf = open(args.inp,'r')
outf = open(args.out,'w')

for line in tqdm(inpf.readlines()):
    fields = line.strip('\n').split('\t')
    fact = fields[0]+' '+fields[1]+' '+fields[2]
    outf.write(fact+'\n')
outf.close()