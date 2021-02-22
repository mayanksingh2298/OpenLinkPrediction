from transformers import AutoTokenizer
import argparse
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--inp', type=str)
parser.add_argument('--out', type=str)
parser.add_argument('--model_str', type=str)
args = parser.parse_args()

# args.model_str = 'bert-large-uncased'
args.out = args.inp[:-3] + args.model_str + '.pkl'
tokenizer = AutoTokenizer.from_pretrained(args.model_str, do_lower_case=False, use_fast=True, data_dir='data/pretrained_cache')
f = open(args.inp)
tokensD = dict()

def clean(token_str):
    if ':impl_' not in token_str:
        return token_str
    else:
        return token_str.split(':impl_')[0]

for i, line in tqdm(enumerate(f)):
    if i == 0: # skip header
        continue
    text = clean(line.split('\t')[0])
    tokensD[text] = tokenizer.encode_plus(text)['input_ids'][1:-1]       

print("length of dictionary", len(tokensD))
pickle.dump(tokensD, open(args.out,'wb'))