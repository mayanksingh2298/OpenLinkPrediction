import os
import ipdb
import argparse
import faiss
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--index_vectors')
parser.add_argument('--query_vectors')
parser.add_argument('--index')
parser.add_argument('--query')
parser.add_argument('--out')

args = parser.parse_args()

def read_vectors_strings(vectors_fp, strings_fp):
    vectors = []
    print('Reading vectors file...')
    for line in tqdm(open(vectors_fp).readlines()):
        vec_values = line.strip('\n').split()
        vec = np.array([float(vec_value) for vec_value in vec_values])
        vectors.append(vec)

    strings = []
    print('Reading strings file')
    for line in tqdm(open(strings_fp).readlines()):
        str_ = line.strip('\n')
        strings.append(str_)

    return np.float32(np.array(vectors)), strings

print('Reading index...')
index_vectors, index_strings = read_vectors_strings(args.index_vectors, args.index)
print('Reading queries...')
query_vectors, query_strings = read_vectors_strings(args.query_vectors, args.query)

index = faiss.IndexFlatL2(100)
index.add(index_vectors)
D, I = index.search(query_vectors, 5)

outf = open(args.out, 'w')
nn = dict()
for query_idx, query in enumerate(query_strings):
    outf.write('Query: '+query+'\n')
    outf.write('Nearest Neighbours: \n')
    nn[query] = []
    for index_idx in I[query_idx]:
        outf.write(index_strings[index_idx]+'\n')
        nn[query].append(index_strings[index_idx])
    outf.write('\n')
    
outf.close()
