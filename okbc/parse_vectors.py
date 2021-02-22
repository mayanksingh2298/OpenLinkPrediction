import os
import pickle
import ipdb
import argparse
import faiss
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--inp')

args = parser.parse_args()

def read_vectors(vectors_fp):
    vectors = []
    print('Reading vectors file...')
    for line in tqdm(open(vectors_fp)):
        vec_values = line.strip('\n').split()
        vec = np.array([float(vec_value) for vec_value in vec_values])
        vectors.append(vec)
    return np.float32(np.array(vectors))

index_vectors = read_vectors(args.inp)
pickle.dump(index_vectors, open(args.inp+'.pkl','wb'), protocol=-1)

# index = faiss.IndexFlatL2(300)
# index.add(index_vectors)
# faiss.write_index(index, args.vectors+'.bin')
