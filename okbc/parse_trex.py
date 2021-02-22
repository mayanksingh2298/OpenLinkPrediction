import os
import json
import ipdb
from tqdm import tqdm

out_f = open('olpbench/all_triples.txt','w')
all_entities, all_predicates = set(), set()
for file_ in os.listdir('/home/keshav/t-rex'):
    file_ = '/home/keshav/t-rex/'+file_
    print('Parsing file: ', file_)
    if not file_.endswith('.json'):
        continue
    js = json.load(open(file_,'r'))
    for example in tqdm(js):
        for triple in example['triples']:
            out_f.write(triple['subject']['uri']+'\t'+triple['predicate']['uri']+'\t'+triple['object']['uri']+'\n')
            all_entities.add(triple['subject']['uri'])
            all_entities.add(triple['object']['uri'])
            all_predicates.add(triple['predicate']['uri'])
            
print('Total Entities: ', all_entities)
print('Total Predicates: ', all_predicates)
out_f.close()            