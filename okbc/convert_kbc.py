import os
import pickle
import sys
import ipdb
import argparse
from tqdm import tqdm
from os.path import expanduser

# kge_dir_path = expanduser('~')+'/KnowledgeGraphEmbedding/'
# kbe_dir_path = expanduser('~')+'/kg-bert/'
olp_dir_path = 'olpbench'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--kge_output') # Knowledge Graph Embeddings
parser.add_argument('--kge_data_dir')
parser.add_argument('--kbe_data_dir') # KB Bert Embeddings
parser.add_argument('--output_dir')
parser.add_argument('--output_file') # input to mcqBERT
parser.add_argument('--model')

parser.add_argument('--filter', action='store_true')
parser.add_argument('--filter_val', action='store_true')
parser.add_argument('--scores', action='store_true')
parser.add_argument('--scores_val', action='store_true')
parser.add_argument('--predictions', action='store_true')
parser.add_argument('--entity_map', action='store_true')
parser.add_argument('--relation_map', action='store_true')
parser.add_argument('--performance', action='store_true')

UNK_NAME = "unknown"
args = parser.parse_args()
os.system("mkdir -p {}".format(args.output_dir))
# args.kge_output = os.path.join(args.kge_output,'models',args.kge_output)
args.kge_data_dir = os.path.join(args.kge_data_dir,'data',args.dataset)
args.kbe_data_dir = os.path.join(args.kbe_data_dir,'data',args.dataset)
# args.output_dir = os.path.join(olp_dir_path,args.dataset)

# if args.filter or args.scores:
#     if '_filter.pkl' not in args.kge_output:
#         print('!!!Warning!!! Computing scores/filterations on file which does not use test filteration')

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

kge_entity_index_id = read_dict(os.path.join(args.kge_data_dir, 'entities.dict'))
kge_relation_index_id = read_dict(os.path.join(args.kge_data_dir, 'relations.dict'))
print('Loaded KGE entiy and relation dictionaries...')

kbe_entity_id_name = read_dict(os.path.join(args.kbe_data_dir, 'entity2text.txt'))
kbe_relation_id_name = read_dict(os.path.join(args.kbe_data_dir, 'relation2text.txt'))
print('Loaded KG-BERT entiy and relation dictionaries...')

predictions = pickle.load(open(args.kge_output,'rb'))
print('Loaded KGE Output...')

if args.predictions:
    output_head_f = open(args.output_dir+'/'+args.output_file+'.head_'+args.model+'.stage1','w')
    output_tail_f = open(args.output_dir+'/'+args.output_file+'.tail_'+args.model+'.stage1','w')

all_known_e2, all_known_e1, scoresD_tail, scoresD_head = {}, {}, {}, {}
print('Processing KGE Output...')
tail_correct, head_correct, base_tail_correct, base_head_correct, total = 0, 0, 0, 0, 0
all_head_false, all_tail_false = 0, 0

if args.performance:
    all_known_e2, all_known_e1 = pickle.load(open(args.output_dir+'/all_knowns_simple_linked.pkl','rb'))
    scoresD_tail, scoresD_head = pickle.load(open(args.output_dir+'/scores_'+args.model+'.pkl','rb'))

for tuple_indx, tuple_ in enumerate(tqdm(predictions)):
    total += 1

    e1_index, r_index, e2_index = str(tuple_[0]), str(tuple_[1]), str(tuple_[2])
    e1_name = kbe_entity_id_name.get(kge_entity_index_id[e1_index],UNK_NAME)
    r_name = kbe_relation_id_name[kge_relation_index_id[r_index]]
    e2_name = kbe_entity_id_name.get(kge_entity_index_id[e2_index],UNK_NAME)
    try:
        head_index = predictions[tuple_]['head-batch']['index'].tolist()
    except:
        head_index = predictions[tuple_]['head-batch']['index']
    head_index = [kbe_entity_id_name.get(kge_entity_index_id[str(h_index)],UNK_NAME) for h_index in head_index]
    if head_index[0] == e1_name:
        base_head_correct += 1
    try:
        head_confidence = predictions[tuple_]['head-batch']['confidence'].tolist()
    except:
        head_confidence = predictions[tuple_]['head-batch']['confidence']

    head_results = str(list(zip(head_index, head_confidence)))
    if args.predictions:    
        output_head_f.write(e1_name+'\t'+r_name+'\t'+e2_name+'\t'+e1_name+'\t'+e2_name+'\t'+head_results+'\n')

    try:
        tail_index = predictions[tuple_]['tail-batch']['index'].tolist()
    except:
        tail_index = predictions[tuple_]['tail-batch']['index']

    tail_index = [kbe_entity_id_name.get(kge_entity_index_id[str(t_index)],UNK_NAME) for t_index in tail_index]
    if tail_index[0] == e2_name:
        base_tail_correct += 1    
    try:
        tail_confidence = predictions[tuple_]['tail-batch']['confidence'].tolist()
    except:
        tail_confidence = predictions[tuple_]['tail-batch']['confidence']
    tail_results = str(list(zip(tail_index, tail_confidence)))
    if args.predictions:
        output_tail_f.write(e1_name+'\t'+r_name+'\t'+e2_name+'\t'+e1_name+'\t'+e2_name+'\t'+tail_results+'\n')

    if args.filter or args.filter_val:
        tail_bias = [kbe_entity_id_name.get(kge_entity_index_id[str(p)],UNK_NAME) for p in predictions[tuple_]['tail-batch']['bias']]
        head_bias = [kbe_entity_id_name.get(kge_entity_index_id[str(p)],UNK_NAME) for p in predictions[tuple_]['head-batch']['bias']]
        if (e1_name, r_name) in all_known_e2:
            all_known_e2[(e1_name, r_name)].extend(tail_bias)
            all_known_e2[(e1_name, r_name)] = list(set(all_known_e2[(e1_name, r_name)]))
        else:    
            all_known_e2[(e1_name, r_name)] = tail_bias
        if (e2_name, r_name) in all_known_e1:
            all_known_e1[(e2_name, r_name)].extend(head_bias)
            all_known_e1[(e2_name, r_name)] = list(set(all_known_e1[(e2_name, r_name)]))
        else:
            all_known_e1[(e2_name, r_name)] = head_bias
    if args.scores or args.scores_val:
        head_scores = predictions[tuple_]['head-batch']['score']
        scoresD_head[(e1_name, r_name, e2_name)] = head_scores
        tail_scores = predictions[tuple_]['tail-batch']['score']
        scoresD_tail[(e1_name, r_name, e2_name)] = tail_scores

    if args.performance:
        ## Code for computing filtered confidence
        tail_bias = [kbe_entity_id_name.get(kge_entity_index_id[str(p)],UNK_NAME) for p in predictions[tuple_]['tail-batch']['bias']]
        head_bias = [kbe_entity_id_name.get(kge_entity_index_id[str(p)],UNK_NAME) for p in predictions[tuple_]['head-batch']['bias']]

        tail_bias = list(set(all_known_e2[(e1_name, r_name)])-set([e2_name]))
        head_bias = list(set(all_known_e1[(e2_name, r_name)])-set([e1_name]))
        tail_scores = scoresD_tail[(e1_name, r_name, e2_name)]
        head_scores = scoresD_head[(e1_name, r_name, e2_name)]

        filtered_tail_confidence, filtered_head_confidence = [], []
        conf_sum = 0
        for (tp, tc) in zip(tail_index, tail_confidence):
            if tp in tail_bias:       
                filtered_tail_confidence.append((tp, -1))
                conf_sum += -1
            else:
                filtered_tail_confidence.append((tp, tc))
                conf_sum += tc
        if conf_sum == -1*len(tail_index) and tail_scores['HITS@1'] == 1:
            tail_correct += 1
        filtered_tail_confidence = sorted(filtered_tail_confidence, key=lambda x: x[1], reverse=True)
        if filtered_tail_confidence[0][0] == e2_name:
            tail_correct += 1
        conf_sum = 0
        for (hp, hc) in zip(head_index, head_confidence):
            if hp in head_bias:       
                filtered_head_confidence.append((hp, -1))
                conf_sum += -1
            else:
                filtered_head_confidence.append((hp, hc))
                conf_sum += hc
        if conf_sum == -1*len(head_index) and head_scores['HITS@1'] == 1:
            head_correct += 1
        filtered_head_confidence = sorted(filtered_head_confidence, key=lambda x: x[1], reverse=True)            
        if filtered_head_confidence[0][0] == e1_name:
            head_correct += 1            

print('Base Head Accuracy: ', (base_head_correct/total*1.0))
print('Base Tail Accuracy: ', (base_tail_correct/total*1.0))
print('Base Accuracy: ', ((base_head_correct+base_tail_correct)/(total*2.0)))

print('Head HITS@1: ',(head_correct/total*1.0))    
print('Tail HITS@1: ',(tail_correct/total*1.0))    
print('Total Accuracy: ', (head_correct+tail_correct)/(total*2.0))

if args.predictions:
    print('Head predictions written to ',output_head_f.name)
    output_head_f.close()
    print('Tail predictions written to ',output_tail_f.name)
    output_tail_f.close()

if args.filter:
    filter_f = open(args.output_dir+'/all_knowns_simple_linked.pkl','wb')
    print('Writing Filter to ', filter_f.name)
    pickle.dump([all_known_e2, all_known_e1], filter_f)
    
if args.filter_val:
    filter_f = open(args.output_dir+'/all_knowns_simple_linked.pkl','rb')
    all_known_e2_test, all_known_e1_test = pickle.load(filter_f)

    for (e1_name, r_name) in all_known_e2:
        if (e1_name, r_name) in all_known_e2_test:
            all_known_e2[(e1_name, r_name)].extend(all_known_e2_test[(e1_name, r_name)])
            all_known_e2[(e1_name, r_name)] = list(set(all_known_e2[(e1_name, r_name)]))
    for (e1_name, r_name) in all_known_e2_test:
        if (e1_name, r_name) in all_known_e2:
            continue
        all_known_e2[(e1_name, r_name)] = all_known_e2_test[(e1_name, r_name)]

    for (e2_name, r_name) in all_known_e1:            
        if (e2_name, r_name) in all_known_e1_test:
            all_known_e1[(e2_name, r_name)].extend(all_known_e1_test[(e2_name, r_name)])
            all_known_e1[(e2_name, r_name)] = list(set(all_known_e1[(e2_name, r_name)]))
    for (e2_name, r_name) in all_known_e1_test:            
        if (e2_name, r_name) in all_known_e1:
            continue
        all_known_e1[(e2_name, r_name)] = all_known_e1_test[(e2_name, r_name)]
    

    filter_f = open(args.output_dir+'/all_knowns_simple_linked.pkl','wb')
    print('Writing Filter to ', filter_f.name)
    pickle.dump([all_known_e2, all_known_e1], filter_f)

if args.scores:
    scores_f = open(args.output_dir+'/scores_'+args.model+'.pkl','wb')
    print('Writing Scores to ', scores_f.name)
    pickle.dump([scoresD_tail, scoresD_head], scores_f)
if args.scores_val:
    scores_f = open(args.output_dir+'/scores_'+args.model+'.pkl','rb')
    scoresD_tail_test, scoresD_head_test = pickle.load(scores_f)

    scoresD_tail.update(scoresD_tail_test)
    scoresD_head.update(scoresD_head_test)
    scores_f = open(args.output_dir+'/scores_'+args.model+'.pkl','wb')
    print('Writing Scores to ', scores_f.name)
    pickle.dump([scoresD_tail, scoresD_head], scores_f)

if args.entity_map:
    os.makedirs(args.output_dir+'/mapped_to_ids', exist_ok=True)
    f = open(args.output_dir+'/mapped_to_ids/entity_id_map.txt','w')
    print('Writing Entity Map to ',f.name)
    f.write('# token\tid\n')
    for index, id_ in kge_entity_index_id.items():
        name = kbe_entity_id_name.get(id_,UNK_NAME)
        f.write(name+'\t'+index+'\n')
    f.close()
if args.relation_map:
    print('Relation Map...')
    os.makedirs(args.output_dir+'/mapped_to_ids', exist_ok=True)
    f = open(args.output_dir+'/mapped_to_ids/relation_id_map.txt','w')
    print('Writing Relation Map to ', f.name)
    f.write('# token\tid\n')
    for index, id_ in kge_relation_index_id.items():
        name = kbe_relation_id_name.get(id_,UNK_NAME)
        f.write(name+'\t'+index+'\n')
    f.close()
