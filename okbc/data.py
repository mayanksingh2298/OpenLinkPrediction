# Utilities for getting data
import os
import ipdb
import pdb
import random
import argparse
import spacy
# import pickle
from pickle5 import pickle
import csv
import nltk
import numpy as np
from tqdm import tqdm
import ast
import utils
from typing import Dict, List, Tuple
from overrides import overrides
import copy

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp_data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field, ListField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, Tokenizer, PretrainedTransformerTokenizer
from allennlp.data import vocabulary, allennlp_collate

from allennlp_data.data_loaders import PyTorchDataLoader, AllennlpDataset, AllennlpLazyDataset
from allennlp_data.data_loaders import MultiProcessDataLoader

from allennlp_data.samplers import MaxTokensBatchSampler

class LabelDatasetReader(DatasetReader):
    def __init__(self, hparams, tokensD):
        super().__init__()
        self.hparams = hparams
        self._tokenizer =  PretrainedTransformerTokenizer('bert-base-cased')
        self._token_indexer = {"tokens": PretrainedTransformerIndexer('bert-base-cased')}
        self._tokensD = tokensD

    @overrides
    def _read(self, labels_fp):
        labels_list, _ = utils.read_mentions(labels_fp)
        for mcat_index, label in enumerate(labels_list):
            yield self.text_to_instance(label, mcat_index)

    def ft_indices(self, text):
        all_indices, all_subwords = [], []
        for word in text.split():
            sub_words, indices = self.ft_model.get_subwords(word)
            all_indices.append(indices)
            all_subwords.append(sub_words)
        return all_indices, all_subwords

    def clean(self, token_str):
        if ':impl_' not in token_str:
            return token_str
        else:
            return token_str.split(':impl_')[0]

    def text_to_instance(self, label, mcat_index):
        fields = {}
        if self.hparams.model_type=="bert" or self.hparams.model_type=="lstm":
            bert_tokens = [101]+self._tokensD[self.clean(label)]+[102]
            # if label in self._tokensD:
                # bert_tokens = [101]+self._tokensD[label]+[102]
            # else:
            #     bert_tokens = [101]+self._tokenizer.tokenize(label)+[102]
            fields["labels"] = ArrayField(np.array(bert_tokens), dtype=np.long)
        elif self.hparams.model_type=="ft":
            all_indices, _ = self.ft_indices(label)
            fields['labels'] = ListField([ArrayField(indices, dtype=np.long, padding_value=self.padding_idx) for indices in all_indices])


        fields['meta_data'] = MetadataField({'labels_index': mcat_index})
        return Instance(fields)

class KBCDatasetReader(DatasetReader):
    def __init__(
        self,
        hparams,
        em_dict,
        rm_dict,
        entity_mentions,
        tokensD, 
        all_known_e2, 
        all_known_e1,
        scoresD_tail,
        scoresD_head,
        mode,
        max_instances,
        world_size
    ) -> None:
        super().__init__(manual_distributed_sharding=True, manual_multi_process_sharding=True, max_instances=max_instances, world_size=world_size)
        self.hparams = hparams
        self.em_dict = em_dict
        self.rm_dict = rm_dict
        self.entity_mentions = entity_mentions
        self._tokenizer =  PretrainedTransformerTokenizer('bert-base-cased')
        self._token_indexer = {"tokens": PretrainedTransformerIndexer('bert-base-cased')}
        self._tokensD = tokensD
        self.random_indexes = list(range(len(em_dict)))
        random.shuffle(self.random_indexes)
        self.key_index = 0
        self.all_known_e1 = all_known_e1
        self.all_known_e2 = all_known_e2
        self.scoresD_tail = scoresD_tail
        self.scoresD_head = scoresD_head
        self.mode = mode
        self.factsD = None
        if hparams.retrieve_facts:
            self.factsD = pickle.load(open(hparams.retrieve_facts,'rb'))

    def load_pickle(self, path):
        d = {}
        f = open(path, 'rb')
        while True:
            try:
                d.update(pickle.load(f))
            except EOFError:
                break
        return d

    @overrides
    def _read(self, inp_file):
        line_no = 0
        if '.both_' in inp_file:
            inp_files = [inp_file.replace('.both_', '.head_'), inp_file.replace('.both_', '.tail_')]
        else:
            inp_files = [inp_file]

        for inp_file in inp_files:
            self.hparams.task_type = 'head' if '.head_' in inp_file else 'tail'
            for line in self.shard_iterable(open(inp_file, 'r')):
                line_no += 1
                fields = line.strip('\n').split('\t')

                stage1_samples, stage1_scores = None, None
                if self.hparams.stage1_model: stage1_samples = ast.literal_eval(fields[5])
                if self.hparams.ckbc:
                    if self.hparams.task_type == 'tail':
                        stage1_scores = self.scoresD_tail.get((fields[0],fields[1],fields[2]), {})
                    elif self.hparams.task_type == 'head':
                        stage1_scores = self.scoresD_head.get((fields[0],fields[1],fields[2]), {})

                instance = None
                if self.hparams.task_type == 'tail' or self.hparams.task_type == 'both':
                    instance = self.text_to_instance(fields[0]+" "+fields[1],fields[0],fields[1],fields[2],fields[3].split("|||"),fields[4].split("|||"), self.mode, 'tail', stage1_samples, stage1_scores, self.all_known_e2)
                    if instance != None: yield instance

                instance = None
                if self.hparams.task_type == 'head' or self.hparams.task_type == 'both':
                    instance = self.text_to_instance(fields[1]+" "+fields[2],fields[2],fields[1],fields[0],fields[4].split("|||"),fields[3].split("|||"), self.mode, 'head', stage1_samples, stage1_scores, self.all_known_e1)
                    if instance != None: yield instance

    def ft_indices(self, text):
        all_indices, all_subwords = [], []
        for word in text.split():
            sub_words, indices = self.ft.get_subwords(word)
            all_indices.append(indices)
            all_subwords.append(sub_words)
        return all_indices, all_subwords

    def clean(self, token_str):
        if ':impl_' not in token_str:
            return token_str
        else:
            return token_str.split(':impl_')[0]

    def text_to_instance(  # type: ignore
        self, x, e1, r, e2, e1_alt_mentions, e2_alt_mentions, mode, task, stage1_samples, stage1_scores, all_known
    ) -> Instance:  

        if self.hparams.stage2:
            if self.hparams.model_type=="ft" or self.hparams.model_type=="lstm":
                raise NotImplementedError("ft and lstm not designed for stage 2 yet")

            fields: Dict[str, Field] = {}    
            y = e2
            e2_index = self.em_dict[e2]
            if stage1_samples:
                stage1_samples = stage1_samples[:self.hparams.negative_samples]
                if self.hparams.round_robin:
                    # Input: [1,2,3,4,5,6,7,8,9,10], chunks of size 2, #chunks = 5
                    # Output: [1,6,2,7,3,8,4,9,5,10]
                    # ASSUMING: len(samples) is a multiple of chunk_size
                    cycled_stage1_samples = []
                    chunk_size = self.hparams.chunk_size
                    chunk_count = len(stage1_samples) // chunk_size
                    for i in range(chunk_count):
                        for j in range(i, len(stage1_samples), chunk_count):
                            cycled_stage1_samples.append(stage1_samples[j])
                    stage1_samples = cycled_stage1_samples

                if type(stage1_samples[0]) == type([]) or type(stage1_samples[0]) == type(()): # [index, score] is provided
                    samples_index = [sample[0] for sample in stage1_samples]
                    scores = [sample[1] for sample in stage1_samples]
                    fields["scores"] = ArrayField(np.array(scores))
                else:
                    samples_index = stage1_samples
                if mode != 'test':
                    if self.hparams.add_missing_e2:
                        if e2_index not in samples_index:
                            samples_index = samples_index[:-1]
                            samples_index = samples_index+[e2_index]
                    if self.hparams.shuffle:
                        random.shuffle(samples_index)                
            else:
                if self.key_index > len(self.random_indexes):
                    random.shuffle(self.random_indexes)
                    self.key_index = 0
                samples_index = self.random_indexes[self.key_index:self.key_index+self.hparams.negative_samples+1]
                self.key_index = self.key_index+self.hparams.negative_samples+1

                if e2_index not in samples_index:
                    samples_index = samples_index[:-1] + [e2_index]
                random.shuffle(samples_index)

            if type(samples_index[0]) != type("a"):
                samples = [self.entity_mentions[sample_index] for sample_index in samples_index]
            else:
                samples = samples_index
                samples_index = [self.em_dict[sample] for sample in samples]
                
            
            targets = []
            for sample in samples:
                if sample == e2:
                    targets.append(1)
                else:
                    targets.append(0)
            
            if self.hparams.model == 'hmcq':
                fields["target"] = ArrayField(np.array(targets), dtype=np.long)
                if task == 'tail': 
                    query = [101] + [10] + self._tokensD[self.clean(e1)] + [12] + self._tokensD[self.clean(r)] + [102]
                elif task == 'head':
                    query = [101] + [11] + self._tokensD[self.clean(r)] + [12] + self._tokensD[self.clean(e1)] + [102]
                    
                fields["query"] = ArrayField(np.array(query), dtype=np.long)      
                chunk_count = self.hparams.negative_samples // self.hparams.chunk_size          
                max_len, total_len = -1, len(query)
                all_samples_tokens, XsampleY = [], []
                for sample_idx, sample in enumerate(samples):
                    if sample_idx % self.hparams.chunk_size == 0:
                        if total_len > 510:
                            print('Input is greater than 512!')
                            return None        
                        total_len = len(query)
                    clean_sample = self.clean(sample)
                    total_len += len(self._tokensD[clean_sample])+1
                    # max_len = max(len(self._tokensD[clean_sample])+1, max_len)
                    all_samples_tokens.append(self._tokensD[clean_sample]+[102])
                    XsampleY.append([101]+query+self._tokensD[clean_sample]+[102])
                if self.hparams.kg_bert:
                    fields["XsampleY"] = ListField([ArrayField(np.array(sample), dtype=np.long) for sample in XsampleY])
                fields["samples"] = ListField([ArrayField(np.array(st), dtype=np.long) for st in all_samples_tokens])

            elif self.hparams.model == 'mcq':
                if task == 'tail': 
                    x = [101] + [10] + self._tokensD[self.clean(e1)] + [12] + self._tokensD[self.clean(r)]
                elif task == 'head':
                    x = [101] + [11] + self._tokensD[self.clean(r)] + [12] + self._tokensD[self.clean(e1)]
                all_sample_indexs = []
                target_idx = -1
                x = x + [102]
                for sample_idx, sample in enumerate(samples):
                    clean_sample = self.clean(sample)                    

                    fact_tokens = []
                    if self.factsD:
                        facts = []
                        if r+' '+sample in self.factsD:
                            facts = self.factsD[r+' '+sample][:3]
                        for fact in facts:
                            if fact == e1:
                                continue
                            fact_tokens.extend(self._tokensD[self.clean(fact)])
                            fact_tokens.append(10)                        

                    if not self.hparams.xt_results and len(x)+len(self._tokensD[clean_sample]) > 510:
                        print('Input is greater than 512!')
                        return None
                        
                    all_sample_indexs.append(range(len(x),len(x)+len(fact_tokens)+len(self._tokensD[clean_sample])))
                    x.extend(fact_tokens+self._tokensD[clean_sample])
                    x.append(102)

                # if task == 'head':
                #     x = x + self._tokensD[self.clean(r)] + self._tokensD[self.clean(e1)] +[102]                

                # if task == 'tail':
                #     x = [101] + self._tokensD[e1] + self._tokensD[r]
                #     all_sample_indexs = []
                #     target_idx = -1
                    # x = x + [102]
                #     for sample_idx, sample in enumerate(samples):
                #         if len(x)+len(self._tokensD[sample]) > 510:
                #             break
                #         all_sample_indexs.append(range(len(x),len(x)+len(self._tokensD[sample])))
                #         x.extend(self._tokensD[sample])
                #         x.append(102)
                # elif task == 'head':
                #     x = [101] 
                #     all_sample_indexs = []
                #     target_idx = -1
                #     for sample_idx, sample in enumerate(samples):
                #         if len(x)+len(self._tokensD[sample]) > 510:
                #             break
                #         all_sample_indexs.append(range(len(x),len(x)+len(self._tokensD[sample])))
                #         x.extend(self._tokensD[sample])
                #         x.append(102)
                #     x = x + self._tokensD[r] + self._tokensD[e1] + [102]

                fields["X"] = ArrayField(np.array(x), dtype=np.long)                
                fields["target"] = ArrayField(np.array(targets), dtype=np.long)
                # (_, num_samples, max_num_tokens_sample)
                fields["index"] = ListField([ArrayField(np.array(sample_indexs), dtype=np.long) for sample_indexs in all_sample_indexs])
                if self.hparams.add_ht_embeddings:
                    fields["type"] = ArrayField(np.array(type_embedding), dtype=np.long)

            elif self.hparams.model == 'lm':
                x = []
                for sample_idx, sample in enumerate(samples):
                    if task == 'tail':
                        xs = [101] + self._tokensD[self.clean(e1)] + self._tokensD[self.clean(r)] + self._tokensD[self.clean(sample)] + [102]
                    elif task == 'head':
                        xs = [101] + self._tokensD[self.clean(sample)] + self._tokensD[self.clean(r)] + self._tokensD[self.clean(e1)] + [102]

                    if len(xs)+len(self._tokensD[self.clean(sample)]) > 510:
                        return None
                    x.append(xs)
                fields["X"] = ListField([ArrayField(np.array(xs), dtype=np.long) for xs in x])
                fields["target"] = ArrayField(np.array(targets), dtype=np.long)

        elif self.hparams.stage1:
            if self.hparams.model_type=="lstm" or self.hparams.model_type=="bert":
                fields: Dict[str, Field] = {}    
                x = e1+' '+r
                y = e2

                if self.key_index > len(self.random_indexes):
                    random.shuffle(self.random_indexes)
                    self.key_index = 0
                samples_index = self.random_indexes[self.key_index:self.key_index+self.hparams.negative_samples+1]
                self.key_index = self.key_index+self.hparams.negative_samples+1

                samples = [self.entity_mentions[sample_index] for sample_index in samples_index]
                if self.mode == 'train' and e2 not in samples:
                    samples = samples[:-1] + [e2]
                random.shuffle(samples)                

                targets = []
                for sample in samples:
                    if sample == e2:
                        targets.append(1)
                    else:
                        targets.append(0)

                x = self._tokensD[self.clean(e1)] + self._tokensD[self.clean(r)] 
                y = self._tokensD[self.clean(e2)]
                bert_tokens = [101]+x+[102]
                xy_bert_tokens = [101] + x + y + [102]
                sampleY, XsampleY, target = [], [], []
                # target = len(samples)
                for sample_idx, sample in enumerate(samples):
                    clean_sample = self.clean(sample)
                    # if sample == e2:
                    #     target = sample_idx
                    sampleY.append([101]+self._tokensD[clean_sample]+[102])
                    XsampleY.append([101]+x+self._tokensD[clean_sample]+[102])
                fields["X"] = ArrayField(np.array(bert_tokens), dtype=np.long)
                fields["sampleY"] = ListField([ArrayField(np.array(sample), dtype=np.long) for sample in sampleY])
                fields["XsampleY"] = ListField([ArrayField(np.array(sample), dtype=np.long) for sample in XsampleY])
                fields["target"] = ArrayField(np.array(targets), dtype=np.long)

            elif self.hparams.model_type=="ft":
                x_all_indices, x_all_subwords = self.ft_indices(x)
                xy_all_indices, xy_all_subwords = self.ft_indices(x+" "+y)
                fields["X"] = ListField([ArrayField(x_indices, dtype=np.long, padding_value=self.padding_idx) for x_indices in x_all_indices])
                negY, XnegY, target = [], [], []
                target = None
                for sample_idx, sample in enumerate(samples):
                    if sample == y:
                        target = sample_idx
                    negY.append(self.ft_indices(sample)[0])
                    XnegY.append(self.ft_indices(x+" "+sample)[0])
                assert target != None

                # (num_negative_samples, num_words, num_sub_words)
                fields["sampleY"] = ListField([ListField([ArrayField(x_indices, dtype=np.long, padding_value=self.padding_idx) \
                     for x_indices in x_all_indices]) for x_all_indices in negY])
                
                # (num_negative_samples, num_words, num_sub_words)
                fields["XsampleY"] = ListField([ListField([ArrayField(x_indices, dtype=np.long, padding_value=self.padding_idx) \
                     for x_indices in x_all_indices]) for x_all_indices in XnegY])
                fields["target"] = ArrayField(np.array(target), dtype=np.long)

        if mode == 'test':      
            # for okbc
            # all_alt_mentions_e2 = all_known.get((self.em_dict[e1],self.rm_dict[r]),[])
            # for closed kbc
            e2_filtered_mentions = [self.em_dict[ak] for ak in all_known.get((e1,r),[])]

            e2_alt_mentions_map = []
            for mention in e2_alt_mentions:
                if mention in self.em_dict:
                    e2_alt_mentions_map.append(self.em_dict[mention])
        else:
            e2_filtered_mentions, e2_alt_mentions_map = [], []

        if self.mode != 'train':
            fields["meta_data"] = MetadataField({"X": x, "tuple": (e1, r, e2), "Y": y, "e2_alt_mentions": e2_alt_mentions_map, "e2_filtered_mentions": e2_filtered_mentions, "samples": samples_index, "stage1_scores": stage1_scores})

        return Instance(fields)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--fp', type=str)
    parser.add_argument('--tokenize', type=str, default='bert')
    parser.add_argument('--negative_samples', type=int, default=10)
    args = parser.parse_args()

    # def main(args):
    # ft = fasttext.load_model('cc.en.300.bin')
    # input_ft = torch.FloatTensor(ft.get_input_matrix()).cuda()
    ft = None

    labels_set = set([l.strip('\n').split('\t')[2] for l in open('data/debug.txt', 'r').readlines()])
    labels_dict = {v: i for v, i in zip(labels_set, range(len(labels_set)))}

    dataset_reader = KBCDatasetReader(args, ft, labels_dict, 0)
    # dataset_reader.manual_multi_process_sharding = True

    # instances = [i for i in dataset_reader._read('data/debug.txt')]
    instances = dataset_reader._read('data/debug.txt')
    dummy_instance = next(instances)
    dataset_reader.apply_token_indexers(dummy_instance)
    vocab = vocabulary.Vocabulary.from_instances([dummy_instance])

    # dataset = AllennlpLazyDataset(dataset_reader.read, 'data/test.txt', vocab=vocab)    
    dataset = AllennlpLazyDataset(dataset_reader, 'data/test.txt', vocab=vocab)    
    sampler = MaxTokensBatchSampler(max_tokens=1000, sorting_keys=('X'))
    # dataloader = PyTorchDataLoader(dataset, batch_size=32)
    dataloader = MultiProcessDataLoader(dataset_reader, 'data/test.txt', batch_sampler=sampler, max_instances_in_memory=1000, num_workers=0)
    dataloader.index_with(vocab)

    for batch in tqdm(dataloader):
        # batch = next(iter(dataloader))
        batch = dotdict(batch)
