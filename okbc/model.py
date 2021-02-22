import sys
import os
import ipdb
import random
import numpy as np
import pickle
import copy
from typing import Dict
from collections import OrderedDict
import logging
from tqdm import tqdm
import regex as re
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import LSTM, CrossEntropyLoss
from torch.optim import Adam, SGD
#from torch.optim.lr_scheduler import MultiplicativeLR

import pytorch_lightning as pl
# from pytorch_lightning.callbacks import EarlyStopping
from transformers import AdamW, AutoModel

import data

# prevents printing of model weights, etc
logging.getLogger('transformers.configuration_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def set_seed(seed):
    # Be warned that even with all these seeds, complete reproducibility cannot be guaranteed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    return

class Model(pl.LightningModule):

    def __init__(self, hparams, labels_dataloader):
        super(Model, self).__init__()
        self.hparams = hparams
        if self.hparams.model_type=="bert":
            self.bert_encoder = AutoModel.from_pretrained(hparams.model_str)
            try:
                if self.hparams.limit_layers:
                    self.bert_encoder.encoder.layer = self.bert_encoder.encoder.layer[:self.hparams.limit_layers]
                self.transformer = AutoModel.from_pretrained(hparams.model_str).encoder.layer[self.hparams.limit_layers:]
            except:
                if self.hparams.limit_layers:
                    self.bert_encoder.transformer.layer = self.bert_encoder.transformer.layer[:self.hparams.limit_layers]
                # self.transformer = AutoModel.from_pretrained(hparams.model_str).transformer.layer[self.hparams.limit_layers:self.hparams.limit_layers*2]
                self.transformer = AutoModel.from_pretrained(hparams.model_str).transformer.layer[self.hparams.limit_layers:]


            self.transformer_pos = copy.deepcopy(self.bert_encoder.embeddings.position_embeddings).weight
            self.hidden_size = self.bert_encoder.config.hidden_size
            self.lstm_encoder = nn.LSTM(300, 150, bidirectional=True, num_layers=2, batch_first=True)
            self.word_embeddings = nn.Embedding(30000, 300)
            self.label_mlp = nn.Sequential(nn.Linear(self.hidden_size, 100), nn.ReLU(), nn.Linear(100, 1))
            if self.hparams.add_ht_embeddings:
                self.type_embedding = nn.Embedding(2, self.hidden_size)

        elif self.hparams.model_type=="lstm":
            BERT_TOKEN_COUNT = 30000
            self.word_embeddings = nn.Embedding(BERT_TOKEN_COUNT+1, hparams.embedding_dim) # +1 for UNK
            nn.init.normal_(self.word_embeddings.weight.data, 0, 0.05)
            self.lstm_encoder = nn.LSTM(input_size=hparams.embedding_dim, hidden_size=hparams.embedding_dim, num_layers=1, batch_first=True, bidirectional=True)

        elif self.hparams.model_type=="ft":
            ft_model = fasttext.load_model('cc.en.300.bin')
            ft_vectors = torch.FloatTensor(ft_model.get_input_matrix())
            ft_vectors = torch.cat([ft_vectors, torch.zeros(1,300)])
            padding_idx = len(ft_vectors)-1
            self.padding_idx=len(ft_vectors)-1
            if self.hparams.freeze_ft:
                self.embedding = nn.Embedding.from_pretrained(ft_vectors, freeze=True, padding_idx=self.padding_idx)
            else:
                self.embedding = nn.Embedding.from_pretrained(ft_vectors, freeze=False, padding_idx=self.padding_idx)
            self.ft_bert_mlp = nn.Linear(300, 768)
            self.lstm = nn.LSTM(300, 150, bidirectional=True, num_layers=2, batch_first=True)

        self.scaled_add = nn.Parameter(torch.tensor([1.0]))
        self.labels_dataloader = labels_dataloader
        self.label_embeddings = None

        self._ce_loss = nn.CrossEntropyLoss()
        self._bce_loss = nn.BCEWithLogitsLoss()
        self._margin_loss = nn.MultiMarginLoss()
        self.labels_dataloader = labels_dataloader
        self.label_embeddings = None
        self.dummy = nn.Parameter(torch.Tensor([0.8]))

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adamW':
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'sgd':
            optimizer = SGD(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def embed_ft(self, text):
        # Input: text: (batch_size, num_words, num_sub_words)

        # embedding: (batch_size, num_words, num_sub_words, embed_dim=300)
        embedding = self.embedding(text)

        # embedding of word is mean of embeddings of sub-words
        # word_embedding: (batch_size, num_words, embed_dim)
        word_embedding = embedding.mean(2)

        # phrase_embedding: (batch_size, embed_dim)
        phrase_embedding = word_embedding.mean(1)
        # concatenate last h for lstm in bi-directional, multi-layer LSTM
        # phrase_embedding = torch.cat([self.lstm(word_embedding)[1][0][-1], self.lstm(word_embedding)[1][0][-2]], dim=-1)
        phrase_embedding = phrase_embedding / (torch.norm(phrase_embedding, 2, dim=-1).unsqueeze(-1)+0.001)
        return phrase_embedding

    def embed_bert(self, text):
        word_embeddings = self.bert_encoder(text)[0]
        cls_embeddings = torch.mean(word_embeddings[:,1:], dim=1)
        # cls_embeddings = word_embeddings[:,0]
        return cls_embeddings

    def embed_lstm(self, text):
        word_embeddings = self.word_embeddings(text)
        lstm_embeddings = self.lstm_encoder(word_embeddings)
        hidden_embeddings = lstm_embeddings[1][0]
        final_embeddings = torch.cat([hidden_embeddings[-1], hidden_embeddings[-2]], dim=1)
        return final_embeddings


    def embed_labels(self):
        # during test time only!
        label_embeddings = {}
        
        # print('Embedding all Labels...')
        # self.mcat_embeddings = None
        with torch.no_grad():
            print('Embed labels...')
            for batch in tqdm(self.labels_dataloader):
                if self.hparams.model_type=="ft":
                    embs = self.embed_ft(batch['labels'].cuda())
                
                elif self.hparams.model_type=="bert":
                    if type(batch['labels']) == type({}):
                        tokens = batch['labels']['tokens']['token_ids']
                    else:
                        tokens = batch['labels']
                    if self.hparams.gpus != 0:
                        tokens = tokens.cuda()
                    embs = self.embed_bert(tokens) 
                elif self.hparams.model_type=="lstm":
                    embs = self.embed_lstm(batch['labels'].cuda())

                elif self.hparams.model_type=="lstm":
                    raise NotImplementedError("lstm not yet designed embed_labels")

                label_indexs = [b['labels_index'] for b in batch['meta_data']]
                for label_index, emb in zip(label_indexs, embs):
                    label_embeddings[label_index] = emb.unsqueeze(0)
            
            final_embeddings = torch.cat([label_embeddings[i] for i in range(len(label_embeddings))], dim=0)
            del label_embeddings
            torch.cuda.empty_cache()

        return final_embeddings

    def cat_embed(self, query, samples, num_token_samples):
        '''
        Concatenates the samples with the query for each element in the batch
        Returns indices for each sample, so that we can pool on it later
        '''
        batch_size = query.shape[0]
        cat_query = torch.cat([query, samples.view(batch_size, -1)], dim=-1)

        cat_query_rmz = []
        max_size = -1
        for b in range(batch_size):
            cat_query_b_nz = cat_query[b][cat_query[b].nonzero()].squeeze(1)
            max_size = max(max_size, len(cat_query_b_nz))
            cat_query_rmz.append(cat_query_b_nz.tolist())
        for b in range(batch_size):
            cat_query_rmz[b] = cat_query_rmz[b]+[0]*(max_size-len(cat_query_rmz[b]))
        cat_query = torch.tensor(cat_query_rmz).to(self.device)
        embed = self.bert_encoder(cat_query, attention_mask=(cat_query!=0))[0]

        sample_indexes = []
        for b in range(batch_size):
            cat_query_b = cat_query[b]
            sids = torch.logical_and(cat_query_b!=102, cat_query_b!=0).nonzero().squeeze(1).tolist()
            si_b, si_b_k, prev_sid = [], [], -1
            for sid in sids:
                if sid-1 != prev_sid:
                    si_b.append(si_b_k+[0]*(num_token_samples-len(si_b_k)))
                    si_b_k = []
                si_b_k.append(sid)
                prev_sid = sid
            si_b.append(si_b_k+[0]*(num_token_samples-len(si_b_k)))
            # sample_indexes.append(si_b[1:]) # remove the query
            sample_indexes.append(si_b)
        try:
            sample_indexes = torch.tensor(sample_indexes).to(self.device)          
        except:
            ipdb.set_trace()
        
        return embed, sample_indexes

    def mean_pool(self, embed, sample_indexes):
        '''
        Returns the mean pooled embeddings for each of the sample
        '''
        batch_size, num_samples, num_tokens_samples = sample_indexes.shape      
        sample_indexes = sample_indexes.reshape(batch_size, num_samples*num_tokens_samples)

        samples_embed = torch.gather(embed, 1, sample_indexes.unsqueeze(-1).repeat(1,1,self.hidden_size))
        samples_embed = samples_embed * (sample_indexes != 0).unsqueeze(-1).float()
        samples_sum = torch.sum(samples_embed.reshape(batch_size, num_samples, num_tokens_samples, self.hidden_size), 2) 
        sample_indexes = sample_indexes.reshape(batch_size, num_samples, num_tokens_samples)
        samples_mean = samples_sum / ((sample_indexes != 0).sum(-1).unsqueeze(-1).float()+1e-4)

        return samples_mean

    def forward(self, batch, mode='train', batch_idx=-1):
        output_dict = {}
        if self.hparams.xt_results:
            output_dict['scores'] = batch.scores
            output_dict['predictions'] = torch.max(batch.scores.squeeze(-1), 1)[1]
        elif self.hparams.stage2:
            if self.hparams.model == 'hmcq':
                batch_size, total_samples, num_token_samples = batch.samples.shape
                _, num_token_query = batch.query.shape

                max_samples, max_indices, max_scores, all_scores, cum_chunk_size = [], [], [], [], 0
                if self.hparams.use_anchor:
                    random_anchor_id = random.randint(0,9)
                    # random_anchor_id = 0
                    anchor_entity = batch.samples[:,random_anchor_id,:]
                    # batch.samples = torch.cat((batch.samples[:,:random_anchor_id,:], batch.samples[:,random_anchor_id+1:,:]), 1)
                    all_anchor_scores = []
                    for chunk_id, samples_chunk in enumerate(torch.split(batch.samples, self.hparams.chunk_size, 1)):
                        random_insertion = random.randint(0,samples_chunk.shape[1]-1)
                        # random_insertion = 7
                        if chunk_id != 0:
                            samples_chunk_anchor = torch.cat((samples_chunk[:,:random_insertion], anchor_entity.unsqueeze(1), samples_chunk[:,random_insertion:]), dim=1)
                            embed, sample_indexes = self.cat_embed(batch.query, samples_chunk_anchor, num_token_samples)
                            samples_chunk_mean = self.mean_pool(embed, sample_indexes)
                            scores = self.label_mlp(samples_chunk_mean).squeeze(-1)
                            anchor_scores = scores[:,random_insertion]
                            scores = torch.cat((scores[:,:random_insertion], scores[:,random_insertion+1:]), dim=1)
                        else:
                            embed, sample_indexes = self.cat_embed(batch.query, samples_chunk, num_token_samples)
                            samples_chunk_mean = self.mean_pool(embed, sample_indexes)
                            scores = self.label_mlp(samples_chunk_mean).squeeze(-1)
                            anchor_scores = scores[:,random_anchor_id]

                        all_anchor_scores.append(anchor_scores)
                        all_scores.append(scores/torch.abs(anchor_scores).unsqueeze(1))

                        max_chunk_scores, max_chunk_indices = torch.max(scores, dim=-1)
                        max_chunk_samples = torch.gather(samples_chunk, 1, max_chunk_indices.unsqueeze(1).unsqueeze(2).repeat(1,1,num_token_samples))
                        max_samples.append(max_chunk_samples)

                        max_indices.append(max_chunk_indices.unsqueeze(1)+cum_chunk_size)
                        cum_chunk_size += samples_chunk.shape[1]


                    # max_samples = torch.cat(max_samples, dim=1)                
                    # max_indices = torch.cat(max_indices, dim=1)            

                    # max_samples_anchor = torch.cat((max_samples, anchor_entity.unsqueeze(1)), dim=1)
                    # embed, sample_indexes = self.cat_embed(batch.query, max_samples_anchor, num_token_samples)
                    # max_samples_mean = self.mean_pool(embed, sample_indexes)
                    # max_scores = self.label_mlp(max_samples_mean) 
                    # anchor_scores = max_scores[:,-1]
                    # max_scores = max_scores[:,:-1]
                    # max_scores = max_scores/torch.abs(anchor_scores).unsqueeze(-1)

                    # scores = max_scores
                    scores = torch.cat(all_scores, dim=1).unsqueeze(-1)
                    # for b in range(batch_size):
                    #     for c, m in enumerate(max_indices[b]):
                    #         scores[b, m] = max_scores[b,c,0]
                    # scores = scores.unsqueeze(-1)

                elif self.hparams.transformer:
                    all_chunks_mean, all_query_embed = [], []
                    for chunk_id, samples_chunk in enumerate(torch.split(batch.samples, self.hparams.chunk_size, 1)):
                        embed, sample_indexes = self.cat_embed(batch.query, samples_chunk, max(num_token_samples, num_token_query))
                        samples_chunk_mean = self.mean_pool(embed, sample_indexes)
                        query_embed = samples_chunk_mean[:,0,:]
                        samples_chunk_mean = samples_chunk_mean[:,1:,:]

                        # cat_query = torch.cat([batch.query, samples_chunk.view(batch_size, -1)], dim=-1)
                        # embed = self.bert_encoder(cat_query, attention_mask=(cat_query!=0))[0]
                        # query_embed = embed[:,:num_token_query].mean(1)

                        # samples_chunk_embed = embed[:,num_token_query:,:].reshape(batch_size, self.hparams.chunk_size, num_token_samples, self.hidden_size)
                        # samples_chunk_embed = samples_chunk_embed * (samples_chunk != 0).unsqueeze(-1).float() * (samples_chunk != 102).unsqueeze(-1).float()
                        # samples_chunk_mean = samples_chunk_embed.sum(dim=2) / ((samples_chunk != 0).sum(-1).unsqueeze(-1).float()+1e-4)

                        all_chunks_mean.append(samples_chunk_mean)
                        all_query_embed.append(query_embed.unsqueeze(0))

                    # aggregate query embedding across chunks by taking mean pool
                    # (batch_size, 768)
                    mean_query_embed = torch.cat(all_query_embed,dim=0).mean(0)
                    # (batch_size, total_num_samples, 768)
                    all_chunks_mean = torch.cat(all_chunks_mean, dim=1)
                    if self.hparams.multiply_scores:
                        sample_weights = batch.scores - torch.min(batch.scores, 1)[0].unsqueeze(-1) + 1                
                        all_chunks_mean = all_chunks_mean * sample_weights.unsqueeze(-1)
                    embed = torch.cat((mean_query_embed.unsqueeze(1), all_chunks_mean), dim=1)
                    embed = embed + self.transformer_pos[:embed.shape[1],:]
                    for layer in self.transformer:
                        embed = layer(embed)[0]
                    # (batch_size, total_num_samples)
                    scores = self.label_mlp(embed[:,1:,:])

                    if self.hparams.add_scores:
                        scores = scores + self.scaled_add * batch.scores.unsqueeze(-1)
                elif self.hparams.kg_bert:
                    bert_input = batch.XsampleY.reshape(batch_size*total_samples,-1)
                    bert_embedding = self.bert_encoder(bert_input)[0][:,0] # take cls embedding
                    bert_embedding = bert_embedding.view(batch_size, total_samples, -1)
                    scores = self.label_mlp(bert_embedding)
                else:
                    # (batch_size, 50, num_token_samples) --> 5*(batch_size,10, num_token_samples)
                    for chunk_id, samples_chunk in enumerate(torch.split(batch.samples, self.hparams.chunk_size, 1)):
                        embed, sample_indexes = self.cat_embed(batch.query, samples_chunk, num_token_samples)
                        samples_chunk_mean = self.mean_pool(embed, sample_indexes)


                        # (101, 2, 3, 4, 102, 0, 0, 0, 5, 6, 102, 0, 0, 7, 8, 9, 102, 0)
                        # cat_query = torch.cat([batch.query, samples_chunk.view(batch_size, -1)], dim=-1)
                        # embed = self.bert_encoder(cat_query, attention_mask=(cat_query!=0))[0]
                        # samples_chunk_embed = embed[:,num_token_query:,:].reshape(batch_size, self.hparams.chunk_size, num_token_samples, self.hidden_size)
                        # samples_chunk_embed = samples_chunk_embed * (samples_chunk != 0).unsqueeze(-1).float() * (samples_chunk != 102).unsqueeze(-1).float()
                        # samples_chunk_mean = samples_chunk_embed.sum(dim=2) / ((samples_chunk != 0).sum(-1).unsqueeze(-1).float()+1e-4)

                        scores = F.softmax(self.label_mlp(samples_chunk_mean))

                        max_chunk_scores, max_chunk_indices = torch.max(scores.squeeze(-1), dim=-1)
                        max_chunk_samples = torch.gather(samples_chunk, 1, max_chunk_indices.unsqueeze(1).unsqueeze(2).repeat(1,1,num_token_samples))
                        max_samples.append(max_chunk_samples)

                        max_indices.append(max_chunk_indices.unsqueeze(1)+cum_chunk_size)
                        cum_chunk_size += samples_chunk.shape[1]
                        all_scores.append(scores.squeeze(-1))
                        # all_scores.append(scores.squeeze(-1)/max_chunk_scores.unsqueeze(-1))

                    max_samples = torch.cat(max_samples, dim=1)                
                    num_max_samples = max_samples.shape[1]
                    max_indices = torch.cat(max_indices, dim=1)            

                    # for b in range(len(batch.meta_data)):     
                    #     batch.meta_data[b]['samples'] = [batch.meta_data[b]['samples'][si] for si in range(20,30)]
                    # scores = all_scores[2].unsqueeze(-1)

                    # for b in range(len(batch.meta_data)):     
                    #     batch.meta_data[b]['samples'] = [batch.meta_data[b]['samples'][si] for si in max_indices[b]]

                    embed, sample_indexes = self.cat_embed(batch.query, max_samples, num_token_samples)
                    max_samples_mean = self.mean_pool(embed, sample_indexes)
                    max_scores = F.softmax(self.label_mlp(max_samples_mean))
                    # scores = max_scores
                    norm_scores = []
                    for scores_ind, scores in enumerate(all_scores):
                        norm_scores.append(scores*max_scores[:,scores_ind])

                    scores = torch.cat(norm_scores, dim=1)
                    # (batch_size, num_samples)
                    # scores = -200000*torch.ones([batch_size, total_samples]).to(self.device)
                    # for b in range(batch_size):
                    #     for c, m in enumerate(max_indices[b]):
                    #         # ipdb.set_trace()
                    #         scores[b, m] = max_scores[b,c,0]
                    #         # if max_scores[b,c,0] > 0:
                    #         #     scores[b, m] += max_scores[b,c,0]                        
                    #         # else:
                    #         #     scores[b, m] -= max_scores[b,c,0]                        
                    scores = scores.unsqueeze(-1)

                    # scores[torch.arange(batch_size), max_indices[:,b]] += max_scores[:,b].squeeze(-1)

                    # max_samples_embed = embed[:,num_token_query:,:].reshape(batch_size, num_max_samples, num_token_samples, self.hidden_size)
                    # max_samples_embed = max_samples_embed * (max_samples != 0).unsqueeze(-1).float() * (max_samples != 102).unsqueeze(-1).float()
                    # max_samples_mean = max_samples_embed.sum(dim=2) / ((max_samples != 0).sum(-1).unsqueeze(-1).float()+1e-4)
                    # scores = self.label_mlp(max_samples_mean)

            elif self.hparams.model_type == "ft" or self.hparams.model_type == "lstm":
                raise NotImplementedError("lstm and ft not yet designed for stage 2")

            elif self.hparams.model == 'mcq':
                tokens = batch.X
                # import torch.distributed as dist
                # print('RANK',dist.get_rank(),'TOKENS SHAPE',tokens.shape)
                # (batch_size, num_tokens, embed_dim)

                # X = self.bert_encoder.embeddings.word_embeddings(tokens)
                # for layer in self.bert_encoder.encoder.layer:
                #     X = layer(X)[0]

                X = self.bert_encoder(tokens)[0]

                if self.hparams.add_ht_embeddings:
                    X = X + self.type_embedding(batch.type)

                batch_size, num_input_tokens, embed_dim = X.shape
                batch_size, num_samples, num_sample_tokens = batch.index.shape
                batch.index = batch.index.reshape(batch_size, num_samples*num_sample_tokens)
                
                # finally, gX  --> (batch_size, num_samples, embed_dim)

                # (batch_size, num_samples*num_sample_tokens, embed_dim)
                gX = torch.gather(X, 1, batch.index.unsqueeze(-1).repeat(1,1,embed_dim))
                # make padding as 0
                gX = gX * (batch.index != 0).unsqueeze(-1).float()
                # max pooling
                # gX = torch.max(gX.reshape(batch_size, num_samples, num_sample_tokens, embed_dim), 2)[0]
                # mean pooling
                gX = torch.sum(gX.reshape(batch_size, num_samples, num_sample_tokens, embed_dim), 2) 
                batch.index = batch.index.reshape(batch_size, num_samples, num_sample_tokens)
                # donot consider padding while taking sum
                gX = gX / ((batch.index != 0).sum(-1).unsqueeze(-1).float()+1e-4)

                scores = self.label_mlp(gX)
                # add stage1-scores
                if self.hparams.add_scores:
                    scores = scores + self.scaled_add * batch.scores.unsqueeze(-1)
                
            elif self.hparams.model == 'lm':
                tokens = batch.X
                batch_size, num_samples, num_tokens = tokens.shape
                bert_embeddings = self.bert_encoder(tokens.reshape(batch_size*num_samples, num_tokens))[0]
                bert_embeddings = bert_embeddings.reshape(batch_size, num_samples, num_tokens, -1)
                bert_embeddings = bert_embeddings[:,:,0,:] # get cls embedding
                scores = self.label_mlp(bert_embeddings)

            if mode == 'train':
                loss = self._bce_loss(scores.reshape(-1), batch.target.reshape(-1).float())
                output_dict['loss'] = loss
                # output_dict['loss'] = self._bce_loss(self.dummy, torch.tensor([1.0]).cuda())
            else:
                # (batch_size, 1)
                output_dict['scores'] = scores
                output_dict['predictions'] = torch.max(scores.squeeze(-1), 1)[1]

        elif self.hparams.stage1:
            # WHY DOES THIS HAVE MODE=TRAIN here?
            if mode == 'test' and batch_idx == 0:
                self.label_embeddings = self.embed_labels()

            if (self.hparams.model_type=="bert" or self.hparams.model_type=="lstm"):
                if type(batch.X) == type({}):
                    batch.X = batch.X['tokens']['token_ids']
                    batch.sampleY = batch.sampleY['tokens']['token_ids']
                    batch.XsampleY = batch.XsampleY['tokens']['token_ids']
                tokens = batch.X
                if self.hparams.model_type=="bert":
                    X = self.embed_bert(tokens)                    
                else:
                    X = self.embed_lstm(tokens)

                batch_size, embed_dim = X.shape
                # if (self.hparams.use_label_embeddings and self.current_epoch > 1) or mode == 'test':
                if mode == 'test':
                    Y = self.label_embeddings.expand(batch_size, -1, -1)
                else:
                    batch_size, neg_samples, num_words = batch.sampleY.shape
                    if self.hparams.model_type=="bert":
                        Y = self.embed_bert(batch.sampleY.reshape(batch_size*neg_samples, num_words))
                    else:
                        Y = self.embed_lstm(batch.sampleY.reshape(batch_size*neg_samples, num_words))
                    Y = Y.reshape(batch_size, neg_samples, embed_dim)
                X = X.unsqueeze(1).expand_as(Y)
                scores = (X * Y).sum(dim=-1) / math.sqrt(embed_dim)
            
            elif self.hparams.model_type=="ft":
                X = self.embed_ft(batch.X)
                batch_size, embed_dim = X.shape
                if mode == 'test':
                    Y = self.label_embeddings.expand(batch_size, -1, -1)
                else:
                    batch_size, neg_samples, num_words, num_sub_words = batch.sampleY.shape
                    # (batch_size*neg_samples, embed_dim)
                    Y = self.embed_ft(batch.sampleY.reshape(batch_size*neg_samples, num_words, num_sub_words))
                    Y = Y.reshape(batch_size, neg_samples, embed_dim)

                # (batch_size, neg_samples, embed_dim)        
                X = X.unsqueeze(1).expand_as(Y) 
                # TODO: Check Normalization 
                # (batch_size, neg_samples)
                scores = (X * Y).sum(dim=-1) / math.sqrt(embed_dim)

            if mode == 'train':
                # loss = self._margin_loss(scores, batch.target)
                loss = self._bce_loss(scores, batch.target.float())
                output_dict['loss'] = loss
            else:
                # (batch_size, 1)
                output_dict['predictions'] = torch.max(scores, 1)[1]
                output_dict['scores'] = scores
        return output_dict


    def get_progress_bar_dict(self):
        running_train_loss = self.trainer.train_loop.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if running_train_loss is not None else float('NaN')
        if type(self.trainer.checkpoint_callback.kth_value) != type(0.0):
            best = self.trainer.checkpoint_callback.kth_value.item()
        else:
            best = self.trainer.checkpoint_callback.kth_value
        tqdm = {'loss': '{:.3f}'.format(avg_training_loss), 'best': best}
        return tqdm

    def training_step(self, batch, batch_idx, optimizer_idx=-1):
        batch = dotdict(batch)
        output_dict = self.forward(batch, mode='train', batch_idx=batch_idx)
        tqdm_dict = {'train_loss': output_dict['loss']}
        output = OrderedDict({'loss': output_dict['loss'], 'log': tqdm_dict})

        return output

    def validation_step(self, batch, batch_idx, mode='val'):
        batch = dotdict(batch)
        # batch.samples would contain the indices of k entity mentions
        # output['scores'] would contain a list that is score for these k mentions
        output = self.forward(batch, mode=mode, batch_idx=batch_idx)
        batch_size = len(batch.meta_data)
        if self.hparams.stage2:
            # add dummy value at end
            gold_target = torch.cat((batch.target.float(), torch.ones_like(batch.target)[:,0:1].float()*0.1), 1)
            output['gold'] = torch.max(gold_target, 1)[1]
            # tail evaluation
            rank = []
            for b in range(batch_size):
                topk_real_indices = batch.meta_data[b]["samples"]
                e2_filtered_mentions = batch.meta_data[b]["e2_filtered_mentions"]
                e2_alt_mentions = batch.meta_data[b]["e2_alt_mentions"]
                topk_scores = output["scores"][b]
                stage1_scores = batch.meta_data[b]["stage1_scores"]
                best_score = -20000000
                contains_gold = False
                for i,j in enumerate(topk_real_indices):
                    if j in e2_alt_mentions:
                        best_score = max(best_score,topk_scores[i].item())
                        topk_scores[i] = -20000000
                        contains_gold = True

                for i,j in enumerate(topk_real_indices):
                    if j in e2_filtered_mentions:
                        topk_scores[i] = -20000000

                if contains_gold:
                    greater = topk_scores.gt(best_score).float()
                    equal = topk_scores.eq(best_score).float()
                    rank_b = greater.sum()+1+equal.sum()/2.0 
                else:
                    if stage1_scores and 'MR' in stage1_scores:
                        rank_b = stage1_scores['MR']
                    else: # should happen only for train set
                        # print('Not Found MR in stage1_scores. Is this train set?')
                        rank_b = len(topk_scores)+1
                rank.append(torch.tensor(rank_b))

            output['rank'] = rank

        else:
            output['gold'] = batch.target
            # tail evaluation
            rank = []
            for b in range(batch_size):
                topk_scores = output["scores"][b]
                e2_filtered_mentions = batch.meta_data[b]["e2_filtered_mentions"]
                e2_alt_mentions = batch.meta_data[b]["e2_alt_mentions"]
                best_score = topk_scores[e2_alt_mentions].max()
                topk_scores[e2_filtered_mentions] = -20000000 # MOST NEGATIVE VALUE
                greater = topk_scores.gt(best_score).float()
                equal = topk_scores.eq(best_score).float()
                rank.append(greater.sum()+1+equal.sum()/2.0)
            output['rank'] = rank

        return output

    def evaluation_end(self, outputs, mode):
        result = {}
        result['mr'] = 0
        result['mrr'] = 0
        result['hits1'] = 0
        result['hits10'] = 0
        result['hits50'] = 0
        count, correct, total = 0, 0, 0
        for output in outputs:
            correct += (output['gold'] == output['predictions']).sum().item()
            total += len(output['gold'])

            rank_batch = output['rank']
            for rank in rank_batch:
                count += 1
                result['mr'] += rank.item()
                result['mrr'] += 1.0/rank.item()
                result['hits1'] += rank.le(1).float().item()
                result['hits10'] += rank.le(10).float().item()
                result['hits50'] += rank.le(50).float().item()
        result['mr'] /= float(count)
        result['mrr'] /= float(count)
        result['hits1'] /= float(count)
        result['hits10'] /= float(count)
        result['hits50'] /= float(count)
        result['simple_accuracy'] = float(correct)/total        
                
        # result['eval_f1'] = correct/total
        print('\nResults: '+str(result))
        return result

    def validation_epoch_end(self, outputs):
        eval_results = self.evaluation_end(outputs, 'dev')
        result = {"log": eval_results, "eval_acc": eval_results['hits1']}
        result = OrderedDict(result)
        self.log('eval_acc', result['eval_acc'])
        return result

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, mode='test')

    def test_epoch_end(self, outputs):
        eval_results = self.evaluation_end(outputs, 'test')
        self.outputs = outputs
        result = {"log": eval_results, "progress_bar": eval_results,
                  "test_acc": eval_results['hits1']}
        self.results = eval_results
        return result

    # obligatory definitions - pass actual through fit
    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    
