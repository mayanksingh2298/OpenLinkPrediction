import sys
import utils
import numpy as np
import random
import regex as re
import time
import glob
import ipdb
import argparse
import shutil
from shutil import copytree, ignore_patterns
import sys
import os
import params
from data import KBCDatasetReader, LabelDatasetReader
import math
import pickle
from tqdm import tqdm
# import model_oie, model_conj, model_srl, model_consti
from model import Model
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import ModelCheckpoint
from custom_model_checkpoint import ModelCheckpoint
from typing import Any, Dict, Optional, Union
from copy import deepcopy

import warnings
# necessary to ignore lots of numpy+tensorflow warnings
warnings.filterwarnings('ignore')

# import fasttext

# from allennlp.data.dataset_readers.dataset_reader import DatasetReader
# from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field, ListField, ArrayField
# from allennlp.data.instance import Instance
# from allennlp.data import vocabulary

from allennlp_data.dataset_readers.dataset_reader import DatasetReader
from allennlp_data.fields import TextField, SequenceLabelField, MetadataField, Field, ListField, ArrayField
from allennlp_data.instance import Instance
from allennlp_data import vocabulary


# from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
# from allennlp.data.tokenizers import Token, Tokenizer, PretrainedTransformerTokenizer

from allennlp_data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp_data.tokenizers import Token, Tokenizer, PretrainedTransformerTokenizer

from allennlp_data.samplers import MaxTokensBatchSampler

from allennlp_data.data_loaders import PyTorchDataLoader, AllennlpDataset, AllennlpLazyDataset, MultiProcessDataLoader
from allennlp_data.data_loaders import allennlp_collate,allennlp_worker_init_fn

has_cuda = torch.cuda.is_available()
torch.set_num_threads(64)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    # makes it slower and still not really deterministic
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    return

def get_logger(mode, hparams):
    log_dir = hparams.save+'/logs/'
    if os.path.exists(log_dir+f'{mode}'):
        mode_logs = list(glob.glob(log_dir+f'/{mode}_*'))
        new_mode_index = len(mode_logs)+1
        print('Moving old log to...')
        print(shutil.move(hparams.save +
                          f'/logs/{mode}', hparams.save+f'/logs/{mode}_{new_mode_index}'))
    logger = TensorBoardLogger(
        save_dir=hparams.save,
        name='logs',
        version=mode)
    return logger

def override_args(loaded_hparams_dict, current_hparams_dict, cline_sys_args):
    # override the values of loaded_hparams_dict with the values i current_hparams_dict
    # (only the keys in cline_sys_args)
    for arg in cline_sys_args:
        if '--' in arg:
            key = arg[2:]
            loaded_hparams_dict[key] = current_hparams_dict[key]

    for key in current_hparams_dict:
        if key not in loaded_hparams_dict:
            loaded_hparams_dict[key] = current_hparams_dict[key]

    return loaded_hparams_dict

def convert_to_namespace(d):
    params = argparse.Namespace()
    for key in d:
        setattr(params, key, d[key])
    return params

def train(hparams, checkpoint_callback, train_dataloader, validation_dataloader, labels_dataloader):
    logger = get_logger('train', hparams)
    if hparams.resume_checkpoint:
        checkpoint = torch.load(hparams.resume_checkpoint, map_location=torch.device('cpu'))
        loaded_hparams_dict = checkpoint['hyper_parameters']
        current_hparams_dict = vars(hparams)
        loaded_hparams_dict = override_args(loaded_hparams_dict, current_hparams_dict, sys.argv[1:])
        hparams = convert_to_namespace(loaded_hparams_dict)
        loaded_state_dict = checkpoint['state_dict']
    
        model = Model(hparams, labels_dataloader)
        model.load_state_dict(loaded_state_dict, strict=False)
        trainer = Trainer(logger=logger, gpus=hparams.gpus)
        trainer.test(model, test_dataloaders=test_dataloader)
    else:
        model = Model(hparams, labels_dataloader)

    backend = None
    if hparams.gpus > 1:
        backend='ddp'
    trainer = Trainer(distributed_backend=backend, num_sanity_val_steps=0, gpus=hparams.gpus, logger=logger, 
                      checkpoint_callback=checkpoint_callback, 
                      min_epochs=hparams.epochs, max_epochs=hparams.epochs,
                      max_steps=hparams.max_steps, val_check_interval=hparams.val_check_interval,
                      accumulate_grad_batches=int(hparams.accumulate_grad_batches), track_grad_norm=hparams.track_grad_norm, 
                      replace_sampler_ddp=True, profiler=hparams.profiler)

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=validation_dataloader)
    # if hparams.gpus > 1 and os.environ['LOCAL_RANK'] == '0':
    #     print('LOCAL_RANK of 0 will run validation...')
    #     trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
    # else:
    #     trainer.fit(model, train_dataloader=train_dataloader)
    return model

def test(hparams, checkpoint_callback, test_dataloader, labels_dataloader, model):
    logger = get_logger('test', hparams)
    test_fp = hparams.save+'/logs/test.txt'
    if not os.path.exists(os.path.dirname(test_fp)):
        os.makedirs(os.path.dirname(test_fp), exist_ok=True)
    test_f = open(hparams.save+'/logs/test.txt', 'w')
    trainer = Trainer(logger=logger, gpus=hparams.gpus)

    if not model:
        checkpoint = torch.load(hparams.checkpoint, map_location=torch.device('cpu'))
        loaded_hparams_dict = checkpoint['hyper_parameters']
        # override hyper-parameters with command-line passed ones
        current_hparams_dict = vars(hparams)
        loaded_hparams_dict = override_args(loaded_hparams_dict, current_hparams_dict, sys.argv[1:])
        hparams = convert_to_namespace(loaded_hparams_dict)
        loaded_state_dict = checkpoint['state_dict']

        model = Model(hparams, labels_dataloader)
        model.load_state_dict(loaded_state_dict, strict=False)
        
    trainer.test(model, test_dataloaders=test_dataloader)

    result = model.results
    test_f.write(f'{hparams.checkpoint}\t{result}\n')
    test_f.flush()
    test_f.close()

def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path, ignore=ignore_patterns('data','models'))

def collect_data(files):
    collectD = dict()
    for name in files:
        data = os.path.dirname(name)
        if data not in collectD:
            collectD[data] = []
        collectD[data].append(name)
    return collectD

def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """

    if hparams.save != None:
        checkpoint_callback = ModelCheckpoint(
            filepath=hparams.save+'/{epoch:02d}_{loss:.3f}_{eval_acc:0.3f}', verbose=True, monitor='eval_acc', save_top_k=hparams.save_k if not hparams.debug else 0)
        if 'train' in hparams.save:
            copy_and_overwrite('.', hparams.save+'/code')
    else:
        checkpoint_callback = None

    if hparams.ckbc: 
        default_train_path = 'train_data.txt'
        default_validation_path = 'validation_data.txt'
        default_test_path = 'test_data.txt'
    else:
        default_train_path = 'train_data_thorough.txt'
        default_validation_path = 'validation_data_linked.txt'
        default_test_path = 'test_data.txt'

    if not hparams.train: hparams.train = hparams.data_dir+'/'+default_train_path
    if not hparams.valid: hparams.valid = hparams.data_dir+'/'+default_validation_path
    if not hparams.test: hparams.test = hparams.data_dir+'/'+default_test_path

    if hparams.stage1_model:
        hparams.train = hparams.train+'.'+hparams.task_type+'_'+hparams.stage1_model+'.stage1'
        hparams.valid = hparams.valid+'.'+hparams.task_type+'_'+hparams.stage1_model+'.stage1'
        hparams.test = hparams.test+'.'+hparams.task_type+'_'+hparams.stage1_model+'.stage1'

    entity_mentions, em_map = utils.read_mentions(os.path.join(hparams.data_dir,"mapped_to_ids","entity_id_map.txt"))
    relation_mentions, rm_map = utils.read_mentions(os.path.join(hparams.data_dir,"mapped_to_ids","relation_id_map.txt"))

    # this contains the bert tokens of all the entities, relations in files (caching)
    tokensD = pickle.load(open(hparams.data_dir+'/mapped_to_ids/entity_id_map.'+hparams.model_str+'.pkl','rb'))
    tokensD.update(pickle.load(open(hparams.data_dir+'/mapped_to_ids/relation_id_map.'+hparams.model_str+'.pkl','rb')))

    if hparams.limit_tokens:
        for key in tokensD:
            tokensD[key] = tokensD[key][:hparams.limit_tokens]

    if hparams.leave_alt_mentions:
        all_known_e2, all_known_e1 = {}, {}
    else:
        print("Loading all_known pickled data...(takes times since large)")
        all_known_e2, all_known_e1 = pickle.load(open(os.path.join(hparams.data_dir,"all_knowns_simple_linked.pkl"),"rb"))
    if hparams.ckbc:
        scoresD_tail, scoresD_head = pickle.load(open(hparams.data_dir+'/scores_'+hparams.stage1_model+'.pkl','rb'))
    else:
        scoresD_tail, scoresD_head = {}, {}

    if hparams.debug:
        hparams.max_instances = 100
    world_size = 1 if hparams.gpus == 0 else hparams.gpus
    train_dataset_reader = KBCDatasetReader(hparams, em_map, rm_map, entity_mentions, tokensD, all_known_e2, all_known_e1, scoresD_tail, scoresD_head, 'train', hparams.max_instances, world_size)
    test_dataset_reader = KBCDatasetReader(hparams, em_map, rm_map, entity_mentions, tokensD, all_known_e2, all_known_e1, scoresD_tail, scoresD_head, 'test', hparams.max_instances, world_size)

    # if os.path.exists(hparams.train+'.cached'):
    #     train_dataset = pickle.load(open(hparams.train+'.cached','rb'))
    #     train_dataset = train_dataset[:hparams.max_instances]
    # else:
    #     train_instances = []
    #     for i, instance in tqdm(enumerate(train_dataset_reader.read(hparams.train))):
    #         if hparams.max_instances and i > hparams.max_instances:
    #             break
    #         train_instances.append(instance)
    #     train_dataset = AllennlpDataset(train_instances)    
    #     pickle.dump(train_dataset, open(hparams.train+'.cached','wb'))
    #     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size, num_workers=1, collate_fn=allennlp_collate)

    # if hparams.gpus != 1:
    #     train_instances = []
    #     for i, instance in tqdm(enumerate(dataset_reader.read(hparams.train))):
    #         train_instances.append(instance)
    #     train_dataset = AllennlpDataset(train_instances)    
    #     validation_instances = []
    #     for i, instance in tqdm(enumerate(dataset_reader.read(hparams.valid))):
    #         validation_instances.append(instance)
    #     validation_dataset = AllennlpDataset(validation_instances)    
    # train_dataset = AllennlpLazyDataset(dataset_reader, hparams.train)
    # validation_dataset = AllennlpLazyDataset(dataset_reader, hparams.valid)

    sampler = MaxTokensBatchSampler(max_tokens=hparams.max_tokens)
    num_workers = hparams.num_workers
    if 'train' in hparams.mode:
        if hparams.max_tokens:
            train_dataloader = MultiProcessDataLoader(train_dataset_reader, hparams.train, batch_sampler=sampler, num_workers=hparams.num_workers)
            
            # there is some randomness in getting the length of train_dataloader
            # so the model sometimes misses that epoch has ended and it should call the validation step
            # therefore, we force it to evaluate at slightly less than train_dataloader steps (usual observed difference is only 3-4)
            if len(train_dataloader) > 20:
                hparams.val_check_interval = len(train_dataloader)-20
        elif hparams.batch_size:
            train_dataloader = MultiProcessDataLoader(train_dataset_reader, hparams.train, batch_size=hparams.batch_size, num_workers=hparams.num_workers)

        validation_dataloader = MultiProcessDataLoader(test_dataset_reader, hparams.valid, batch_size=128, num_workers=1)


    if hparams.stage1:  
        labels_dataloader = None
        # if hparams.debug:
        #     labels_dataloader = None
        # else:
        #     labels_dataset_reader = LabelDatasetReader(hparams, tokensD)
        #     labels_sampler = MaxTokensBatchSampler(max_tokens=hparams.max_tokens)
        #     labels_dataloader = MultiProcessDataLoader(labels_dataset_reader, os.path.join(hparams.data_dir,"mapped_to_ids","entity_id_map.txt"), batch_sampler=labels_sampler)        
        test_dataloader = MultiProcessDataLoader(test_dataset_reader, hparams.test, batch_size=1, num_workers=1)
    elif hparams.stage2:
        labels_dataloader = None
        test_dataloader = MultiProcessDataLoader(test_dataset_reader, hparams.test, batch_size=128, num_workers=0)
        # test_dataloader = None

    model = None
    if 'train' in hparams.mode:
        model = train(hparams, checkpoint_callback, train_dataloader, validation_dataloader, labels_dataloader)
    if 'test' in hparams.mode:
        test(hparams, checkpoint_callback, test_dataloader, labels_dataloader, model)
    if 'validation' in hparams.mode:
        for checkpoint in glob.glob(hparams.save+'/*.ckpt'):
            hparams.checkpoint = checkpoint
            test(hparams, checkpoint_callback, validation_dataloader, labels_dataloader, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = params.add_args(parser)
    hyperparams = parser.parse_args()
    set_seed(hyperparams.seed)
    assert hyperparams.model_type=="bert" or hyperparams.model_type=="lstm" or hyperparams.model_type=="ft"
    main(hyperparams)
