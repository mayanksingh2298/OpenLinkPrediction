# get ent freq
# write test code
# FIX convert to good batches 
import argparse
import logging
import os
import pickle
import pprint
import sys
import time
import numpy as np
import random
import torch
# torch.backends.cudnn.benchmark=True
from tqdm import tqdm
import datetime
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from kb import kb
import utils
import time
from dataset import Dataset
from datasetauthor import OneToNMentionRelationDataset


from models import rotatELSTM, complexLSTM, complexLSTM_2, complexLSTM_2_all_e, complexLSTM_2_all_e_PLT


def convert_string_to_indices(data, mapp, maxlen=15, PAD_MAPPING = 0, START_MAPPING = 1, END_MAPPING = 2, use_tqdm=False):
	"""
		data: list of strings like ["the boy is playing", "the kite is flying", ...]
		map: contains mapping from words to integer. len(mapp) represents UNK
		maxlen: pad with PAD_MAPPING to fit the maxlen
		This function also adds START_MAPPING and END_MAPPING 
		returns mapped_data and actual lengths tensors
	"""
	mapped_data = []
	lengths = []
	for line in tqdm(data, disable=not use_tqdm):
		#add start token
		mapped_line = [START_MAPPING]
		line = line.strip().split()
		for word in line:
			mapped_line.append(mapp.get(word,len(mapp)))
			if(len(mapped_line)==maxlen-1):
				break
		#add end token
		mapped_line.append(END_MAPPING)
		lengths.append(len(mapped_line))
		assert(len(mapped_line) <= maxlen)
		for i in range(maxlen-len(mapped_line)):
			mapped_line.append(PAD_MAPPING)
		mapped_data.append(mapped_line)
	return torch.tensor(mapped_data),torch.tensor(lengths)

def convert_mention_index_to_string(data, mentions):
	"""
		data    : [id0,id1,id2,...]
		mentions: [<string for 0>, <string for 1>, <string for 2>, ...] 

		returns : [<string for id0>, <string for id1>, <string for id2>, ...]
	"""
	to_return = []
	for datum in data:
		to_return.append(mentions[datum])
	return to_return

def convert_mention_to_token_indices(data,mapp,length_mapp):
	"""
		data: [id0, id1, ...]
		mapp: [[tokenids for id0], [tokenids for id1], ...]
		length_mapp: [length_for_id0, length_for_id1, ...]

		returns: tensor for mapped_token_indices, tensor for actual lengths
	"""
	mapped_data = []
	lengths = []
	for index in data:
		mapped_data.append(mapp[index])
		lengths.append(length_mapp[index])

	return torch.tensor(mapped_data), torch.tensor(lengths)

def get_entity_scores_from_node_scores(node_scores, entity_node_indices, mask):
	"""
		node_scores         = shape -> #entity mentions
		entity_node_indices = shape -> #entity mentions x max_tree_depth
		mask                = shape -> #entity mentions x max_tree_depth
		returns:
			shape -> #entity mentions
	"""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	num_mentions = len(node_scores)
	scores = []
	for i in tqdm(range(num_mentions)):
		this_entity_scores = node_scores[entity_node_indices[i]]
		this_mask = mask[i]
		score = torch.log(torch.sigmoid(this_entity_scores * this_mask)).sum()
		scores.append(score)
	scores = torch.tensor(scores, device = device)
	return scores

has_cuda = torch.cuda.is_available()
if not has_cuda:
	utils.colored_print("yellow", "CUDA is not available, using cpu")

def main(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	
	# read token maps
	etokens, etoken_map = utils.get_tokens_map(os.path.join(args.data_dir,"mapped_to_ids","entity_token_id_map.txt"))
	rtokens, rtoken_map = utils.get_tokens_map(os.path.join(args.data_dir,"mapped_to_ids","relation_token_id_map.txt"))
	entity_mentions,em_map = utils.read_mentions(os.path.join(args.data_dir,"mapped_to_ids","entity_id_map.txt"))
	relation_mentions,rm_map = utils.read_mentions(os.path.join(args.data_dir,"mapped_to_ids","relation_id_map.txt"))

	# create entity_token_indices and entity_lengths
	# [[max length indices for entity 0 ], [max length indices for entity 1], [max length indices for entity 2], ...]
	# [length of entity 0, length of entity 1, length of entity 2, ...]
	entity_token_indices, entity_lengths = utils.get_token_indices_from_mention_indices(entity_mentions, etoken_map, maxlen=args.max_seq_length, use_tqdm=True)
	relation_token_indices, relation_lengths = utils.get_token_indices_from_mention_indices(relation_mentions, rtoken_map, maxlen=args.max_seq_length, use_tqdm=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if not args.do_train and not args.do_eval:
		raise ValueError("At least one of `do_train` or `do_eval` must be True.")

	#train code (+1 for unk token)
	model = complexLSTM_2_all_e_PLT(len(etoken_map)+1,len(rtoken_map)+1, len(entity_mentions), args.embedding_dim, {}, initial_token_embedding =args.initial_token_embedding, entity_tokens = etokens, relation_tokens = rtokens, lstm_dropout=args.lstm_dropout)
	logsigmoid = torch.nn.LogSigmoid()
	if(args.do_train):
		data_config = {'input_file': 'train_data_thorough.txt', 'batch_size': args.train_batch_size, 'use_batch_shared_entities': True, 'min_size_batch_labels': args.train_batch_size, 'max_size_prefix_label': 64, 'device': 0}
		expt_settings = {'loss': 'bce', 'replace_entities_by_tokens': True, 'replace_relations_by_tokens': True, 'max_lengths_tuple': [10, 10]}
		train_data = OneToNMentionRelationDataset(dataset_dir=os.path.join(args.data_dir,"mapped_to_ids"), is_training_data=True, **data_config, **expt_settings)
		train_data.create_data_tensors(
			dataset_dir=os.path.join(args.data_dir,"mapped_to_ids"),
			train_input_file='train_data_thorough.txt',
			valid_input_file='validation_data_linked.txt',
			test_input_file='test_data.txt',
		)
		train_loader = train_data.get_loader(
			shuffle=True,
			num_workers=8,
			drop_last=True,
		)
		# optimizer = torch.optim.Adagrad(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
		optimizer = torch.optim.Adagrad(model.parameters(),lr=args.learning_rate)

		if(args.resume):
			print("Resuming from:",args.resume)
			# checkpoint = torch.load(args.resume,map_location=lambda storage, loc: storage)
			checkpoint = torch.load(args.resume,map_location="cpu")
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			del checkpoint
			torch.cuda.empty_cache()

			#Load other things too if required

		model.train()

		# crossEntropyLoss = torch.nn.CrossEntropyLoss(reduction='mean')
		BCEloss = torch.nn.BCEWithLogitsLoss(reduction='sum')
		for epoch in tqdm(range(0,args.num_train_epochs), desc="epoch"):
			iteration = 0
			# ct = 0
			for batch in tqdm(train_loader, desc="Train dataloader"):
				# ct+=1
				# if (ct==1000):
				# 	break
				inputs, \
				normalizer_loss, \
				normalizer_metric, \
				labels, \
				label_ids, \
				filter_mask, \
				batch_shared_entities = train_data.input_and_labels_to_device(
					batch,
					training=True,
					device="cpu"
				)
				if inputs[0]==None:
					num_samples_for_head=0
				else:
					num_samples_for_head = inputs[0][0].shape[0]
				tree_nodes_head, head_mask, tree_nodes_tail, tail_mask,\
				tree_nodes_head_neg, head_mask_neg, tree_nodes_tail_neg, tail_mask_neg = model.get_tree_nodes(batch_shared_entities - 2, labels, num_samples_for_head)

				loss = torch.tensor(0,device=device)
				for mode,model_inputs in zip(["head","tail"],inputs):
					if model_inputs==None:
						continue
					# subtract two from author's indices because our map is 2 less
					if mode=="head":
						batch_e2_indices = model_inputs[1] - 2
						batch_e2_indices = batch_e2_indices.where(batch_e2_indices!=-2,torch.tensor(len(entity_mentions)-2,dtype=torch.int32))
						batch_r_indices  = model_inputs[0] - 2
						batch_r_indices = batch_r_indices.where(batch_r_indices!=-2,torch.tensor(len(relation_mentions)-2,dtype=torch.int32))
						batch_e1_indices = batch_shared_entities - 2

						train_r_mention_tensor, train_r_lengths   = convert_mention_to_token_indices(batch_r_indices.squeeze(1), relation_token_indices, relation_lengths)
						train_e2_mention_tensor, train_e2_lengths = convert_mention_to_token_indices(batch_e2_indices.squeeze(1), entity_token_indices, entity_lengths)

						train_r_mention_tensor, train_r_lengths   = train_r_mention_tensor.cuda(), train_r_lengths.cuda()
						train_e2_mention_tensor, train_e2_lengths = train_e2_mention_tensor.cuda(), train_e2_lengths.cuda()

						# e1_real_lstm, e1_img_lstm = model.get_atomic_entity_embeddings(batch_e1_indices.squeeze(1).long().cuda())
						r_real_lstm, r_img_lstm   = model.get_mention_embedding(train_r_mention_tensor,1,train_r_lengths)
						e2_real_lstm, e2_img_lstm = model.get_mention_embedding(train_e2_mention_tensor,0,train_e2_lengths)

						# import pdb
						# pdb.set_trace()
						tmp_nodes_0 = torch.cat([tree_nodes_head[0],tree_nodes_head_neg[0]],dim=1)
						tmp_nodes_1 = torch.cat([tree_nodes_head[1],tree_nodes_head_neg[1]],dim=1)
						head_mask_neg *= -1
						tmp_mask = torch.cat([head_mask,head_mask_neg],dim=1)

						model_output = model.complex_score_e2_r_with_given_ementions(e2_real_lstm,e2_img_lstm,r_real_lstm,r_img_lstm,tmp_nodes_0,tmp_nodes_1)
						loss = loss - (logsigmoid(model_output*tmp_mask)).mean()
						# neg
						# model_output = model.complex_score_e2_r_with_given_ementions(e2_real_lstm,e2_img_lstm,r_real_lstm,r_img_lstm,tree_nodes_head_neg[0],tree_nodes_head_neg[1])
						# loss = loss - (logsigmoid(-1*model_output*head_mask_neg)).mean()
					else:
						batch_e1_indices = model_inputs[0] - 2
						batch_e1_indices = batch_e1_indices.where(batch_e1_indices!=-2,torch.tensor(len(entity_mentions)-2,dtype=torch.int32))
						batch_r_indices  = model_inputs[1] - 2
						batch_r_indices = batch_r_indices.where(batch_r_indices!=-2,torch.tensor(len(relation_mentions)-2,dtype=torch.int32))
						batch_e2_indices = batch_shared_entities - 2
					
						train_e1_mention_tensor, train_e1_lengths = convert_mention_to_token_indices(batch_e1_indices.squeeze(1), entity_token_indices, entity_lengths)
						train_r_mention_tensor, train_r_lengths   = convert_mention_to_token_indices(batch_r_indices.squeeze(1), relation_token_indices, relation_lengths)


						train_e1_mention_tensor, train_e1_lengths = train_e1_mention_tensor.cuda(), train_e1_lengths.cuda()
						train_r_mention_tensor, train_r_lengths   = train_r_mention_tensor.cuda(), train_r_lengths.cuda()

						e1_real_lstm, e1_img_lstm = model.get_mention_embedding(train_e1_mention_tensor,0,train_e1_lengths)
						r_real_lstm, r_img_lstm   = model.get_mention_embedding(train_r_mention_tensor,1,train_r_lengths)

						tmp_nodes_0 = torch.cat([tree_nodes_tail[0],tree_nodes_tail_neg[0]],dim=1)
						tmp_nodes_1 = torch.cat([tree_nodes_tail[1],tree_nodes_tail_neg[1]],dim=1)
						tail_mask_neg *= -1
						tmp_mask = torch.cat([tail_mask,tail_mask_neg],dim=1)

						model_output = model.complex_score_e1_r_with_given_ementions(e1_real_lstm,e1_img_lstm,r_real_lstm,r_img_lstm,tmp_nodes_0,tmp_nodes_1)
						loss = loss - (logsigmoid(model_output*tmp_mask)).mean()
						#neg
						# model_output = model.complex_score_e1_r_with_given_ementions(e1_real_lstm,e1_img_lstm,r_real_lstm,r_img_lstm,tree_nodes_tail_neg[0],tree_nodes_tail_neg[1])
						# loss = loss - (logsigmoid(-1*model_output*tail_mask_neg)).mean()

					# all_outputs.append(output)
				
				# all_outputs = torch.cat(all_outputs)
				# loss = BCEloss(all_outputs.view(-1),labels.view(-1))
				# loss /= normalizer_loss
				# import pdb
				# pdb.set_trace()
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				if(iteration%args.print_loss_every==0):
					print("Current loss:",loss.item())
				iteration+=1
			if(epoch%args.save_model_every==0):
				utils.save_checkpoint({
						'state_dict':model.state_dict(),
						'optimizer':optimizer.state_dict()
						},args.output_dir+"/checkpoint_epoch_{}".format(epoch+1))


	if args.do_eval:
		#eval code
		metrics = {}
		metrics['mr'] = 0
		metrics['mrr'] = 0
		metrics['hits1'] = 0
		metrics['hits10'] = 0
		metrics['hits50'] = 0
		metrics['mr_t'] = 0
		metrics['mrr_t'] = 0
		metrics['hits1_t'] = 0
		metrics['hits10_t'] = 0
		metrics['hits50_t'] = 0
		metrics['mr_h'] = 0
		metrics['mrr_h'] = 0
		metrics['hits1_h'] = 0
		metrics['hits10_h'] = 0
		metrics['hits50_h'] = 0

		if(args.resume and not args.do_train):
			print("Resuming from:",args.resume)
			checkpoint = torch.load(args.resume,map_location=lambda storage, loc: storage)
			model.load_state_dict(checkpoint['state_dict'])

		model.eval()

		cut_nodes_indices = []
		cut_mask          = []
		for i in model.hsoftmax.nodes_at_cut:
			cut_nodes_indices.append(model.hsoftmax.node_indices_for_t[i])
			cut_mask.append(model.hsoftmax.mask_for_t[i])
		cut_mask          = torch.tensor(cut_mask, device=device)
		cut_nodes_indices = torch.tensor(cut_nodes_indices, device=device)
		cut_nodes_real, cut_nodes_img = model.E_atomic(cut_nodes_indices).chunk(2,dim=-1)
		
		# all_nodes_indices = torch.tensor(range(len(entity_mentions)), device=device).unsqueeze(0)
		# import pdb
		# pdb.set_trace()
		# I checked that it can save at max 5 * len(entity_mentions)
		# ementions_real, ementions_img = model.E_atomic(tree_nodes_indices).chunk(2,dim=-1)
		# all_nodes_real, all_nodes_img = model.E_atomic(all_nodes_indices).chunk(2,dim=-1)


		########################################################################
		if "olpbench" in args.data_dir:
			# test_kb = kb(os.path.join(args.data_dir,"train_data_thorough.txt"), em_map = em_map, rm_map = rm_map)
			test_kb = kb(os.path.join(args.data_dir,"test_data.txt"), em_map = em_map, rm_map = rm_map)
		else:
			test_kb = kb(os.path.join(args.data_dir,"test.txt"), em_map = em_map, rm_map = rm_map)

		print("Loading all_known pickled data...(takes times since large)")
		all_known_e2 = {}
		all_known_e1 = {}
		all_known_e2,all_known_e1 = pickle.load(open(os.path.join(args.data_dir,"all_knowns_{}_linked.pkl".format(args.train_data_type)),"rb"))


		test_e1_tokens_tensor, test_e1_tokens_lengths = convert_string_to_indices(test_kb.triples[:,0], etoken_map,maxlen=args.max_seq_length)
		test_r_tokens_tensor, test_r_tokens_lengths = convert_string_to_indices(test_kb.triples[:,1], rtoken_map,maxlen=args.max_seq_length)
		test_e2_tokens_tensor, test_e2_tokens_lengths = convert_string_to_indices(test_kb.triples[:,2], etoken_map,maxlen=args.max_seq_length)
		
		# e2_tensor = convert_string_to_indices(test_kb.triples[:,2], etoken_map)
		indices = torch.Tensor(range(len(test_kb.triples))) #indices would be used to fetch alternative answers while evaluating
		test_data = TensorDataset(indices, test_e1_tokens_tensor, test_r_tokens_tensor, test_e2_tokens_tensor, test_e1_tokens_lengths, test_r_tokens_lengths, test_e2_tokens_lengths)
		test_sampler = SequentialSampler(test_data)
		test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
		split_dim_for_eval = 1
		# e1_support = []
		# e2_support = []

		for index, test_e1_tokens, test_r_tokens, test_e2_tokens, test_e1_lengths, test_r_lengths, test_e2_lengths in tqdm(test_dataloader,desc="Test dataloader"):
			print(metrics)
			test_e1_tokens, test_e1_lengths = test_e1_tokens.to(device), test_e1_lengths.to(device)
			test_r_tokens, test_r_lengths = test_r_tokens.to(device), test_r_lengths.to(device)
			test_e2_tokens, test_e2_lengths = test_e2_tokens.to(device), test_e2_lengths.to(device)
			with torch.no_grad():
				e1_real_lstm, e1_img_lstm = model.get_mention_embedding(test_e1_tokens,0, test_e1_lengths)
				r_real_lstm, r_img_lstm = model.get_mention_embedding(test_r_tokens,1, test_r_lengths)	
				e2_real_lstm, e2_img_lstm = model.get_mention_embedding(test_e2_tokens,0, test_e2_lengths)


			for count in tqdm(range(index.shape[0]), desc="Evaluating"):
				# breakpoint()
				this_e1_real = e1_real_lstm[count]
				this_e1_img  = e1_img_lstm[count]
				this_r_real  = r_real_lstm[count]
				this_r_img   = r_img_lstm[count]
				this_e2_real = e2_real_lstm[count]
				this_e2_img  = e2_img_lstm[count]
				
				ind = index[count]
				this_correct_mentions_e2 = test_kb.e2_all_answers[int(ind.item())]
				this_correct_mentions_e1 = test_kb.e1_all_answers[int(ind.item())] 

				all_correct_mentions_e2 = all_known_e2.get((em_map[test_kb.triples[int(ind.item())][0]],rm_map[test_kb.triples[int(ind.item())][1]]),[])
				all_correct_mentions_e1 = all_known_e1.get((em_map[test_kb.triples[int(ind.item())][2]],rm_map[test_kb.triples[int(ind.item())][1]]),[])
				
				with torch.no_grad():
					pass
					simi_t = model.test_query(this_e1_real, this_e1_img, this_r_real, this_r_img, None, None, "tail", cut_mask, cut_nodes_real, cut_nodes_img)
					simi_h = model.test_query(None, None, this_r_real, this_r_img, this_e2_real, this_e2_real, "head", cut_mask, cut_nodes_real, cut_nodes_img)
					# simi_t = model.test_query_debug(this_e1_real, this_e1_img, this_r_real, this_r_img, None, None, "tail", cut_mask, cut_nodes_real, cut_nodes_img,this_correct_mentions_e2)
					# simi_h = model.test_query_debug(None, None, this_r_real, this_r_img, this_e2_real, this_e2_real, "head", cut_mask, cut_nodes_real, cut_nodes_img,this_correct_mentions_e1)

				# get known answers for filtered ranking
				# e1_support.append(len(all_correct_mentions_e1))
				# e2_support.append(len(all_correct_mentions_e2))

				
				# compute metrics
				# for mention in all_correct_mentions_e2:
				# 	if mention in simi_t:
				# 		rank = torch.tensor(1.).cuda()
				# 		break
				# else:
				# 	rank = torch.tensor(2.).cuda()
				
				# rank = simi_t

				best_score = simi_t[this_correct_mentions_e2].max()
				simi_t[all_correct_mentions_e2] = -20000000 # MOST NEGATIVE VALUE
				greatereq = simi_t.ge(best_score).float()
				equal = simi_t.eq(best_score).float()
				rank = greatereq.sum()+1+equal.sum()/2.0

				metrics['mr_t'] += rank
				metrics['mrr_t'] += 1.0/rank
				metrics['hits1_t'] += rank.le(1).float()
				metrics['hits10_t'] += rank.le(10).float()
				metrics['hits50_t'] += rank.le(50).float()


				# for mention in all_correct_mentions_e1:
				# 	if mention in simi_h:
				# 		rank = torch.tensor(1.).cuda()
				# 		break
				# else:
				# 	rank = torch.tensor(2.).cuda()
				
				# rank = simi_h

				best_score = simi_h[this_correct_mentions_e1].max()
				simi_h[all_correct_mentions_e1] = -20000000 # MOST NEGATIVE VALUE
				greatereq = simi_h.ge(best_score).float()
				equal = simi_h.eq(best_score).float()
				rank = greatereq.sum()+1+equal.sum()/2.0

				metrics['mr_h'] += rank
				metrics['mrr_h'] += 1.0/rank
				metrics['hits1_h'] += rank.le(1).float()
				metrics['hits10_h'] += rank.le(10).float()
				metrics['hits50_h'] += rank.le(50).float()

				metrics['mr'] = (metrics['mr_h']+metrics['mr_t'])/2
				metrics['mrr'] = (metrics['mrr_h']+metrics['mrr_t'])/2
				metrics['hits1'] = (metrics['hits1_h']+metrics['hits1_t'])/2
				metrics['hits10'] = (metrics['hits10_h']+metrics['hits10_t'])/2
				metrics['hits50'] = (metrics['hits50_h']+metrics['hits50_t'])/2
		# e1_support = torch.tensor(e1_support)
		# e2_support = torch.tensor(e2_support)

		# import pdb
		# pdb.set_trace()
		for key in metrics:
			metrics[key] = metrics[key] / len(test_kb.triples)
		print(metrics)


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default=None, type=str, required=True)
	parser.add_argument("--output_dir",default=None,type=str,required=False,help="The output directory where the model checkpoints will be written.")
	parser.add_argument("--max_seq_length",default=128,type=int)
	parser.add_argument("--model",default="complex",type=str)
	parser.add_argument("--do_train",
						default=False,
						action='store_true',
						help="Whether to run training.")
	parser.add_argument("--separate_lstms",
						default=False,
						action='store_true',
						help="spearate lstm for ent and rel.")
	parser.add_argument("--do_eval",
						default=False,
						action='store_true',
						help="Whether to run eval on the dev set.")
	parser.add_argument("--train_batch_size",
						default=32,
						type=int,
						help="Total batch size for training.")
	parser.add_argument("--eval_batch_size",
						default=8,
						type=int,
						help="Total batch size for eval.")
	parser.add_argument("--gamma_rotate",
						default=0,
						type=int,
						help="Gamma for the rotatE model.")
	parser.add_argument("--learning_rate",
						default=1e-5,
						type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--lstm_dropout",
						default=0,
						type=float)
	parser.add_argument("--num_train_epochs",
						default=3,
						type=int,
						help="Total number of training epochs to perform.")
	parser.add_argument("--embedding_dim",
						default=256,
						type=int,
						help="Dimension of embeddings for token.")
	parser.add_argument("--weight_decay",
						default=1e-6,
						type=float,
						help="Proportion of training to perform linear learning rate warmup for. "
							 "E.g., 0.1 = 10%% of training.")
	parser.add_argument("--no_cuda",
						default=False,
						action='store_true',
						help="Whether not to use CUDA when available")
	parser.add_argument('--seed',
						type=int,
						default=42,
						help="random seed for initialization")
	parser.add_argument('--gradient_accumulation_steps',
						type=int,
						default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument('--resume',
						type=str,
						help="Path of already saved checkpoint")
	parser.add_argument('--initial_token_embedding',
						type=str,
						help="Path to intial glove embeddings")
	parser.add_argument("--debug", default=False,action='store_true',)
	parser.add_argument("--skip_train_prob", type=float, default=0, help="Skip batches for training with this probability")
	parser.add_argument('--print_loss_every',type=int,default=1000)
	parser.add_argument('--save_model_every',type=int,default=1)
	parser.add_argument('--train_data_type',type=str,default="thorough")


	args = parser.parse_args()

	if args.output_dir is None:
		args.output_dir = os.path.join("models","{}".format(str(datetime.datetime.now())))

	if not args.debug:
		if not os.path.isdir(args.output_dir):
			print("Making directory (s) %s" % args.output_dir)
			os.makedirs(args.output_dir)
		else:
			utils.colored_print("yellow", "directory %s already exists" % args.output_dir)
		utils.duplicate_stdout(os.path.join(args.output_dir, "log.txt"))
	
	
	print(args)
	main(args)


