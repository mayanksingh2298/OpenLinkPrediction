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
torch.backends.cudnn.benchmark=True
from tqdm import tqdm
import datetime
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from kb import kb
import utils
from dataset import Dataset
from models import rotatELSTM, complexLSTM, complexLSTM_2


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
		#add end token
		mapped_line.append(END_MAPPING)
		lengths.append(len(mapped_line))
		assert(len(mapped_line) <= maxlen)
		for i in range(maxlen-len(mapped_line)):
			mapped_line.append(PAD_MAPPING)
		mapped_data.append(mapped_line)
	return torch.tensor(mapped_data),torch.tensor(lengths)


has_cuda = torch.cuda.is_available()
if not has_cuda:
	utils.colored_print("yellow", "CUDA is not available, using cpu")

def main(args):
	# read token maps
	etokens, etoken_map = utils.get_tokens_map(os.path.join(args.data_dir,"mapped_to_ids","entity_token_id_map.txt"))
	rtokens, rtoken_map = utils.get_tokens_map(os.path.join(args.data_dir,"mapped_to_ids","relation_token_id_map.txt"))
	entity_mentions,em_map = utils.read_mentions(os.path.join(args.data_dir,"mapped_to_ids","entity_id_map.txt"))
	_,rm_map = utils.read_mentions(os.path.join(args.data_dir,"mapped_to_ids","relation_id_map.txt"))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	if not args.do_train and not args.do_eval:
		raise ValueError("At least one of `do_train` or `do_eval` must be True.")

	#train code (+1 for unk token)
	if args.model=="complex":
		if args.separate_lstms:
			model = complexLSTM_2(len(etoken_map)+1,len(rtoken_map)+1,args.embedding_dim, initial_token_embedding =args.initial_token_embedding, entity_tokens = etokens, relation_tokens = rtokens, lstm_dropout=args.lstm_dropout)
		else:
			model = complexLSTM(len(etoken_map)+1,len(rtoken_map)+1,args.embedding_dim, initial_token_embedding =args.initial_token_embedding, entity_tokens = etokens, relation_tokens = rtokens, lstm_dropout=args.lstm_dropout)
	elif args.model == "rotate":
		model = rotatELSTM(len(etoken_map)+1,len(rtoken_map)+1,args.embedding_dim, initial_token_embedding =args.initial_token_embedding, entity_tokens = etokens, relation_tokens = rtokens, gamma = args.gamma_rotate, lstm_dropout=args.lstm_dropout)
	if(args.do_train):
		optimizer = torch.optim.Adagrad(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)

		if(args.resume):
			print("Resuming from:",args.resume)
			checkpoint = torch.load(args.resume)
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			#Load other things too if required

		model.train()
		if "olpbench" in args.data_dir:
			train_kb = kb(os.path.join(args.data_dir,"train_data_{}.txt".format(args.train_data_type)), em_map = em_map, rm_map = rm_map)
			# train_kb = kb(os.path.join(args.data_dir,"train_data_thorough_r_sorted.txt"), em_map = em_map, rm_map = rm_map)
			# train_kb = kb(os.path.join(args.data_dir,"test_data.txt"), em_map = em_map, rm_map = rm_map)

		else:
			train_kb = kb(os.path.join(args.data_dir,"train.txt"), em_map = em_map, rm_map = rm_map)
		
		train_data = Dataset(train_kb.triples)
		train_sampler = RandomSampler(train_data,replacement=False)
		#train_sampler = SequentialSampler(train_data)
		train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

		# crossEntropyLoss = torch.nn.CrossEntropyLoss(reduction='mean')
		BCEloss = torch.nn.BCEWithLogitsLoss(reduction='sum')

		for epoch in tqdm(range(0,args.num_train_epochs),desc="epoch"):
			iteration = 0
			for train_e1_batch, train_r_batch, train_e2_batch in tqdm(train_dataloader,desc="Train dataloader"):
				# skip this batch
				if(random.random()<args.skip_train_prob):
					continue	
				batch_size = len(train_e1_batch)
				train_e1_mention_tensor, train_e1_lengths = convert_string_to_indices(train_e1_batch,etoken_map,maxlen=args.max_seq_length)
				train_r_mention_tensor, train_r_lengths = convert_string_to_indices(train_r_batch,rtoken_map,maxlen=args.max_seq_length)
				train_e2_mention_tensor, train_e2_lengths = convert_string_to_indices(train_e2_batch,etoken_map,maxlen=args.max_seq_length)

				train_e1_mention_tensor, train_e1_lengths = train_e1_mention_tensor.cuda(), train_e1_lengths.cuda()
				train_r_mention_tensor, train_r_lengths = train_r_mention_tensor.cuda(), train_r_lengths.cuda()
				train_e2_mention_tensor, train_e2_lengths = train_e2_mention_tensor.cuda(), train_e2_lengths.cuda()


				e1_real_lstm, e1_img_lstm = model.get_mention_embedding(train_e1_mention_tensor,0,train_e1_lengths)
				r_real_lstm, r_img_lstm = model.get_mention_embedding(train_r_mention_tensor,1,train_r_lengths)
				e2_real_lstm, e2_img_lstm = model.get_mention_embedding(train_e2_mention_tensor,0,train_e2_lengths)
				#tail
				simi_t = model.complex_score_e1_r_with_all_ementions(e1_real_lstm,e1_img_lstm,r_real_lstm,r_img_lstm,e2_real_lstm,e2_img_lstm)
				#head
				simi_h = model.complex_score_e2_r_with_all_ementions(e2_real_lstm,e2_img_lstm,r_real_lstm,r_img_lstm,e1_real_lstm,e1_img_lstm)
				# change the loss suitably
				target = torch.eye(batch_size).cuda()
				# import pdb
				# pdb.set_trace()
				loss_t = BCEloss(simi_t.view(-1),target.view(-1))
				loss_h = BCEloss(simi_h.view(-1),target.view(-1))
				loss = (loss_h+loss_t)/2
				loss /= target.size(0) * target.size(1)

				# Do the routine
				optimizer.zero_grad()
				loss.backward()
				#gradient clip?
				optimizer.step()

				if(iteration%args.print_loss_every==0):
					print("Current loss(avg, tail, head):",loss.item(), loss_t.item(), loss_h.item())
				iteration+=1
			if(epoch%args.save_model_every==0 and epoch!=0):
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

		# get embeddings for all entity mentions
		entity_mentions_tensor, entity_mentions_lengths = convert_string_to_indices(entity_mentions,etoken_map,maxlen=args.max_seq_length,use_tqdm=True)
		entity_mentions_tensor = entity_mentions_tensor.cuda()
		entity_mentions_lengths = entity_mentions_lengths.cuda()

		ementions_real_lis = []
		ementions_img_lis = []
		split = 100 #cant fit all in gpu together. hence split
		with torch.no_grad():
			for i in tqdm(range(0,len(entity_mentions_tensor),len(entity_mentions_tensor)//split)):
				data = entity_mentions_tensor[i:i+len(entity_mentions_tensor)//split,:]
				data_lengths = entity_mentions_lengths[i:i+len(entity_mentions_tensor)//split]
				ementions_real_lstm,ementions_img_lstm = model.get_mention_embedding(data,0,data_lengths)			
				# a = model.Et_im(entity_mentions_tensor[i:i+len(entity_mentions_tensor)//split,:])
				# b = model.Et_re(entity_mentions_tensor[i:i+len(entity_mentions_tensor)//split,:])
				
				# a_lstm,_ = model.lstm(a)
				# a_lstm = a_lstm[:,-1,:]

				
				# b_lstm,_ = model.lstm(b)
				# b_lstm = b_lstm[:,-1,:]

				ementions_real_lis.append(ementions_real_lstm.cpu())
				ementions_img_lis.append(ementions_img_lstm.cpu())
		del entity_mentions_tensor,ementions_real_lstm,ementions_img_lstm
		torch.cuda.empty_cache()
		ementions_real = torch.cat(ementions_real_lis).cuda()
		ementions_img = torch.cat(ementions_img_lis).cuda()
		########################################################################
		if "olpbench" in args.data_dir:
			# test_kb = kb(os.path.join(args.data_dir,"test_data_sophis.txt"), em_map = em_map, rm_map = rm_map)
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
		if(args.embedding_dim>=256 and "olpbench" in args.data_dir and "rotat" in args.model):
			split_dim_for_eval = 4
		if(args.embedding_dim>=512 and "olpbench" in args.data_dir):
			split_dim_for_eval = 4
		if(args.embedding_dim>=512 and "olpbench" in args.data_dir and "rotat" in args.model):
			split_dim_for_eval = 6
		split_dim_for_eval = 1
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
				this_e1_real = e1_real_lstm[count].unsqueeze(0)
				this_e1_img  = e1_img_lstm[count].unsqueeze(0)
				this_r_real  = r_real_lstm[count].unsqueeze(0)
				this_r_img   = r_img_lstm[count].unsqueeze(0)
				this_e2_real = e2_real_lstm[count].unsqueeze(0)
				this_e2_img  = e2_img_lstm[count].unsqueeze(0)
				# import pdb
				# pdb.set_trace()
				simi_t = model.complex_score_e1_r_with_all_ementions(this_e1_real,this_e1_img,this_r_real,this_r_img,ementions_real,ementions_img,split=split_dim_for_eval).squeeze(0)
				simi_h = model.complex_score_e2_r_with_all_ementions(this_e2_real,this_e2_img,this_r_real,this_r_img,ementions_real,ementions_img,split=split_dim_for_eval).squeeze(0)
				# get known answers for filtered ranking
				ind = index[count]
				this_correct_mentions_e2 = test_kb.e2_all_answers[int(ind.item())]
				this_correct_mentions_e1 = test_kb.e1_all_answers[int(ind.item())] 

				all_correct_mentions_e2 = all_known_e2.get((em_map[test_kb.triples[int(ind.item())][0]],rm_map[test_kb.triples[int(ind.item())][1]]),[])
				all_correct_mentions_e1 = all_known_e1.get((em_map[test_kb.triples[int(ind.item())][2]],rm_map[test_kb.triples[int(ind.item())][1]]),[])
				
				# compute metrics
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


