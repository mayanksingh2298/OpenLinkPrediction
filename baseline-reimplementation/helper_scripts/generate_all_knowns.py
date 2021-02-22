"""
	use this to generate all knowns matrix for model evaluation(filtered ranking)
	usage: python3 helper_scripts/generate_all_knowns.py --data_dir <path to data set> --train [basic|simple|thorough] --val [all|linked_mention|linked]
"""
import argparse
import os
from tqdm import tqdm
import pickle


def _update_all_knowns(file_path,em_map,rm_map,all_knowns_e2,all_knowns_e1):
	lines = open(file_path).readlines()
	ct = 0
	for line in tqdm(lines):
		line = line.strip().split("\t")
		e1_answers = line[3].split("|||")		
		e2_answers = line[4].split("|||")
		mapped_e1_answers = []
		mapped_e2_answers = []
		for i in range(len(e1_answers)):
			if(e1_answers[i] not in em_map): # sometimes the answer isn't present in em_map. ignoring that answer
				continue
			mapped_e1_answers.append(em_map[e1_answers[i]])
		for i in range(len(e2_answers)):
			if(e2_answers[i] not in em_map): # sometimes the answer isn't present in em_map. ignoring that answer
				continue
			mapped_e2_answers.append(em_map[e2_answers[i]])
		e1 = em_map[line[0]]
		r  = rm_map[line[1]]
		e2  = em_map[line[2]]

		if((e1,r) not in all_knowns_e2):
			all_knowns_e2[(e1,r)] = []
		all_knowns_e2[(e1,r)].extend(mapped_e2_answers)
		if((e2,r) not in all_knowns_e1):
			all_knowns_e1[(e2,r)] = []
		all_knowns_e1[(e2,r)].extend(mapped_e1_answers)
		ct+=1
		assert mapped_e2_answers!=[]
		assert mapped_e1_answers!=[]


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default=None, type=str, required=True)
	parser.add_argument("--train", default=None, type=str, required=True)
	parser.add_argument("--val", default=None, type=str, required=True)
	parser.add_argument("--output_dir", default=None, type=str, required=True)
	args = parser.parse_args()

	# Loading entity mentions
	entity_mentions = []
	em_map = {}
	lines = open(os.path.join(args.data_dir,"mapped_to_ids","entity_id_map.txt"),'r').readlines()
	print("Reading entity mentions...")
	for line in tqdm(lines[1:]):
		line = line.strip().split("\t")
		entity_mentions.append(line[0])
		em_map[line[0]] = len(em_map)

	# Loading relation mentions
	relation_mentions = []
	rm_map = {}
	lines = open(os.path.join(args.data_dir,"mapped_to_ids","relation_id_map.txt"),'r').readlines()
	print("Reading relation mentions...")
	for line in tqdm(lines[1:]):
		line = line.strip().split("\t")
		relation_mentions.append(line[0])
		rm_map[line[0]] = len(rm_map)

	# all_knowns[(i,j)] = lis
	# lis is the set of all e2 seen for e1:i and r:j. Note these are mapped indices and not strings
	all_knowns_e2 = {} # tail prediction
	all_knowns_e1 = {} # head prediction


	print("parsing test...")
	_update_all_knowns(os.path.join(args.data_dir,"test_data.txt"),em_map,rm_map,all_knowns_e2,all_knowns_e1)
	# _update_all_knowns(os.path.join(args.data_dir,"test.txt"),em_map,rm_map,all_knowns_e2,all_knowns_e1)


	print("parsing valid...")
	_update_all_knowns(os.path.join(args.data_dir,"validation_data_"+args.val+".txt"),em_map,rm_map,all_knowns_e2,all_knowns_e1)
	# _update_all_knowns(os.path.join(args.data_dir,"valid.txt"),em_map,rm_map,all_knowns_e2,all_knowns_e1)

	print("parsing train...")
	_update_all_knowns(os.path.join(args.data_dir,"train_data_"+args.train+".txt"),em_map,rm_map,all_knowns_e2,all_knowns_e1)
	# _update_all_knowns(os.path.join(args.data_dir,"train.txt"),em_map,rm_map,all_knowns_e2,all_knowns_e1)

	f = open(args.output_dir,'wb')
	pickle.dump([all_knowns_e2,all_knowns_e1], f)
	f.close()

