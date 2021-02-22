import random
import argparse
import os
from tqdm import tqdm
import pickle

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default=None, type=str, required=True)
	parser.add_argument("--output_file", default=None, type=str, required=True)
	parser.add_argument("--n_times", default=1, type=int, required=False, help="Number of new evidences to be inserted")

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

	lines = open(os.path.join(args.data_dir,"test_data.txt")).readlines()
	ct = 0
	f_new = open(args.output_file,'w')
	for line in tqdm(lines):
		line = line.strip().split("\t")
		for times in range(args.n_times):
			random_r1 = relation_mentions[random.randint(0,len(relation_mentions)-1)]
			random_r2 = relation_mentions[random.randint(0,len(relation_mentions)-1)]
			random_e = entity_mentions[random.randint(0,len(entity_mentions)-1)]
			line_copy1 = line.copy()
			line_copy2 = line.copy()
			line_copy1[1] = random_r1
			line_copy1[2] = random_e
			line_copy2[0] = random_e
			line_copy2[1] = random_r2
			f_new.write("\t".join(line_copy1)+"\n")
			f_new.write("\t".join(line_copy2)+"\n")
	f_new.close()