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
	# entity_mentions = []
	# em_map = {}
	# lines = open(os.path.join(args.data_dir,"mapped_to_ids","entity_id_map.txt"),'r').readlines()
	# print("Reading entity mentions...")
	# for line in tqdm(lines[1:]):
	# 	line = line.strip().split("\t")
	# 	entity_mentions.append(line[0])
	# 	em_map[line[0]] = len(em_map)

	# Loading relation mentions
	relation_mentions = []
	rm_map = {}
	lines = open(os.path.join(args.data_dir,"mapped_to_ids","relation_id_map.txt"),'r').readlines()
	print("Reading relation mentions...")
	for line in tqdm(lines[1:]):
		line = line.strip().split("\t")
		relation_mentions.append(line[0])
		rm_map[line[0]] = len(rm_map)
	# relation_mentions = ["was","in","has:impl_poss-clause","is","be:impl_vmod","of","to","is:impl_np-person","by","the","on","are","were","a","for","with","at","from","as","has","had","is:impl_appos-clause","also","be","is:impl_hearst-np-such-as-np","is:impl_hearst-np-including-np","been","born","have","made","into","died","released","played","used","is:impl_hearst-np-and-or-other-np","is:impl_org-in-loc","won","became","up","an","began","held","moved","out","is:impl_org-np-person","of:impl_org-np-person","his","received","returned","served","is:impl_hearst-np-like-np"]

	lines = open(os.path.join(args.data_dir,"test_data.txt")).readlines()
	ct = 0
	f_new = open(args.output_file,'w')
	for line in tqdm(lines):
		line = line.strip().split("\t")
		e1_answers = line[3].split("|||")		
		e2_answers = line[4].split("|||")
		for times in range(args.n_times):
			line[1] = relation_mentions[random.randint(0,len(relation_mentions)-1)]
			# line[1] = "is is"

			f_new.write("\t".join(line)+"\n")
	f_new.close()