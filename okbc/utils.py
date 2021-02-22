# Generate labels file

# from tqdm import tqdm
# inp_f = open('data/debug.txt','r')
# out_f = open('data/debug_labels.txt','w')
# e2_set = set()
# for line in tqdm(inp_f):
#     line = line.strip('\n')
#     e2 = line.split('\t')[2]
#     e2_set.add(e2)
# for e2 in e2_set:
#     out_f.write(e2+'\n')
# out_f.close()
from tqdm import tqdm

# get entity/relation tokens map given path to a file which lists all entity/relation tokens
# first line is header
# use the len(map) for unk(basically the last token)
def get_tokens_map(path):
	"""
		<PAD>: 0
		<INIT>: 1
		<END>: 2
		...
	"""
	lines = open(path,'r').readlines()
	token_map = {"<PAD>":0,"<INIT>":1,"<END>":2}
	tokens = ["<PAD>","<INIT>","<END>"]
	for line in tqdm(lines[1:],desc="Reading file for token map"):
		line = line.strip().split("\t")
		tokens.append(line[0])
		token_map[line[0]] = len(token_map)
	return tokens,token_map

# read entity or relation mentions from a file containing them 
# 1st line is header
def read_mentions(path):
	mapp = {}
	mentions = []
	lines = open(path,'r').readlines()
	for line in tqdm(lines[1:]): 
		line = line.strip().split("\t")
		mentions.append(line[0])
		mapp[line[0]] = len(mapp)
	return mentions,mapp
