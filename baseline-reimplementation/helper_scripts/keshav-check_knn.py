import sys
import os
sys.path.append(sys.path[0]+"/../")
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
from tqdm import tqdm
import datetime
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from kb import kb
import utils
from dataset import Dataset
import ast


has_cuda = torch.cuda.is_available()
if not has_cuda:
	utils.colored_print("yellow", "CUDA is not available, using cpu")


def main():
	K = 2
	data_dir = "helper_scripts"
	knn_lines = open(os.path.join(data_dir,"keshav_xt-preds","knn_validation_simple.txt")).readlines()
	i = 0
	count = 0
	while(i<len(knn_lines)):
		query = knn_lines[i]
		assert query.startswith("Query:")

		query = query[query.index("Query: ")+7:].strip().split("\t")

		i+=1
		assert knn_lines[i].strip()=="Nearest Neighbours:"

		evidence = []
		for j in range(5):
			i+=1
			evidence.append(knn_lines[i].strip().split("\t"))

		i+=1
		assert knn_lines[i].strip()==""		

		for j in range(5):
			if (query[0]==evidence[j][0] and query[2]==evidence[j][2]) or (query[2]==evidence[j][0] and query[0]==evidence[j][2]):
				count += 1
				break

		i+=1

	print(count)



if __name__=="__main__":
	main()


