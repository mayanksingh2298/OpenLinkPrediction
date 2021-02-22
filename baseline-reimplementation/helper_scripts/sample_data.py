import sys
import os
sys.path.append(sys.path[0]+"/../")
import argparse
from kb import kb
import random
import numpy as np
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--file", default=None, type=str, required=True)
	parser.add_argument("--sample", default=None, type=int, required=True, help="How many triples to sample?")

	args = parser.parse_args()

	data = kb(args.file)
	indices = list(range(len(data.triples)))
	random.shuffle(indices)

	for i in indices[:args.sample]:
		print(i,"|",data.triples[i],"|",data.e1_all_answers[i],"|",data.e2_all_answers[i])


