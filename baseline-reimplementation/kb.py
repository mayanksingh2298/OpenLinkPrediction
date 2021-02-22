import numpy as np
import torch
import copy
from  tqdm import tqdm
class kb(object):
	"""
		stores data about knowledge base
		if split_as_regular_sentences is True:
			triples stores as [["this","is","a","ball"],...] (i.e. not as e1,r,e2)
	"""
	def __init__(self,filename,em_map=None,rm_map=None,split_as_regular_sentences = False):
		print("Reading file {}".format(filename))
		lines = open(filename,'r').readlines()
		self.triples = []
		self.e1_all_answers = [] # saves all answers for ith triple using their 'int' value from map 
		self.e2_all_answers = []
		self.em_map = em_map
		self.rm_map = rm_map

		#save alternate mentions, triple wise
		for line in tqdm(lines):
			line = line.strip().split("\t")
			if (not split_as_regular_sentences):
				self.triples.append([line[0],line[1],line[2]])
				e1_answers = line[3].split("|||")
				e2_answers = line[4].split("|||")
				if(em_map!=None):
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
					self.e1_all_answers.append(mapped_e1_answers)
					self.e2_all_answers.append(mapped_e2_answers)
				else:
					self.e1_all_answers.append(e1_answers)
					self.e2_all_answers.append(e2_answers)
			else:
				self.triples.append(line[0].split()+line[1].split()+line[2].split())
		if(not split_as_regular_sentences):
			self.triples = np.array(self.triples)
