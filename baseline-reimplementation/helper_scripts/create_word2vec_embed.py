import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
import argparse
from gensim.models import Word2Vec
from time import time
import sys
import os
sys.path.append(sys.path[0]+"/../")
import utils
from kb import kb
# # define training data
# sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
# 			['this', 'is', 'the', 'second', 'sentence'],
# 			['yet', 'another', 'sentence'],
# 			['one', 'more', 'sentence'],
# 			['and', 'the', 'final', 'sentence']]
# # train model
# model = Word2Vec(sentences, min_count=1)
# # summarize the loaded model
# print(model)
# # summarize vocabulary
# words = list(model.wv.vocab)
# print(words)
# # access vector for one word
# print(model['sentence'])
# # save model
# model.save('model.bin')
# # load model
# new_model = Word2Vec.load('model.bin')
# print(new_model)
def main(args):
	sentences = kb(os.path.join(args.data_dir,"train_data_simple.txt"), em_map = None, rm_map = None, split_as_regular_sentences=True).triples
	# sentences = kb(os.path.join(args.data_dir,"delta_simple_thorough.txt"), em_map = None, rm_map = None, split_as_regular_sentences=True).triples
	w2v_model = Word2Vec(min_count=1, size=args.embedding_dim, window = args.window, workers = args.workers, negative=args.negative)
	
	t = time()
	print("Building vocab...")
	w2v_model.build_vocab(sentences, progress_per=10000)
	print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

	t = time()
	print("Begin training...")
	w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=args.num_train_epochs, report_delay=1)
	print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

	w2v_model.init_sims(replace=True)
	word_vectors = w2v_model.wv
	vocab = list(word_vectors.vocab.keys())
	f = open(args.output_file,'w')
	for word in vocab:
		towrite = word
		embeddings = word_vectors.get_vector(word)
		for embed in embeddings:
			towrite+=" "+str(embed)
		print(towrite,file=f)
	f.close()
	# import pdb
	# pdb.set_trace()

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default=None, type=str, required=True)
	parser.add_argument("--output_file", default=None, type=str, required=True)
	parser.add_argument("--workers", default=3, type=int)
	parser.add_argument("--window", default=7, type=int)
	parser.add_argument("--negative", default=20, type=int)
	parser.add_argument("--embedding_dim",
						default=256,
						type=int,
						help="Dimension of embeddings for token.")
	parser.add_argument("--num_train_epochs",
						default=10,
						type=int,
						help="Total number of training epochs to perform.")
	args = parser.parse_args()

	print(args)
	main(args)