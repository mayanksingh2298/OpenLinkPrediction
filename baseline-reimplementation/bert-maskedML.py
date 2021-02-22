from transformers import BertTokenizer, BertForMaskedLM
import torch
import argparse

def sample():
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True)
	# model = BertForMaskedLM.from_pretrained('bert-base-uncased')


	inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
	labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
	top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
	mask_token_logits = 6

	outputs = model(**inputs, labels=labels)

	loss = outputs.loss
	logits = outputs.logits

	top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices.tolist()

	for token in top_5_tokens:
		print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))

	import ipdb
	ipdb.set_trace()


def main(args):
	# step 1 get tokenizer
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	# step 2 get bert model 
	model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True)
python3 run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/olpbench/test_data.txt \
    --validation_file data/olpbench/test_data.txt \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm --line_by_line

python3 run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm --line_by_line


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default=None, type=str, required=False)
	args = parser.parse_args()
	main(args)