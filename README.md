# OpenLinkPrediction
The problem of link prediction has been extensively studied for closed knowledge graphs for quite some time. Researchers have come to with multiple techniques to solve this problem. While closed knowledge graphs are fascinating, much lesser work has been done for open knowledge graphs mostly because of lack of a proper dataset. Recently, [BGWG20] published a dataset - OLPBENCH for standardizing the task of open link prediction. A large part of this thesis is centered around this dataset and trying to improve their baseline algorithms. We try a bunch of adhoc heuristics and find that as simple as they may be - they are good enough to beat the author’s baseline. We also work on a two stage architecture which uses the pre-trained language model [DCLT19] to further improve the link prediction metrics. Not only on open knowledge graphs we find that this architecture also improves the performance of state of the art models on closed knowledge graphs.

# Downloading the OLPBENCH dataset
```
wget http://data.dws.informatik.uni-mannheim.de/olpbench/olpbench.tar.gz
tar xzf olpbench.tar.gz
```
## Contents
./mapped_to_ids
./mapped_to_ids/entity_id_map.txt
./mapped_to_ids/entity_token_id_map.txt
./mapped_to_ids/relation_id_map.txt
./mapped_to_ids/relation_token_id_map.txt
./mapped_to_ids/entity_id_tokens_ids_map.txt
./mapped_to_ids/relation_id_tokens_ids_map.txt
./mapped_to_ids/validation_data_linked_mention.txt
./mapped_to_ids/validation_data_linked.txt
./mapped_to_ids/validation_data_all.txt
./mapped_to_ids/test_data.txt
./mapped_to_ids/train_data_thorough.txt
./mapped_to_ids/train_data_basic.txt
./mapped_to_ids/train_data_simple.txt

./train_data_simple.txt
./train_data_basic.txt
./train_data_thorough.txt

./validation_data_all.txt
./validation_data_linked_mention.txt
./validation_data_linked.txt

./test_data.txt

./README.md

## Format

### train, validation and test files

The train, validation and test files contain triples as described in the publication. The format is 5 TAB separated columns:

COL 1			COL 2			COL 3			COL 4					COL 5
subject mention tokens	open relation tokens	object mention tokens	alternative subj mention|||...|||...	alternative obj mention|||...|||...

Except for validation_data_linked.txt test_data.txt COL 4 and COL 5 are empty.


### mapped_to_ids

mapped_to_ids contains the training, validation and test data mapped to ids according to the maps:

entity_id_map.txt:		maps entities (actually entity mentions) to ids, starting from 2 (!)
relation_id_map.txt:		maps relations to ids, starting from 2 (!)

entity_token_id_map.txt:	maps the tokens of entities to ids, starting from 4 (!)
relation_token_id_map.txt:	maps the tokens of relations to ids, starting from 4 (!)

entity_id_tokens_ids_map.txt:	maps entity ids to a sequence of token ids
relation_id_tokens_ids_map.txt:	maps relation ids to a sequence of token ids

The train, validation and test files contain triples as described in the publication. The format is 5 TAB separated columns, COL 4 and COL 5 are lists of space seperated ids:

COL 1		COL 2           COL 3		COL 4			COL 5
entity id	relation id	entity id	alt. subj entity ids	alt. obj entity ids

# Baseline implementation - OLPBENCH authors
The authors of `OLPBENCH` provide a simple baseline algorith which uses LSTM to compose tokens to yield a single embedding for the entire entity(relation). Further, the complex scoring function scores the validity of the triple using these embeddings.  
The complete instructions to run their implementations are available in their public repository: https://github.com/samuelbroscheit/open_knowledge_graph_embeddings   
Note: Instead of downloading the OLPBENCH dataset again, make a softlink to the dataset downloaded in earlier step as follows:
```
cd open_knowledge_graph_embeddings/data
ln -s ../olpbench olpbench
```

# Our implementation - complex-LSTM
The authors who produced the OLPBENCH dataset delayed a lot in making their code public. So we re-implemented their algorithms by following their paper and did our analysis using our implementation. This section contains the instructions to use our re-implementation.  


## Generating all knowns
The first step is to generate the pickle file for filtered evaluation:
```
cd baseline-reimplementation
python3 helper_scripts/generate_all_knowns.py --data_dir ../olpbench --train simple --val linked --output_dir ../olpbench/all_knowns_simple_linked.pkl
```
The folder `helper_scripts` contains some more hacky scripts for various tasks.

## Training
This section contains training instructions for different settings. Refer to the file `e2e.sh` for more instructions.  
Training the paper baseline algorithm
```
python3 main-baseline-paper.py --data_dir ../olpbench --do_train --train_batch_size 512 --num_train_epochs 101 --max_seq_length 15 --output_dir models/paper_baseline_simple_2lstm --learning_rate 0.2 --embedding_dim 512 --weight_decay 1e-10 --skip_train_prob 0.0 --lstm_dropout 0.1 --save_model_every 10 --separate_lstms
```
Training the model using exactly the authors' pickle files (they seem to have done better pre-processing)
(Follow the authors' github repo to run the initial pre-processing before proceeding)
```
python3 main-baseline-author_data-all_e.py --data_dir ../olpbench --do_train --train_batch_size 512 --num_train_epochs 10 --max_seq_length 15 --output_dir models/paper_baseline_simple_2lstm_512_thorough_r-sorted --learning_rate 0.2 --embedding_dim 512 --weight_decay 1e-10 --skip_train_prob 0.0 --lstm_dropout 0.1 --save_model_every 1 --separate_lstms
```

## Evaluating
```
python3 main-baseline.py --data_dir ../olpbench --do_eval --eval_batch_size 512 --max_seq_length 15 --debug --embedding_dim 256 --resume models/paper_baseline_simple_2lstm/checkpoint_epoch_28
```


# Extreme text - mcqBERT: 2 stage architecture
This section contains instruction for training the extreme text model (original github repo: https://github.com/mwydmuch/extremeText) and then training BERT over the predictions of this extreme text model.   
For more instruction refer to `exteremeText/e2e_xt.sh`

## Install extremetext
```
cd extremeText
make
```

## Closed KBs
Step 1: Prepare the input files
```
cd extremeText/data/closed_kb
python xt_input.py --inp <dataset>/train.txt --out <dataset>/train --type tail --relation2text_file <dataset>/relation2text.txt --entity2text_file <dataset>/entity2text.txt --entity_id_file <dataset>/entities.dict
python xt_input.py --inp <dataset>/train.txt --out <dataset>/train --type head --relation2text_file <dataset>/relation2text.txt --entity2text_file <dataset>/entity2text.txt --entity_id_file <dataset>/entities.dict
python xt_input.py --inp <dataset>/test.txt --out <dataset>/test --type tail --relation2text_file <dataset>/relation2text.txt --entity2text_file <dataset>/entity2text.txt --entity_id_file <dataset>/entities.dict
python xt_input.py --inp <dataset>/test.txt --out <dataset>/test --type head --relation2text_file <dataset>/relation2text.txt --entity2text_file <dataset>/entity2text.txt --entity_id_file <dataset>/entities.dict
python xt_input.py --inp <dataset>/valid.txt --out <dataset>/valid --type tail --relation2text_file <dataset>/relation2text.txt --entity2text_file <dataset>/entity2text.txt --entity_id_file <dataset>/entities.dict
python xt_input.py --inp <dataset>/valid.txt --out <dataset>/valid --type head --relation2text_file <dataset>/relation2text.txt --entity2text_file <dataset>/entity2text.txt --entity_id_file <dataset>/entities.dict
```

Step 2: Train the stage 1 extreme text model
```
cd extremeText
./extremetext supervised -input data/closed_kb/<dataset>/train.tail.xt -output data/closed_kb/<dataset>/tail_300d -lr 0.1 -thread 40 -epoch 50 -dim 300 -loss plt -wordNgrams 2
./extremetext supervised -input data/closed_kb/<dataset>/train.head.xt -output data/closed_kb/<dataset>/head_300d -lr 0.1 -thread 40 -epoch 50 -dim 300 -loss plt -wordNgrams 2
```

Step 3: How to get predictions?
```
./extremetext predict-prob data/closed_kb/<dataset>/head_300d.bin data/closed_kb/<dataset>/test.head.xt 50 0 data/closed_kb/<dataset>/test.head.preds.txt 1 > data/closed_kb/<dataset>/test.head.preds.txt
./extremetext predict-prob data/closed_kb/<dataset>/tail_300d.bin data/closed_kb/<dataset>/test.tail.xt 50 0 data/closed_kb/<dataset>/test.tail.preds.txt 1 > data/closed_kb/<dataset>/test.tail.preds.txt
```

Step 4: How to test stage1 results and generate the pickle file for stage2?
```
cd extremeText/data/closed_kb
python3 test_filtered_metrics.py --dataset <dataset> --mode test
python3 test_filtered_metrics.py --dataset <dataset> --mode valid
python3 test_filtered_metrics.py --dataset <dataset> --mode train
```
This will create files `train_data.pkl`, `test_data.pkl` and `validation_data.pkl` in `<dataset>` folder. Move these files to the appropriate dataset as mentioned below:
```
cd okbc
mkdir closed_kbc_data/xt
mkdir closed_kbc_data/xt/data
mkdir closed_kbc_data/xt/data/WN18RR
mkdir closed_kbc_data/xt/data/YAGO3-10
mkdir closed_kbc_data/xt/data/FB15k-237
cp ../extremeText/data/closed_kb/<dataset>/*dict closed_kbc_data/xt/data/<DATASET>/
mv ../extremeText/data/closed_kb/<dataset>/*data.pkl closed_kbc_data/xt/data/<DATASET>/
```
In instructions below, we will refer to these dataset names as `<DATASET>`

Step 5: Generate data for training stage 2?
```
cd okbc
python convert_kbc.py --kge_output closed_kbc_data/xt/data/<DATASET>/test_data.pkl --kbe_data_dir closed_kbc_data/kg-bert --kge_data_dir closed_kbc_data/xt/ --dataset <DATASET> --output_dir closed_kbc_data/data_for_stage2/xt-<DATASET> --output_file test_data.txt --model Xt --entity_map --relation_map --filter --predictions --scores
python convert_kbc.py --kge_output closed_kbc_data/xt/data/<DATASET>/validation_data.pkl --kbe_data_dir closed_kbc_data/kg-bert --kge_data_dir closed_kbc_data/xt/ --dataset <DATASET> --output_dir closed_kbc_data/data_for_stage2/xt-<DATASET> --output_file validation_data.txt --model Xt --filter_val --predictions --scores_val
python convert_kbc.py --kge_output closed_kbc_data/xt/data/<DATASET>/train_data.pkl --kbe_data_dir closed_kbc_data/kg-bert --kge_data_dir closed_kbc_data/xt/ --dataset <DATASET> --output_dir closed_kbc_data/data_for_stage2/xt-<DATASET> --output_file train_data.txt --model Xt --predictions
python3 tokenize_pkl.py --inp closed_kbc_data/data_for_stage2/xt-<DATASET>/mapped_to_ids/entity_id_map.txt --model_str bert-base-cased
python3 tokenize_pkl.py --inp closed_kbc_data/data_for_stage2/xt-<DATASET>/mapped_to_ids/relation_id_map.txt --model_str bert-base-cased
```

Step 6: How to train stage2?
```
cd okbc
python run.py --save closed_kbc_data/models/<DATASET> --mode train_test --gpus 1 --epochs 10 --stage2 --negative_samples 10 --data_dir closed_kbc_data/data_for_stage2/xt-<DATASET> --model mcq --stage1_model Xt --model_str bert-base-cased --task_type both --max_tokens 5000 --ckbc
```

Step 7: How to test stage2?
```
cd okbc
python run.py --save closed_kbc_data/models/dummy --mode test --gpus 1 --epochs 5 --stage2 --negative_samples 10 --data_dir closed_kbc_data/data_for_stage2/xt-<DATASET>/ --model mcq --stage1_model Xt --model_str bert-base-cased --task_type both --checkpoint closed_kbc_data/models/<DATASET>/<checkpoint> --test closed_kbc_data/data_for_stage2/xt-<DATASET>/test_data.txt
```

## Open KBs
Step 1: convert open kb data to xt format


```
cd baseline-reimplementation
python3 helper_scripts/get_freq_for-relation.py
cd ..
cd extremeText/data
ln -s ../../olpbench olpbench
python xt_input.py --inp olpbench/test_data.txt --type head --num_frequent 5
python xt_input.py --inp olpbench/test_data.txt --type tail --num_frequent 5
python xt_input.py --inp olpbench/train_data_thorough.txt --type head --num_frequent 5
python xt_input.py --inp olpbench/train_data_thorough.txt --type tail --num_frequent 5
python xt_input.py --inp olpbench/validation_data_linked.txt --type head --num_frequent 5
python xt_input.py --inp olpbench/validation_data_linked.txt --type tail --num_frequent 5
```

Step 2: How to train?
```
cd extremeText
./extremetext supervised -input data/olpbench/train_data_thorough.txt.tail.xt -output data/olpbench/xt_models/tail_thorough_f5_d300 -lr 0.1 -thread 40 -epoch 50 -dim 300 -loss plt -wordNgrams 2
./extremetext supervised -input data/olpbench/train_data_thorough.txt.head.xt -output data/olpbench/xt_models/head_thorough_f5_d300 -lr 0.1 -thread 40 -epoch 50 -dim 300 -loss plt -wordNgrams 2
```

Step 3: How to get predictions and output for stage2?
```
MODEL=thorough_f5_d300
FILE=olpbench/test_data.txt
THREADS=2

cd extremeText
cd data
../extremetext predict-prob olpbench/xt_models/head_$MODEL.bin $FILE.head.xtp 50 0 $FILE.head_$MODEL'.preds' $THREADS
cat $FILE.head_$MODEL'.preds'.0* > $FILE.head_$MODEL'.preds_parallel'
rm $FILE.head_$MODEL'.preds'.0*
mv $FILE.head_$MODEL'.preds_parallel' $FILE.head_$MODEL'.preds'

../extremetext predict-prob olpbench/xt_models/tail_$MODEL.bin $FILE.tail.xtp 50 0 $FILE.tail_$MODEL'.preds' $THREADS
cat $FILE.tail_$MODEL'.preds'.0* > $FILE.tail_$MODEL'.preds_parallel'
rm $FILE.tail_$MODEL'.preds'.0*
mv $FILE.tail_$MODEL'.preds_parallel' $FILE.tail_$MODEL'.preds'

python xt_output.py --inp $FILE --model $MODEL --type head
python xt_output.py --inp $FILE --model $MODEL --type tail
```
Repeat the process for validation set and subset(for reasonable train time)(you might need to create this subset of train set. You can use the first 1 mil train points in thorough dataset) of train set. (Use `THREADS=60` for train data). This will create multiple `.stage1` files. Move them to the following locations.
```
cd okbc
mkdir open_kbc_data
mv ../olpbench/*stage1 open_kbc_data/ 
cd open_kbc_data
ln -s ../../olpbench/mapped_to_ids mapped_to_ids
ln -s ../../olpbench/mapped_to_ids mapped_to_ids
cp ../../olpbench/all_knowns_simple_linked.pkl ./
cd ../
python3 tokenize_pkl.py --inp open_kbc_data/mapped_to_ids/relation_id_map.txt --model_str bert-base-uncased
python3 tokenize_pkl.py --inp open_kbc_data/mapped_to_ids/entity_id_map.txt --model_str bert-base-uncased
```
(Instructions to create `all_knowns_simple_linked.pkl` are above)

Step 4: How to test stage1 results using mcqbert code?
```
python run.py --save open_kbc_data/models/dummy --mode test --gpus 1 --epochs 5 --stage2 --negative_samples 10 --data_dir open_kbc_data --model mcq --stage1_model thorough_f5_d300 --model_str bert-base-uncased --task_type both --xt_results --checkpoint path_to_some_ckpt_file --test open_kbc_data/test_data.txt
```

Step 5: How to train stage 2?
```
cd okbc
python run.py --save open_kbc_data/models --mode train_test --gpus 1 --epochs 5 --stage2 --negative_samples 10 --data_dir open_kbc_data --model mcq --stage1_model thorough_f5_d300 --model_str bert-base-uncased --task_type both --max_tokens 5000 --train open_kbc_data/train_data_thorough_1mil.txt
```
Step 6: How to test stage 2?
```
cd okbc
python run.py --save open_kbc_data/models/dummy --mode test --gpus 1 --epochs 5 --stage2 --negative_samples 10 --data_dir open_kbc_data --model mcq --stage1_model thorough_f5_d300 --model_str bert-base-uncased --task_type both --checkpoint open_kbc_data/models/<checkpoint> --test open_kbc_data/test_data.txt
```

## Extend stage 2 to any architecture
This `mcqBERT` architecture can be used on top of any model on any dataset. To do this the stage 1 model must have yielded the following files:(Refer to `extremeText/data/closed_kb/fb15k237` for example)
1. entities.dict - contains id for each entity used by stage 1 model
2. relations.dict - contains id for each relation used by stage 1 model
3. train_data.pkl - a dictionary of the form {
	(entity1_id, relation_id, entity2_id): {
		"head-batch":{
			"index": a list of ids of best k entities for head prediction, 
			"confidence": a list of condifence scores for above k predictions, 
			"bias": a list of known entities for (entity1_id, relation_id) for filtered evaluation,
			"score":{
					"MRR": mean filtered rank of entity2_id for stage1 model, 
					"HITS1": mean filtered hits1 of entity2_id for stage1 model,
					"HITS3": mean filtered hits3 of entity2_id for stage1 model,
					"HITS10": mean filtered hits10 of entity2_id for stage1 model,
				}
		},
		"tail-batch":{
			"index": a list of ids of best k entities for tail prediction, 
			"confidence": a list of condifence scores for above k predictions, 
			"bias": a list of known entities for (entity2_id, relation_id) for filtered evaluation,
			"score":{
					"MRR": mean filtered rank of entity1_id for stage1 model, 
					"HITS1": mean filtered hits1 of entity1_id for stage1 model,
					"HITS3": mean filtered hits3 of entity1_id for stage1 model,
					"HITS10": mean filtered hits10 of entity1_id for stage1 model,
				}
		} 
	},...
}
4. test_data.pkl
5. validation_data.pkl
Next use `okbc/convert_kbc.py` as done in previous section with similar folder structures to convert this into required format for stage 2 training.

## Important config variables
1. `negative_samples` in `okbc/run.py` can be used to change the number of negative samples.
2. `task_type` in `okbc/run.py` can be `tail|head|both` for approriate training or testing.





