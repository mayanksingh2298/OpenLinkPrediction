# ---------------------------------------------------------- OPEN KB ----------------------------------------------------------
# Step 1: convert open kb data to xt format
cd extremeText_keshav
cd data
python xt_input.py --inp olpbench/test_data.txt --type head --num_frequent 5
python xt_input.py --inp olpbench/test_data.txt --type tail --num_frequent 5
## repeat the above for all olpbench files like train and validation also
## This generates xt file and an xtp file. xt file is used for training and xtp file is used for interence (parallel)

# Step 2: How to train?
cd extremeText_keshav
./extremetext supervised -input data/olpbench/train_data_thorough.txt.tail.xt -output ~/mayank/xt_modelss/tail_thorough_f5_d300 -lr 0.1 -thread 40 -epoch 50 -dim 300 -loss plt -wordNgrams 2
./extremetext supervised -input data/olpbench/train_data_thorough.txt.head.xt -output ~/mayank/xt_modelss/head_thorough_f5_d300 -lr 0.1 -thread 40 -epoch 50 -dim 300 -loss plt -wordNgrams 2
## the parameters can be changed based on your liking

# Step 3: How to get predictions and output for stage2?
MODEL=thorough_f5_d300
FILE=olpbench/test_data.txt
THREADS=2

cd extremeText_keshav
cd data
../extremetext predict-prob ~/mayank/xt_models/head_$MODEL.bin $FILE.head.xtp 50 0 $FILE.head_$MODEL'.preds' $THREADS
cat $FILE.head_$MODEL'.preds'.0* > $FILE.head_$MODEL'.preds_parallel'
rm $FILE.head_$MODEL'.preds'.0*
mv $FILE.head_$MODEL'.preds_parallel' $FILE.head_$MODEL'.preds'

../extremetext predict-prob ~/mayank/xt_models/tail_$MODEL.bin $FILE.tail.xtp 50 0 $FILE.tail_$MODEL'.preds' $THREADS
cat $FILE.tail_$MODEL'.preds'.0* > $FILE.tail_$MODEL'.preds_parallel'
rm $FILE.tail_$MODEL'.preds'.0*
mv $FILE.tail_$MODEL'.preds_parallel' $FILE.tail_$MODEL'.preds'

python xt_output.py --inp $FILE --model $MODEL --type head
python xt_output.py --inp $FILE --model $MODEL --type tail

# Step 4: How to test stage1 results using mcqbert code?
## move the .stage1 predictions generated above into ~/mayank/open_kbc_data/xt_relfeatures in s2. It already contains the all_knowns pkl file and mapped_data pkl files (latter generated using keshav_okbc/tokenize_pkl.py
python run.py --save closed_kbc_data/models/dummy --mode test --gpus 1 --epochs 5 --stage2 --negative_samples 10 --data_dir ~/mayank/open_kbc_data/xt_relfeatures --model mcq --stage1_model thorough_f5_d300 --model_str bert-base-uncased --task_type both --xt_results --checkpoint closed_kbc_data/models/fb15k-237/cv2_mcq_both_1/epoch=02_loss=0.148_eval_acc=0.267.ckpt --test ~/mayank/open_kbc_data/xt_relfeatures/test_data.txt
## in the above command --checkpoint is not used and stage1 predictions get used

# Step 5: How to train stage 2?
CUDA_VISIBLE_DEVICES=1 python run.py --save ~/mayank/closed_kbc_data/models --mode train_test --gpus 1 --epochs 5 --stage2 --negative_samples 10 --data_dir ~/mayank/open_kbc_data/xt_relfeatures --model mcq --stage1_model thorough_f5_d300 --model_str bert-base-uncased --task_type both --max_tokens 5000 --train ~/mayank/open_kbc_data/xt_relfeatures/train_data_thorough_1mil.txt

# Step 6: How to test stage 2?
python run.py --save closed_kbc_data/models/tmp --mode test --gpus 1 --epochs 5 --stage2 --negative_samples 10 --data_dir ~/mayank/open_kbc_data/xt_relfeatures --model mcq --stage1_model thorough_f5_d300 --model_str bert-base-uncased --task_type both --checkpoint ~/mayank/closed_kbc_data/models/best-base/epoch=01_loss=0.063_eval_acc=0.072.ckpt --test ~/mayank/open_kbc_data/xt_relfeatures/test_data.txt

# ------------------------------------------------------- CLOSED KB ----------------------------------------------------------
## Step 1: How to get input_file
cd extremeText
cd data
cd closed_kb
## make sure you have the dataset folder ready here with (obtain using prachi's code):
## all_knowns_head.pkl  entities.dict    relations.dict     test.txt
## all_knowns_tail.pkl  entity2text.txt  relation2text.txt  train.txt
python xt_input.py --inp fb15k237/valid.txt --out fb15k237/valid --type tail --relation2text_file fb15k237/relation2text.txt --entity2text_file fb15k237/entity2text.txt --entity_id_file fb15k237/entities.dict

# Step 2: How to train?
cd extremeText
./extremetext supervised -input data/closed_kb/fb15k237/train.tail.xt -output data/closed_kb/fb15k237/tail_300d -lr 0.1 -thread 40 -epoch 50 -dim 300 -loss plt -wordNgrams 2
./extremetext supervised -input data/closed_kb/fb15k237/train.head.xt -output data/closed_kb/fb15k237/head_300d -lr 0.1 -thread 40 -epoch 50 -dim 300 -loss plt -wordNgrams 2

# Step 3: How to get predictions?
./extremetext predict-prob data/closed_kb/fb15k237/head_300d.bin data/closed_kb/fb15k237/test.head.xt 50 0 data/closed_kb/fb15k237/test.head.preds.txt 1 > data/closed_kb/fb15k237/test.head.preds.txt
./extremetext predict-prob data/closed_kb/fb15k237/tail_300d.bin data/closed_kb/fb15k237/test.tail.xt 50 0 data/closed_kb/fb15k237/test.tail.preds.txt 1 > data/closed_kb/fb15k237/test.tail.preds.txt

# Step 4: How to test stage1 results and generate the pickle file for stage2?
python3 test_filtered_metrics.py --dataset fb15k237 --mode train
## This should generate a pickle file which should be given as an input to Keshav's convert_kbc.py (instructions below)
python convert_kbc.py --kge_output ~/mayank/closed_kbc_data/xt/data/WN18RR/test_data.pkl --kbe_data_dir ~/mayank/closed_kbc_data/kg-bert --kge_data_dir ~/mayank/closed_kbc_data/xt/ --dataset WN18RR --output_dir ~/mayank/closed_kbc_data/data_for_stage2/xt-wn18rr --output_file test_data.txt --model Xt --entity_map --relation_map --filter --predictions --scores
python convert_kbc.py --kge_output ~/mayank/closed_kbc_data/xt/data/WN18RR/validation_data.pkl --kbe_data_dir ~/mayank/closed_kbc_data/kg-bert --kge_data_dir ~/mayank/closed_kbc_data/xt/ --dataset WN18RR --output_dir ~/mayank/closed_kbc_data/data_for_stage2/xt-wn18rr --output_file validation_data.txt --model Xt --filter_val --predictions --scores_val
python convert_kbc.py --kge_output ~/mayank/closed_kbc_data/xt/data/WN18RR/train_data.pkl --kbe_data_dir ~/mayank/closed_kbc_data/kg-bert --kge_data_dir ~/mayank/closed_kbc_data/xt/ --dataset WN18RR --output_dir ~/mayank/closed_kbc_data/data_for_stage2/xt-wn18rr --output_file train_data.txt --model Xt --predictions
python3 tokenize_pkl.py --inp ~/mayank/closed_kbc_data/data_for_stage2/xt-wn18rr/mapped_to_ids/entity_id_map.txt --model_str bert-base-cased
python3 tokenize_pkl.py --inp ~/mayank/closed_kbc_data/data_for_stage2/xt-wn18rr/mapped_to_ids/relation_id_map.txt --model_str bert-base-cased


# Step 5: How to train stage2?
python run.py --save ~/mayank/closed_kbc_data/models --mode train_test --gpus 1 --epochs 10 --stage2 --negative_samples 10 --data_dir ~/mayank/closed_kbc_data/data_for_stage2/xt-fb15k237 --model mcq --stage1_model Xt --model_str bert-base-cased --task_type both --max_tokens 5000 --ckbc

# Step 6: Dummy test stage 1 again?
python run.py --save closed_kbc_data/models/dummy --mode test --gpus 1 --epochs 5 --stage2 --negative_samples 10 --data_dir ~/mayank/closed_kbc_data/data_for_stage2/xt-wn18rr --model mcq --stage1_model Xt --model_str bert-base-cased --task_type both --ckbc --xt_results --checkpoint closed_kbc_data/models/fb15k-237/cv2_mcq_both_1/epoch=02_loss=0.148_eval_acc=0.267.ckpt --test ~/mayank/closed_kbc_data/data_for_stage2/xt-wn18rr/test_data.txt

# Step 7: How to test stage2?
python run.py --save closed_kbc_data/models/tmp --mode test --gpus 1 --epochs 5 --stage2 --negative_samples 10 --data_dir ~/mayank/closed_kbc_data/data_for_stage2/xt-fb15k237/ --model mcq --stage1_model Xt --model_str bert-base-cased --task_type both --checkpoint ~/mayank/closed_kbc_data/models/epoch=00_loss=0.205_eval_acc=0.195.ckpt_OR_ACTUAL_CHECKPOINT --test ~/mayank/closed_kbc_data/data_for_stage2/xt-fb15k237/test_data.txt





