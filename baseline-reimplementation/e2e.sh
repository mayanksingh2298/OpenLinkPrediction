# why are there so many main scripts?
# main-baseline.py - my first implementation of authors'
# main-baseline-paper.py - when authors' released their code I replicated what they did by taking 2 lstms
# main-baseline-author_data.py - the train loader loads data from authors' pickle files. REALLY HELP achieve author performance
# main-baseline-author_data-all_e.py - atomic entity embedding for entity when used as a target otherwise get from LSTM
# main-baseline-author_data-all_e_fr.py - atomic entity embedding for both head and tail. Relation get from LSTM
# main-baseline-author_data-pure_complex.py - pure complex. no LSTM

# eval
python bert_multiple_choice_swag.py --data_dir data/swag --bert_model bert-base-uncased --output_dir models --do_eval --eval_batch_size 256

#train
CUDA_VISIBLE_DEVICES=1 python bert_multiple_choice_swag.py --data_dir data/swag --bert_model bert-base-uncased --output_dir models --do_eval --eval_batch_size 256 --do_train --train_batch_size 12
#######################################################################
#BERT OLPBENCH
# Generate all knowns train-thorough valid-linked 
python3 helper_scripts/generate_all_knowns.py --data_dir data/olpbench --train thorough --val linked --output_dir data/olpbench/all_knowns_thorough_linked.pkl

# Generate entity mentions embeddings
python3 helper_scripts/generate_entity_mention_embeddings.py --data_dir data/olpbench --eval_batch_size 256 --output_dir data/olpbench/pretrained_bert_entity_mentions.pkl --max_seq_length 45

# Generate 1 hop evidence for test set
python3 helper_scripts/generate_1_hop_evidence.py --data_dir data/olpbench --output_file test_evidence.txt --n_times 1

# sample 100 random elements from train set
python3 helper_scripts/sample_data.py --file data/olpbench/train_data_thorough.txt --sample 100

# sample 100 hits1 and nothits50
CUDA_VISIBLE_DEVICES=1 python3 helper_scripts/sample_hits1_nothits50.py --data_dir data/olpbench --eval_batch_size 512 --max_seq_length 15 --embedding_dim 512 --resume models/512dim_baseline_head_tail/checkpoint_epoch_10 >output.txt
# sample 100 hits1 along with their injected evidence and not selecting those hits1 from baseline
CUDA_VISIBLE_DEVICES=1 python3 helper_scripts/sample_hits1_evidence.py --data_dir data/olpbench --eval_batch_size 512 --max_seq_length 15 --embedding_dim 256 --resume models/standard_baseline_testevidence_1hop_1/checkpoint_epoch_10 --n_times 1 --evidence_file data/olpbench/inject_test_evidence/test_evidence.txt > output.txt

# Train word2vec embeddings using train_simple.txt
python3 helper_scripts/create_word2vec_embed.py --data_dir data/olpbench --output_file w2v_embeddings_256 --workers 45 --num_train_epochs 10

# cosine similarity check with injected evidence
python3 helper_scripts/cosine_simi_rels.py --resume_for_simi models/first_try_baseline_head_tail/checkpoint_epoch_10 --resume models/standard_baseline_testevidence_1hop_1/checkpoint_epoch_10  --data_dir data/olpbench --test_file data/olpbench/test_data.txt --evidence_file data/olpbench/inject_test_evidence/test_evidence.txt --n_times 1 --do_eval

# Eval
python3 main.py --data_dir data/olpbench --bert_model bert-base-uncased --debug --do_eval --eval_batch_size 256 --max_seq_length 45

# Train
CUDA_VISIBLE_DEVICES=1 python3 main.py --data_dir data/olpbench --bert_model bert-base-uncased --do_train --train_batch_size 64 --num_train_epochs 10 --max_seq_length 45 --skip_train_prob 0.9 --output_dir models/first_try 

# Train and eval
saved_model=models/first_try/checkpoint_epoch_10
output_embeddings=data/olpbench/e2_embeddings/checkpoint_epoch_10.pt
CUDA_VISIBLE_DEVICES=1 python3 helper_scripts/generate_entity_mention_embeddings.py --data_dir data/olpbench --eval_batch_size 256 --output_dir $output_embeddings --max_seq_length 45 --resume $saved_model &&\
CUDA_VISIBLE_DEVICES=1 python3 main.py --data_dir data/olpbench --bert_model bert-base-uncased --debug --do_eval --eval_batch_size 256 --max_seq_length 45 --resume $saved_model --e2_embeddings_for_eval $output_embeddings 

#######################################################################
# Baseline
# Train
CUDA_VISIBLE_DEVICES=5 python3 main-baseline.py --data_dir data/olpbench --do_train --train_batch_size 1024 --num_train_epochs 10 --max_seq_length 15 --output_dir models/standard_baseline_testevidence_2hop_10 --learning_rate 0.1 --embedding_dim 256 --weight_decay 1e-6 --skip_train_prob 0.0

# Train 2 lstms one for entity - one for relation - with 512 dimension
CUDA_VISIBLE_DEVICES=5 python3 main-baseline-2-lstms.py --data_dir data/olpbench --do_train --train_batch_size 512 --num_train_epochs 10 --max_seq_length 15 --output_dir models/2lstms_512dim_baseline_head_tail --learning_rate 0.1 --embedding_dim 512 --weight_decay 1e-6 --skip_train_prob 0.0

#Train with glove embedding sive 300
CUDA_VISIBLE_DEVICES=2 python3 main-baseline.py --data_dir data/olpbench --do_train --train_batch_size 1024 --num_train_epochs 10 --max_seq_length 15 --output_dir models/first_try_baseline --learning_rate 0.1 --embedding_dim 300 --weight_decay 1e-6 --skip_train_prob 0.0 --initial_token_embedding data/glove/glove.6B.300d.txt

# eval
CUDA_VISIBLE_DEVICES=1 python3 main-baseline.py --data_dir data/olpbench --do_eval --eval_batch_size 512 --max_seq_length 15 --debug --embedding_dim 256 --resume models/glove_first_try_baseline/checkpoint_epoch_28

# train fb15k237
CUDA_VISIBLE_DEVICES=2 python3 main-baseline.py --data_dir data/fb15k237 --do_train --train_batch_size 512 --num_train_epochs 400 --max_seq_length 21 --output_dir models/fb15k237_0.1_512 --learning_rate 0.1 --embedding_dim 512 --weight_decay 1e-6 --skip_train_prob 0.0 --print_loss_every 200 --save_model_every 10

# eval fb15k237
CUDA_VISIBLE_DEVICES=1 python3 main-baseline.py --data_dir data/fb15k237 --do_eval --eval_batch_size 512 --max_seq_length 21 --debug --embedding_dim 512 --resume models/fb15k237_0.1_512/checkpoint_epoch_11


# train rotatE
CUDA_VISIBLE_DEVICES=2 python3 main-baseline.py --model rotate --gamma_rotate 0 --data_dir data/olpbench --do_train --train_batch_size 512 --num_train_epochs 50 --max_seq_length 15 --output_dir models/rotatE_512_0.1 --learning_rate 0.1 --embedding_dim 512 --weight_decay 1e-6 --skip_train_prob 0.0

# eval rotatE
CUDA_VISIBLE_DEVICES=1 python3 main-baseline.py --model rotate --gamma_rotate 0 --data_dir data/olpbench --do_eval --eval_batch_size 512 --max_seq_length 15 --debug --embedding_dim 512 --resume models/rotatE_512_0.1/checkpoint_epoch_10

# paper single LSTM
CUDA_VISIBLE_DEVICES=5 python3 main-baseline-paper.py --data_dir data/olpbench --do_train --train_batch_size 512 --num_train_epochs 101 --max_seq_length 15 --output_dir models/paper_baseline_simple_2lstm --learning_rate 0.2 --embedding_dim 512 --weight_decay 1e-10 --skip_train_prob 0.0 --lstm_dropout 0.1 --save_model_every 10 --separate_lstms


# bert learn masked language model
CUDA_VISIBLE_DEVICES=1 python3 run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/olpbench/train_data_thorough.txt \
    --validation_file data/olpbench/test_data.txt \
    --do_train \
    --do_eval \
    --output_dir mlm_models/train_thorough_1epoch --logging_dir mlm_models/train_thorough_1epoch --line_by_line --overwrite_output_dir --eval_accumulation_steps 50 --per_device_train_batch_size 128 --num_train_epochs 1 --save_steps 50000 --eval_steps 10000 --logging_steps 500 --mlm_probability 0.15
