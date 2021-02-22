cd ~/KnowledgeGraphEmbedding
MODEL=$1
MODEL_ID=$2
DATA=$3
GPU=$4

# conda activate kge

echo 'Predicting test...'
bash run.sh predict_test $MODEL $DATA $GPU 0 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
echo 'Predicting val...'
bash run.sh predict_val $MODEL $DATA $GPU 0 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
echo 'Predicting train...'
bash run.sh predict_train $MODEL $DATA $GPU 0 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001

# conda deactivate

cd ~/okbc
# python convert_kbc.py --kge_output closed_kbc_data/prachi-complex/data/FB15k-237/test_data.pkl --kbe_data_dir closed_kbc_data/kg-bert --kge_data_dir closed_kbc_data/prachi-complex --dataset FB15k-237  --output_dir closed_kbc_data/data_for_stage2/prachi-complexV2-fb15k237 --output_file test_data.txt --model ComplexV2 --entity_map --relation_map --filter --predictions --scores
python convert_kbc.py --dataset $DATA --kge_output $MODEL'_'$DATA'_'$MODEL_ID/output_test.pkl --output_file test_data.txt --model $MODEL'_'$MODEL_ID --scores --filter --predictions --entity_map --relation_map
python convert_kbc.py --dataset $DATA --kge_output $MODEL'_'$DATA'_'$MODEL_ID/output_valid.pkl --output_file validation_data.txt --model $MODEL'_'$MODEL_ID --scores_val --filter_val --predictions 
python convert_kbc.py --dataset $DATA --kge_output $MODEL'_'$DATA'_'$MODEL_ID/output_train.pkl --output_file train_data.txt --model $MODEL'_'$MODEL_ID --predictions

echo "Checking performance"
python convert_kbc.py --dataset $DATA --kge_output $MODEL'_'$DATA'_'$MODEL_ID/output_test.pkl --output_file test_data.txt --model $MODEL'_'$MODEL_ID --performance
python convert_kbc.py --dataset $DATA --kge_output $MODEL'_'$DATA'_'$MODEL_ID/output_valid.pkl --output_file validation_data.txt --model $MODEL'_'$MODEL_ID --performance
