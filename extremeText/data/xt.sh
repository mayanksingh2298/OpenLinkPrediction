MODEL=$1
FILE=$2
THREADS=$3 # Use THREADS of 2 for validation/test

python xt_input.py --inp $FILE --type head
python xt_input.py --inp $FILE --type tail



../extremetext predict-prob models/head_$MODEL.bin $FILE.head.xtp 50 0 $FILE.head_$MODEL'.preds' $THREADS
cat $FILE.head_$MODEL'.preds'.0* > $FILE.head_$MODEL'.preds_parallel'
rm $FILE.head_$MODEL'.preds'.0*
mv $FILE.head_$MODEL'.preds_parallel' $FILE.head_$MODEL'.preds'

../extremetext predict-prob models/tail_$MODEL.bin $FILE.tail.xtp 50 0 $FILE.tail_$MODEL'.preds' $THREADS
cat $FILE.tail_$MODEL'.preds'.0* > $FILE.tail_$MODEL'.preds_parallel'
rm $FILE.tail_$MODEL'.preds'.0*
mv $FILE.tail_$MODEL'.preds_parallel' $FILE.tail_$MODEL'.preds'

python xt_output.py --inp $FILE --model $MODEL --type head
python xt_output.py --inp $FILE --model $MODEL --type tail

# Train XT: ./extremetext supervised -input data/olpbench/train_data_thorough.txt.head.xt -output data/models/head_thorough_f5_d300 -lr 0.1 -thread 40 -epoch 50 -dim 300 -loss plt -wordNgrams 2 

# Command to test this: python run.py --save closed_kbc_data/models/dummy --mode test --gpus 1 --epochs 5 --stage2 --negative_samples 10 --data_dir ~/mayank/open_kbc_data/xt_relfeatures --model mcq --stage1_model thorough_f5_d300 --model_str bert-base-cased --task_type tail --xt_results --checkpoint closed_kbc_data/models/fb15k-237/cv2_mcq_both_1/epoch=02_loss=0.148_eval_acc=0.267.ckpt --test ~/mayank/open_kbc_data/xt_relfeatures/test_data.txt

