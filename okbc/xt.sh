MODEL=$1
FILE=$2
THREADS=$3 # Use THREADS of 2 for validation/test

python xt_input.py --inp $FILE --type head
python xt_input.py --inp $FILE --type tail

$HOME/extremeText/extremetext predict-prob $HOME/extremeText/models/head_$MODEL.bin $FILE.head.xtp 50 0 $FILE.head_$MODEL'.preds' $THREADS
cat $FILE.head_$MODEL'.preds'.0* > $FILE.head_$MODEL'.preds_parallel'
rm $FILE.head_$MODEL'.preds'.0*
mv $FILE.head_$MODEL'.preds_parallel' $FILE.head_$MODEL'.preds'

$HOME/extremeText/extremetext predict-prob $HOME/extremeText/models/tail_$MODEL.bin $FILE.tail.xtp 50 0 $FILE.tail_$MODEL'.preds' $THREADS
cat $FILE.tail_$MODEL'.preds'.0* > $FILE.tail_$MODEL'.preds_parallel'
rm $FILE.tail_$MODEL'.preds'.0*
mv $FILE.tail_$MODEL'.preds_parallel' $FILE.tail_$MODEL'.preds'

python xt_output.py --inp $FILE --model $MODEL --type head
python xt_output.py --inp $FILE --model $MODEL --type tail