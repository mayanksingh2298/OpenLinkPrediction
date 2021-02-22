# Train
./extremetext supervised -input ~/olpbench/train_data_thorough.txt.tail.frequent -output models/thorough_tail_f5_d300 -lr 0.1 -thread 40 -epoch 50 -dim 300 -loss plt -wordNgrams 2

# Test
./extremetext test models/thorough_tail_f5_d300.bin ~/olpbench/test_data.txt.tail.frequent.xt 1

# Generate xt_input data
python xt_input.py --inp olpbench/train_data_thorough.txt --type head --num_frequent 5
