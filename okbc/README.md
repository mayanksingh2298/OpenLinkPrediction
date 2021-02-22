# Fasttext Experiments

Preparing environment:
```
pip install -r requirements.txt
```

Requires pytorch 1.6 which inturn needs CUDA 10+

Code for running:

BERT model
```
python run.py --save models/dummy  --mode train_test --gpus 1 --max_tokens 1000 --tokenize bert --bize bert --bert --optimizer adamW --lr 2e-5 --epochs 100 --train data/train_small.txt 
```

Fasttext model
```
python run.py --save models/dummy  --mode train_test --gpus 1 --max_tokens 
1000 --ft --optimizer sgd --lr 0.01 --epochs 100 --debug
```


