# BERT on IMDB Sentiment Analysis

## Setup

### Download Dataset

```console
sh setup.sh
```

### Install Libraries

```console
pip install -r requirements.txt
```

### Pretrain using Masked-Language Modelling (MLM)

```console
sh run_pretrained.sh
```

## Run Experiments

### Experiment 1: Download pretrained model, then fine tune the entire network

```console
python3 main.py \
    --bert_model=large \
    --pretrain=out_domain_pretrain \
    --out_domain_pretrain_model=bert-large-uncased \
    --learning_rate=2e-5 \
    --weight_decay=0.05 \
    --num_epochs=3 \
    --train_batch_size=16 \
    --val_batch_size=16 \
    --test_batch_size=16
```

### Experiment 2: Pretrain on IMDB, then fine tune the entire network

Based on the paper [How to Fine-Tune BERT for Text Classiﬁcation?](https://arxiv.org/abs/1905.05583), in-domain pretraining on IMDB dataset achieves better performance.

```console
python3 main.py \
    --bert_model=large \
    --pretrain=in_domain_pretrain \
    --in_domain_pretrain_dir=pretrained/checkpoint-2500 \
    --learning_rate=2e-5 \
    --weight_decay=0.05 \
    --num_epochs=3 \
    --train_batch_size=16 \
    --val_batch_size=16 \
    --test_batch_size=16
```

### Experiment 3: Pretrain on IMDB, then partial freezing

Based on the paper [What Happens To BERT Embeddings During Fine-tuning?](https://arxiv.org/pdf/2004.14448.pdf), we can fine tune the model by partial freezing: keeping the early BERT layers fixed during the fine tuning process, and measure how much the performance on the downstream task changes when varying the number of frozen layers. They show that the performance on both MNLI and SQuAD tasks does not notably drop even when freezing the first 8 of the 12 BERT layers (i.e. tuning only the last 4).

```console
python3 main.py \
    --bert_model=large \
    --pretrain=in_domain_pretrain \
    --in_domain_pretrain_dir=pretrained/checkpoint-2500 \
    --freeze_layers=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
    --learning_rate=2e-5 \
    --weight_decay=0.05 \
    --num_epochs=3 \
    --train_batch_size=16 \
    --val_batch_size=16 \
    --test_batch_size=16
```