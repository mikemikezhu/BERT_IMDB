#!/bin/sh
rm -r transformers
git clone https://github.com/huggingface/transformers.git
pip install -q ./transformers
pip install datasets evaluate

# Create pretrained directory
PRETRAINED_DIR="pretrained"
if [[ ! -e $PRETRAINED_DIR ]]; then
    mkdir $PRETRAINED_DIR
fi

python3 main_pretrained.py \
    --model_name_or_path bert-large-uncased \
    --dataset_name imdb \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir pretrained