#!/bin/sh
rm -r transformers
git clone https://github.com/huggingface/transformers.git
pip install -q ./transformers
pip install datasets evaluate