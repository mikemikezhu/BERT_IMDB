# Constants
BERT_LR = 2e-5
BERT_WEIGHT_DECAY = 0.05
BERT_EPOCHS = 3

VAL_DATA_LENGTH = 5000

TRAIN_NEG_DATA_PATH = "data/aclImdb/train/neg"
TRAIN_POS_DATA_PATH = "data/aclImdb/train/pos"
TEST_NEG_DATA_PATH = "data/aclImdb/test/neg"
TEST_POS_DATA_PATH = "data/aclImdb/test/pos"

TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16

# BERT
BERT_LARGE = "large"
BERT_BASE = "base"

BERT_OUTPUT = {
    BERT_LARGE: 1024,
    BERT_BASE: 768
}

# Pretrain
IN_DOMAIN_PRETRAIN = "in_domain_pretrain"
IN_DOMAIN_PRETRAIN_DIR = "pretrained/checkpoint-2500"
OUT_DOMAIN_PRETRAIN = "out_domain_pretrain"
OUT_DOMAIN_PRETRAIN_MODEL = "bert-large-uncased"

# Log
LOG_PATH = "log/"
LOGGER_DEFAULT = "logger_default"
