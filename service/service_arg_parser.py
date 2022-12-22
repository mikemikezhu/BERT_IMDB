import argparse
from utils.constants import *


class ArgumentParserService:

    """ Initialize """

    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._init_parser(self._parser)

    """ Public methods """

    def get_parser(self):
        return self._parser

    """ Private methods """

    def _init_parser(self, parser):

        parser.add_argument(
            '--pretrain',
            type=str,
            default=IN_DOMAIN_PRETRAIN,
            choices=[
                IN_DOMAIN_PRETRAIN,
                OUT_DOMAIN_PRETRAIN
            ]
        )

        parser.add_argument(
            '--bert_model',
            type=str,
            default=BERT_LARGE,
            choices=[
                BERT_LARGE,
                BERT_BASE
            ]
        )

        parser.add_argument('--in_domain_pretrain_dir',
                            type=str, default=IN_DOMAIN_PRETRAIN_DIR)
        parser.add_argument('--out_domain_pretrain_model',
                            type=str, default=OUT_DOMAIN_PRETRAIN_MODEL)

        parser.add_argument('--learning_rate', type=float, default=BERT_LR)
        parser.add_argument('--weight_decay', type=float,
                            default=BERT_WEIGHT_DECAY)
        parser.add_argument('--num_epochs', type=int, default=BERT_EPOCHS)

        parser.add_argument('--val_data_length', type=int,
                            default=VAL_DATA_LENGTH)

        parser.add_argument('--train_neg_data_path',
                            type=str, default=TRAIN_NEG_DATA_PATH)
        parser.add_argument('--train_pos_data_path',
                            type=str, default=TRAIN_POS_DATA_PATH)
        parser.add_argument('--test_neg_data_path',
                            type=str, default=TEST_NEG_DATA_PATH)
        parser.add_argument('--test_pos_data_path',
                            type=str, default=TEST_POS_DATA_PATH)

        parser.add_argument('--train_batch_size', type=int,
                            default=TRAIN_BATCH_SIZE)
        parser.add_argument('--val_batch_size', type=int,
                            default=VAL_BATCH_SIZE)
        parser.add_argument('--test_batch_size', type=int,
                            default=TEST_BATCH_SIZE)
