import numpy as np
import os
import re
import random

from torch.utils.data import DataLoader
from dataset.imdb_dataset import ImdbDataset
from utils.utils_log import LogUtils
from utils.constants import *


class DataLoaderService:

    """ Initialize """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    """ Public methods """

    def load_data(self, load_data_query):

        # Process BERT input
        train_neg_x_path = load_data_query.train_neg_x_path
        train_pos_x_path = load_data_query.train_pos_x_path
        test_neg_x_path = load_data_query.test_neg_x_path
        test_pos_x_path = load_data_query.test_pos_x_path

        train_neg_x = self._process_bert_input(train_neg_x_path)
        train_pos_x = self._process_bert_input(train_pos_x_path)
        test_neg_x = self._process_bert_input(test_neg_x_path)
        test_pos_x = self._process_bert_input(test_pos_x_path)

        assert train_neg_x is not None
        assert train_pos_x is not None
        assert test_neg_x is not None
        assert test_pos_x is not None

        # Generate BERT labels
        train_neg_y = np.zeros(len(train_neg_x))
        train_pos_y = np.ones(len(train_pos_x))
        test_neg_y = np.zeros(len(test_neg_x))
        test_pos_y = np.ones(len(test_pos_x))

        # Create train and test set
        bert_train_imdb_x = train_neg_x + train_pos_x
        bert_train_imdb_y = np.concatenate([train_neg_y, train_pos_y])
        bert_test_imdb_x = test_neg_x + test_pos_x
        bert_test_imdb_y = np.concatenate([test_neg_y, test_pos_y])

        # Shuffle dataset
        bert_train_imdb_combined = list(
            zip(bert_train_imdb_x, bert_train_imdb_y))
        random.shuffle(bert_train_imdb_combined)
        bert_train_imdb_x, bert_train_imdb_y = zip(*bert_train_imdb_combined)

        bert_test_imdb_combined = list(zip(bert_test_imdb_x, bert_test_imdb_y))
        random.shuffle(bert_test_imdb_combined)
        bert_test_imdb_x, bert_test_imdb_y = zip(*bert_test_imdb_combined)

        # Create validation set
        val_data_length = load_data_query.val_data_length
        bert_train_imdb_x = bert_train_imdb_x[val_data_length:]
        bert_val_imdb_x = bert_train_imdb_x[:val_data_length]
        bert_train_imdb_y = bert_train_imdb_y[val_data_length:]
        bert_val_imdb_y = bert_train_imdb_y[:val_data_length]

        bert_train_imdb_y = np.asarray(bert_train_imdb_y, dtype="float32")
        bert_val_imdb_y = np.asarray(bert_val_imdb_y, dtype="float32")
        bert_test_imdb_y = np.asarray(bert_test_imdb_y, dtype="float32")

        LogUtils.instance().log_info("bert_train_imdb_x: {}".format(len(bert_train_imdb_x)))
        LogUtils.instance().log_info("bert_val_imdb_x: {}".format(len(bert_val_imdb_x)))
        LogUtils.instance().log_info("bert_test_imdb_x: {}".format(len(bert_test_imdb_x)))
        LogUtils.instance().log_info("bert_train_imdb_y: {}".format(bert_train_imdb_y.shape))
        LogUtils.instance().log_info("bert_val_imdb_y: {}".format(bert_val_imdb_y.shape))
        LogUtils.instance().log_info("bert_test_imdb_y: {}".format(bert_test_imdb_y.shape))

        # Create data loader
        bert_train_dataset = ImdbDataset(bert_train_imdb_x, bert_train_imdb_y)
        bert_val_dataset = ImdbDataset(bert_val_imdb_x, bert_val_imdb_y)
        bert_test_dataset = ImdbDataset(bert_test_imdb_x, bert_test_imdb_y)

        train_batch_size = load_data_query.train_batch_size
        val_batch_size = load_data_query.val_batch_size
        test_batch_size = load_data_query.test_batch_size

        bert_train_data_loader = DataLoader(bert_train_dataset,
                                            batch_size=train_batch_size)
        bert_val_data_loader = DataLoader(bert_val_dataset,
                                          batch_size=val_batch_size)
        bert_test_data_loader = DataLoader(bert_test_dataset,
                                           batch_size=test_batch_size)

        return bert_train_data_loader, bert_val_data_loader, bert_test_data_loader

    """ Private methods """

    def _truncate(self, tokens):

        if tokens is None or len(tokens) == 0:
            return None

        if len(tokens) > 510:
            # Truncate long texts by selecting the first 128 and the last 382 tokens
            tokens = tokens[:128] + tokens[-382:]

        return " ".join(tokens)

    def _process_bert_input(self, path):

        result = None

        all_imdb_files = os.listdir(path)
        if all_imdb_files is None or len(all_imdb_files) == 0:
            return result

        LogUtils.instance().log_info("Process BERT input: {}".format(path))

        result = []
        for imdb_file in all_imdb_files:

            if not imdb_file.endswith("txt"):
                continue

            with open(path + "/" + imdb_file) as f:
                lines = f.readlines()[0]
                lines = lines.replace("<br />", "")
                tokens = re.findall(r"[\w']+|[.,!?;]", lines)
                lines = self._truncate(tokens)  # Truncate
                bert_input = self._tokenizer(lines,  # Bert tokenize
                                             padding='max_length',
                                             max_length=512,
                                             truncation=True,
                                             return_tensors="pt")
                result.append(bert_input)

        return result
