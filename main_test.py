from model.classifier import BertClassifier
from model.param_load_data import LoadDataParam
from model.param_plot import PlotParam
from model.param_train_test import TrainTestParam

from service.service_arg_parser import ArgumentParserService
from service.service_data_loader import DataLoaderService
from service.service_bert import BertService
from service.service_plot import PlotService

from utils.utils_log import LogUtils
from utils.utils_pid import PidUtils
from utils.constants import *

from transformers import BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import os
import numpy as np
import requests
import pandas as pd
import string


def main():

    if not os.path.exists(os.path.join(os.getcwd(), "data")):
        raise FileNotFoundError(
            "Data directory not found. Please run setup.sh first!")

    if not os.path.exists(os.path.join(os.getcwd(), "log")):
        raise FileNotFoundError(
            "Log directory not found. Please run setup.sh first!")

    if not os.path.exists(os.path.join(os.getcwd(), "output")):
        raise FileNotFoundError(
            "Output directory not found. Please run setup.sh first!")

    pid = PidUtils.instance().get_pid()
    LogUtils.instance().log_info("PID: {}".format(pid))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    LogUtils.instance().log_info("Device: {}".format(device))

    # Parse arguments
    parser_service = ArgumentParserService()
    parser = parser_service.get_parser()
    flags = parser.parse_args()

    if flags.best_model_path is None or not os.path.exists(os.path.join(os.getcwd(), flags.best_model_path)):
        raise FileNotFoundError(
            "Please specify valid model path!")

    for k, v in sorted(vars(flags).items()):
        LogUtils.instance().log_info("\t{}: {}".format(k, v))

    # Pretrained model 
    # Note: we can use default BERT pretrained model
    # because we will initialize model with best model weights
    if flags.bert_model == BERT_LARGE:
        pretrained_model_name_or_path = "bert-large-uncased"
    elif flags.bert_model == BERT_BASE:
        pretrained_model_name_or_path = "bert-base-uncased"
    else:
        pretrained_model_name_or_path = flags.out_domain_pretrain_model

    LogUtils.instance().log_info(
        "Pretrained model name or path: {}".format(pretrained_model_name_or_path))

    pretrained_bert = BertModel.from_pretrained(pretrained_model_name_or_path,
                                                output_attentions=True)
    model = BertClassifier(pretrained_bert=pretrained_bert,
                           bert_model=flags.bert_model)
    model.to(device)

    # Load data
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
    data_loader = DataLoaderService(tokenizer)
    load_data_param = LoadDataParam(flags.val_data_length,
                                    flags.train_neg_data_path,
                                    flags.train_pos_data_path,
                                    flags.test_neg_data_path,
                                    flags.test_pos_data_path,
                                    flags.train_batch_size,
                                    flags.val_batch_size,
                                    flags.test_batch_size)
    _, _, test_data_loader = data_loader.load_data(load_data_param,
                                                   test_only=True)
    
    # Test BERT
    bert_service = BertService()
    criterion = nn.BCELoss()
    best_model = BertClassifier(pretrained_bert=pretrained_bert,
                                bert_model=flags.bert_model)
    best_model = best_model.to(device)
    best_model.load_state_dict(torch.load(flags.best_model_path))

    test_param = TrainTestParam()
    test_param.model = best_model
    test_param.criterion = criterion
    test_param.test_data_loader = test_data_loader
    test_param.device = device

    test_result = bert_service.test_bert(test_param)

    pd.DataFrame(test_result.y_preds).to_csv(
        "output/PID: {} - bert_y_preds.csv".format(pid))
    pd.DataFrame(test_result.y_test).to_csv(
        "output/PID: {} - bert_y_test.csv".format(pid))

    # Plot attention matrix
    stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
    stopwords = set(stopwords_list.decode().splitlines())
    stopwords.update([*string.punctuation])

    stopwords_ids = tokenizer.convert_tokens_to_ids(stopwords)
    plot_service = PlotService()
    tp_attention = test_result.tp_attention
    if tp_attention is not None:
        plot_param = PlotParam()
        input = (tp_attention.input[0]).cpu().detach().numpy().copy()
        attention = (tp_attention.attention[0]).cpu().detach().numpy().copy()

        non_stop_words_pos = [pos for pos in range(len(input)) if input[pos] not in stopwords_ids]
        non_stop_words_pos = np.asarray(non_stop_words_pos)
        filtered_attention = attention[non_stop_words_pos, :]
        filtered_attention = filtered_attention[:, non_stop_words_pos]

        attention_sum = np.sum(filtered_attention, axis=0)
        attention_sort_index = np.argsort(attention_sum)[::-1]
        top_word_index = attention_sort_index[:20]
        top_word_index = np.sort(top_word_index)
        top_word_index = np.insert(top_word_index, 0, 0)

        filtered_attention = filtered_attention[top_word_index, :]
        filtered_attention = filtered_attention[:, top_word_index]

        input = input[non_stop_words_pos]
        input = input[top_word_index]
        plot_param.tokens = tokenizer.convert_ids_to_tokens(input)
        plot_param.attention = filtered_attention
        plot_param.plot_title = "True positive attention"
        plot_param.file_name = "output/PID: {} - bert_tp_attention.png".format(
            pid)
        plot_service.plot_attention(plot_param)
    else:
        LogUtils.instance().log_warning("No true positive attention")

    tn_attention = test_result.tn_attention
    if tn_attention is not None:
        plot_param = PlotParam()
        input = (tn_attention.input[0]).cpu().detach().numpy().copy()
        attention = (tn_attention.attention[0]).cpu().detach().numpy().copy()

        non_stop_words_pos = [pos for pos in range(len(input)) if input[pos] not in stopwords_ids]
        non_stop_words_pos = np.asarray(non_stop_words_pos)
        filtered_attention = attention[non_stop_words_pos, :]
        filtered_attention = filtered_attention[:, non_stop_words_pos]

        attention_sum = np.sum(filtered_attention, axis=0)
        attention_sort_index = np.argsort(attention_sum)[::-1]
        top_word_index = attention_sort_index[:20]
        top_word_index = np.sort(top_word_index)
        top_word_index = np.insert(top_word_index, 0, 0)

        filtered_attention = filtered_attention[top_word_index, :]
        filtered_attention = filtered_attention[:, top_word_index]

        input = input[non_stop_words_pos]
        input = input[top_word_index]
        plot_param.tokens = tokenizer.convert_ids_to_tokens(input)
        plot_param.attention = filtered_attention
        plot_param.plot_title = "True negative attention"
        plot_param.file_name = "output/PID: {} - bert_tn_attention.png".format(
            pid)
        plot_service.plot_attention(plot_param)
    else:
        LogUtils.instance().log_warning("No true negative attention")

    fp_attention = test_result.fp_attention
    if fp_attention is not None:
        plot_param = PlotParam()
        input = (fp_attention.input[0]).cpu().detach().numpy().copy()
        attention = (fp_attention.attention[0]).cpu().detach().numpy().copy()

        non_stop_words_pos = [pos for pos in range(len(input)) if input[pos] not in stopwords_ids]
        non_stop_words_pos = np.asarray(non_stop_words_pos)
        filtered_attention = attention[non_stop_words_pos, :]
        filtered_attention = filtered_attention[:, non_stop_words_pos]

        attention_sum = np.sum(filtered_attention, axis=0)
        attention_sort_index = np.argsort(attention_sum)[::-1]
        top_word_index = attention_sort_index[:20]
        top_word_index = np.sort(top_word_index)
        top_word_index = np.insert(top_word_index, 0, 0)

        filtered_attention = filtered_attention[top_word_index, :]
        filtered_attention = filtered_attention[:, top_word_index]

        input = input[non_stop_words_pos]
        input = input[top_word_index]
        plot_param.tokens = tokenizer.convert_ids_to_tokens(input)
        plot_param.attention = filtered_attention
        plot_param.plot_title = "False positive attention"
        plot_param.file_name = "output/PID: {} - bert_fp_attention.png".format(
            pid)
        plot_service.plot_attention(plot_param)
    else:
        LogUtils.instance().log_warning("No false positive attention")

    fn_attention = test_result.fn_attention
    if fn_attention is not None:
        plot_param = PlotParam()
        input = (fn_attention.input[0]).cpu().detach().numpy().copy()
        attention = (fn_attention.attention[0]).cpu().detach().numpy().copy()

        non_stop_words_pos = [pos for pos in range(len(input)) if input[pos] not in stopwords_ids]
        non_stop_words_pos = np.asarray(non_stop_words_pos)
        filtered_attention = attention[non_stop_words_pos, :]
        filtered_attention = filtered_attention[:, non_stop_words_pos]

        attention_sum = np.sum(filtered_attention, axis=0)
        attention_sort_index = np.argsort(attention_sum)[::-1]
        top_word_index = attention_sort_index[:20]
        top_word_index = np.sort(top_word_index)
        top_word_index = np.insert(top_word_index, 0, 0)

        filtered_attention = filtered_attention[top_word_index, :]
        filtered_attention = filtered_attention[:, top_word_index]

        input = input[non_stop_words_pos]
        input = input[top_word_index]
        plot_param.tokens = tokenizer.convert_ids_to_tokens(input)
        plot_param.attention = filtered_attention
        plot_param.plot_title = "False negative attention"
        plot_param.file_name = "output/PID: {} - bert_fn_attention.png".format(
            pid)
        plot_service.plot_attention(plot_param)
    else:
        LogUtils.instance().log_warning("No false negative attention")


if __name__ == "__main__":

    main()
