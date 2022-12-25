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
from torch.optim import Adam
from torch import nn
import os


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

    if flags.pretrain == IN_DOMAIN_PRETRAIN and not os.path.exists(os.path.join(os.getcwd(), "pretrained")):
        raise FileNotFoundError(
            "In domain pretrain needs to run run_pretrained.sh first!")

    for k, v in sorted(vars(flags).items()):
        LogUtils.instance().log_info("\t{}: {}".format(k, v))

    # Pretrained model
    pretrained_model_name_or_path = None
    if flags.pretrain == IN_DOMAIN_PRETRAIN:
        pretrained_model_name_or_path = flags.in_domain_pretrain_dir
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = IN_DOMAIN_PRETRAIN_DIR
    else:
        pretrained_model_name_or_path = flags.out_domain_pretrain_model
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = OUT_DOMAIN_PRETRAIN_MODEL

    LogUtils.instance().log_info(
        "Pretrained model name or path: {}".format(pretrained_model_name_or_path))
    pretrained_bert = BertModel.from_pretrained(pretrained_model_name_or_path)
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
    train_data_loader, val_data_loader, test_data_loader = data_loader.load_data(
        load_data_param)

    # Train BERT
    optimizer = Adam(model.parameters(),
                     lr=flags.learning_rate,
                     weight_decay=flags.weight_decay)
    bert_service = BertService()
    criterion = nn.BCELoss()

    train_param = TrainTestParam()
    train_param.model = model
    train_param.optimizer = optimizer
    train_param.criterion = criterion
    train_param.train_data_loader = train_data_loader
    train_param.val_data_loader = val_data_loader
    train_param.epochs = flags.num_epochs
    train_param.freeze_layers = flags.freeze_layers
    train_param.device = device

    best_model_weights, train_result, val_result = bert_service.train_bert(
        train_param)

    # Test BERT
    best_model = BertClassifier(pretrained_bert=pretrained_bert,
                                bert_model=flags.bert_model)
    best_model = best_model.to(device)
    best_model.load_state_dict(best_model_weights)

    test_param = TrainTestParam()
    test_param.model = best_model
    test_param.criterion = criterion
    test_param.test_data_loader = test_data_loader
    test_param.device = device

    bert_service.test_bert(test_param)

    # Plot
    plot_service = PlotService()
    plot_param = PlotParam(train_result.loss_hist,
                           val_result.loss_hist,
                           "Train loss",
                           "Val loss",
                           "Epoch",
                           "Loss",
                           "BERT loss history",
                           "output/PID: {} - bert_loss_history.png".format(pid))
    plot_service.plot_hist(plot_param)


if __name__ == "__main__":

    main()
