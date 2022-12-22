from model.classifier import BertClassifier
from model.query_load_data import LoadDataQuery
from model.query_plot import PlotQuery

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


def main():

    pid = PidUtils.instance().get_pid()
    LogUtils.instance().log_info("PID: {}".format(pid))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    LogUtils.instance().log_info("Device: {}".format(device))

    # Parse arguments
    parser_service = ArgumentParserService()
    parser = parser_service.get_parser()
    flags = parser.parse_args()

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

    pretrained_bert = BertModel.from_pretrained(pretrained_model_name_or_path)
    model = BertClassifier(pretrained_bert=pretrained_bert,
                           bert_model=flags.bert_model)
    model.to(device)

    # Load data
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
    data_loader = DataLoaderService(tokenizer)
    load_data_query = LoadDataQuery(flags.val_data_length,
                                    flags.train_neg_data_path,
                                    flags.train_pos_data_path,
                                    flags.test_neg_data_path,
                                    flags.test_pos_data_path,
                                    flags.train_batch_size,
                                    flags.val_batch_size,
                                    flags.test_batch_size)
    train_data_loader, val_data_loader, test_data_loader = data_loader.load_data(
        load_data_query)

    # Train BERT
    optimizer = Adam(model.parameters(),
                     lr=flags.learning_rate,
                     weight_decay=flags.weight_decay)
    bert_service = BertService()
    criterion = nn.BCELoss()
    best_model_weights, train_result, val_result = bert_service.train_bert(model,
                                                                           optimizer,
                                                                           criterion,
                                                                           train_data_loader,
                                                                           val_data_loader,
                                                                           flags.num_epochs,
                                                                           device)

    # Test BERT
    best_model = BertClassifier(pretrained_bert=pretrained_bert,
                                bert_model=flags.bert_model)
    best_model = best_model.to(device)
    best_model.load_state_dict(best_model_weights)
    bert_service.test_bert(best_model,
                           criterion,
                           test_data_loader,
                           device)

    # Plot
    plot_service = PlotService()
    plot_query = PlotQuery(train_result.loss_hist,
                           val_result.loss_hist,
                           "Train loss",
                           "Val loss",
                           "Epoch",
                           "Loss",
                           "BERT loss history",
                           "PID: {} - bert_loss_history.png".format(pid))
    plot_service.plot_hist(plot_query)


if __name__ == "__main__":

    main()
