from model.classifier import BertClassifier
import torch

from utils.utils_log import LogUtils

from transformers import BertModel


def main():

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    LogUtils.instance().log_info("Device: {}".format(device))

    pretrained_bert = BertModel.from_pretrained('pretrained/checkpoint-2500')
    model = BertClassifier(pretrained_bert=pretrained_bert)
    model.to(device)

    # TODO


if __name__ == "__main__":
    main()
