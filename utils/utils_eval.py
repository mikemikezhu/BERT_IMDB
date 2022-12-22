from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import torch.nn as nn


class EvaluationUtils:
    """ Utility class to evaluate models """

    @staticmethod
    def mean_nll(pred, y):
        criterion = nn.BCELoss()
        return criterion(pred, y)

    @staticmethod
    def mean_accuracy(pred, y):
        pred = (pred > 0.5).float()
        return ((pred - y).abs() < 1e-2).float().mean()

    @staticmethod
    def mean_roc_auc(pred, y):
        y = y.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        return roc_auc_score(y, pred)

    @staticmethod
    def mean_pr_auc(pred, y):
        y = y.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        precision, recall, _ = precision_recall_curve(y, pred)
        return auc(recall, precision)
