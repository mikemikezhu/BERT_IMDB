from tqdm import tqdm
import torch
import copy
import numpy as np

from utils.utils_eval import EvaluationUtils
from utils.utils_log import LogUtils
from utils.utils_pid import PidUtils
from model.result import Result
from model.result import AttentionResult


class BertService:

    """ Public methods """

    def train_bert(self, train_param):

        train_result = Result()
        val_result = Result()

        train_loss_history, val_loss_history = [], []
        train_acc_history, val_acc_history = [], []

        best_model_weights = None
        best_val_loss = float("inf")

        for epoch in range(train_param.epochs):

            # Train
            train_result = self._run_model(train_param.model,
                                           train_param.criterion,
                                           train_param.train_data_loader,
                                           train_param.device,
                                           train_param.freeze_layers,
                                           optimizer=train_param.optimizer,
                                           enable_train=True)

            train_loss_history.append(train_result.loss)
            train_acc_history.append(train_result.acc)

            # Validation
            val_result = self._run_model(train_param.model,
                                         train_param.criterion,
                                         train_param.val_data_loader,
                                         train_param.device)

            val_loss_history.append(val_result.loss)
            val_acc_history.append(val_result.acc)

            if val_result.loss < best_val_loss:
                best_val_loss = val_result.loss
                bert_model_weights = train_param.model.state_dict()
                best_model_weights = copy.deepcopy(bert_model_weights)

            LogUtils.instance().log_info("Epoch: {}, train loss: {}, train acc: {}, val loss: {}, val acc: {}".format(
                epoch, train_result.loss, train_result.acc, val_result.loss, val_result.acc))

        train_result.loss_hist = train_loss_history
        train_result.acc_hist = train_acc_history
        val_result.loss_hist = val_loss_history
        val_result.acc_hist = val_acc_history

        return best_model_weights, train_result, val_result

    def test_bert(self, test_param):
        # Test
        test_result = self._run_model(test_param.model,
                                      test_param.criterion,
                                      test_param.test_data_loader,
                                      test_param.device)
        LogUtils.instance().log_info(
            "Test loss: {}, test acc: {}, test roc: {}".format(test_result.loss, test_result.acc, test_result.roc))

        # Output model
        pid = PidUtils.instance().get_pid()
        output_path = "output/PID: {} - best_model.pt".format(pid)
        torch.save(test_param.model.state_dict(), output_path)

        return test_result

    """ Private methods """

    def _run_model(self, model,
                   criterion,
                   data_loader,
                   device,
                   freeze_layers=None,
                   optimizer=None,
                   enable_train=False):

        if enable_train and optimizer == None:
            raise ValueError("Please pass optimizer if enable train!")

        result = Result()

        total_loss = 0
        total_acc = 0
        total_roc = 0

        tp_attention = None
        tn_attention = None
        fp_attention = None
        fn_attention = None

        torch.set_grad_enabled(enable_train)

        if freeze_layers is not None:

            if enable_train:
                for name, param in model.named_parameters():
                    if param.requires_grad and self._need_freeze(freeze_layers, name):
                        LogUtils.instance().log_info("Freeze layer: {}".format(name))
                        param.requires_grad = False
            else:
                raise ValueError("Only freeze layers when enable train")

        y_preds, y_test = [], []
        for input, label in tqdm(data_loader):

            label = label.to(device)
            mask = input['attention_mask'].to(device)
            input_id = input['input_ids'].squeeze(1).to(device)

            output, attentions = model(input_id, mask)

            y_pred = output[:, 0]
            batch_loss = criterion(y_pred, label)
            total_loss += batch_loss.item()

            acc = EvaluationUtils.mean_accuracy(y_pred, label)
            total_acc += acc

            roc = EvaluationUtils.mean_roc_auc(y_pred, label)
            total_roc += roc

            if enable_train:
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            else:
                y_preds.append(y_pred.detach().cpu().numpy())
                y_test.append(label.detach().cpu().numpy())

                # Attention matrix
                head_idx = 0  # Select the first attention head
                # Select the last attention layer
                last_layer_attention = (list(attentions))[-1]
                if tp_attention is None:

                    LogUtils.instance().log_info("Create attention matrix for true positive samples")
                    bool_matrix = (y_pred == label) & (label == 1.0)
                    tp_attention = self._create_attention(bool_matrix,
                                                          input_id,
                                                          last_layer_attention,
                                                          head_idx)

                if tn_attention is None:

                    LogUtils.instance().log_info("Create attention matrix for true negative samples")
                    bool_matrix = (y_pred == label) & (label == 0.0)
                    tn_attention = self._create_attention(bool_matrix,
                                                          input_id,
                                                          last_layer_attention,
                                                          head_idx)

                if fp_attention is None:

                    LogUtils.instance().log_info("Create attention matrix for false positive samples")
                    bool_matrix = (y_pred != label) & (label == 1.0)
                    fp_attention = self._create_attention(bool_matrix,
                                                          input_id,
                                                          last_layer_attention,
                                                          head_idx)

                if fn_attention is None:

                    LogUtils.instance().log_info("Create attention matrix for false negative samples")
                    bool_matrix = (y_pred != label) & (label == 0.0)
                    fn_attention = self._create_attention(bool_matrix,
                                                          input_id,
                                                          last_layer_attention,
                                                          head_idx)

        total_loss /= len(data_loader)
        total_acc /= len(data_loader)
        total_roc /= len(data_loader)

        result.loss = total_loss
        result.acc = total_acc
        result.roc = total_roc

        result.tp_attention = tp_attention
        result.tn_attention = tn_attention
        result.fp_attention = fp_attention
        result.fn_attention = fn_attention

        if len(y_preds) > 0:
            result.y_preds = np.ravel(y_preds)

        if len(y_test) > 0:
            result.y_test = np.ravel(y_test)

        return result

    def _need_freeze(self, freeze_layers, layer_name):

        if freeze_layers is None:
            return False

        freeze_layers_list = freeze_layers.split(',')
        if len(freeze_layers_list) == 0:
            return False

        for freeze_layer in freeze_layers_list:
            keyword = ".{}.".format(freeze_layer)
            if keyword in layer_name:
                return True

        return False

    def _create_attention(self, bool_matrix,
                          input_id,
                          last_layer_attention,
                          head_idx):

        result = None

        indices = list(bool_matrix.nonzero())
        LogUtils.instance().log_info("Total samples index: {}".format(indices))

        if len(indices) > 0:

            sample_idx = indices[0]
            LogUtils.instance().log_info("First sample index: {}".format(sample_idx))

            input = input_id[sample_idx]
            sample_attention = last_layer_attention[sample_idx, head_idx]

            result = AttentionResult(input,
                                     sample_attention)

        return result
