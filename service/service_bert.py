from tqdm import tqdm
import torch
import copy

from utils.utils_eval import EvaluationUtils
from utils.utils_log import LogUtils
from model.result import Result


class BertService:

    """ Public methods """

    def train_bert(self, model,
                   optimizer,
                   criterion,
                   train_data_loader,
                   val_data_loader,
                   epochs, device):

        train_result = Result()
        val_result = Result()

        train_loss_history, val_loss_history = [], []
        train_acc_history, val_acc_history = [], []

        best_model_weights = None
        best_val_loss = float("inf")

        for epoch in range(epochs):

            # Train
            train_result = self._run_model(model,
                                           criterion,
                                           train_data_loader,
                                           device,
                                           optimizer,
                                           True)

            train_loss_history.append(train_result.loss)
            train_acc_history.append(train_result.acc)

            # Validation
            val_result = self._run_model(model,
                                         criterion,
                                         val_data_loader,
                                         device)

            val_loss_history.append(val_result.loss)
            val_acc_history.append(val_result.acc)

            if val_result.loss < best_val_loss:
                best_val_loss = val_result.loss
                bert_model_weights = model.state_dict()
                best_model_weights = copy.deepcopy(bert_model_weights)

            LogUtils.instance().log_info("Epoch: {}, train loss: {}, train acc: {}, val loss: {}, val acc: {}".format(
                epoch, train_result.loss, train_result.acc, val_result.loss, val_result.acc))

        train_result.loss_hist = train_loss_history
        train_result.acc_hist = train_acc_history
        val_result.loss_hist = val_loss_history
        val_result.acc_hist = val_acc_history

        return best_model_weights, train_result, val_result

    def test_bert(self, model,
                  criterion,
                  test_data_loader,
                  device):
        # Test
        test_result = self._run_model(model,
                                      criterion,
                                      test_data_loader,
                                      device)
        LogUtils.instance().log_info(
            "Test loss: {}, test acc: {}".format(test_result.loss, test_result.acc))
        return test_result

    """ Private methods """

    def _run_model(self, model,
                   criterion,
                   data_loader,
                   device,
                   optimizer=None,
                   enable_train=False):

        if enable_train and optimizer == None:
            raise ValueError("Please pass optimizer if enable train!")

        result = Result()

        total_loss = 0
        total_acc = 0

        torch.set_grad_enabled(enable_train)

        for input, label in tqdm(data_loader):

            label = label.to(device)
            mask = input['attention_mask'].to(device)
            input_id = input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, label[:, None])
            total_loss += batch_loss.item()

            y_pred = output[:, 0]
            y_pred = y_pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            acc = EvaluationUtils.calculate_accuracy(y_pred, label)
            total_acc += acc

            if enable_train:
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

        total_loss /= len(data_loader)
        total_acc /= len(data_loader)

        result.loss = total_loss
        result.acc = total_acc

        return result
