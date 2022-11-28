import os
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import math
import warnings
import logging
import numpy as np

logger = logging.getLogger(__name__)

def my_accuracy_score(y_true, y_pred, average=None, zero_division=0):
    # print(y_true.shape)
    # print(y_pred.shape)
    # print(y_true[:,0])
    # print(y_pred[:,0])
    n_labels= y_true.shape[1]

    per_class_accuracy = np.zeros((n_labels,))
    # print(per_class_accuracy.shape)

    for i in range(n_labels):
        # print((y_pred[:,i] == y_true[:,i]).float().mean())
        per_class_accuracy[i] = (y_pred[:,i] == y_true[:,i]).float().mean()

    # print(y_true)
    # print(y_pred)
    # print(per_class_accuracy)

    # raise Exception()
    if average is None:
        return per_class_accuracy
    else:
        return np.mean(per_class_accuracy)


class Metric():
    def __init__(self, mode, metric):
        self.mode = mode
        self.best_result = None
        self.all_results = []
        self.epoches = []

        if metric == "precision":
            self.fn = precision_score
        elif metric == "recall":
            self.fn = recall_score
        elif metric == "f1":
            self.fn = f1_score
        elif metric == "accuracy":
            self.fn = my_accuracy_score
        else:
            raise Exception("{metric} is unsupported".format(metric=metric))

        self.name = "{mode}_avg_{metric}".format(mode=mode, metric=metric)

    def avg_best_result(self, top=-1):
        if top == -1:
            return np.mean(self.best_result)
        else:
            bests = sorted(self.best_result)
            return np.mean(bests[-top:])


    def update(self, epoch, preds, labels):
        # print(self.name, preds.shape, labels.shape)

        per_class_result = self.fn(preds, labels, average=None, zero_division=0)

        self.epoches.append(epoch)
        self.all_results.append(per_class_result)

        # print(per_class_result)
        if self.best_result is None:
            self.best_result = per_class_result
        else:
            for i, result in enumerate(per_class_result):
                self.best_result[i] = max(self.best_result[i], result)

        return np.mean(per_class_result),self.avg_best_result()

    def toValue(self):
        return {
            self.name: self.avg_best_result()
        }

    def __str__(self):
        return "{name}:{result}(top3:{top})".format(name=self.name,result=self.avg_best_result(),top=self.avg_best_result(3))

    def __repr__(self):
        return str(self)


def train(model, graph, n_epochs, step=5, min_delta=0.0001, patience=5, train_ratio=0.6, validation_ratio=0.2):
    # os.environ["CUDA_VISIBLE_DEVICES"]=""

    # device = torch.device(device)
    # model.to(device)
    # model.to("cpu")
    n_users = graph.num_nodes()

    mode_masks = {
        "train": torch.zeros(n_users, dtype=torch.bool),
        "val": torch.zeros(n_users, dtype=torch.bool),
        "test": torch.zeros(n_users, dtype=torch.bool),
    }

    n_train = int(n_users * train_ratio)
    n_val = math.ceil(n_users * validation_ratio)

    logger.debug("{}_{}_{}".format(n_users, n_users * validation_ratio, n_val))
    mode_masks["train"][:n_train] = True
    mode_masks["val"][n_train:n_train + n_val] = True
    mode_masks["test"][n_train + n_val:] = True


    user_feats = graph.nodes['user'].data['feature']
    labels = graph.nodes['user'].data['label']

    # logger.debug(user_feats)


    optimizer = torch.optim.Adam(model.parameters())

    node_features = {'user': user_feats}

    metrics = {}

    for mode in ["train", "val", "test"]:
        mode_metrics = []
        for metric in ["precision", "recall", "f1", "accuracy"]:
            mode_metrics.append(Metric(mode, metric))

        metrics[mode] = mode_metrics

    # print(metrics)
    # return
    last_loss_val = 100000
    patience_count = 0
    for epoch in range(n_epochs):
        model.train()
        # forward propagation by using all nodes and extracting the user embeddings
        logits = model(graph, node_features)['user']
        # compute loss
        loss = F.cross_entropy(logits[mode_masks["train"]], labels[mode_masks["train"]])

        # Compute prediction
        pred_probabilities = torch.sigmoid(logits)
        pred = torch.round(pred_probabilities)

        current_loss_val = loss.detach().numpy()

        if last_loss_val - current_loss_val < min_delta:
            patience_count += 1

        if patience_count > patience:
            break

        if epoch % step == 0:
            raw_pred = pred.detach()

            for mode in metrics.keys():
                mode_metrics = metrics.get(mode)

                for metric in mode_metrics:
                    metric.update(epoch, raw_pred[mode_masks[mode]], labels[mode_masks[mode]])


            logger.info("epoch: {e} - loss:{loss:.3f} - metrics:{metric} - patience: {patience} - dif: {dif}".format(e=epoch, loss=current_loss_val, metric=metrics, patience=patience_count, dif=last_loss_val - current_loss_val))

        last_loss_val = current_loss_val

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    results = {}
    for mode in metrics.keys():
        mode_metrics = metrics.get(mode)

        for metric in mode_metrics:
            results[metric.name] = metric

    return results