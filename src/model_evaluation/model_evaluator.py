from typing import Callable

import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import DataLoader
from utils.device import get_device


class NNEvaluator:
    """Handles neural network evaluation with given metric function."""

    def __init__(self, model, data_loader, metric_fn, device):
        """
        :param model: current model to evaluate
        :param data_loader: dataloader for currently used dataset
        :param metric_fn: function to compute metric for given batch
        """
        self.model: Module = model
        self.data_loader: DataLoader = data_loader
        self.metric_fn: Callable[[torch.Tensor, torch.Tensor], None] = metric_fn
        self.device: torch.device = device

    def evaluate(self, aggregate_fn: Callable = torch.mean):
        """
        Evaluates model and collect metrics within each batch.
        :param aggregate_fn: function to aggregate the batch metrics, computes mean by default
        :return: aggregated model metrics
        """
        self.model.eval()
        metrics = []
        with torch.no_grad():
            for X_batch, y_batch in self.data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                metric = self.metric_fn(y_pred, y_batch)
                metrics.append(metric)

        return aggregate_fn(torch.stack(metrics))

    def evaluate_tm(self, metric):
        self.model.eval()
        metric.reset()  # at the beginning reset metric
        with torch.no_grad():
            for X_batch, y_batch in self.data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                metric.update(y_pred, y_batch)  # update the metric at each iteration

        # compute final result
        return metric.compute()

    def get_evaluation_val(self):
        return self.model.eva


def evaluate_on_valid_set(
    model: nn.Module, valid_set_dataloader: DataLoader, metric_fc: Callable
):
    """
    Perform evaluation on given dataloader with specific metric function. Returns the value of evaluation e.g:
    accuracy of the model on validation set.
    :param model: Instance of predictive model.
    :param valid_set_dataloader: Validation set dataloader.
    :param metric_fc: metric function to be applied during evaluation.
    :return: Value of computed metric on whole set.
    """
    model_evaluator = NNEvaluator(
        model=model,
        data_loader=valid_set_dataloader,
        metric_fn=metric_fc,
        device=get_device(),
    )
    evaluation = model_evaluator.evaluate()
    return evaluation
