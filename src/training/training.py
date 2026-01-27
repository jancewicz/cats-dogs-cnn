import torch
from loguru import logger
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class NNTrainer:
    def __init__(self, model, optimizer, criterion, train_loader, device):
        self.model: Module = model
        self.optimizer: Optimizer = optimizer
        self.criterion: Module = criterion
        self.train_loader: DataLoader = train_loader
        self.device: torch.device = device

    def single_training_loop(self):
        total_loss = 0
        loop = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            unit="batch",
            leave=False,
        )

        for _, (X_batch, y_batch) in loop:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)

            loss = self.criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            loop.set_postfix(loss=loss.item())

        mean_loss = total_loss / len(self.train_loader)
        return mean_loss

    def train(self, n_epochs):
        self.model.train()
        for epoch in range(n_epochs):
            mean_loss = self.single_training_loop()
            logger.info(f"Epoch {epoch+1} / {n_epochs}, Loss: {mean_loss: .4f}")
