import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from torchmetrics.classification import MulticlassAccuracy, BinaryAUROC
from typing import Callable
from tqdm import tqdm


class ModelTrainer():
    def __init__(
            self,
            model:nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optim: torch.optim.Optimizer,
            loss_fn: Callable,
            num_classes: int,
            save_filename:str='model.pt',
            device='cuda',
            # amp: bool = True
            ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optim = optim
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.save_filename = save_filename
        self.device = device
        self.scaler = torch.amp.grad_scaler.GradScaler()


    def train_epoch(self):
        accuracy_metric = MulticlassAccuracy(num_classes=self.num_classes).to(self.device)
        auc_metric = BinaryAUROC(thresholds=None).to(self.device)
        losses = []
        self.model.train()
        for images, labels in tqdm(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            labels_onehot = torch.nn.functional.one_hot(labels, num_classes=self.num_classes)
            labels_onehot = labels_onehot.type(torch.float32)

            self.optim.zero_grad()
            with torch.autocast(self.device):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels_onehot)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

            losses.append(loss.item())
            accuracy_metric.update(torch.argmax(outputs, dim=-1), labels)
            auc_metric.update(torch.softmax(outputs, dim=1)[:, 1], labels)
        return np.mean(losses).item(), accuracy_metric.compute().item(), auc_metric.compute().item()

    def validation_epoch(self):
        accuracy_metric = MulticlassAccuracy(num_classes=self.num_classes).to(self.device)
        auc_metric = BinaryAUROC(thresholds=None).to(self.device)
        losses = []

        self.model.eval()
        accuracy_metric.reset()
        with torch.inference_mode():
            for images, labels in tqdm(self.val_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                labels_onehot = torch.nn.functional.one_hot(labels, num_classes=self.num_classes)
                labels_onehot = labels_onehot.type(torch.float32)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels_onehot)

                losses.append(loss.item())
                accuracy_metric.update(torch.argmax(outputs, dim=-1), labels)
                auc_metric.update(torch.softmax(outputs, dim=1)[:, 1], labels)
        return np.mean(losses).item(), accuracy_metric.compute().item(), auc_metric.compute().item()

    def save_state(self, epoch, fname, val_loss):
        save_dict = {
            'model_state': self.model.state_dict(),
            'optim_state': self.optim.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
        }
        torch.save(save_dict, fname)

    def load_state(self, fname):
        obj = torch.load(fname)
        print(f'Saved at Epoch {obj["epoch"]}')
        self.model.load_state_dict(obj['model_state'])
        self.optim.load_state_dict(obj['optim_state'])


    def train_model(self, num_epochs=1, early_stop_patience=10):
        losses = []
        accs = []
        val_losses = []
        val_accs = []

        # For early stopping
        no_improve_epoch_count = 0
        for epoch in range(num_epochs):
            print(f'Epoch {epoch:2d}/{num_epochs:2d}:')
            train_loss, train_acc, auc = self.train_epoch()
            val_loss, val_acc, val_auc = self.validation_epoch()

            if len(val_losses) == 0 or val_loss < np.min(val_losses):
                # Save model state
                dirname = f'checkpoints/{self.model.__class__.__name__}'
                filename = f'{dirname}/{self.save_filename}'
                os.makedirs(dirname, exist_ok=True)
                self.save_state(epoch, filename, val_loss)
                print(f'Best State Saved at {filename} || val_loss: {val_loss:0.4f}')
                # Reset early stopping counter
                no_improve_epoch_count = 0
            else:
                no_improve_epoch_count += 1

            losses.append(train_loss)
            accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(f'loss: {train_loss:0.4f} | acc: {train_acc:0.4f} |  auc: {auc:0.4f} | val_loss: {val_loss:0.4f} | val_acc: {val_acc:0.4f} |  val_auc: {val_auc:0.4f} ')
            
            if no_improve_epoch_count > early_stop_patience:
                print('Early Stopping')
                break
        return losses, accs, val_losses, val_accs
