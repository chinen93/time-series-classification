import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm, trange

from utils import *

import neptune


def train(model_name: str,
          dataset_name: str,
          dataloader_train: DataLoader,
          dataloader_test: DataLoader,
          device: str,
          model: nn.Module,
          optimizer,
          epochs: int,
          save: bool):

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=50,
        min_lr=0.0001,
        verbose=True
    )

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    desc = '{}_{}'.format(model_name, dataset_name)

    history = []

    bar_format = (
        '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, '
        'acc={postfix[0]:.2f}, '
        'cross_entropy={postfix[1]:.2f}, '
        'error rate={postfix[2]:.2f}]'
    )
    postfix = [0.0, 0.0, 0.0]

    progress_bar = tqdm(
        total=epochs,
        bar_format=bar_format,
        postfix=postfix
    )
    best_loss = 999

    for epoch in range(epochs):

        # ================================
        # Train
        # ================================
        # training mode enables dropout
        model.train()
        for batch, data in enumerate(dataloader_train):
            inputs, targets = data
            inputs, targets = inputs.to(device), (targets.view(-1)).to(device)

            optimizer.zero_grad()

            out = model(inputs)
            loss = criterion(out, targets)

            loss.backward()
            optimizer.step()

            neptune.log_metric('{}_train_loss'.format(desc), loss.item())

        # ================================
        # Test
        # ================================
        model.eval()
        running_loss = 0
        running_acc = 0
        for batch, data in enumerate(dataloader_test):
            inputs, targets = data
            inputs, targets = inputs.to(device), (targets.view(-1)).to(device)

            with torch.no_grad():
                outs = model(inputs)

                sum_right = ((torch.argmax(outs, 1)) == targets).cpu().detach().numpy().sum()
                test_acc = sum_right/len(targets)
                test_loss = criterion(outs, targets).item()

            running_acc += test_acc
            running_loss += test_loss

        scheduler.step(test_loss)

        test_size = len(dataloader_test)
        test_acc = running_acc / test_size
        test_loss = running_loss / test_size
        test_error_rate = 1 - test_acc

        progress_bar.postfix[0] = test_acc * 100
        progress_bar.postfix[1] = test_loss
        progress_bar.postfix[2] = test_error_rate
        progress_bar.update(n=1)

        history.append((test_acc * 100, test_loss, test_error_rate))

        neptune.log_metric('{}_test_accuracy'.format(desc), test_acc)
        neptune.log_metric('{}_test_loss'.format(desc), test_loss)
        neptune.log_metric('{}_test_error_rate'.format(desc), test_error_rate)
        if test_loss <= best_loss:
            best_loss = test_loss
            neptune.log_metric('{}_test_best_loss'.format(desc), x=epoch, y=test_loss)
            neptune.log_metric('{}_test_best_error_rate'.format(desc), x=epoch, y=test_error_rate)

        if save:
            #save
            pass

    return model, history
