import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from utils import *

def train(dataloader_train: DataLoader,
          dataloader_test: DataLoader,
          device: str,
          model: nn.Module,
          epochs: int,
          learning_rate: float,
          save: bool):

    optimiser = optim.Adam(model.parameters(),lr=learning_rate)
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

    for epoch in range(epochs):

        # train
        model.train()
        for batch, data in enumerate(dataloader_train):
            inputs, targets = data
            inputs, targets = inputs.to(device), (targets.view(-1)).to(device)

            optimiser.zero_grad()

            out = model(inputs)
            loss = F.cross_entropy(out, targets)

            loss.backward()
            optimiser.step()

        # test
        running_loss = 0
        running_acc = 0
        for batch, data in enumerate(dataloader_test):
            inputs, targets = data
            inputs, targets = inputs.to(device), (targets.view(-1)).to(device)
            outs = model(inputs)

            sum_right = ((torch.argmax(outs, 1)) == targets).cpu().detach().numpy().sum()
            test_acc = sum_right/len(targets)
            test_loss = F.cross_entropy(outs, targets).item()
            running_acc += test_acc * inputs.size(0)
            running_loss += test_loss * inputs.size(0)

        test_size = len(dataloader_test.dataset)
        test_acc = running_acc / test_size
        test_loss = running_loss / test_size
        test_error_rate = 1 - test_acc
        progress_bar.postfix[0] = test_acc * 100
        progress_bar.postfix[1] = test_loss
        progress_bar.postfix[2] = test_error_rate
        progress_bar.update(n=1)

        history.append((test_acc * 100, test_loss, test_error_rate))

        if save:
            #save
            pass

    return model, history
