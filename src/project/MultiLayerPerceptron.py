import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):
    def __init__(self, n_in, n_classes):
        super(MultiLayerPerceptron, self).__init__()
        self.n_in = n_in
        self.n_classes = n_classes

        self.dropout1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(n_in, 500)

        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(500, 500)

        self.dropout3 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(500, 500)

        self.dropout4 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(500, self.n_classes)


    def forward(self, x: torch.Tensor):

        x = self.dropout1(x)
        x = F.relu(self.fc1(x))

        x = self.dropout2(x)
        x = F.relu(self.fc2(x))

        x = self.dropout3(x)
        x = F.relu(self.fc3(x))

        x = self.dropout4(x)
        x = self.fc4(x)

        x = x.view(-1, self.n_classes)

        return x
