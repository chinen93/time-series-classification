import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):
    def __init__(self, n_in, n_classes):
        super(MultiLayerPerceptron, self).__init__()
        self.n_in = n_in
        self.n_classes = n_classes

        self.dropout1 = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(
            in_features=self.n_in,
            out_features=500
        )

        self.dropout2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(
            in_features=500,
            out_features=500
        )

        self.dropout3 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(
            in_features=500,
            out_features=500
        )

        self.dropout4 = nn.Dropout(p=0.3)
        self.fully_connected = nn.Linear(
            in_features=500,
            out_features=self.n_classes
        )


    def forward(self, x: torch.Tensor):
        x = x.view(-1, self.n_in)

        x = self.dropout1(x)
        x = F.relu(self.fc1(x))

        x = self.dropout2(x)
        x = F.relu(self.fc2(x))

        x = self.dropout3(x)
        x = F.relu(self.fc3(x))

        x = self.dropout4(x)
        x = self.fully_connected(x)

        return x
