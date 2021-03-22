import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, n_in, n_classes):
        super(ConvNet, self).__init__()
        self.n_in = n_in
        self.n_classes = n_classes

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=128,
            kernel_size=8,
            stride=1,
            padding=3,
            padding_mode='replicate'
        )
        self.bn1 = nn.BatchNorm1d(num_features=128)

        self.conv2 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=5,
            stride=1,
            padding=2,
            padding_mode='replicate'
        )
        self.bn2 = nn.BatchNorm1d(num_features=256)

        self.conv3 = nn.Conv1d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='replicate'
        )
        self.bn3 = nn.BatchNorm1d(num_features=128)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv1d(
            in_channels=128,
            out_channels=self.n_classes,
            kernel_size=1
        )
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)


    def forward(self, x: torch.Tensor):
        x = x.view(-1, 1, self.n_in)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.dropout(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.avgpool(x)

        x = x.view(x.shape[0], -1)

        return x
