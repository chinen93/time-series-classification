import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, n_in, n_classes):
        super(ResNet, self).__init__()
        self.n_in = n_in
        self.n_classes = n_classes

        blocks = [1, 64, 128, 128]
        self.blocks = nn.ModuleList()
        for b, _ in enumerate(blocks[:-1]):
            self.blocks.append(ResidualBlock(*blocks[b:b+2], self.n_in))

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)

        self.fc1 = nn.Linear(blocks[-1], self.n_classes)


    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_maps, out_maps, time_steps):
        super(ResidualBlock, self).__init__()
        self.in_maps = in_maps
        self.out_maps = out_maps
        self.time_steps = time_steps

        self.conv1 = nn.Conv1d(
            in_channels=self.in_maps,
            out_channels=self.out_maps,
            kernel_size=8,
            stride=1,
            padding=4,
            padding_mode='replicate'
        )
        self.bn1 = nn.BatchNorm1d(self.out_maps)

        self.conv2 = nn.Conv1d(
            in_channels=self.out_maps,
            out_channels=self.out_maps,
            kernel_size=5,
            stride=1,
            padding=2,
            padding_mode='replicate'
        )
        self.bn2 = nn.BatchNorm1d(self.out_maps)

        self.conv3 = nn.Conv1d(
            in_channels=self.out_maps,
            out_channels=self.out_maps,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='replicate'
        )
        self.bn3 = nn.BatchNorm1d(self.out_maps)

        self.convI = nn.Conv1d(
            in_channels=self.in_maps,
            out_channels=self.out_maps,
            kernel_size=1
        )
        self.bnI = nn.BatchNorm1d(self.out_maps)

    def forward(self, x):

        is_expand_channels = not self.in_maps == self.out_maps
        if is_expand_channels:
            identity = self.bnI(self.convI(x))
        else:
            identity = self.bnI(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Clip timesteps to the smaller one.
        shape = identity.shape[2]
        id_shape = x.shape[2]
        if shape > id_shape:
            shape = id_shape
        x = x[:, :, :shape]
        identity = identity[:, :, :shape]

        # Add residual part.
        x += identity

        return x
