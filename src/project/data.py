import pandas as pd
import numpy as np
from scipy.io.arff import loadarff
import torch
from torch.utils.data import Dataset

class Data(Dataset):
    """
    Implements a PyTorch Dataset class

    """
    def __init__(self, dataset, testing):
        train_data = pd.DataFrame(
            loadarff('data/{}_TRAIN.arff'.format(dataset))[0])
        dtypes = {i:np.float32 for i in train_data.columns[:-1]}
        dtypes.update({train_data.columns[-1]: np.int})
        train_data = train_data.astype(dtypes)

        self.inputs = train_data.iloc[:, :-1].values
        self.targets = train_data.iloc[:, -1].values

        self.mean = np.mean(self.inputs, axis=0)
        self.std = np.std(self.inputs, axis=0)

        if testing:
            test_data = pd.DataFrame(
                loadarff('data/{}_TEST.arff'.format(dataset))[0])
            dtypes = {i:np.float32 for i in test_data.columns[:-1]}
            dtypes.update({test_data.columns[-1]: np.int})
            test_data = test_data.astype(dtypes)

            self.inputs = test_data.iloc[:, :-1].values
            self.targets = test_data.iloc[:, -1].values

        class_labels = np.unique(self.targets)
        if (class_labels != np.arange(len(class_labels))).any():
            new_labels = {old_label: i for i, old_label in enumerate(class_labels)}
            self.targets = [new_labels[old_label] for old_label in self.targets]

        self.inputs = (self.inputs - self.mean) / self.std

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        inputs = torch.Tensor([self.inputs[idx]]).float()
        targets = torch.Tensor([self.targets[idx]]).long()
        return inputs, targets

class DataTSV(Dataset):
    """
    Implements a PyTorch Dataset class

    """
    def __init__(self, dataset, testing):

        dataset_folder = 'UCRArchive_2018/{}/'.format(dataset)

        train_data = pd.read_csv(
            '{}{}_TRAIN.tsv'.format(dataset_folder, dataset),
            delimiter="\t",
            header=None
        )
        dtypes = {i:np.float32 for i in train_data.columns[:-1]}
        dtypes.update({train_data.columns[-1]: np.int})
        train_data = train_data.astype(dtypes)

        if not np.isfinite(train_data).all().all():
            raise ValueError

        self.inputs = train_data.iloc[:, 1:].values
        self.targets = train_data.iloc[:, 0].values


        self.mean = np.mean(self.inputs, axis=0)
        self.std = np.std(self.inputs, axis=0)

        if (self.std == 0).any():
            raise ValueError

        if testing:
            test_data = pd.read_csv(
                '{}{}_TEST.tsv'.format(dataset_folder, dataset),
                delimiter="\t",
                header=None
            )
            dtypes = {i:np.float32 for i in test_data.columns[:-1]}
            dtypes.update({test_data.columns[-1]: np.int})
            test_data = test_data.astype(dtypes)

            if not np.isfinite(test_data).all().all():
                raise ValueError

            self.inputs = test_data.iloc[:, 1:].values
            self.targets = test_data.iloc[:, 0].values

        class_labels = np.unique(self.targets)
        if (class_labels != np.arange(len(class_labels))).any():
            new_labels = {old_label: i for i, old_label in enumerate(class_labels)}
            self.targets = [new_labels[old_label] for old_label in self.targets]

        self.inputs = (self.inputs - self.mean) / self.std

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        inputs = torch.Tensor([self.inputs[idx]]).float()
        targets = torch.Tensor([self.targets[idx]]).long()
        return inputs, targets
