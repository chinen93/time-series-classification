import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from data import DataTSV

def data_dictionary(datasets, verbose=True):
    """

    Create a dictionary of train/test DataLoaders for
    each of the datasets downloaded

    """
    dataset_dict = {}
    disable = not verbose
    pbar = tqdm(datasets, disable=disable)
    for dataset in pbar:
        pbar.set_description('Processing {}'.format(dataset))
        try:
            train_set = DataTSV(dataset, testing=False)
            test_set = DataTSV(dataset, testing=True)
        except ValueError:
            print("{} has invalid values or std = 0".format(dataset))
            continue

        batch_size = min(16, len(train_set)//10)

        dataset_dict[dataset] = {}

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        dataset_dict[dataset]['train'] = DataLoader(
            train_set,
            batch_size=batch_size,
            worker_init_fn=seed_worker
        )
        dataset_dict[dataset]['test'] = DataLoader(
            test_set,
            batch_size=batch_size,
            worker_init_fn=seed_worker
        )

    return dataset_dict


def print_dataset_info(dataset, dataloader):
    """

    Print information about the dataset

    """
    train = dataloader['train']
    test = dataloader['test']
    time_steps = train.dataset.inputs.shape[-1]
    n_classes = len(np.unique(train.dataset.targets))

    print(dataset)
    print('train samples={}\ttest samples={}\ttime steps={}\tnum. classes={}'.format(
        len(train.dataset.inputs),
        len(test.dataset.inputs),
        time_steps, n_classes
    ))
