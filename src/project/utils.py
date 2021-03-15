import os
import subprocess
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import *

def download_datasets(datasets):
    """

    Download the time series datasets used in the paper
    from http://www.timeseriesclassification.com

    NonInvThorax1 and NonInvThorax2 are missing

    """
    subprocess.call('mkdir -p data'.split())
    datasets_pbar = tqdm(datasets)

    for dataset in datasets_pbar:
        datasets_pbar.set_description('Downloading {0}'.format(dataset))
        subprocess.call('curl http://www.timeseriesclassification.com/Downloads/{0}.zip \
                         -o data/{0}.zip'.format(dataset).split())
        datasets_pbar.set_description('Extracting {0} data'.format(dataset))
        subprocess.call('unzip data/{0} -d data/'.format(dataset).split())
        assert os.path.exists('data/{}_TRAIN.arff'.format(dataset)), dataset


def data_dictionary(datasets):
    """

    Create a dictionary of train/test DataLoaders for
    each of the datasets downloaded

    """
    dataset_dict = {}
    pbar = tqdm(datasets)
    for dataset in pbar:
        pbar.set_description('Processing {}'.format(dataset))
        train_set, test_set = DataTSV(dataset, testing=False), DataTSV(dataset, testing=True)
        batch_size = min(16, len(train_set)//10)

        dataset_dict[dataset] = {}
        dataset_dict[dataset]['train'] = DataLoader(train_set, batch_size=batch_size)
        dataset_dict[dataset]['test'] = DataLoader(test_set, batch_size=batch_size)

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


def print_history(history):
    """
    Print only the essential history information
    """
    best_loss = 99999
    for epoch, (acc, loss, error_rate) in enumerate(history):
        if loss <= best_loss:
            best_loss = loss
            print('{:5} ACC: {:5.2f} Loss: {:8.6f} Error Rate: {:5.3f}'.format(
                epoch,
                acc,
                loss,
                error_rate
            ))


def mpce(model, test_dataloader, device):
    """

    Mean per-class error as described in the paper:
        The test-set cross-entropy errors on the data
        partitioned by class labels. Weight these errors
        by the count of each label in the data and take the
        mean.

    """
    inputs = np.array(test_dataloader.dataset.inputs)
    targets = np.array(test_dataloader.dataset.targets)

    counts = {label: count for label, count in
              enumerate(np.bincount(targets))}
    label_idxs = {label: np.where(targets == label)[0]
                  for label in counts.keys()}

    inputs = torch.Tensor(inputs).to(device)
    targets = torch.Tensor(targets).long().to(device)

    errors = {}
    model.eval()
    for label, idxs in label_idxs.items():
        batch_size = min(16, len(idxs)//10+1)
        batches = np.arange(0, len(idxs), batch_size)

        errors[label] = 0
        for b, _ in enumerate(batches[:-1]):
            out = model(inputs[batches[b]:batches[b+1]])
            loss = F.cross_entropy(out, targets[batches[b]:batches[b+1]])
            errors[label] += loss.item()*out.size(0)

        errors[label] = errors[label]/len(idxs)

    mean_per_class_error = np.mean([v for _, v in errors.items()])
    return mean_per_class_error
