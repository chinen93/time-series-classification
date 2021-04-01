"""
Unittest for the utils functions
"""
import unittest
import mock
import torch
from torch.utils.data import Dataset

from utils import data_dictionary

class CustomDataset(Dataset):
    def __init__(self, samples, length):
        self.data = torch.randn(
            [samples, length]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestUtilFunctions(unittest.TestCase):
    """
    Class to test utils functions
    """

    @mock.patch('utils.DataTSV')
    def test_data_dictionary_small(self, mock_data):
        """test data dictionary"""

        # Mock values.
        samples = 10
        mock_data.return_value = CustomDataset(
            samples=samples,
            length=20
        )
        datasets = ["Adiac", "Beef"]

        # Run function.
        dataset_dict = data_dictionary(datasets, verbose=False)

        # Assert tests.
        for dataset_name in dataset_dict:
            self.assertEqual(dataset_dict[dataset_name]["train"].batch_size, 1)
            self.assertEqual(dataset_dict[dataset_name]["test"].batch_size, 1)

    @mock.patch('utils.DataTSV')
    def test_data_dictionary_bigger(self, mock_data):
        """test data dictionary"""

        # Mock values.
        samples = 160
        mock_data.return_value = CustomDataset(
            samples=samples,
            length=20
        )
        datasets = ["Adiac", "Beef"]

        # Run function.
        dataset_dict = data_dictionary(datasets, verbose=False)

        # Assert tests.
        for dataset_name in dataset_dict:
            self.assertEqual(dataset_dict[dataset_name]["train"].batch_size, 16)
            self.assertEqual(dataset_dict[dataset_name]["test"].batch_size, 16)

    @mock.patch('utils.DataTSV')
    def test_data_dictionary_huge(self, mock_data):
        """test data dictionary"""

        # Mock values.
        samples = 1600
        mock_data.return_value = CustomDataset(
            samples=samples,
            length=20
        )
        datasets = ["Adiac", "Beef"]

        # Run function.
        dataset_dict = data_dictionary(datasets, verbose=False)

        # Assert tests.
        for dataset_name in dataset_dict:
            self.assertEqual(dataset_dict[dataset_name]["train"].batch_size, 16)
            self.assertEqual(dataset_dict[dataset_name]["test"].batch_size, 16)
