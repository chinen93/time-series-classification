"""
Unittest for the dataset
"""
import unittest
import mock
import torch
from torch.utils.data import Dataset
import pandas
import numpy as np

from data import DataTSV

class TestDataTSV(unittest.TestCase):
    """
    Class to test data TSV
    """

    @mock.patch("data.pd")
    def test_create_dataset(self, mock_pandas):
        """Test create dataset"""

        # Mock values.
        samples = 10
        length = 201
        data = np.random.randint(
            low=10,
            high=30,
            size=(samples, length)
        )
        data = data.astype(float)
        mock_pandas.read_csv.return_value = pandas.DataFrame(data)

        # Run function.
        dataset = DataTSV(dataset="TRAIN_DATASET", testing=False)

        # Assert tests.
        self.assertEqual(len(dataset), samples)
        self.assertEqual(len(dataset.inputs[0]), 200)

    @mock.patch("data.pd")
    def test_create_invalid_nan_dataset(self, mock_pandas):
        """Test create invalid nan dataset"""

        # Mock values.
        samples = 10
        length = 201
        data = np.random.randint(
            low=10,
            high=30,
            size=(samples, length)
        )
        data = data.astype(float)
        data[0, 0] = np.nan
        mock_pandas.read_csv.return_value = pandas.DataFrame(data)

        # Run function.
        for testing in [True, False]:
            with self.assertRaises(ValueError):
                DataTSV(dataset="TRAIN_DATASET", testing=testing)

    @mock.patch("data.pd")
    def test_create_invalid_std_dataset(self, mock_pandas):
        """Test create invalid std dataset"""

        # Mock values.
        samples = 10
        length = 201
        data = np.ones((samples, length)) * 50
        data = data.astype(float)
        mock_pandas.read_csv.return_value = pandas.DataFrame(data)

        # Run function.
        for testing in [True, False]:
            with self.assertRaises(ValueError):
                DataTSV(dataset="TRAIN_DATASET", testing=testing)
