"""
Unittest for main functions
"""
import unittest
import pandas
import numpy as np

import mock
from main import *

from utils import data_dictionary


class TestMain(unittest.TestCase):
    """
    Class to test main functions
    """

    @mock.patch('main.neptune')
    @mock.patch('main.train')
    def test_run_train_models(self, mock_train, mock_neptune):


        # Mock Variables
        best_error_rate = 0.5
        model = None
        history = []
        mock_train.return_value = (
            best_error_rate,
            model,
            history
        )

        parameters = {
            "verbose": False,
            "neptune_project": 'pedro-chinen/time-series-classification',
            "tags": ['Local', 'scheduler'],
            "seed_number": 42,
            "device": "cpu",
            "run_mlp": True,
            "run_fcn": True,
            "run_resnet": True,
            "mlp_epochs": 5000,
            "mlp_lr": 0.1,
            "mlp_rho": 0.95,
            "mlp_eps": 1e-8,
            "fcn_epochs": 2000,
            "fcn_lr": 0.001,
            "fcn_betas": (0.9, 0.999),
            "fcn_eps": 1e-8
        }

        # Dataset
        num_classes = 5
        length = 200
        train_samples = 180
        train_data = np.random.randint(
            low=10,
            high=30,
            size=(train_samples, length)
        )
        train_target = np.random.randint(
            low=0,
            high=num_classes,
            size=(train_samples, 1)
        )
        train_data = np.concatenate((train_target, train_data), axis=1)
        train_data = train_data.astype(float)

        datasets = ["DATASET_NAME_1"]
        with mock.patch('data.pd') as mock_pandas:
            mock_pandas.read_csv.return_value = pandas.DataFrame(train_data)
            datasets = data_dictionary(datasets=datasets, verbose=False)

        # Run function
        run_train_models(datasets, parameters)

        # Assert tests.
        mock_neptune.log_metric.assert_any_call(
            "MLP_mpce",
            best_error_rate / num_classes
        )
        mock_neptune.log_metric.assert_any_call(
            "FCN_mpce",
            best_error_rate / num_classes
        )
        mock_neptune.log_metric.assert_any_call(
            "ResNet_mpce",
            best_error_rate / num_classes
        )
