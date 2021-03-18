import numpy as np
from time import sleep

import neptune

# import our modules
import torch
import torch.optim as optim
from utils import *
from train import *
from MultiLayerPerceptron import *
from ConvNet import *
from ResNet import *

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}\n'.format(device))

    # Parameters:
    parameters = {
        "epochs": 1000,
        "seed_number": 42,
        "device": device,
        "run_mlp": False,
        "run_fcn": True,
        "run_resnet": False,
        "mlp_lr": 0.1,
        "mlp_rho": 0.95,
        "mlp_eps": 1e-8,
        "fcn_lr": 0.001,
        "fcn_betas": (0.9, 0.999),
        "fcn_eps": 1e-8
    }

    torch.manual_seed(parameters["seed_number"])
    np.random.seed(parameters["seed_number"])

    datasets = np.loadtxt('datasets.txt', dtype=str)
    # download_datasets(datasets)  # uncomment this to download the data
    dataset_dictionary = data_dictionary(datasets)

    # Create Neptune client.
    neptune.init(project_qualified_name='pedro-chinen/time-series-classification')
    neptune.create_experiment(
        upload_source_files=[],
        params=parameters,
        tags=[
            "FCN"
        ]
    )
    neptune.log_artifact("ConvNet.py")
    neptune.log_artifact("MultiLayerPerceptron.py")
    neptune.log_artifact("ResNet.py")

    for dataset, dataloader in dataset_dictionary.items():

        # setting up
        print_dataset_info(dataset, dataloader)
        sleep(1)

        time_steps = dataloader['test'].dataset.inputs.shape[-1]
        n_classes = len(np.unique(dataloader['test'].dataset.targets))

        # MLP
        if parameters["run_mlp"]:
            print("MLP")
            model = MultiLayerPerceptron(time_steps, n_classes)
            optimizer = optim.Adadelta(
                model.parameters(),
                lr=parameters["mlp_lr"],
                rho=parameters["mlp_rho"],
                eps=parameters["mlp_eps"]
            )
            if torch.cuda.device_count() > 0:
                model = nn.DataParallel(model)
            model.to(device)
            model, history = train(model_name="MLP",
                                   dataset_name=dataset,
                                   dataloader_train=dataloader['train'],
                                   dataloader_test=dataloader['test'],
                                   device=device,
                                   model=model,
                                   optimizer=optimizer,
                                   epochs=parameters["epochs"],
                                   save=False)

        # ConvNet
        if parameters["run_fcn"]:
            print("FCN")
            model = ConvNet(time_steps, n_classes)
            if torch.cuda.device_count() > 0:
                model = nn.DataParallel(model)
            model.to(device)

            optimizer = optim.Adam(
                model.parameters(),
                lr=parameters["fcn_lr"],
                betas=parameters["fcn_betas"],
                eps=parameters["fcn_eps"]
            )
            model, history = train(model_name="FCN",
                                   dataset_name=dataset,
                                   dataloader_train=dataloader['train'],
                                   dataloader_test=dataloader['test'],
                                   device=device,
                                   model=model,
                                   optimizer=optimizer,
                                   epochs=parameters["epochs"],
                                   save=False)

        # ResNet
        if parameters["run_resnet"]:
            print("ResNet")
            model = ResNet(time_steps, n_classes)
            if torch.cuda.device_count() > 0:
                model = nn.DataParallel(model)
            model.to(device)
            optimizer = optim.Adam(
                model.parameters(),
                lr=parameters["fcn_lr"],
                betas=parameters["fcn_betas"],
                eps=parameters["fcn_eps"]
            )
            model, history = train(model_name="ResNet",
                                   dataset_name=dataset,
                                   dataloader_train=dataloader['train'],
                                   dataloader_test=dataloader['test'],
                                   device=device,
                                   model=model,
                                   optimizer=optimizer,
                                   epochs=parameters["epochs"],
                                   save=False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
else:
    raise "This script should be called as a single program"
