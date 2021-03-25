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

def run_train_models(datasets, parameters):
    device = parameters["device"]
    for dataset, dataloader in datasets.items():

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
                                   epochs=parameters["mlp_epochs"],
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
                                   epochs=parameters["fcn_epochs"],
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
                                   epochs=parameters["fcn_epochs"],
                                   save=False)


def run_experiments(datasets, parameters):

    # Populate tags with some info.
    tags = parameters['tags']
    if parameters['run_mlp']:
        tags.append("MLP")
    if parameters['run_fcn']:
        tags.append("FCN")
    if parameters['run_resnet']:
        tags.append("ResNet")

    # Create Neptune client.
    neptune.init(project_qualified_name='pedro-chinen/time-series-classification')
    neptune.create_experiment(
        upload_source_files=[],
        params=parameters,
        tags=tags
    )
    neptune.log_artifact("ConvNet.py")
    neptune.log_artifact("MultiLayerPerceptron.py")
    neptune.log_artifact("ResNet.py")

    try:
        run_train_models(datasets, parameters)
    except KeyboardInterrupt:
        pass

    neptune.stop()

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}\n'.format(device))

    # Parameters:
    parameters = {
        "tags": ['scheduler'],
        "seed_number": 42,
        "device": device,
        "run_mlp": True,
        "run_fcn": True,
        "run_resnet": True,
        "mlp_epochs": 5000,
        "mlp_lr": 0.001,
        "mlp_rho": 0.95,
        "mlp_eps": 1e-7,
        "fcn_epochs": 2000,
        "fcn_lr": 0.001,
        "fcn_betas": (0.9, 0.999),
        "fcn_eps": 1e-7
    }

    datasets = np.loadtxt('datasets.txt', dtype=str)
    # download_datasets(datasets)  # uncomment this to download the data
    dataset_dictionary = data_dictionary(datasets)

    seed = parameters["seed_number"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    run_experiments(
        datasets=dataset_dictionary,
        parameters=parameters
    )

if __name__ == "__main__":
    main()
else:
    raise "This script should be called as a single program"
