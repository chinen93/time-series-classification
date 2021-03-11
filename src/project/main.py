import numpy as np
from time import sleep

# import our modules
import torch
import torch.optim as optim
from utils import *
from train import *
from MultiLayerPerceptron import *
from ConvNet import *
from ResNet import *

def main():
    seed_number = 42
    torch.manual_seed(seed_number)
    np.random.seed(seed_number)

    datasets = np.loadtxt('datasets_small.txt', dtype=str)
    # download_datasets(datasets)  # uncomment this to download the data
    dataset_dictionary = data_dictionary(datasets)

    # training loop
    epochs = 5000
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}\n'.format(device))

    for dataset, dataloader in dataset_dictionary.items():

        # setting up
        print_dataset_info(dataset, dataloader)
        print()
        sleep(1)

        time_steps = dataloader['test'].dataset.inputs.shape[-1]
        n_classes = len(np.unique(dataloader['test'].dataset.targets))

        # MLP
        model = MultiLayerPerceptron(time_steps, n_classes)
        optimizer = optim.Adadelta(model.parameters(), lr=0.1, rho=0.95, eps=1e-8)
        if torch.cuda.device_count() > 0:
            model = nn.DataParallel(model)
        model.to(device)
        model, history = train(dataloader_train=dataloader['train'],
                               dataloader_test=dataloader['test'],
                               device=device,
                               model=model,
                               optimizer=optimizer,
                               epochs=epochs,
                               save=False)
        print('MPCE: {0:.4f}'.format(mpce(model, dataloader['test'], device)))
        print_history(history)

        sleep(1)

        # # ConvNet
        # model = ConvNet(time_steps, n_classes)
        # print(time_steps, n_classes)
        # if torch.cuda.device_count() > 0:
        #     model = nn.DataParallel(model)
        # model.to(device)

        # model, history = train(dataloader['train'],
        #                        dataloader['test'],
        #                        device, model,
        #                        epochs, learning_rate,
        #                        save=False)
        # print('MPCE: {0:.4f}'.format(mpce(model, dataloader['test'], device)))
        # sleep(1)

        # ResNet
        # model = ResNet(time_steps,n_classes)
        # if torch.cuda.device_count() > 0:
        #     model = nn.DataParallel(model)
        # model.to(device)
        # model, history = train(dataloader['train'],
        #                        dataloader['test'],
        #                        device, model,
        #                        epochs, learning_rate,
        #                        save=False)
        # print('MPCE: {0:.4f}'.format(mpce(model,dataloader['test'],device)))

        print()

if __name__ == "__main__":
    main()
else:
    raise "This script should be called as a single program"
