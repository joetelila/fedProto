import argparse
import logging
import random
import copy

from flwr_datasets import FederatedDataset
from flwr_datasets.visualization import plot_label_distributions

#torch
import torch
from torchvision import transforms

#utils
#from utils.utils_vis import *
from utils.data_utils import *
from utils.train_utils import *
from utils.model import *

def main():

    # collect args
    args = parse_arguments()
    
    # print general info about the experiment
    print(f"Dataset: {args.data}")
    print(f"Total number of clients: {args.clients}")
    print(f"Total number of global rounds: {args.round}")
    print(f"Local epochs: {args.epoch}")
    print(f"Batch size: {args.batchsize}")
    print(f"learning rate: {args.lr}")
    print(f"device: {args.device}")
    print(f"seed: {args.seed}")
    print(f"alpha: {args.alpha}")
    print(f"iid: {args.isiid}")
    print(f"split: {args.split}")
    print(f"clsplit: {args.clsplit}")


    
    # global model
    match args.data:
        case "mnist":
            global_model = MnistNet().to(args.device)
        case "cifar10":
            global_model = Cifar10Net().to(args.device)
        case _:
            raise ValueError(f"Unknown model: {args.data}")

    # initalize client models with global model
    clients = [global_model for _ in range(args.clients)]

    # train FL
    for _round in range(args.round):

        # select csplit clients randomly
        random.seed(args.seed)
        round_clients = random.sample(range(len(clients)), int(args.clients*args.clsplit))
        
        if args.clog:
            print(f"Round {_round} selected clients: {round_clients}")

        # collect round models for averaging if not using fedproto
        if not args.fedproto:
            running_avg = None
        
        # train the selected clients
        for _client in round_clients:
            
            if args.clog:
                print(f"Training client {_client}")
            
            _client_model = copy.deepcopy(clients[_client])
            _client_model = set_parameters(_client_model, get_parameters(global_model))
            # check if the global model is correctly set to client model
            if not are_models_equal(_client_model, global_model):
                print("Global model is not correctly set to client model")
                exit(0)
    
            match args.data:
                case "mnist":
                    train_loader, testloader = load_mnist_partition(args, _client)
                case "cifar10":
                    train_loader, testloader = load_cifar10_partition(args, _client)
                case _:
                    raise ValueError(f"Unknown dataset: {args.data}")
        
            # train the client model
            _client_model_trained = train(args, _client_model, train_loader)
            # add local model parameters to running average
            
            if not args.fedproto:
                running_avg = running_model_avg(running_avg, _client_model_trained.state_dict(), 1/len(round_clients))

            #round_models.append(_client_model_trained)

        # average the models
        #round_average_model = fed_average(round_models)
        if not args.fedproto:
            global_model.load_state_dict(running_avg)
            _loss, _acc = test(args, global_model, testloader)
            print(f"Global round {_round+1} loss: {_loss}, accuracy: {_acc}")

def parse_arguments():
    """
    Parse command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser(description="A brief description of what the script does.")
    
    # Define command-line arguments
    parser.add_argument('-clients', '--clients', default=10, type=str, help='Total number of clients in FL')
    parser.add_argument('-batchsize', '--batchsize', default=32, type=str, help='Total number of clients in FL')
    parser.add_argument('-iid', '--isiid', default=True, type=bool, help='Total number of clients in FL')
    parser.add_argument('-seed', '--seed', default=42, type=bool, help='Total number of clients in FL')
    parser.add_argument('-alpha', '--alpha', default=0.07, type=int, help='Dritchelet alpha value')
    parser.add_argument('-log', '--log', default=True, type=bool, help='log all outputs')
    parser.add_argument('-clog', '--clog', default=False, type=bool, help='client log')
    parser.add_argument('-split', '--split', default=0.2, type=int, help='log all outputs')
    parser.add_argument('-epochs', '--epoch', default=10, type=int, help='total epoch per clients')
    parser.add_argument('-lr', '--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-device', '--device', default='mps', type=str, help='device to train the model')
    parser.add_argument('-round', '--round', default=20, type=int, help='total number of global rounds')
    parser.add_argument('-clsplit', '--clsplit', default=0.99, type=float, help='client split for training')
    parser.add_argument('-data', '--data', default='cifar10', type=str, help='model to train')
    parser.add_argument('-fedproto', '--fedproto', default=True, type=str, help='use federated prototyping')
   

    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
