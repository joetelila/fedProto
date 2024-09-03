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
    
    # initalize client models
    clients = [Net().to(args.device) for _ in range(args.clients)]
    
    global_model = Net().to(args.device)

    # train FL
    for _round in range(args.round):
        # select 5 clients randomly
        round_clients = random.sample(range(len(clients)), int(args.clients*args.clsplit))
        print(f"Round {_round} selected clients: {round_clients}")
        loss, acc = 0, 0
        
        # collect round models for averaging
        round_models = []

        # train the selected clients
        for _client in round_clients:
            print(f"Training client {_client}")

            _client_model = copy.deepcopy(clients[_client])
            _client_model = set_parameters(_client_model, get_parameters(global_model))
            
            # check if the global model is correctly set to client model
            if not are_models_equal(_client_model, global_model):
                print("Global model is not correctly set to client model")
                exit(0)
    
            
            train_loader, test_loader = load_datasets(args, _client)
            _client_model_trained = train(args, _client_model, train_loader)

            _loss, _acc = test(args, _client_model_trained, test_loader)
            print(f"Client {_client} loss: {_loss}, accuracy: {_acc}")
            round_models.append(_client_model_trained)
            
            # collect loss and accuracy
            loss += _loss
            acc += _acc
        
        # average the models
        round_average_model = fed_average(round_models)
        global_model = set_parameters(global_model, round_average_model)

        print(f"Round {_round} loss: {loss/len(round_clients)}, accuracy: {acc/len(round_clients)}")
        


    



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
    parser.add_argument('-alpha', '--alpha', default=0.08, type=int, help='Dritchelet alpha value')
    parser.add_argument('-log', '--log', default=True, type=bool, help='log all outputs')
    parser.add_argument('-clog', '--clog', default=True, type=bool, help='client log')
    parser.add_argument('-split', '--split', default=0.2, type=int, help='log all outputs')
    parser.add_argument('-epochs', '--epoch', default=10, type=int, help='total epoch per clients')
    parser.add_argument('-lr', '--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-device', '--device', default='mps', type=str, help='device to train the model')
    parser.add_argument('-round', '--round', default=1, type=int, help='total number of global rounds')
    parser.add_argument('-clsplit', '--clsplit', default=0.8, type=float, help='client split for training')

    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
