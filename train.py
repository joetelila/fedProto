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
    print(f"Fed proto: {args.fedproto}")
    print(f"proto loss ld: {args.ld}")
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
    
    # initialize global class prototype
    global_proto = {
        #i: torch.zeros(120) for i in range(10)
    }

    # train FL
    for _round in range(args.round):

        
        # inialize round client prototypes.
        # Store client prototypes after training.
        client_protos = {}

        # select csplit clients randomly
        random.seed(args.seed)
        round_clients = random.sample(range(len(clients)), int(args.clients*args.clsplit))
        
        if args.clog:
            print(f"Round {_round} selected clients: {round_clients}")

        # collect round models for averaging if not using fedproto
        if not args.fedproto:
            running_avg = None
        
        # collect client accuracies
        client_test_acc = 0
        client_test_loss = 0

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
            if args.fedproto:
                _client_model_trained, protos = train(args, _client_model, train_loader, global_proto)
                _loss, _acc = test(args, _client_model_trained, testloader)
                client_test_acc += _acc
                client_test_loss += _loss

                # replace client model with trained model
                clients[_client] = _client_model_trained
                
                # collect client prototypes
                for key in protos.keys():
                    if key in client_protos.keys():
                        client_protos[key].append(protos[key])
                    else:
                        client_protos[key] = [protos[key]]
               
            else:
                _client_model_trained = train(args, _client_model, train_loader, global_proto)
                # evaluate the client model
            # add local model parameters to running average
            
            # Running average of the models
            if not args.fedproto:
                running_avg = running_model_avg(running_avg, _client_model_trained.state_dict(), 1/len(round_clients))

            #round_models.append(_client_model_trained)

        # average the client prototypes and update the global prototype
        if args.fedproto:
            for key in client_protos.keys():
                global_proto[key] = torch.stack(client_protos[key]).mean(dim=0)
    
        if args.fedproto:
            print(f"Global round {_round+1} loss: {client_test_loss/len(round_clients)}, accuracy: {client_test_acc/len(round_clients)}")   
        else:
            global_model.load_state_dict(running_avg)
            _loss, _acc = test(args, global_model, testloader)
            print(f"Global round {_round+1} loss: {_loss}, accuracy: {_acc}")

def parse_arguments():
    """
    Parse command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser(description="A brief description of what the script does.")
    
    # Define command-line arguments
    parser.add_argument('-clients', '--clients', default=5, type=str, help='Total number of clients in FL')
    parser.add_argument('-batchsize', '--batchsize', default=32, type=str, help='Total number of clients in FL')
    parser.add_argument('-iid', '--isiid', default=False, type=bool, help='Total number of clients in FL')
    parser.add_argument('-seed', '--seed', default=42, type=bool, help='Total number of clients in FL')
    parser.add_argument('-alpha', '--alpha', default=0.07, type=int, help='Dritchelet alpha value')
    parser.add_argument('-log', '--log', default=True, type=bool, help='log all outputs')
    parser.add_argument('-clog', '--clog', default=False, type=bool, help='client log')
    parser.add_argument('-split', '--split', default=0.2, type=int, help='log all outputs')
    parser.add_argument('-epochs', '--epoch', default=10, type=int, help='total epoch per clients')
    parser.add_argument('-lr', '--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-device', '--device', default='mps', type=str, help='device to train the model')
    parser.add_argument('-round', '--round', default=50, type=int, help='total number of global rounds')
    parser.add_argument('-clsplit', '--clsplit', default=0.99, type=float, help='client split for training')
    parser.add_argument('-data', '--data', default='cifar10', type=str, help='model to train')
    parser.add_argument('-fedproto', '--fedproto', default=True, type=str, help='use federated prototyping')
    parser.add_argument('-ld', '--ld', default=1, type=int, help='lambda value for prototype loss')
   

    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
