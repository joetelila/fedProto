import argparse
import logging

from flwr_datasets import FederatedDataset
from flwr_datasets.visualization import plot_label_distributions

#torch
import torch
from torchvision import transforms

#utils
#from utils.utils_vis import *
from utils.data_utils import *
from utils.train_utils import *

def main():

    # collect args
    args = parse_arguments()
    
    # load datasets for client 1
    train_loader, test_loader = load_datasets(args, 1)

    # length of train and test data
    #print("Train data length: ", len(train_loader))
    #print("Test data length: ", len(test_loader))

    



def parse_arguments():
    """
    Parse command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser(description="A brief description of what the script does.")
    
    # Define command-line arguments
    parser.add_argument('-clients', '--clients', default=10, type=str, help='Total number of clients in FL')
    parser.add_argument('-batchsize', '--batchsize', default=32, type=str, help='Total number of clients in FL')
    parser.add_argument('-iid', '--isiid', default=False, type=bool, help='Total number of clients in FL')
    parser.add_argument('-seed', '--seed', default=42, type=bool, help='Total number of clients in FL')
    parser.add_argument('-alpha', '--alpha', default=0.08, type=int, help='Dritchelet alpha value')
    parser.add_argument('-log', '--log', default=True, type=bool, help='log all outputs')
    parser.add_argument('-split', '--split', default=0.2, type=int, help='log all outputs')
    parser.add_argument('-epochs', '--epoch', default=10, type=int, help='total epoch per clients')

    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()