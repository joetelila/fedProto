# flower
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions
from flwr_datasets import FederatedDataset
from flwr_datasets.preprocessor import Merger



# torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torch

# utils
import pandas as pd
import numpy as np

# huggingface dataset
from datasets import Dataset
from datasets import concatenate_datasets

def load_mnist_partition(args, partition_id: int):

    # If fedproto; combine the test and train set and split
    merger = Merger(
            merge_config={
                "train": ("train", "test"),
                })
    
    #fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
    fds = FederatedDataset(
        dataset="ylecun/mnist",
        preprocessor= merger if args.fedproto else None,
        partitioners={
            "train": IidPartitioner(
                        num_partitions=args.clients,
                        #partition_by="label",
                    ) if args.isiid else DirichletPartitioner(
                        num_partitions=args.clients,
                        partition_by="label",
                        alpha=args.alpha,
                        seed=args.seed,
                        min_partition_size=500,
                        self_balancing=True,
                        #num_classes_per_partition = 4
                    ),
        },
    )

    client_train = fds.load_partition(partition_id, split="train")
    partition_train_test = client_train.train_test_split(test_size=args.split, seed=args.seed)
    
    if args.fedproto:
        client_train = partition_train_test["train"]
        client_test = partition_train_test["test"]
    else:
        client_test = fds.load_split("test")

    # pytorch_transforms = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )
    pytorch_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def apply_transforms(batch):
       batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
       return batch
    # Apply the transforms
    client_train = client_train.with_transform(apply_transforms)
    client_test = client_test.with_transform(apply_transforms)

    # Prepare the DataLoader
    trainloader = DataLoader(client_train, batch_size=args.batchsize)
    testloader = DataLoader(client_test, batch_size=args.batchsize)
  
    return trainloader, testloader



def load_cifar10_partition(args, partition_id: int):

    merger = Merger(
            merge_config={
                "train": ("train", "test"),
                })
    
    fds = FederatedDataset(
        dataset="cifar10",
        preprocessor= merger if args.fedproto else None,
        partitioners={
            "train": IidPartitioner(
                        num_partitions=args.clients,
                        #partition_by="label",
                    ) if args.isiid else DirichletPartitioner(
                        num_partitions=args.clients,
                        partition_by="label",
                        alpha=args.alpha,
                        seed=args.seed,
                        min_partition_size=500,
                        self_balancing=True
                    ),
        },
    )

    client_train = fds.load_partition(partition_id, split="train")
    
    partition_train_test = client_train.train_test_split(test_size=args.split, seed=args.seed)
    
    if args.fedproto:
        client_train = partition_train_test["train"]
        client_test = partition_train_test["test"]
    else:
        client_test = fds.load_split("test")


    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Apply the transforms
    client_train = client_train.with_transform(apply_transforms)
    client_test = client_test.with_transform(apply_transforms)

    # Prepare the DataLoader
    trainloader = DataLoader(client_train, batch_size=args.batchsize)
    testloader = DataLoader(client_test, batch_size=args.batchsize)
  
    return trainloader, testloader

def are_models_equal(model1, model2):
    # Compare the state_dict of both models
    for param1, param2 in zip(model1.state_dict().values(), model2.state_dict().values()):
        if not torch.equal(param1, param2):
            return False
    return True


def convert_to_hf_dataset(pytorch_dataset):
    data_dict = {
        "img": [],
        "label": []
    }
    
    for image, label in pytorch_dataset:
        data_dict["img"].append(image.clone().detach())  # Convert to tensor
        data_dict["label"].append(label)
    
    # Stack images and labels into the desired shape
    data_dict["img"] = torch.stack(data_dict["img"], dim=0)  # Shape: [32, 3, 32, 32]
    data_dict["label"] = torch.tensor(data_dict["label"])     # Shape: [32]ff

    return data_dict