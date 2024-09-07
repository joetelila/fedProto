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



# def load_datasets(args, dataset_name: str):
    
#     # Define the transform to apply to the images (e.g., converting to tensors)
#     transform_cifar10 = transforms.Compose(
#         [transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     # Load the training set
#     trainset = torchvision.datasets.CIFAR10(root='project/cifar10', train=True, download=False, transform=transform_cifar10)
#     # Load the test set
#     testset = torchvision.datasets.CIFAR10(root='project/cifar10', train=False, download=False, transform=transform_cifar10)
    
#     # Convert train and test to hugging face datasets
#     train_hf_dataset = Dataset.from_dict(convert_to_hf_dataset(trainset))
#     test_hf_dataset = Dataset.from_dict(convert_to_hf_dataset(testset))
    
#     return train_hf_dataset, test_hf_dataset

# def load_partition(args, train_dataset, partition_id: int):

#     partitioner = IidPartitioner(
#                         num_partitions=args.clients,
#                         #partition_by="label",
#                         ) if args.isiid else DirichletPartitioner(
#                             num_partitions=args.clients,
#                             partition_by="label",
#                             alpha=args.alpha,
#                             seed=args.seed,
#                             min_partition_size=500,
#                             self_balancing=True,
#                             #num_classes_per_partition = 4
#                         )

#     partitioner.dataset = train_dataset
#     partition = partitioner.load_partition(partition_id)

#     trainloader = DataLoader(partition, batch_size=args.batchsize)
  
#     return trainloader #, testloader

def load_partition(args, partition_id: int):

    #fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
    fds = FederatedDataset(
        dataset="cifar10",
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
    client_partition = fds.load_partition(partition_id)

    global_test = fds.load_split("test")


    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    client_partition = client_partition.with_transform(apply_transforms)
    global_test = global_test.with_transform(apply_transforms)

    trainloader = DataLoader(client_partition, batch_size=args.batchsize)

    testloader = DataLoader(global_test, batch_size=args.batchsize)
  
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