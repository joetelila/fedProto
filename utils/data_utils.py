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

def load_local_datasets(args, partition_id: int):
    
    # Define the transform to apply to the images (e.g., converting to tensors)
    _transforms = transforms.Compose(
            [transforms.ToTensor()]#, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    # Load the training set
    trainset = torchvision.datasets.CIFAR10(root='project/cifar10', train=True, download=False, transform=_transforms)

    # Load the test set
    testset = torchvision.datasets.CIFAR10(root='project/cifar10', train=False, download=False, transform=_transforms)

    # Convert train and test datasets
    train_hf_dataset = convert_to_hf_dataset(trainset)
    test_hf_dataset = convert_to_hf_dataset(testset)

    # Combine train and test datasets
    combined_dataset = concatenate_datasets([train_hf_dataset, test_hf_dataset])
    # Shuffle the combined dataset if desired
    combined_dataset = combined_dataset.shuffle(seed=args.seed)

    partitioner = IidPartitioner(
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
                        )

    partitioner.dataset = combined_dataset
    partition = partitioner.load_partition(partition_id)

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=args.split, seed=args.seed)
    
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=args.batchsize, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=args.batchsize)
  
    return trainloader, testloader

def load_datasets(args, partition_id: int):


    # merge train and test data
    merger = Merger(
    merge_config={
        "train_g": ("train", "test"),
        }
    ) 
    #fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
    fds = FederatedDataset(
        dataset="cifar10",
        preprocessor = merger,
        partitioners={
            "train_g": IidPartitioner(
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

    # # visualize partitioned data
    # partitioner_train = fds.partitioners["train_g"]
    # fig, ax, df = plot_label_distributions(
    #     partitioner_train,
    #     label_name="label",

    #     plot_type="bar",
    #     size_unit="absolute",
    #     partition_id_axis="x",
    #     legend=True,
    #     verbose_labels=True,
    #     # in the title include args.clients value
    #     title="Per Partition Labels Distribution; alpha="+str(args.alpha),
    # )
    # # logging
    # if args.log:
    #     # saving the plot
    #     fig.savefig("plots/label_distribution_alpha_"+str(args.alpha)+".png")
    #     # print label distribution
    #     for partition_id in range(args.clients):    
    #         partition = fds.load_partition(partition_id)
    #         unique_labels = partition.unique("label")
    #         print(f"Partition {partition_id}: {unique_labels}")


    partition = fds.load_partition(partition_id)

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=args.split, seed=args.seed)

    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=args.batchsize, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=args.batchsize)
  
    return trainloader, testloader


def are_models_equal(model1, model2):
    # Compare the state_dict of both models
    for param1, param2 in zip(model1.state_dict().values(), model2.state_dict().values()):
        if not torch.equal(param1, param2):
            return False
    return True


def convert_to_hf_dataset(torch_dataset):
    # Extract images and labels from the PyTorch dataset
    images = np.array([torch_dataset[i][0].numpy() for i in range(len(torch_dataset))])
    labels = np.array([torch_dataset[i][1] for i in range(len(torch_dataset))])

    # Create a Pandas DataFrame
    df = pd.DataFrame({
        'img': images,
        'label': labels
    })
    # Convert the DataFrame to a Hugging Face Dataset
    hf_dataset = Dataset.from_dict(df)
    return hf_dataset