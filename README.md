# FedProto

This repository contains an implementation of the **FedProto** algorithm, based on the paper _"FedProto: Federated Prototype Learning for Efficient Federated Learning"_.

## Overview

Federated learning (FL) is a distributed machine learning approach where data remains decentralized across multiple clients. The clients collaborate to train a model without sharing their local datasets, preserving data privacy. 

FedProto introduces **Federated Prototype Learning** by incorporating prototype-based learning for efficient and privacy-preserving federated learning.

## Features

- **Federated Learning** with multiple clients.
- **Prototype Learning** for more efficient federated model updates.
- Customizable command-line arguments for various configurations, including the number of clients, epochs, learning rate, and device options.
- Supports popular datasets like CIFAR-10 and allows usage of different models.
  
## Requirements

- Python > 3.10.1
- PyTorch

You can install the required packages by running:
```bash
pip install -r requirements.txt 
```
## How to Use

### Command-line Arguments
The script supports various command-line arguments for configuration. You can specify these directly in the terminal when running the script. Here are the available arguments:

| Argument | Short Form | Description | Default Value |
|----------|------------|-------------|---------------|
| `--clients` | `-clients` | Total number of clients in federated learning | 10 |
| `--batchsize` | `-batchsize` | Batch size for training | 32 |
| `--iid` | `-iid` | Use IID or non-IID data distribution | False |
| `--seed` | `-seed` | Random seed | 42 |
| `--alpha` | `-alpha` | Dirichlet alpha value for data split | 0.07 |
| `--log` | `-log` | Enable logging | True |
| `--clog` | `-clog` | Enable client-specific logging | True |
| `--split` | `-split` | Train/test split ratio | 0.2 |
| `--epochs` | `-epoch` | Number of training epochs per client | 10 |
| `--lr` | `-lr` | Learning rate | 0.001 |
| `--device` | `-device` | Device to train the model (e.g., `mps` for Apple Silicon, `cuda` for GPUs) | "mps" |
| `--round` | `-round` | Total number of global training rounds | 20 |
| `--clsplit` | `-clsplit` | Client split ratio for training | 0.99 |
| `--data` | `-data` | Dataset to use (e.g., `cifar10, mnist`) | 'cifar10' |
| `--fedproto` | `-fedproto` | Enable federated prototyping | True |
| `--ld` | `-ld` | Lambda value for prototype loss | 0.8 |

### Example Usage
```bash
python script.py --clients 20 --batchsize 64 --lr 0.0005
