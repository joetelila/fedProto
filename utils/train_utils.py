
import torch

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch.nn.functional as F



def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net


def train(args, net, trainloader, global_proto=None):
    
    """
    Train the network on the training set.
    
    """
    
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    for epoch in range(args.epoch):
        
        correct, total, epoch_loss = 0, 0, 0.0

        # initialize client prototypes
        client_protos = {}

        for batch in trainloader:
            match args.data:
                case "mnist":
                    images, labels = batch["image"], batch["label"]
                case "cifar10":
                    images, labels = batch["img"], batch["label"]
                case _:
                    raise ValueError(f"Unknown dataset: {args.dataset}")
            
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs, proto = net(images)
            
            # loss is the sum of the classification loss and the prototype loss
            loss1 = criterion(outputs, labels)
            if len(global_proto.keys()) == 0:
                loss2 = 0*loss1
            else:
                batch_global_proto = []
                for label in labels:
                    if label.item() in global_proto.keys():
                        # get the global prototype for the current label.
                        label_proto = global_proto[label.item()] 
                        batch_global_proto.append(label_proto)
                    else:
                        # if the prototype for the current label is not available, use zero vector.
                        batch_global_proto.append(torch.zeros(len(proto[0])))
                batch_global_proto = torch.stack(batch_global_proto)
                
                # compute loss 2
                loss2 = F.mse_loss(proto, batch_global_proto.to(args.device))

            # total loss
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            # collect client prototypes on the last epoch only to save computation time.
            if epoch == args.epoch - 1:
                for i, label in enumerate(labels):
                    if label.item() in client_protos.keys():
                        client_protos[label.item()].append(proto[i].detach().cpu())
                    else:
                        client_protos[label.item()] = [proto[i].detach().cpu()]
                
            
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
    
           
        if args.clog:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    
    # average the client prototypes
    for key in client_protos.keys():
        client_protos[key] = torch.stack(client_protos[key]).mean(dim=0)
    
    if args.fedproto:
        return net, client_protos
    else:
        return net


def test(args, net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            match args.data:
                case "mnist":
                    images, labels = batch["image"], batch["label"]
                case "cifar10":
                    images, labels = batch["img"], batch["label"]
                case _:
                    raise ValueError(f"Unknown dataset: {args.dataset}")
            images, labels = images.to(args.device), labels.to(args.device)
            outputs, protos = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    if args.clog:
        print(f"Test loss {loss}, accuracy {accuracy}")
    return loss, accuracy


def fed_average(models):
    """Compute the average of the given models."""
    # Use the state_dict() method to get the parameters of a model
    avg_state = sum([model.state_dict() for model in models]) / len(models)
    return avg_state

def running_model_avg(current, next, scale):
    if current == None:
        current = next
        for key in current:
            current[key] = current[key] * scale
    else:
        for key in current:
            current[key] = current[key] + (next[key] * scale)
    return current