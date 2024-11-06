import numpy as np
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
from torch.autograd import grad

from model import *
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox/moon')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication rounds')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    parser.add_argument('--malicious_fraction', type=float, default=0.2, help='Fraction of clients behaving maliciously')
    args = parser.parse_args()
    return args


# Malicious client selection and poisoning behavior
def select_malicious_clients(n_parties, malicious_fraction):
    n_malicious = int(n_parties * malicious_fraction)
    return random.sample(range(n_parties), n_malicious)


def poison_data(data, n_classes):
    x, target = data
    return x, (target + 1) % n_classes  # Label flipping


def poison_model_weights(model):
    with torch.no_grad():
        for param in model.parameters():
            param.data += torch.randn_like(param.data) * 0.1  # Add random noise


def train_net_malicious(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device, is_malicious=False):
    criterion = nn.CrossEntropyLoss().cuda()
    net = nn.DataParallel(net)
    net.cuda()
    logger.info(f"Training network {net_id} (Malicious: {is_malicious})")

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    for epoch in range(epochs):
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            if is_malicious:
                x, target = poison_data((x, target), args.out_dim)  # Apply data poisoning

            optimizer.zero_grad()
            _, _, out = net(x)
            loss = criterion(out, target)

            if is_malicious:
                poison_model_weights(net)  # Apply model poisoning

            loss.backward()
            optimizer.step()

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, _, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info(f"Final accuracy (net_id={net_id}): Train={train_acc}, Test={test_acc}")
    return train_acc, test_acc


# Simulation and likelihood for client updates
def likelihood_function(x, y):
    product_sigmoid_x = torch.prod(torch.sigmoid(x))
    product_sigmoid_y = torch.prod(torch.sigmoid(1 - y))
    return product_sigmoid_x * product_sigmoid_y


def simul(prev_clients, curr_clients, party_weights, learning_rate):
    x_client = [element for element in curr_clients if element not in prev_clients]
    y_client = [element for element in prev_clients if element not in curr_clients]

    x_values = [party_weights[element] for element in x_client]
    y_values = [party_weights[element] for element in y_client]

    x = torch.tensor(x_values, dtype=torch.float32, requires_grad=True)
    y = torch.tensor(y_values, dtype=torch.float32, requires_grad=True)

    epc = 100
    while epc > 0:
        output = likelihood_function(x, y)
        prev_val = output.item()
        output.backward()

        x.data = x.data + learning_rate * x.grad.data
        y.data = y.data + learning_rate * y.grad.data

        x.grad.zero_()
        y.grad.zero_()

        # Update party_weights
        prev_party_weights = party_weights.copy()
        for i, element in enumerate(x_client):
            party_weights[element] = x[i].item()
        
        for i, element in enumerate(y_client):
            party_weights[element] = y[i].item()

        party_weights = np.array(party_weights)  
        for i in range(len(party_weights)):
            party_weights[i] = np.exp(party_weights[i])
            
        party_weights /= party_weights.sum() 

        output1 = likelihood_function(x, y)
        if output1.item() < prev_val:
            party_weights = prev_party_weights
            break

        epc -= 1

    return party_weights

if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
    global_model = ModelFedCon(args.model, args.out_dim, len(np.unique(y_train)), args.net_config).to(args.device)

    party_weights = np.array([1 / args.n_parties for _ in range(args.n_parties)], dtype='float32')
    malicious_clients = select_malicious_clients(args.n_parties, args.malicious_fraction)
    logger.info(f"Malicious clients: {malicious_clients}")

    prev_acc = -1
    prev_round = []
    learning_rate = 0.01

    for round in range(args.comm_round):
        party_weights = party_weights / np.sum(party_weights)
        party_list_this_round = np.random.choice(range(args.n_parties), size=min(int(args.sample_fraction * args.n_parties), len(party_weights)), replace=False, p=party_weights)

        for net_id in party_list_this_round:
            is_malicious = net_id in malicious_clients
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, net_dataidx_map[net_id])

            train_net_malicious(net_id, nets[net_id], train_dl_local, test_dl_local, args.epochs, args.lr, args.optimizer, args, args.device, is_malicious=is_malicious)

        # Simulate likelihood updates for party weights
        if prev_round:
            party_weights = simul(prev_round, party_list_this_round, party_weights, learning_rate)

        prev_round = party_list_this_round
