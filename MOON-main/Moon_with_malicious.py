import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random

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
    parser.add_argument('--malicious_fraction', type=float, default=0.2, help='Fraction of malicious clients')
    args = parser.parse_args()
    return args


def select_malicious_clients(n_parties, malicious_fraction):
    n_malicious = int(n_parties * malicious_fraction)
    return random.sample(range(n_parties), n_malicious)


def poison_data(data, n_classes):
    x, target = data
    return x, (target + 1) % n_classes 


def poison_model_weights(model):
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.1) 


def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    n_classes = 100 if args.dataset == 'cifar100' else 10  

    for net_i in range(n_parties):
        if args.use_project_head:
            net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs)
        else:
            net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs)
        nets[net_i] = net.to(device)

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def train_net_fedcon_malicious(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, temperature, args, round, device="cpu"):
    net = nn.DataParallel(net)
    net.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=args.reg)
    cos = torch.nn.CosineSimilarity(dim=-1)

    is_malicious = net_id in malicious_clients

    for epoch in range(epochs):
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            if is_malicious:
                x, target = poison_data((x, target), n_classes)

            optimizer.zero_grad()
            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)

            posi = cos(pro1, pro2)
            logits = posi.reshape(-1, 1)

            for previous_net in previous_nets:
                _, pro3, _ = previous_net(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

            logits /= temperature
            labels = torch.zeros(x.size(0)).cuda().long()

            loss2 = mu * criterion(logits, labels)
            loss1 = criterion(out, target)
            loss = loss1 + loss2

            if is_malicious:
                poison_model_weights(net)

            loss.backward()
            optimizer.step()


def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model=None, prev_model_pool=None, round=None, device="cpu"):
    avg_acc = 0.0
    acc_list = []
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)

        if args.alg == 'moon':
            prev_models = [prev_model_pool[i][net_id] for i in range(len(prev_model_pool))]
            train_net_fedcon_malicious(net_id, net, global_model, prev_models, train_dl_local, test_dl, args.epochs, args.lr, args.optimizer, args.mu, args.temperature, args, round, device)

        logger.info(f"Net {net_id} final test accuracy: {avg_acc}")


if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    device = torch.device(args.device)

    np.random.seed(args.init_seed)
    torch.manual_seed(args.init_seed)
    random.seed(args.init_seed)

    logger = logging.getLogger()
    logging.basicConfig(filename=os.path.join(args.logdir, 'log.txt'), level=logging.INFO)

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_classes = len(np.unique(y_train))
    malicious_clients = select_malicious_clients(args.n_parties, args.malicious_fraction)
    logger.info(f"Malicious Clients: {malicious_clients}")

    nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
    global_models, _, _ = init_nets(args.net_config, 1, args, device='cpu')
    global_model = global_models[0]
    old_nets_pool = []

    for round in range(args.comm_round):
        logger.info(f"Communication Round: {round}")
        global_w = global_model.state_dict()
        party_list_this_round = random.sample(range(args.n_parties), int(args.sample_fraction * args.n_parties))
        nets_this_round = {k: nets[k] for k in party_list_this_round}

        for net in nets_this_round.values():
            net.load_state_dict(global_w)

        local_train_net(nets_this_round, args, net_dataidx_map, train_dl=None, test_dl=None, global_model=global_model, prev_model_pool=old_nets_pool, round=round, device=device)

        total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    global_w[key] = net_para[key] * fed_avg_freqs[net_id]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * fed_avg_freqs[net_id]

        global_model.load_state_dict(global_w)
        old_nets_pool.append(copy.deepcopy(nets_this_round))
