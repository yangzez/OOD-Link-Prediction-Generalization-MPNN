import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
#from torch_geometric.nn import GATConv
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx
import random
from torch_geometric.utils import negative_sampling
from torch.utils.data import DataLoader
from ogb.linkproppred import Evaluator
import numpy as np
import argparse
import utils

from friendster import Friendster
from torch_sparse import SparseTensor
import networkx as nx

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Training for MNIST')
    parser.add_argument('-e', '--n_epochs', type=int, default=200,
                        help='number of epochs (DEFAULT: 100)')
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-4,
                        help='learning rate for gradient descent (DEFAULT: 1e-4)')
    parser.add_argument('-n_runs', '--num_runs', type=int, default=5,
                        help='number of runs (DEFAULT: 5)')
    parser.add_argument('-n', '--num_nodes', type=int, default=1,
                        help='learning rate for gradient descent (DEFAULT: 1e-4)')
    parser.add_argument('-n_layers', '--num_layers', type=int, default=2,
                        help='learning rate for gradient descent (DEFAULT: 1e-4)')
    parser.add_argument('-method', '--method', choices=['SAGE', 'GCN'], default='SAGE',
                        help='choose the network implementation (DEFAULT: SAGE)')
    parser.add_argument('-W', '--W', action='store_true', default=False,
                        help='use oracle')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args

def test(prediction, x, split_edge, evaluator, batch_size):
    pred = prediction
    f = prediction
    pos_train_edge = split_edge['eval_train']['edge'].to(f.device)
    pos_valid_edge = split_edge['valid']['edge'].to(f.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(f.device)
    pos_test_edge = split_edge['test']['edge'].to(f.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(f.device)

    pos_train_pred = pred[pos_train_edge[:,0],pos_train_edge[:,1]]
    pos_valid_pred = pred[pos_valid_edge[:,0],pos_valid_edge[:,1]]
    neg_valid_pred = pred[neg_valid_edge[:,0],neg_valid_edge[:,1]]
    pos_test_pred = pred[pos_test_edge[:,0],pos_test_edge[:,1]]
    neg_test_pred = pred[neg_test_edge[:,0],neg_test_edge[:,1]]

    train_acc_pos = (pos_train_pred>=0.5).sum()/len(pos_train_pred)
    #train_acc_neg = (neg_train_pred<0.5).sum()/len(neg_train_pred)

    valid_acc_pos = (pos_valid_pred>=0.5).sum()/len(pos_valid_pred)
    valid_acc_neg = (neg_valid_pred<0.5).sum()/len(neg_valid_pred)
    valid_acc = (valid_acc_pos+valid_acc_neg)/2

    test_acc_pos = (pos_test_pred>=0.5).sum()/len(pos_test_pred)
    test_acc_neg = (neg_test_pred<0.5).sum()/len(neg_test_pred)
    test_acc = (test_acc_pos+test_acc_neg)/2

    TP_valid = (pos_valid_pred>=0.5).sum()/len(pos_valid_pred)
    TN_valid = (neg_valid_pred<0.5).sum()/len(neg_valid_pred)
    FP_valid = 1 - TN_valid
    FN_valid = 1 - TP_valid
    if TP_valid * TN_valid - FP_valid * FN_valid==0:
        mcc_valid = 0
    else:
        mcc_valid = (TP_valid * TN_valid - FP_valid * FN_valid) / torch.sqrt((TP_valid + FP_valid) * (TP_valid + FN_valid) * (TN_valid + FP_valid) * (TN_valid + FN_valid))

    #mcc_valid = (TP_valid * TN_valid - FP_valid * FN_valid) / torch.sqrt((TP_valid + FP_valid) * (TP_valid + FN_valid) * (TN_valid + FP_valid) * (TN_valid + FN_valid))


    TP_test = (pos_test_pred>=0.5).sum()/len(pos_test_pred)
    TN_test = (neg_test_pred<0.5).sum()/len(neg_test_pred)
    FP_test = 1 - TN_test
    FN_test = 1 - TP_test
    #mcc_test = (TP_test * TN_test - FP_test * FN_test) / torch.sqrt((TP_test + FP_test) * (TP_test + FN_test) * (TN_test + FP_test) * (TN_test + FN_test))
    #evaluator = Evaluator(name='ogbl-ddi')
    if TP_test * TN_test - FP_test * FN_test==0:
        mcc_test = 0
    else:
        mcc_test = (TP_test * TN_test - FP_test * FN_test) / torch.sqrt((TP_test + FP_test) * (TP_test + FN_test) * (TN_test + FP_test) * (TN_test + FN_test))

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits, mcc_valid, mcc_test, valid_acc, test_acc)

    return results

@torch.no_grad()
def test_ind(length, n, evaluator, batch_size, seed):
    # Inductive
    #n = 1
    n_1 = 45*n
    n_2 = 10*n
    n_3 = n_1
    sizes = [45 * n, 10 * n, 45 * n]
    probs = [[0.55, 0.05, 0.02], [0.05, 0.55, 0.05], [0.02, 0.05, 0.55]]
    #probs = [[0.55, 0.005, 0.002], [0.005, 0.55, 0.005], [0.002, 0.005, 0.55]]
    g = nx.stochastic_block_model(sizes, probs, seed=seed)
    data = from_networkx(g)
    num_nodes = g.number_of_nodes()
    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], sparse_sizes=(num_nodes, num_nodes))
    #length = int(0.1 * data.edge_index.size(1))
    #print(length)
    a = torch.sparse.FloatTensor(data.edge_index, torch.ones(data.edge_index.size(1)),
                                 torch.Size([num_nodes, num_nodes]))#.to(device)

    f = torch.ones(num_nodes, num_nodes).to(device)  # initialization of f_{i,j}

    d = torch.ones(num_nodes, num_nodes).to(device)

    # torch.manual_seed(135)
    idx = torch.randperm(data.edge_index.size(1))
    idx_tr = idx[:int(0.8 * data.edge_index.size(1))]
    idx_va = idx[int(0.8 * data.edge_index.size(1)):int(0.9 * data.edge_index.size(1))]
    idx_te = idx[int(0.9 * data.edge_index.size(1)):]
    # edge_neg = negative_sampling(data.edge_index, num_nodes=num_nodes, num_neg_samples=int(0.2*data.edge_index.size(1)), method='dense')
    # num_neg_edge = int(0.1*data.edge_index.size(1))
    num_neg_edge = length
    indice_10 = torch.from_numpy(np.random.choice(45 * n, num_neg_edge))
    indice_20 = torch.from_numpy(np.random.choice(45 * n, num_neg_edge) + 55 * n)
    indice_11 = torch.from_numpy(np.random.choice(45 * n, num_neg_edge) + 55 * n)
    indice_21 = torch.from_numpy(np.random.choice(45 * n, num_neg_edge))
    id_0 = torch.cat((indice_10, indice_20)).reshape(1, -1)
    id_1 = torch.cat((indice_11, indice_21)).reshape(1, -1)
    edge_neg = torch.cat((id_0, id_1), dim=0)
    idx = torch.randperm(data.edge_index.size(1))
    split_edge = {}
    # split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}
    # idx_neg = torch.randperm(2*int(0.1*data.edge_index.size(1)))
    idx_neg = torch.randperm(2 * length)
    edge_neg = edge_neg[:, idx_neg]
    # split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}
    idx = torch.randperm(int(0.8 * data.edge_index.size(1)))
    idx_eval_tr = idx[:int(0.1 * data.edge_index.size(1))]

    split_edge['train'] = {'edge': data.edge_index.t()[idx_tr]}
    split_edge['eval_train'] = {'edge': data.edge_index.t()[idx_tr][idx_eval_tr]}
    split_edge['valid'] = {'edge': data.edge_index.t()[idx_va], 'edge_neg': edge_neg[:, :length].t()}
    split_edge['test'] = {'edge': data.edge_index.t()[idx_te], 'edge_neg': edge_neg[:, length:].t()}

    pos_test_edge = split_edge['test']['edge'].to(f.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(f.device)


    if not args.W:
        #Calculate the joint degrees in the graphs beforehand to save computation, i.e. the number of common numbers for node i and node j. If the common neighbor is 0, we make it 1.
        de_2 = torch.sparse.mm(a, a.t())  # distance
        de_2 = ((torch.ones(num_nodes, num_nodes) - de_2) > 0) + de_2
        de_2 = de_2.to(device)
        a = a.to(device)

        for k in range(args.num_layers): #iteration for 2-gMPNN, we do 3 iterations
            agg = torch.sparse.mm(a, f.t())
            agg = agg + agg.t()
            agg = agg / (2 * de_2)
            f = f / agg
        f.fill_diagonal_(1.0)
    else:
        w1 = torch.cat((torch.ones(int(n_1)) * 0.55, torch.ones(int(n_2)) * 0.05, torch.ones(int(n_3)) * 0.02))
        w2 = torch.cat((torch.ones(int(n_1)) * 0.05, torch.ones(int(n_2)) * 0.55, torch.ones(int(n_3)) * 0.05))
        w3 = torch.cat((torch.ones(int(n_1)) * 0.02, torch.ones(int(n_2)) * 0.05, torch.ones(int(n_3)) * 0.55))
        W = torch.cat((w1.repeat(n_1, 1), w2.repeat(n_2, 1), w3.repeat(n_3, 1)), dim=0)#.to(device)
        #W.fill_diagonal_(1.0)
        f=W

    pred = f
    pos_test_pred = pred[pos_test_edge[:, 0], pos_test_edge[:, 1]]
    neg_test_pred = pred[neg_test_edge[:, 0], neg_test_edge[:, 1]]

    test_acc_pos = (pos_test_pred >= 0.5).sum() / len(pos_test_pred)
    test_acc_neg = (neg_test_pred < 0.5).sum() / len(neg_test_pred)
    test_acc = (test_acc_pos + test_acc_neg) / 2

    TP_test = (pos_test_pred >= 0.5).sum() / len(pos_test_pred)
    TN_test = (neg_test_pred < 0.5).sum() / len(neg_test_pred)
    FP_test = 1 - TN_test
    FN_test = 1 - TP_test
    if TP_test * TN_test - FP_test * FN_test == 0:
        mcc_test = 0
    else:
        mcc_test = (TP_test * TN_test - FP_test * FN_test) / torch.sqrt((TP_test + FP_test) * (TP_test + FN_test) * (TN_test + FP_test) * (TN_test + FN_test))
    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']
        results[f'Hits@{K}'] = (test_hits, mcc_test, test_acc)

    return results

def main(seed1,seed2):
    n = args.num_nodes
    n_1 = 45*n
    n_3 = n_1
    n_2 = 10*n
    sizes = [45 * n, 10 * n, 45 * n]
    probs = [[0.55, 0.05, 0.02], [0.05, 0.55, 0.05], [0.02, 0.05, 0.55]]
    g = nx.stochastic_block_model(sizes, probs, seed=seed1)
    data = from_networkx(g)
    num_nodes = g.number_of_nodes()
    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], sparse_sizes=(num_nodes, num_nodes))
    # out_1 = model(data.x, adj_t)
    length = int(0.1 * data.edge_index.size(1))
    print(length)
    a = torch.sparse.FloatTensor(data.edge_index, torch.ones(data.edge_index.size(1)),
                                 torch.Size([num_nodes, num_nodes]))#.to(device)
    #length = 243
    f = torch.ones(num_nodes, num_nodes).to(device) #initialization of f_{i,j}

    d = torch.ones(num_nodes, num_nodes).to(device)

    #torch.manual_seed(135)
    idx = torch.randperm(data.edge_index.size(1))
    idx_tr = idx[:int(0.8*data.edge_index.size(1))]
    idx_va = idx[int(0.8*data.edge_index.size(1)):int(0.9*data.edge_index.size(1))]
    idx_te = idx[int(0.9*data.edge_index.size(1)):]
    #edge_neg = negative_sampling(data.edge_index, num_nodes=num_nodes, num_neg_samples=int(0.2*data.edge_index.size(1)), method='dense')
    #num_neg_edge = int(0.1*data.edge_index.size(1))
    num_neg_edge = length
    indice_10 = torch.from_numpy(np.random.choice(45*n, num_neg_edge))
    indice_20 = torch.from_numpy(np.random.choice(45*n, num_neg_edge)+55*n)
    indice_11 = torch.from_numpy(np.random.choice(45*n, num_neg_edge)+55*n)
    indice_21 = torch.from_numpy(np.random.choice(45*n, num_neg_edge))
    id_0 = torch.cat((indice_10, indice_20)).reshape(1,-1)
    id_1 = torch.cat((indice_11, indice_21)).reshape(1,-1)
    edge_neg = torch.cat((id_0,id_1),dim=0)
    idx = torch.randperm(data.edge_index.size(1))
    split_edge ={}
    #split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}
    #idx_neg = torch.randperm(2*int(0.1*data.edge_index.size(1)))
    idx_neg = torch.randperm(2*length)
    edge_neg = edge_neg[:,idx_neg]
    #split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}
    idx = torch.randperm(int(0.8*data.edge_index.size(1)))
    idx_eval_tr = idx[:int(0.1*data.edge_index.size(1))]

    split_edge['train'] = {'edge': data.edge_index.t()[idx_tr]}
    split_edge['eval_train'] = {'edge': data.edge_index.t()[idx_tr][idx_eval_tr]}
    split_edge['valid'] = {'edge': data.edge_index.t()[idx_va], 'edge_neg': edge_neg[:,:length].t()}
    split_edge['test'] = {'edge': data.edge_index.t()[idx_te], 'edge_neg': edge_neg[:,length:].t()}

    if not args.W:
        #Calculate the joint degrees in the graphs beforehand to save computation, i.e. the number of common numbers for node i and node j. If the common neighbor is 0, we make it 1.
        de_2 = torch.sparse.mm(a, a.t())  # distance
        de_2 = ((torch.ones(num_nodes, num_nodes) - de_2) > 0) + de_2
        de_2 = de_2.to(device)
        a = a.to(device)

        for k in range(args.num_layers): #iteration for 2-gMPNN, we do 3 iterations
            agg = torch.sparse.mm(a, f.t())
            agg = agg + agg.t()
            agg = agg / (2 * de_2)
            f = f / agg
        f.fill_diagonal_(1.0)
    else:
        w1 = torch.cat((torch.ones(int(n_1)) * 0.55, torch.ones(int(n_2)) * 0.05, torch.ones(int(n_3)) * 0.02))
        w2 = torch.cat((torch.ones(int(n_1)) * 0.05, torch.ones(int(n_2)) * 0.55, torch.ones(int(n_3)) * 0.05))
        w3 = torch.cat((torch.ones(int(n_1)) * 0.02, torch.ones(int(n_2)) * 0.05, torch.ones(int(n_3)) * 0.55))
        W = torch.cat((w1.repeat(n_1, 1), w2.repeat(n_2, 1), w3.repeat(n_3, 1)), dim=0)#.to(device)
        #W.fill_diagonal_(1.0)
        f = W

    # x is the output we will obtain for f_{i,j}
    print(f)


    #x = W
    pred = f
    pos_train_edge = split_edge['eval_train']['edge'].to(f.device)
    pos_valid_edge = split_edge['valid']['edge'].to(f.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(f.device)
    pos_test_edge = split_edge['test']['edge'].to(f.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(f.device)

    pos_train_pred = pred[pos_train_edge[:,0],pos_train_edge[:,1]]
    pos_valid_pred = pred[pos_valid_edge[:,0],pos_valid_edge[:,1]]
    neg_valid_pred = pred[neg_valid_edge[:,0],neg_valid_edge[:,1]]
    pos_test_pred = pred[pos_test_edge[:,0],pos_test_edge[:,1]]
    neg_test_pred = pred[neg_test_edge[:,0],neg_test_edge[:,1]]

    train_acc_pos = (pos_train_pred>=0.5).sum()/len(pos_train_pred)
    #train_acc_neg = (neg_train_pred<0.5).sum()/len(neg_train_pred)

    valid_acc_pos = (pos_valid_pred>=0.5).sum()/len(pos_valid_pred)
    valid_acc_neg = (neg_valid_pred<0.5).sum()/len(neg_valid_pred)
    valid_acc = (valid_acc_pos+valid_acc_neg)/2

    test_acc_pos = (pos_test_pred>=0.5).sum()/len(pos_test_pred)
    test_acc_neg = (neg_test_pred<0.5).sum()/len(neg_test_pred)
    test_acc = (test_acc_pos+test_acc_neg)/2

    TP_valid = (pos_valid_pred>=0.5).sum()/len(pos_valid_pred)
    TN_valid = (neg_valid_pred<0.5).sum()/len(neg_valid_pred)
    FP_valid = 1 - TN_valid
    FN_valid = 1 - TP_valid
    mcc_valid = (TP_valid * TN_valid - FP_valid * FN_valid) / torch.sqrt((TP_valid + FP_valid) * (TP_valid + FN_valid) * (TN_valid + FP_valid) * (TN_valid + FN_valid))

    TP_test = (pos_test_pred>=0.5).sum()/len(pos_test_pred)
    TN_test = (neg_test_pred<0.5).sum()/len(neg_test_pred)
    FP_test = 1 - TN_test
    FN_test = 1 - TP_test
    mcc_test = (TP_test * TN_test - FP_test * FN_test) / torch.sqrt((TP_test + FP_test) * (TP_test + FN_test) * (TN_test + FP_test) * (TN_test + FN_test))
    evaluator = Evaluator(name='ogbl-ddi')
    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits, mcc_valid, mcc_test, valid_acc, test_acc)

    print(results)

    train_hits, valid_hits, test_hits_10, mcc_valid, mcc_test, valid_acc, test_acc = results['Hits@10']
    train_hits, valid_hits, test_hits_50, mcc_valid, mcc_test, valid_acc, test_acc = results['Hits@50']
    train_hits, valid_hits, test_hits_100, mcc_valid, mcc_test, valid_acc, test_acc = results['Hits@100']

    n_1 = n
    results_ind = test_ind(length, n_1, evaluator, batch_size=1280 * int(n_1) * int(n_1), seed=seed2)
    n_1 = 10 * n
    results_ind_1 = test_ind(length, n_1, evaluator, batch_size=1280 * int(n_1) * int(n_1), seed=seed2)
    print(results_ind_1)
    print('Inductive result for ' + str(args.method) + 'of ' + str(n_1))
    print(results_ind)

    test_hits_10_ind_1, mcc_test_ind_1, test_acc_ind_1 = results_ind_1['Hits@10']
    test_hits_50_ind_1, mcc_test_ind_1, test_acc_ind_1 = results_ind_1['Hits@50']
    test_hits_100_ind_1, mcc_test_ind_1, test_acc_ind_1 = results_ind_1['Hits@100']
    test_hits_10_ind, mcc_test_ind, test_acc_ind = results_ind['Hits@10']
    test_hits_50_ind, mcc_test_ind, test_acc_ind = results_ind['Hits@50']
    test_hits_100_ind, mcc_test_ind, test_acc_ind = results_ind['Hits@100']

    return test_hits_10, test_hits_50, test_hits_100, mcc_test, test_acc, test_hits_10_ind, test_hits_50_ind, test_hits_100_ind, mcc_test_ind, test_acc_ind, test_hits_10_ind_1, test_hits_50_ind_1, test_hits_100_ind_1, mcc_test_ind_1, test_acc_ind_1


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    n_runs = args.num_runs
    res = torch.zeros((n_runs, 5))
    res_ind = torch.zeros((n_runs, 10))
    # seed = [232,11,67,55,12,77,33,66,888,5]
    # seed = [4566,32,6,55,42,66,888,54,27,2]
    # seed = [4566, 32, 6, 55, 276, 66, 888, 54, 27, 2]
    np.random.seed(32)
    seed = np.random.randint(1000, size=2 * n_runs)
    for i in range(n_runs):
        test_hits_10, test_hits_50, test_hits_100, mcc_test, test_acc, test_hits_10_ind, test_hits_50_ind, test_hits_100_ind, mcc_test_ind, test_acc_ind, test_hits_10_ind_1, test_hits_50_ind_1, test_hits_100_ind_1, mcc_test_ind_1, test_acc_ind_1 = main(int(seed[i]), int(seed[i+n_runs]))
        res[i] = torch.tensor([test_hits_10, test_hits_50, test_hits_100, mcc_test, test_acc])
        res_ind[i] = torch.tensor([test_hits_10_ind, test_hits_50_ind, test_hits_100_ind, mcc_test_ind, test_acc_ind, test_hits_10_ind_1, test_hits_50_ind_1, test_hits_100_ind_1, mcc_test_ind_1, test_acc_ind_1])
    print(args)
    print(res)
    print(res_ind)
    print("Transductive results")
    print('Mean')
    print(res.mean(dim=0))
    print('STD')
    print(res.std(dim=0))
    print("Inductive results")
    print('Mean')
    print(res_ind.mean(dim=0))
    print('STD')
    print(res_ind.std(dim=0))