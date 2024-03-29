import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset
from torch_geometric.utils import negative_sampling
import numpy as np
import argparse
import utils

from torch_sparse import SparseTensor
import networkx as nx

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Training for LINK-PREDICTION')
    parser.add_argument('-hid', '--hid_dim', type=int, default=5,
                        help='hidden dimension')
    parser.add_argument('-b', '--batch', type=int, default=128,
                        help='learning rate for gradient descent (DEFAULT: 1e-4)')
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

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

def train(model, f, split_edge, edge_index, optimizer, batch_size):

    model.train()

    pos_train_edge = split_edge['train']['edge'].to(f.device)
    n = int(f.size(0)/100)
    num_nodes = f.size(0)
    f = f.reshape(-1,1)
    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        f_new = model(f)
        f_new = f_new.reshape(num_nodes,num_nodes)
        edge = pos_train_edge[perm].t()
        pos_out = f_new[edge[0], edge[1]]
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = negative_sampling(edge_index, num_nodes=427,
                                 num_neg_samples=perm.size(0), method='dense')
        # num_neg_edge = perm.size(0)
        # indice_10 = torch.from_numpy(np.random.choice(45 * n, num_neg_edge))
        # indice_20 = torch.from_numpy(np.random.choice(45 * n, num_neg_edge) + 55 * n)
        # indice_11 = torch.from_numpy(np.random.choice(55 * n, num_neg_edge) + 45 * n)
        # indice_21 = torch.from_numpy(np.random.choice(55 * n, num_neg_edge))
        # id_0 = torch.cat((indice_10, indice_20)).reshape(1, -1)
        # id_1 = torch.cat((indice_11, indice_21)).reshape(1, -1)
        # edge = torch.cat((id_0, id_1), dim=0)

        neg_out = f_new[edge[0], edge[1]]
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(x, 1.0)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

@torch.no_grad()
def test(predictor, f, split_edge, evaluator, batch_size):
    num_nodes = 427
    if not args.W:
        pred = predictor(f.reshape(-1,1)).reshape(num_nodes, num_nodes)
    else:
        pred = f
    #f = prediction
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


    TP_test = (pos_test_pred>=0.5).sum()/len(pos_test_pred)
    TN_test = (neg_test_pred<0.5).sum()/len(neg_test_pred)
    FP_test = 1 - TN_test
    FN_test = 1 - TP_test

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
def test_ind(predictor, evaluator):
    dataset = PygLinkPropPredDataset(name='ogbl-ddi', transform=T.ToSparseTensor())
    data = dataset[0]
    # adj_t = data.adj_t.to(device)
    # data.x = adj_t.to_dense().sum(dim=1)
    split_edge = dataset.get_edge_split()
    idxs_small_tr = (((split_edge['train']['edge'][:, 0] < 854).to(torch.int) + (
            split_edge['train']['edge'][:, 1] < 854).to(torch.int) +
                      (split_edge['train']['edge'][:, 0] >= 427).to(torch.int) + (
                              split_edge['train']['edge'][:, 1] >= 427).to(torch.int)) == 4)
    idxs_small_te = (((split_edge['test']['edge'][:, 0] < 854).to(torch.int) + (
            split_edge['test']['edge'][:, 1] < 854).to(torch.int) +
                      (split_edge['test']['edge'][:, 0] >= 427).to(torch.int) + (
                              split_edge['test']['edge'][:, 1] >= 427).to(torch.int)) == 4)
    idxs_small_te_neg = (((split_edge['test']['edge_neg'][:, 0] < 854).to(torch.int) + (
            split_edge['test']['edge_neg'][:, 1] < 854).to(torch.int) +
                          (split_edge['test']['edge_neg'][:, 0] >= 427).to(torch.int) + (
                                  split_edge['test']['edge_neg'][:, 1] >= 427).to(torch.int)) == 4)

    split_edge['train']['edge'] = split_edge['train']['edge'][idxs_small_tr] - 427
    split_edge['test']['edge'] = split_edge['test']['edge'][idxs_small_te] - 427
    split_edge['test']['edge_neg'] = split_edge['test']['edge_neg'][idxs_small_te_neg] - 427

    row, col, edge_attr = data.adj_t.t().coo()
    data.edge_index = torch.stack([row, col], dim=0)
    idx_adj = (((data.edge_index[0] < 854).to(torch.int) + (
            data.edge_index[1] < 854).to(torch.int) + (data.edge_index[0] >= 427).to(torch.int) + (
                        data.edge_index[1] >= 427).to(torch.int)) == 4)
    data.edge_index = data.edge_index[:, idx_adj] - 427

    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], sparse_sizes=(427, 427))
    adj_t = adj.t().to(device)
    num_nodes = 427
    a = torch.sparse.FloatTensor(data.edge_index, torch.ones(data.edge_index.size(1)),
                                 torch.Size([num_nodes, num_nodes]))  # .to(device)
    f = torch.ones(num_nodes, num_nodes).to(device)  # initialization of f_{i,j}

    pos_test_edge = split_edge['test']['edge'].to(f.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(f.device)

    # Calculate the joint degrees in the graphs beforehand to save computation, i.e. the number of common numbers for node i and node j. If the common neighbor is 0, we make it 1.
    de_2 = torch.sparse.mm(a, a.t())  # distance
    de_2 = ((torch.ones(num_nodes, num_nodes) - de_2) > 0) + de_2
    de_2 = de_2.to(device)
    a = a.to(device)

    for k in range(2):  # iteration for 2-gMPNN, we do 3 iterations
        agg = torch.sparse.mm(a, f.t())
        agg = ((torch.ones(num_nodes, num_nodes).to(device) - agg) >= 0) + agg
        agg = agg + agg.t()
        agg = agg / (2 * de_2)
        f = f / agg
    f.fill_diagonal_(1.0)

    pred = predictor(f.reshape(-1, 1))
    pred = pred.reshape(num_nodes, num_nodes)

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
        mcc_test = (TP_test * TN_test - FP_test * FN_test) / torch.sqrt(
            (TP_test + FP_test) * (TP_test + FN_test) * (TN_test + FP_test) * (TN_test + FN_test))
    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']
        results[f'Hits@{K}'] = (test_hits, mcc_test, test_acc)

    return results

@torch.no_grad()
def test_ind_large(predictor, evaluator):
    dataset = PygLinkPropPredDataset(name='ogbl-ddi', transform=T.ToSparseTensor())
    data = dataset[0]
    # adj_t = data.adj_t.to(device)
    # data.x = adj_t.to_dense().sum(dim=1)
    split_edge = dataset.get_edge_split()
    idxs_small_tr = (((split_edge['train']['edge'][:, 0] >= 427).to(torch.int) + (
            split_edge['train']['edge'][:, 1] >= 427).to(torch.int)) == 2)
    idxs_small_te = (((split_edge['test']['edge'][:, 0] >= 427).to(torch.int) + (
            split_edge['test']['edge'][:, 1] >= 427).to(torch.int)) == 2)
    idxs_small_te_neg = (((split_edge['test']['edge_neg'][:, 0] >= 427).to(torch.int) + (
            split_edge['test']['edge_neg'][:, 1] >= 427).to(torch.int)) == 2)
    idxs_small_va = (((split_edge['test']['edge'][:, 0] < 854).to(torch.int) + (
            split_edge['test']['edge'][:, 1] < 854).to(torch.int) +
                      (split_edge['test']['edge'][:, 0] >= 427).to(torch.int) + (
                              split_edge['test']['edge'][:, 1] >= 427).to(torch.int)) == 4)
    idxs_small_va_neg = (((split_edge['test']['edge_neg'][:, 0] < 854).to(torch.int) + (
            split_edge['test']['edge_neg'][:, 1] < 854).to(torch.int) +
                          (split_edge['test']['edge_neg'][:, 0] >= 427).to(torch.int) + (
                                  split_edge['test']['edge_neg'][:, 1] >= 427).to(torch.int)) == 4)

    split_edge['train']['edge'] = split_edge['train']['edge'][idxs_small_tr] - 427
    split_edge['valid']['edge'] = split_edge['test']['edge'][idxs_small_va] - 427
    split_edge['valid']['edge_neg'] = split_edge['test']['edge_neg'][idxs_small_va_neg] - 427
    split_edge['test']['edge'] = split_edge['test']['edge'][idxs_small_te] - 427
    split_edge['test']['edge_neg'] = split_edge['test']['edge_neg'][idxs_small_te_neg] - 427

    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['test']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['test']['edge'] = split_edge['test']['edge'][idx]
    idx = torch.randperm(split_edge['test']['edge_neg'].size(0))
    idx = idx[:split_edge['valid']['edge_neg'].size(0)]
    split_edge['test']['edge_neg'] = split_edge['test']['edge_neg'][idx]

    # split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    row, col, edge_attr = data.adj_t.t().coo()
    data.edge_index = torch.stack([row, col], dim=0)
    idx_adj = (((data.edge_index[0] >= 427).to(torch.int) + (
            data.edge_index[1] >= 427).to(torch.int)) == 2)
    data.edge_index = data.edge_index[:, idx_adj] - 427

    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], sparse_sizes=(4267 - 427, 4267 - 427))
    adj_t = adj.t().to(device)

    num_nodes = 4267 - 427
    a = torch.sparse.FloatTensor(data.edge_index, torch.ones(data.edge_index.size(1)),
                                 torch.Size([num_nodes, num_nodes]))  # .to(device)
    f = torch.ones(num_nodes, num_nodes).to(device) #initialization of f_{i,j}


    pos_test_edge = split_edge['test']['edge'].to(f.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(f.device)


    #Calculate the joint degrees in the graphs beforehand to save computation, i.e. the number of common numbers for node i and node j. If the common neighbor is 0, we make it 1.
    de_2 = torch.sparse.mm(a, a.t())  # distance
    de_2 = ((torch.ones(num_nodes, num_nodes) - de_2) > 0) + de_2
    de_2 = de_2.to(device)
    a = a.to(device)

    for k in range(2): #iteration for 2-gMPNN, we do 3 iterations
        agg = torch.sparse.mm(a, f.t())
        agg = ((torch.ones(num_nodes, num_nodes).to(device) - agg) >= 0) + agg
        agg = agg + agg.t()
        agg = agg / (2 * de_2)
        f = f / agg
    f.fill_diagonal_(1.0)



    pred = predictor(f.reshape(-1,1))
    pred = pred.reshape(num_nodes, num_nodes)

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
    dataset = PygLinkPropPredDataset(name='ogbl-ddi', transform=T.ToSparseTensor())
    data = dataset[0]
    # adj_t = data.adj_t.to(device)
    # data.x = adj_t.to_dense().sum(dim=1)
    split_edge = dataset.get_edge_split()
    idxs_small_tr = (((split_edge['train']['edge'][:, 0] < 427).to(torch.int) + (
            split_edge['train']['edge'][:, 1] < 427).to(torch.int)) == 2)
    idxs_small_va = (((split_edge['valid']['edge'][:, 0] < 427).to(torch.int) + (
            split_edge['valid']['edge'][:, 1] < 427).to(torch.int)) == 2)
    idxs_small_te = (((split_edge['test']['edge'][:, 0] < 427).to(torch.int) + (
            split_edge['test']['edge'][:, 1] < 427).to(torch.int)) == 2)
    idxs_small_va_neg = (((split_edge['valid']['edge_neg'][:, 0] < 427).to(torch.int) + (
            split_edge['valid']['edge_neg'][:, 1] < 427).to(torch.int)) == 2)
    idxs_small_te_neg = (((split_edge['test']['edge_neg'][:, 0] < 427).to(torch.int) + (
            split_edge['test']['edge_neg'][:, 1] < 427).to(torch.int)) == 2)
    split_edge['train']['edge'] = split_edge['train']['edge'][idxs_small_tr]
    split_edge['valid']['edge'] = split_edge['valid']['edge'][idxs_small_va]
    split_edge['valid']['edge_neg'] = split_edge['valid']['edge_neg'][idxs_small_va_neg]
    split_edge['test']['edge'] = split_edge['test']['edge'][idxs_small_te]
    split_edge['test']['edge_neg'] = split_edge['test']['edge_neg'][idxs_small_te_neg]

    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    row, col, edge_attr = data.adj_t.t().coo()
    data.edge_index = torch.stack([row, col], dim=0)
    idx_adj = (((data.edge_index[0] < 427).to(torch.int) + (
            data.edge_index[1] < 427).to(torch.int)) == 2)
    data.edge_index = data.edge_index[:, idx_adj]

    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], sparse_sizes=(427, 427))
    adj_t = adj.t().to(device)

    data.x = (adj_t.sum(dim=0) / 427).reshape(-1, 1)

    num_nodes = 427
    a = torch.sparse.FloatTensor(data.edge_index, torch.ones(data.edge_index.size(1)),
                                 torch.Size([num_nodes, num_nodes]))  # .to(device)
    f = torch.ones(num_nodes, num_nodes).to(device)
    de_2 = torch.sparse.mm(a, a.t())  # distance
    de_2 = ((torch.ones(num_nodes, num_nodes) - de_2) > 0) + de_2
    de_2 = de_2.to(device)
    a = a.to(device)

    for k in range(2):  # iteration for 2-gMPNN, we do 3 iterations
        agg = torch.sparse.mm(a, f.t())
        agg = ((torch.ones(num_nodes, num_nodes).to(device) - agg) >= 0) + agg
        agg = agg + agg.t()
        agg = agg / (2 * de_2)
        f = f / agg
    f.fill_diagonal_(1.0)

    # x is the output we will obtain for f_{i,j}
    print(f)
    predictor = LinkPredictor(1, args.hid_dim, 1, args.num_layers, 0).to(device)
    #torch.manual_seed(123)
    predictor.reset_parameters()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.learning_rate)
    evaluator = Evaluator(name='ogbl-ddi')
    n=1
    if not args.W:
        best_valid_acc = 0
        checkpoint_file_name = "7-30-ddi" + str(args.num_nodes) + str(args.hid_dim) + str(args.num_runs) + "model_2GNN_fixed_checkpoint.pth.tar"

        for epoch in range(1, 1 + args.n_epochs):
            loss = train(predictor, f, split_edge, data.edge_index, optimizer, batch_size=args.batch*int(n)*int(n))
            #if epoch % 10 == 0:
            #    print(epoch)
            if epoch % 100 == 0:
                results = test(predictor, f, split_edge, evaluator, batch_size=320*int(n)*int(n))
                for key, result in results.items():
                    train_hits, valid_hits, test_hits, mcc_valid, mcc_test, valid_acc, test_acc = result
                    print(key)
                    print(f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_hits:.2f}%, '
                          f'Valid: {100 * valid_hits:.2f}%, '
                          f'Test: {100 * test_hits:.2f}%, '
                          f'Valid_mcc: {100 * mcc_valid:.2f}%, '
                          f'Test_mcc: {100 * mcc_test:.2f}%, '
                          f'Valid_acc: {100 * valid_acc:.2f}%, '
                          f'Test_acc: {100 * test_acc:.2f}%')
                    print('---')
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    torch.save(predictor, checkpoint_file_name)
        print('End of GraphSAGE training')
        predictor = torch.load(checkpoint_file_name)

    results = test(predictor, f, split_edge, evaluator, batch_size=1280 * int(n) * int(n))
    train_hits, valid_hits, test_hits_10, mcc_valid, mcc_test, valid_acc, test_acc = results['Hits@10']
    train_hits, valid_hits, test_hits_50, mcc_valid, mcc_test, valid_acc, test_acc = results['Hits@50']
    train_hits, valid_hits, test_hits_100, mcc_valid, mcc_test, valid_acc, test_acc = results['Hits@100']


    results_ind = test_ind(predictor, evaluator)
    print(results_ind)

    results_ind_1 = test_ind_large(predictor, evaluator)
    print(results_ind_1)
    print('Inductive result for ' + str(args.method) + 'of large ')

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