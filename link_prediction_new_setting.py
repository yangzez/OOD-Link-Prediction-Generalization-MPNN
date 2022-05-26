import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import GINConv

from torch_geometric.utils import from_networkx
from torch.utils.data import DataLoader
from ogb.linkproppred import Evaluator
import numpy as np
import argparse
import utils
import logging

from torch_sparse import SparseTensor
import networkx as nx

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Training for LINK PREDICTION')
    parser.add_argument('-e', '--n_epochs', type=int, default=500,
                        help='number of epochs (DEFAULT: 100)')
    parser.add_argument('-n_runs', '--num_runs', type=int, default=5,
                        help='number of runs (DEFAULT: 5)')
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-4,
                        help='learning rate for gradient descent (DEFAULT: 1e-4)')
    parser.add_argument('-hid', '--hid_dim', type=int, default=10,
                        help='hidden dimension')
    parser.add_argument('-n_layer', '--num_layers', type=int, default=2,
                        help='number of layers in the GNN')
    parser.add_argument('-n', '--num_nodes', type=int, default=1,
                        help='number of nodes is n*100')
    parser.add_argument('-b', '--batch', type=int, default=128,
                        help='batch size (default is 128)')
    parser.add_argument('-method', '--method', choices=['SAGE', 'GCN', 'GAT', 'GIN'], default='SAGE',
                        help='choose the network implementation (DEFAULT: SAGE)')
    parser.add_argument('-agg', '--agg', choices=['max', 'mean'], default='mean',
                        help='aggregation method (DEFAULT: MEAN)')
    parser.add_argument('-noinn', '--noinn', action='store_true', default=False,
                        help='choose not to use inner product')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


class GIN(torch.nn.Module):
    def __init__(self, in_channels=1, hidden_channels=10, out_channels=10, num_layer=2):
        super(GIN, self).__init__()
        self.num_layer = num_layer
        self.nn1 = torch.nn.Linear(in_channels, hidden_channels)
        self.nn2 = torch.nn.Linear(hidden_channels, out_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        if num_layer > 2:
            self.nn3 = torch.nn.Linear(hidden_channels, out_channels)
            self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
            self.conv3 = GINConv(self.nn3)
        self.conv1 = GINConv(self.nn1)
        self.conv2 = GINConv(self.nn2)

    def reset_parameters(self):
        self.nn1.reset_parameters()
        self.nn2.reset_parameters()
        if self.num_layer > 2:
            self.nn3.reset_parameters()
            self.bn2.reset_parameters()
        self.bn1.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = self.bn1(x)
        x = F.relu(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, adj_t)
        if self.num_layer > 2:
            x = self.bn2(x)
            x = F.relu(x)
            x = self.conv3(x, adj_t)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels=1, hid_channels=10, out_channels=10, num_layer=2, aggr='mean'):
        super(GAT, self).__init__()
        self.hid = 5
        self.in_head = 2
        self.out_head = 1
        self.num_layer = num_layer
        if self.num_layer > 2:
            self.bn2 = torch.nn.BatchNorm1d(self.hid * self.in_head)
            self.conv3 = GATConv(self.hid * self.in_head, self.hid, heads=self.in_head, fill_value=aggr)#, dropout=0.5)
        self.bn1 = torch.nn.BatchNorm1d(self.hid * self.in_head)
        self.conv1 = GATConv(in_channels, self.hid, heads=self.in_head, fill_value=aggr)#, dropout=0.5)
        self.conv2 = GATConv(self.hid * self.in_head, out_channels, concat=False,
                             heads=self.out_head, fill_value=aggr)#, dropout=0.5)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.bn1.reset_parameters()
        if self.num_layer > 2:
            self.conv3.reset_parameters()
            self.bn2.reset_parameters()

    def forward(self, x, adj_t):
        x, edge_index = x, adj_t
        # Dropout before the GAT layer is used to avoid overfitting in small datasets like Cora.
        # One can skip them if the dataset is sufficiently large.
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        if self.num_layer > 2:
            x = self.conv3(x, edge_index)
            x = self.bn2(x)
            x = F.elu(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return x

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, aggr='mean'):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        i = 0
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            if i==0:
                x = self.bn1(x)
            else:
                x = self.bn2(x)
            x = F.relu(x)
            i += 1
            #x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False))
            self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        i = 0
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            if i==0:
                x = self.bn1(x)
            else:
                x = self.bn2(x)
            x = F.relu(x)
            i += 1
            #x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

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

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

class LinkPredictor_noinn(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor_noinn, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels+in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = torch.cat((x_i, x_j), dim=1)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

def train(model, predictor, x, adj_t, split_edge, edge_index, optimizer, batch_size):

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)
    n = int(x.size(0)/100)
    #n = args.num_nodes

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(x, adj_t)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        num_neg_edge = perm.size(0)
        indice_10 = torch.from_numpy(np.random.choice(45 * n, num_neg_edge))
        indice_20 = torch.from_numpy(np.random.choice(45 * n, num_neg_edge) + 55 * n)
        indice_11 = torch.from_numpy(np.random.choice(55 * n, num_neg_edge) + 45 * n)
        indice_21 = torch.from_numpy(np.random.choice(55 * n, num_neg_edge))
        id_0 = torch.cat((indice_10, indice_20)).reshape(1, -1)
        id_1 = torch.cat((indice_11, indice_21)).reshape(1, -1)
        edge = torch.cat((id_0, id_1), dim=0)

        neg_out = predictor(h[edge[0]], h[edge[1]])
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
def test(model, predictor, x, adj_t, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(x, adj_t)

    pos_train_edge = split_edge['eval_train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    valid_loss = 0
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_out = predictor(h[edge[0]], h[edge[1]])
        valid_loss += -torch.log(pos_valid_out + 1e-15).sum()
        pos_valid_preds += [pos_valid_out.squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_out = predictor(h[edge[0]], h[edge[1]])
        valid_loss += -torch.log(1-neg_valid_out + 1e-15).sum()
        neg_valid_preds += [neg_valid_out.squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)
    valid_loss = valid_loss / (len(neg_valid_pred)+len(pos_valid_pred))

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    train_acc_pos = (pos_train_pred >= 0.5).sum() / len(pos_train_pred)
    # train_acc_neg = (neg_train_pred<0.5).sum()/len(neg_train_pred)

    valid_acc_pos = (pos_valid_pred >= 0.5).sum() / len(pos_valid_pred)
    valid_acc_neg = (neg_valid_pred < 0.5).sum() / len(neg_valid_pred)
    valid_acc = (valid_acc_pos + valid_acc_neg) / 2

    test_acc_pos = (pos_test_pred >= 0.5).sum() / len(pos_test_pred)
    test_acc_neg = (neg_test_pred < 0.5).sum() / len(neg_test_pred)
    test_acc = (test_acc_pos + test_acc_neg) / 2

    TP_valid = (pos_valid_pred >= 0.5).sum() / len(pos_valid_pred)
    TN_valid = (neg_valid_pred < 0.5).sum() / len(neg_valid_pred)
    FP_valid = 1 - TN_valid
    FN_valid = 1 - TP_valid
    if TP_valid * TN_valid - FP_valid * FN_valid==0:
        mcc_valid = 0
    else:
        mcc_valid = (TP_valid * TN_valid - FP_valid * FN_valid) / torch.sqrt((TP_valid + FP_valid) * (TP_valid + FN_valid) * (TN_valid + FP_valid) * (TN_valid + FN_valid))

    TP_test = (pos_test_pred >= 0.5).sum() / len(pos_test_pred)
    TN_test = (neg_test_pred < 0.5).sum() / len(neg_test_pred)
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

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits, mcc_valid, mcc_test, valid_acc, test_acc, valid_loss)

    return results

@torch.no_grad()
def test_ind(model, predictor, length, n, evaluator, batch_size, seed):
    # Inductive
    sizes = [45 * n, 10 * n, 45 * n]
    #probs = [[0.55, 0.05, 0.02], [0.05, 0.55, 0.05], [0.02, 0.05, 0.55]]
    probs = [[0.6, 0.05, 0.02], [0.05, 0.6, 0.05], [0.02, 0.05, 0.6]]
    #probs = [[0.55, 0.005, 0.002], [0.005, 0.55, 0.005], [0.002, 0.005, 0.55]]
    g = nx.stochastic_block_model(sizes, probs, seed=seed)
    data = from_networkx(g)
    num_nodes = g.number_of_nodes()
    data.x = torch.zeros(num_nodes, 1)
    for i in range(num_nodes):
        data.x[i] = g.degree[i]/num_nodes
    data.x = data.x.to(device)
    idx = torch.randperm(data.edge_index.size(1))
    idx_remain = idx[:int(0.9 * data.edge_index.size(1))]
    idx_hide = idx[int(0.9 * data.edge_index.size(1)):]
    edge_index_hide = data.edge_index[:, idx_hide]
    data.edge_index = data.edge_index[:, idx_remain]
    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], sparse_sizes=(num_nodes, num_nodes))
    adj_t = adj.t().to(device)
    idx = torch.randperm(edge_index_hide.size(1))
    idx_tr = idx[:int(0.8 * edge_index_hide.size(1))]
    idx_va = idx[int(0.8 * edge_index_hide.size(1)):int(0.9 * edge_index_hide.size(1))]
    idx_te = idx[int(0.9 * edge_index_hide.size(1)):]
    #length = idx_va.size(0)
    num_neg_edge = length
    indice_10 = torch.from_numpy(np.random.choice(45 * n, num_neg_edge))
    indice_20 = torch.from_numpy(np.random.choice(45 * n, num_neg_edge) + 55 * n)
    indice_11 = torch.from_numpy(np.random.choice(45 * n, num_neg_edge) + 55 * n)
    indice_21 = torch.from_numpy(np.random.choice(45 * n, num_neg_edge))
    id_0 = torch.cat((indice_10, indice_20)).reshape(1, -1)
    id_1 = torch.cat((indice_11, indice_21)).reshape(1, -1)
    edge_neg = torch.cat((id_0, id_1), dim=0)
    split_edge = {}
    idx_neg = torch.randperm(2 * length)
    edge_neg = edge_neg[:, idx_neg]
    idx = torch.randperm(int(0.8 * edge_index_hide.size(1)))
    idx_eval_tr = idx[:int(0.1 * edge_index_hide.size(1))]
    split_edge['train'] = {'edge': edge_index_hide.t()[idx_tr]}
    split_edge['eval_train'] = {'edge': edge_index_hide.t()[idx_tr][idx_eval_tr]}
    split_edge['valid'] = {'edge': edge_index_hide.t()[idx_va], 'edge_neg': edge_neg[:, :length].t()}
    split_edge['test'] = {'edge': edge_index_hide.t()[idx_te], 'edge_neg': edge_neg[:, length:].t()}
    x = data.x
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)
    model.eval()
    predictor.eval()

    h = model(data.x, adj_t)
    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

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

def main(seed1, seed2):
    n = args.num_nodes
    sizes = [45 * n, 10 * n, 45 * n]
    probs = [[0.6, 0.05, 0.02], [0.05, 0.6, 0.05], [0.02, 0.05, 0.6]]
    g = nx.stochastic_block_model(sizes, probs, seed=seed1)
    data = from_networkx(g)
    num_nodes = g.number_of_nodes()
    data.x = torch.zeros(num_nodes, 1)
    for i in range(num_nodes):
        data.x[i] = g.degree[i]/num_nodes
    data.x = data.x.to(device)
    idx = torch.randperm(data.edge_index.size(1))
    idx_remain = idx[:int(0.9 * data.edge_index.size(1))]
    idx_hide = idx[int(0.9 * data.edge_index.size(1)):]
    edge_index_hide = data.edge_index[:, idx_hide]
    data.edge_index = data.edge_index[:, idx_remain]
    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], sparse_sizes=(num_nodes, num_nodes))
    adj_t = adj.t().to(device)
    idx = torch.randperm(edge_index_hide.size(1))
    idx_tr = idx[:int(0.8 * edge_index_hide.size(1))]
    idx_va = idx[int(0.8 * edge_index_hide.size(1)):int(0.9 * edge_index_hide.size(1))]
    idx_te = idx[int(0.9 * edge_index_hide.size(1)):]
    length = idx_va.size(0)
    num_neg_edge = length
    indice_10 = torch.from_numpy(np.random.choice(45 * n, num_neg_edge))
    indice_20 = torch.from_numpy(np.random.choice(45 * n, num_neg_edge) + 55 * n)
    indice_11 = torch.from_numpy(np.random.choice(45 * n, num_neg_edge) + 55 * n)
    indice_21 = torch.from_numpy(np.random.choice(45 * n, num_neg_edge))
    id_0 = torch.cat((indice_10, indice_20)).reshape(1, -1)
    id_1 = torch.cat((indice_11, indice_21)).reshape(1, -1)
    edge_neg = torch.cat((id_0, id_1), dim=0)
    split_edge = {}
    idx_neg = torch.randperm(2 * length)
    edge_neg = edge_neg[:, idx_neg]
    idx = torch.randperm(int(0.8 * edge_index_hide.size(1)))
    idx_eval_tr = idx[:int(0.1 * edge_index_hide.size(1))]
    split_edge['train'] = {'edge': edge_index_hide.t()[idx_tr]}
    split_edge['eval_train'] = {'edge': edge_index_hide.t()[idx_tr][idx_eval_tr]}
    split_edge['valid'] = {'edge': edge_index_hide.t()[idx_va], 'edge_neg': edge_neg[:, :length].t()}
    split_edge['test'] = {'edge': edge_index_hide.t()[idx_te], 'edge_neg': edge_neg[:, length:].t()}

    if args.method == 'SAGE':
        model = SAGE(1, args.hid_dim, args.hid_dim, args.num_layers, 0, aggr=args.agg).to(device)
    elif args.method == 'GCN':
        model = GCN(1, args.hid_dim, args.hid_dim, args.num_layers, 0).to(device)
    elif args.method == 'GAT':
        model = GAT(1,args.hid_dim, num_layer=args.num_layers, aggr=args.agg).to(device)
    else:
        model = GIN(1, args.hid_dim, args.hid_dim, num_layer=args.num_layers).to(device)
    if not args.noinn:
        predictor = LinkPredictor(args.hid_dim, args.hid_dim, 1, args.num_layers, 0).to(device)
    else:
        predictor = LinkPredictor_noinn(args.hid_dim, args.hid_dim, 1, args.num_layers, 0).to(device)
    torch.manual_seed(12)
    model.reset_parameters()
    torch.manual_seed(123)
    predictor.reset_parameters()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()), lr=args.learning_rate)
    evaluator = Evaluator(name='ogbl-ddi')

    best_valid_acc = 0
    valid_losses = []
    checkpoint_file_name = "5-15" + str(args.agg) + str(args.hid_dim)+str(args.num_layers)+ str(args.num_nodes) + str(args.method) + str(args.noinn) + "model_GNN_checkpoint.pth.tar"
    checkpoint_file_name_Y ="5-15" + str(args.agg) + str(args.hid_dim)+str(args.num_layers)+ str(args.num_nodes) + str(args.method) + str(args.noinn) + "model_link_checkpoint.pth.tar"

    for epoch in range(1, 1 + args.n_epochs):
        #if epoch% 10==0:
        #    print(epoch)
        loss = train(model, predictor, data.x, adj_t, split_edge, data.edge_index, optimizer, batch_size=args.batch*int(n)*int(n))
        if epoch % 50 == 0:
            results = test(model, predictor, data.x, adj_t, split_edge, evaluator, batch_size=1280*int(n)*int(n))
            for key, result in results.items():
                train_hits, valid_hits, test_hits, mcc_valid, mcc_test, valid_acc, test_acc, valid_loss = result
                print(key)
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_hits:.2f}%, ' 
                      f'Valid: {100 * valid_hits:.2f}%, '
                      f'Test: {100 * test_hits:.2f}%, '
                      f'Valid_mcc: {100 * mcc_valid:.2f}%, '
                      f'Test_mcc: {100 * mcc_test:.2f}%, '
                      f'Valid_acc: {100 * valid_acc:.2f}%, '
                      f'Test_acc: {100 * test_acc:.2f}%, '
                      f'Valid_loss: {valid_loss:.4f}')
                print('---')
            valid_losses.append(valid_loss)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(model, checkpoint_file_name)
                torch.save(predictor, checkpoint_file_name_Y)

    print('End of GraphSAGE training')
    model = torch.load(checkpoint_file_name)
    predictor = torch.load(checkpoint_file_name_Y)
    results = test(model, predictor, data.x, adj_t, split_edge, evaluator, batch_size = 1280 * int(n) * int(n))

    n_1 = n
    results_ind = test_ind(model, predictor, length, n_1, evaluator, batch_size=1280 * int(n_1) * int(n_1), seed=seed2)
    print(results_ind)

    n_1 = 10*n
    results_ind_1 = test_ind(model, predictor, length, n_1, evaluator, batch_size=1280*int(n_1)*int(n_1), seed= seed2)
    print(results_ind_1)
    print('Inductive result for '+ str(args.method) + 'of '+str(n_1))

    train_hits, valid_hits, test_hits_10, mcc_valid, mcc_test, valid_acc, test_acc, valid_loss = results['Hits@10']
    train_hits, valid_hits, test_hits_50, mcc_valid, mcc_test, valid_acc, test_acc, valid_loss = results['Hits@50']
    train_hits, valid_hits, test_hits_100, mcc_valid, mcc_test, valid_acc, test_acc, valid_loss = results['Hits@100']

    test_hits_10_ind_1, mcc_test_ind_1, test_acc_ind_1 = results_ind_1['Hits@10']
    test_hits_50_ind_1, mcc_test_ind_1, test_acc_ind_1 = results_ind_1['Hits@50']
    test_hits_100_ind_1, mcc_test_ind_1, test_acc_ind_1 = results_ind_1['Hits@100']

    test_hits_10_ind, mcc_test_ind, test_acc_ind = results_ind['Hits@10']
    test_hits_50_ind, mcc_test_ind, test_acc_ind = results_ind['Hits@50']
    test_hits_100_ind, mcc_test_ind, test_acc_ind = results_ind['Hits@100']

    print(valid_losses)

    return test_hits_10, test_hits_50, test_hits_100, mcc_test, test_acc, test_hits_10_ind, test_hits_50_ind, test_hits_100_ind, mcc_test_ind, test_acc_ind, test_hits_10_ind_1, test_hits_50_ind_1, test_hits_100_ind_1, mcc_test_ind_1, test_acc_ind_1

if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    n_runs = args.num_runs
    res = torch.zeros((n_runs, 5))
    res_ind = torch.zeros((n_runs, 10))
    np.random.seed(32)
    seed = np.random.randint(1000, size=2*n_runs)
    for i in range(n_runs):
        test_hits_10, test_hits_50, test_hits_100, mcc_test, test_acc, test_hits_10_ind, test_hits_50_ind, test_hits_100_ind, mcc_test_ind, test_acc_ind, test_hits_10_ind_1, test_hits_50_ind_1, test_hits_100_ind_1, mcc_test_ind_1, test_acc_ind_1 = main(int(seed[i]), int(seed[i+n_runs]))
        res[i] = torch.tensor([test_hits_10, test_hits_50, test_hits_100, mcc_test, test_acc])
        res_ind[i] = torch.tensor([test_hits_10_ind, test_hits_50_ind, test_hits_100_ind, mcc_test_ind, test_acc_ind, test_hits_10_ind_1, test_hits_50_ind_1, test_hits_100_ind_1, mcc_test_ind_1, test_acc_ind_1])
        logging.info("number of runs finished = {}".format(i))
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