import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch.utils.data import DataLoader
from ogb.linkproppred import Evaluator
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

        num_neg_edge = perm.size(0)
        indice_10 = torch.from_numpy(np.random.choice(45 * n, num_neg_edge))
        indice_20 = torch.from_numpy(np.random.choice(45 * n, num_neg_edge) + 55 * n)
        indice_11 = torch.from_numpy(np.random.choice(55 * n, num_neg_edge) + 45 * n)
        indice_21 = torch.from_numpy(np.random.choice(55 * n, num_neg_edge))
        id_0 = torch.cat((indice_10, indice_20)).reshape(1, -1)
        id_1 = torch.cat((indice_11, indice_21)).reshape(1, -1)
        edge = torch.cat((id_0, id_1), dim=0)

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
    if not args.W:
        pred = predictor(f.reshape(-1,1)).reshape(args.num_nodes*100, args.num_nodes*100)
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
def test_ind(predictor, length, n, evaluator, batch_size, seed):
    # Inductive
    #n = 1
    n_1 = 45*n
    n_3 = n_1
    n_2 = 10*n
    sizes = [45 * n, 10 * n, 45 * n]
    probs = [[0.6, 0.05, 0.02], [0.05, 0.6, 0.05], [0.02, 0.05, 0.6]]
    g = nx.stochastic_block_model(sizes, probs, seed=seed)
    data = from_networkx(g)
    num_nodes = g.number_of_nodes()
    idx = torch.randperm(data.edge_index.size(1))
    idx_remain = idx[:int(0.9 * data.edge_index.size(1))]
    idx_hide = idx[int(0.9 * data.edge_index.size(1)):]
    edge_index_hide = data.edge_index[:, idx_hide]  # hidden edges for train, valid, test purpose
    data.edge_index = data.edge_index[:, idx_remain]  # the remaining graphs
    a = torch.sparse.FloatTensor(data.edge_index, torch.ones(data.edge_index.size(1)),
                                 torch.Size([num_nodes, num_nodes]))  # .to(device)
    f = torch.ones(num_nodes, num_nodes).to(device) #initialization of f_{i,j}

    idx = torch.randperm(edge_index_hide.size(1))
    idx_tr = idx[:int(0.8 * edge_index_hide.size(1))]
    idx_va = idx[int(0.8 * edge_index_hide.size(1)):int(0.9 * edge_index_hide.size(1))]
    idx_te = idx[int(0.9 * edge_index_hide.size(1)):]
    #length = idx_va.size(0)
    #print(length)
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
        f = W

    if not args.W:
        pred = predictor(f.reshape(-1,1))
        pred = pred.reshape(num_nodes, num_nodes)
    else:
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
    probs = [[0.6, 0.05, 0.02], [0.05, 0.6, 0.05], [0.02, 0.05, 0.6]]
    g = nx.stochastic_block_model(sizes, probs, seed=seed1)
    data = from_networkx(g)
    num_nodes = g.number_of_nodes()
    idx = torch.randperm(data.edge_index.size(1))
    idx_remain = idx[:int(0.9 * data.edge_index.size(1))]
    idx_hide = idx[int(0.9 * data.edge_index.size(1)):]
    edge_index_hide = data.edge_index[:, idx_hide]  # hidden edges for train, valid, test purpose
    data.edge_index = data.edge_index[:, idx_remain]  # the remaining graphs
    a = torch.sparse.FloatTensor(data.edge_index, torch.ones(data.edge_index.size(1)),
                                 torch.Size([num_nodes, num_nodes]))  # .to(device)
    f = torch.ones(num_nodes, num_nodes).to(device) #initialization of f_{i,j}

    idx = torch.randperm(edge_index_hide.size(1))
    idx_tr = idx[:int(0.8 * edge_index_hide.size(1))]
    idx_va = idx[int(0.8 * edge_index_hide.size(1)):int(0.9 * edge_index_hide.size(1))]
    idx_te = idx[int(0.9 * edge_index_hide.size(1)):]
    length = idx_va.size(0)
    print(length)
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
        w1 = torch.cat((torch.ones(int(n_1)) * 0.6, torch.ones(int(n_2)) * 0.05, torch.ones(int(n_3)) * 0.02))
        w2 = torch.cat((torch.ones(int(n_1)) * 0.05, torch.ones(int(n_2)) * 0.6, torch.ones(int(n_3)) * 0.05))
        w3 = torch.cat((torch.ones(int(n_1)) * 0.02, torch.ones(int(n_2)) * 0.05, torch.ones(int(n_3)) * 0.6))
        W = torch.cat((w1.repeat(n_1, 1), w2.repeat(n_2, 1), w3.repeat(n_3, 1)), dim=0)#.to(device)
        #W.fill_diagonal_(1.0)
        f = W

    # x is the output we will obtain for f_{i,j}
    print(f)
    predictor = LinkPredictor(1, args.hid_dim, 1, args.num_layers, 0).to(device)
    torch.manual_seed(123)
    predictor.reset_parameters()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.learning_rate)
    evaluator = Evaluator(name='ogbl-ddi')
    if not args.W:
        best_valid_acc = 0
        checkpoint_file_name = "5-15" + str(args.num_nodes) + str(args.hid_dim) + str(args.num_runs) + "model_2GNN_fixed_checkpoint.pth.tar"

        for epoch in range(1, 1 + args.n_epochs):
            loss = train(predictor, f, split_edge, data.edge_index, optimizer, batch_size=args.batch*int(n)*int(n))
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

    n_1 = n
    results_ind = test_ind(predictor, length, n_1, evaluator, batch_size=1280 * int(n_1) * int(n_1), seed=seed2)
    print(results_ind)
    n_1 = 10 * n
    results_ind_1 = test_ind(predictor, length, n_1, evaluator, batch_size=1280 * int(n_1) * int(n_1), seed=seed2)
    print(results_ind_1)
    print('Inductive result for ' + str(args.method) + 'of ' + str(n_1))

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