# OOD Link Prediction Generalization Capabilities of Message Passing GNNs in Larger Test Graphs

## Overview

This repository is the official implementation of OOD Link Prediction Generalization Capabilities of Message Passing GNNs in Larger Test Graphs.

We empirically validate our theoretical results using Stochastic block models. And we perform link predictions tasks using structural nodel representation GNN models: [GraphSAGE](https://github.com/williamleif/GraphSAGE), [GCN](https://github.com/tkipf/gcn), [GAT](https://personal.utdallas.edu/~fxc190007/courses/20S-7301/GAT-questions.pdf), [GIN](https://github.com/weihua916/powerful-gnns) before a link prediction MLP model using [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric).

## Install

To create the conda environment `generalize`, use the following command

```bash
conda env create -f dependency.yml
```

Activate and use the `generalize` environment for further running of the code.

```bash
conda activate generalize
```
 
 ## How to run
 
 The convergence and stability results is shown in the notebook Convergence_and_stability.ipynb.
 
 We give the example code to run the link prediction task using GraphSAGE model over randomly generated Stochastic block model with training graph size 1000 and inductive test graph size 10000. Here we optimize using Adam with learning rate 0.0005 for 1,000 epochs and a minibatch size 128 with hidden dimension 10 and number of hidden layers 3 over 50 randomly initialized runs with concatenated node representations.
 
 ```train
 python link_prediction_new.py -n 10 -n_runs 50 -n_layer 3 -e 1000 -hid 10 -b 128 -noinn -l 5e-4 -method SAGE
 ```
 
 * `-method` can be set from `[SAGE, GCN, GAT, GIN]`
 * Change `-n` to have the training graph size `100n`
 * Change `-n_runs` to set different number of independent runs
 * Change `-hid` to set the number of hidden dimensions in the GNN and link prediction MLP
 * Change `-n_layers` to set number of layers in the GNN and link prediction MLP
 * Change `-e` to set number of epochs
 * Change `-b` to set batch size
 * Change `-l` to set learning rate
 * Remove `-noinn` to perform link prediction using inner products of node representations obtained from GNNs


 To run our proposed 2-gMPNN procedure, the example code is 
  ```train
  python learn-2-gmpnn.py -n 10 -n_runs 50 -e 1000 -hid 5 -b 128 -l 5e-4 -n_runs 50
  ```
  
 To run our proposed 2-gMPNN procedure with fixed function (no learning), the example code is 
  ```train
  python learn-2-gmpnn-fix.py -n 10 -n_layers 3 -hid 10 -e 1000 -n_runs 50 -b 128 -l 5e-4
  ```
  
  The arguments have the same meansing as above.
