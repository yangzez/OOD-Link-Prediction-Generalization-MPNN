# OOD Link Prediction Generalization Capabilities of Message Passing GNNs in Larger Test Graphs

## Overview

This repository is the official implementation of OOD Link Prediction Generalization Capabilities of Message Passing GNNs in Larger Test Graphs.

We empirically evaluate the performance of Set Twister on a variety of arithmetic tasks over image inputs and randomly encoded inputs, in terms of prediction accuracy, mean absolute error (MAE) and computational complexity. We compare Set Twister's performance against widely used permutation-invariant representations on a variety of tasks for which we know the task's high-order dependencies: [Deep Sets](https://github.com/manzilzaheer/DeepSets), [2-ary Janossy Pooling](https://github.com/PurdueMINDS/JanossyPooling), [Full Janossy Pooling using GRUs with attention mechanisms](https://github.com/PurdueMINDS/HATS) (JP Full) and [Set Transformer](https://github.com/juho-lee/set_transformer) without inducing points.

## Install

To create the conda environment `test`, use the following command

```bash
conda env create -f dependency.yml
```

Activate and use the `test` environment for further running of the code.
 
 ## How to run
 
 The convergence and stability results is shown in the notebook Convergence_and_stability.ipynb.
 
 We give the example code to run the link prediction task using GraphSAGE model over randomly generated Stochastic block model with $N^\text{tr}=1000$ and $N^\text{te}=10000$. Here we optimize using Adam with learning rate 0.0005 for 1,000 epochs and a minibatch size 128.
 ```train
 python train.py -t maxmin -v -l 5e-4 -g 0 -p adam -b 128 -e 2000 -a tanh -m 2 -seq 5 -img
 ```
 * Change the argument after `-t` for different tasks
 * Set `-m 1` to run the DeepSets model since M=k=1 corresponds to the DeepSets model
 * Set `-m 1 -jk 2` to run the 2-ary Janossy Pooling model
 * Add `-full` to run the Full JP model
 * Add `-sab` to run the Set Transformer model
 * Remove `-img` to run the tasks over randomly encoded inputs
