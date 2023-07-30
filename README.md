# HyperDNE

PyTorch implementation of paper "HyperDNE: Enhanced hypergraph neural network for dynamic network embedding".

## 1. Data preprocessing

HyperDNE follows the preprocessing step of [DySAT], which separates all interactions into `u, v, t` triplet. `u` and `v` denote a link between two nodes in the graph, `t` denotes the discrete time step when the interaction happens.

**Note**: Fig 2 in the paper demonstrates how a simple graph converts to hypergraph.

## 2. Training

Here is the default arguments for running HyperDNE (including line convolutional layers, multi-step prediction):

> `python main.py --dataset={enron|uci|yelp|m1} --predict_time={predict_time} --n_predict_steps=7 --n_line_conv_layers=2 --include_line_conv_weight --l2_reg={l2_regularization_factor}`

The main arguments for training are listed below:
- `--predict_time`: The time step to predict (starting from 0), denoted as *t*. The model uses all data in 0~*t-1* to train and predicts all edges in time step from *t* to *t+N-1* where *N* is the value of `--n_predict_steps`.
- `--n_predict_steps`: How many time steps to predict. For single-step prediction, set to 2; for multi-step prediction, set to 7. (**see the notes below**)
- `--n_line_conv_layers`: The layers of line convolutional module (for enhancing hyperedge embedding), default: 2, disable it by setting to 0.
- `--n_hgcn_layers`: The layers of Hypergraph Convolutional Network (HGCN), default: 2.
- `--n_transformer_layers`: The layers of self-attention module, default: 2, disable it by setting to 0.
- `--include_line_conv_weight`: Add trainable parameters to line convolutional module.
- `--l2_reg`: The factor for L2 regularization, its value depends on which dataset is used. Generally, a small value like 0.1 or 0.5 is ok.

Other arguments are generally unchanged.

**Note**: We found a potential data leakage problem in [DySAT] that uses the data in time step *t* for training. Therefore, we run all evaluation starting from *t+1* instead of *t* to fix it (the evaluation result for time step *t* is ignored). That's why we use `--n_predict_steps=2` for single-step prediction and `--n_predict_steps=7` for multi-step prediction (matches 6 steps described in the paper).

[DySAT]: https://github.com/aravindsankar28/DySAT
