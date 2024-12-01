import task2_utils
import task2_dataset
import task2_model
import task2_trainer

import pandas as pd
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import pickle
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader

def main(args):
    # train : train + valid
    # valid : valid
    # inference : test

    args.test = True

    seed = args.seed

    task2_utils.set_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(device)

    # Load data
    survival_train, survival_valid_q, survival_valid_a, survival_test_q = task2_utils.load_data()

    survival_valid, survival_test_q = task2_utils.preprocess_valid_test(survival_valid_q, survival_valid_a, survival_test_q)

    # label encoding(train + valid + test)
    mapping_dict, survival_total = task2_utils.player_encoding(survival_train, survival_valid, survival_test_q)

    train_size = len(survival_train)
    valid_size = len(survival_valid)

    test_start_idx = train_size + valid_size

    # train valid split
    survival_train = survival_total.iloc[:test_start_idx, :] # train + valid
    survival_valid = survival_total.iloc[train_size:test_start_idx] # valid
    survival_test = survival_total.iloc[test_start_idx:, :] # test

    print('Data Preprocessing')

    args = task2_utils.preprocess_data(survival_train, survival_test, survival_total, mapping_dict, args)
    
    print('load dataloader')
    train_dataloader, valid_dataloader, test_dataloader, infer_dataloader = task2_dataset.load_dataloader_test(survival_train, survival_valid, survival_test, args)

    best_model = task2_trainer.trainer(train_dataloader, valid_dataloader, args)

    # inference
    survival_test_inference = pd.DataFrame(columns = survival_valid_a.columns)

    survival_test_inference = task2_trainer.inference_test(survival_test_inference, best_model, infer_dataloader, args)

    # Save csv
    survival_test_inference.to_csv(f'../result/survival_test_prediction.csv', index=False)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose", dest='verbose', action='store_const', default=False, const=True, help='Print out verbose info during optimization')
    parser.add_argument("--seed", default = 42, type=int, help='seed')
    parser.add_argument("--exp_wt", dest='use_exp_wt', action='store_const', default=False, const=True, help='Fix seed for reproducibility and fair comparison.')
    parser.add_argument("--predict_edge", action='store_const', default=False, const=True, help='whether to predict edges')
    parser.add_argument("--edge_linear", action='store_const', default=False, const=True, help='linerity')
    parser.add_argument("--use_edge_lin", action='store_const', default=False, const=True, help='use additional linear layer in hypermod')

    parser.add_argument("--n_hidden", default=100, type=int, help='number of hidden dimension')
    parser.add_argument("--final_edge_dim", default=50, type=int, help='number of edge_lin dimension')
    parser.add_argument("--n_epoch", default=200, type=int, help='number of epoch')
    parser.add_argument("--batch_size", default=64, type=int, help='batch_size')

    parser.add_argument("--normalize", action='store_const', default=False, const=True, help='whether to normalize node feature')
    parser.add_argument("--node_feature_name", type=str, default='features2_trainvalid_notnorm', help='node feature name')
    parser.add_argument("--alpha_e", default=0, type=float, help='alpha')
    parser.add_argument("--alpha_v", default=0, type=float, help='alpha')
    parser.add_argument("--dropout_p", default=0.3, type=float, help='dropout')
    parser.add_argument("--n_layers", default=1, type=int, help='number of layers')
    parser.add_argument("--lr", default=0.04, type=float, help='learning rate')

    parser.add_argument("--test", default=True, help='not change')

    args = parser.parse_args()

    print(args)

    main(args)