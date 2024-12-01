import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from data import *
from model import *
from utils import VALIDEarlyStopping, BalancedCrossEntropyLoss
from train import gnn_train, mf_train
from test import gnn_test
import json
import argparse
import pickle
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--in_dim', type=int, default=6, help='number of initial features')
parser.add_argument('--hid_dim', type=int, default=10, help='hidden dimension')
parser.add_argument('--num_classes', type=int, default=3, help='number of class')
parser.add_argument('--aspect_dim', type=int, default=5, help='aspect dimension')
parser.add_argument('--num_aspect', type=int, default=4, help='number of aspects')
parser.add_argument('--gnn_lr', type=float, default=.001, help='learning rate for gnn-based model')
parser.add_argument('--mf_lr', type=float, default=.01, help='learning rate for mf model')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--gnn_batch_size', type=int, default=128, help='batch size for gnn-based model')
parser.add_argument('--mf_batch_size', type=int, default=32, help='batch size for mf model')
parser.add_argument('--patience', type=int, default=5, help='patience')
parser.add_argument('--eps', type=float, default=.0001, help='epsilon value for early stopping')
parser.add_argument('--cuda', type=str, default="true", help='gpu use')
parser.add_argument('--model', type=str, default="gnn_based_train", help='method')
parser.add_argument('--num_player', type=int, default=5997, help='number of whole players')
parser.add_argument('--mf_factor', type=int, default=256, help='factors for matrix factorization')


args = parser.parse_args()

TEST_QUERY_PATH = "../data/one2one_test_query.csv"


if args.model == "mf_based_train":
    mf_train(args)
elif args.model == "gnn_based_train":
    best_valid_accuracy = gnn_train(args)
    print("Best Valid Accuracy: {: .4f}".format(best_valid_accuracy))
elif args.model == "gnn_based_test":
    test_predicted_list = gnn_test(args)
    test_df = pd.read_csv(TEST_QUERY_PATH)
    test_df['winner'] = test_predicted_list
    test_df.to_csv("../result/one2one_test_prediction.csv", index=False)
    print("Test Result has been successfully obtained")
else:
    print("Invalid Argument!!")
    
