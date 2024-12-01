import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data import *
from model import *
from utils import VALIDEarlyStopping, BalancedCrossEntropyLoss
import json
import argparse


def gnn_train(args):
    if not os.path.isdir("../ckpts"):
        os.mkdir("../ckpts")
    
    ckpt_path = os.path.join("../ckpts", args.model)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    TRAIN_PATH = "../data/preprocessed_one2one_training.csv"
    FEATURE_PATH = "../data/scaled_features1_updated.p"
    VALID_QUERY_PATH = "../data/one2one_valid_query.csv"
    VALID_ANSWER_PATH = "../data/one2one_valid_answer.csv"

    
    
    
    random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    torch.cuda.manual_seed(args.seed)

    if args.cuda == "true" and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    
    train_data, mapping_dict, input_train_data, node_features = gnn_train_data_preprocess(TRAIN_PATH, FEATURE_PATH)

    valid_data, input_valid_data, updated_mapping_dict, node_features = gnn_valid_data_preprocess(VALID_QUERY_PATH,
                                                        VALID_ANSWER_PATH,
                                                        mapping_dict,
                                                        node_features)
    
    
    integrated_data = pd.concat([train_data, valid_data]).reset_index(drop=True)
    graph = graph_constructor(integrated_data, updated_mapping_dict)

    train_input = input_train_data[:, :2]
    train_label = input_train_data[:, 2:]
    valid_input = input_valid_data[:, :2]
    valid_label = input_valid_data[:, 2:]

    train_dataset = TensorDataset(torch.tensor(train_input), torch. tensor(train_label))
    valid_dataset = TensorDataset(torch.tensor(valid_input), torch.tensor(valid_label))

    train_dataloader = DataLoader(train_dataset, batch_size=args.gnn_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.gnn_batch_size, shuffle=True)

    model = GCN(args, graph, node_features, device).to(device).to(torch.float64)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.gnn_lr)
    criterion = BalancedCrossEntropyLoss(weights=torch.tensor([1.0, 2.4, 1.0],dtype=float))

    early_stopper = VALIDEarlyStopping(patience=args.patience, save_path=ckpt_path, eps=args.eps)

    best_valid_accuracy = 0.0
    previous_accuracy = 0.0
    for epoch in range(1, args.epoch+1):
        
        train_loss = 0.0
        model.train()
        for i, train in enumerate(tqdm(train_dataloader)):
            input = train[0].to(device)
            true_class_label = train[1].squeeze().to(device)
            predicted_class_label = model(input)
            
            loss = criterion(predicted_class_label, true_class_label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss_mean = train_loss/i
        
        model.eval()
        valid_loss = 0.0
        label_list = []
        predicted_list = []
        with torch.no_grad():
            for ii, valid in tqdm(enumerate(valid_dataloader)):
                input = valid[0].to(device)
                true_class_label = valid[1].squeeze().to(device)
                
                predicted_class_label = model(input)

                loss = criterion(predicted_class_label, true_class_label)
                
                predicted_list += list(np.argmax(np.array(predicted_class_label.detach().cpu()), axis=1))
                
                label_list += list(np.array(true_class_label.detach().cpu()))
                
                valid_loss += loss.item()
            
            valid_loss_mean = valid_loss/ii
            label_difference = np.abs(np.array(predicted_list)-np.array(label_list), dtype=np.float32)
            label_difference[label_difference == 2] = -1
            label_difference[label_difference == 1] = 0.5
            label_difference[label_difference == 0] = 1
            label_difference[label_difference == -1] = 0
            


            
            valid_accuracy = label_difference.sum().item() * (100/len(valid_label))
            
            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
            
        print("Epoch: {}, Train Loss: {: .4f}, Valid Loss: {: .4f}, Valid Accuracy: {: .4f}, Best Accuracy: {: .4f}"
                .format(epoch, train_loss_mean, valid_loss_mean, valid_accuracy, best_valid_accuracy))

        result_mapping = {0:'p1', 1:'DRAW', 2:'p2'}
        valid_converted_list = [result_mapping[item] for item in predicted_list]
        if early_stopper.should_stop(model, previous_accuracy, valid_accuracy, valid_converted_list):
            print(f"Early Stopping: [Epoch: {epoch}]")
            
        previous_accuracy = best_valid_accuracy
        
    return best_valid_accuracy

def mf_train(args):
    if not os.path.isdir("../ckpts"):
        os.mkdir("../ckpts")
    
    ckpt_path = os.path.join("../ckpts", args.model)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    TRAIN_PATH = '../data/one2one_training.csv'
    FEATURE_PATH = "../data/scaled_features1_updated.p"
    VALID_QUERY_PATH = "../data/one2one_valid_query.csv"
    VALID_ANSWER_PATH = "../data/one2one_valid_answer.csv"

    
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    if args.cuda == "true" and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    
    df_tr, df_val, num_player1, num_player2 = mf_read_data(TRAIN_PATH, VALID_QUERY_PATH,
                                                     VALID_ANSWER_PATH, args)
    sparsity = 1 - len(df_tr) / (num_player1 * num_player2)
    print(f'number of player 1: {num_player1}, number of player 2: {num_player2}')
    print(f'matrix sparsity: {sparsity:f}')
    
    train_dataset = MFDataset(df=df_tr)
    valid_dataset = MFDataset(df=df_val)
    

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.mf_batch_size, shuffle=True) 
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.mf_batch_size, shuffle=False)
    
    model = MF(args.mf_factor, num_player1, num_player2).to(device)
    criterion = nn.MSELoss()
    optm = optim.Adam(model.parameters(),lr=args.mf_lr)
    
    train_losses = []
    valid_losses = []

    for epoch in range(args.epoch):
        model.train()
        loss_sum = 0.0
        
        for i, (player1, player2, rating) in enumerate(tqdm(train_dataloader)):
            model.zero_grad()
        
            player1 = player1.to(device)
            player2 = player2.to(device)
            rating = rating.to(device)

            preds = model(player1, player2)
            loss = criterion(preds, rating)
        
            optm.zero_grad()     
            loss.backward()   
            optm.step()    
            
            loss_sum += loss.item()
        
        train_loss = loss_sum / len(train_dataloader)
        
        with torch.no_grad():
            model.eval()
            loss_sum = 0.0

            for i, (player1, player2, rating) in enumerate(tqdm(valid_dataloader)):
                player1 = player1.to(device)
                player2 = player2.to(device)
                rating = rating.to(device)

                preds = model(player1, player2)
                loss = criterion(preds, rating)

                loss_sum += loss.item()

            valid_loss = loss_sum / len(valid_dataloader)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        print(f'epoch: {epoch}, train Loss: {train_loss:.4f}, valid Loss: {valid_loss:.4f}')
        
        all_predictions = []
        evaluated_result = pd.DataFrame()
        
    print(f"MF inference")
        
    with torch.no_grad():
        model.eval()
        
        for (player1, player2, _) in tqdm(valid_dataloader):
            player1, player2 = player1.to(device), player2.to(device)

            preds = model(player1, player2)
        
            modified_preds = torch.where(preds > 2, 2, torch.where(preds < 0, 0, preds))

            all_predictions.extend(torch.round(modified_preds).cpu().numpy())

    evaluated_result['winner'] = all_predictions

    print("Predicted Results: \n")
    print(evaluated_result['winner'].value_counts())
    
    score = 0

    for (idx1, row1), (idx2, row2) in zip(df_val.iterrows(), evaluated_result.iterrows()):    
        if row1['winner'] == row2['winner']:
            score += 1
        
        if row1['winner'] == 1:
            if row2['winner'] == 0 or row2['winner'] == 2:
                score += 0.5
                
        else:
            if row2['winner'] == 1:
                score += 0.5
            
    print(np.round((score / len(df_val) * 100),4))
    
    
