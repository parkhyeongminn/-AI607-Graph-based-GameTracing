import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from data import *
from model import *
from utils import TESTEarlyStopping, BalancedCrossEntropyLoss
import json

    

def gnn_test(args):
    if not os.path.isdir("../ckpts"):
        os.mkdir("../ckpts")
    
    ckpt_path = os.path.join("../ckpts", args.model)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    TRAIN_PATH = "../data/preprocessed_one2one_training.csv"
    FEATURE_PATH = "../data/scaled_features1_updated_trainvalid.p"
    VALID_PATH = "../data/preprocessed_one2one_validation.csv"
    TEST_QUERY_PATH = "../data/one2one_test_query.csv"

    
    
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

    
        
    train_data, mapping_dict, input_train_data, node_features = gnn_train_valid_data_preprocess(TRAIN_PATH, VALID_PATH, FEATURE_PATH)

  
    test_data, input_test_data, updated_mapping_dict, node_features = gnn_test_data_preprocess(TEST_QUERY_PATH,
                                                        mapping_dict,
                                                        node_features)
 
    

    integrated_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    graph = graph_constructor(integrated_data, updated_mapping_dict)

    train_input = input_train_data[:, :2]
    train_label = input_train_data[:, 2:]
    test_input = input_test_data[:, :2]
        

    train_dataset = TensorDataset(torch.tensor(train_input), torch. tensor(train_label))
    test_dataset = TensorDataset(torch.tensor(test_input))

    train_dataloader = DataLoader(train_dataset, batch_size=args.gnn_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.gnn_batch_size, shuffle=True)

    model = GCN(args, graph, node_features, device).to(device).to(torch.float64)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.gnn_lr)
    criterion = BalancedCrossEntropyLoss(weights=torch.tensor([1.0, 2.4, 1.0],dtype=float))
    
    early_stopper = TESTEarlyStopping(patience=args.patience, save_path=ckpt_path, eps=args.eps)

    best_train_accuracy = 0.0
    previous_accuracy = 0.0
    for epoch in range(1, args.epoch+1):
        
        train_loss = 0.0
        train_label_list = []
        train_predicted_list = []
        model.train()
        for i, train in enumerate(tqdm(train_dataloader)):
            input_data = train[0].to(device)
            true_class_label = train[1].squeeze().to(device)
           
            predicted_class_label = model(input_data)
        
        
            loss = criterion(predicted_class_label, true_class_label)
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            train_predicted_list += list(np.argmax(np.array(predicted_class_label.detach().cpu()), axis=1))
                
            train_label_list += list(np.array(true_class_label.detach().cpu()))
            
        train_loss_mean = train_loss/i
        label_difference = np.abs(np.array(train_predicted_list)-np.array(train_label_list), dtype=np.float32)
        label_difference[label_difference == 2] = -1
        label_difference[label_difference == 1] = 0.5
        label_difference[label_difference == 0] = 1
        label_difference[label_difference == -1] = 0
        
        train_accuracy = label_difference.sum().item() * (100/len(train_label))
        
        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            
        print("Epoch: {}, Train Loss: {: .4f}, Train Accuracy: {: .4f}, Best Accuracy: {: .4f}"
            .format(epoch, train_loss_mean, train_accuracy, best_train_accuracy))
        
        result_mapping = {0:'p1', 1:'DRAW', 2:'p2'}
        valid_converted_list = [result_mapping[item] for item in train_predicted_list]
        if early_stopper.should_stop(model, previous_accuracy, train_accuracy, valid_converted_list):
            print(f"Early Stopping: [Epoch: {epoch}]")
            break
        
        previous_accuracy = train_accuracy
    
        
    model.eval()
    
    test_predicted_list = []
    with torch.no_grad():
        checkpoint = torch.load(os.path.join(
            ckpt_path, "best_model_epoch.ckpt"))
        model.load_state_dict(checkpoint)
        for ii, test in tqdm(enumerate(test_dataloader)):
            input_data = test[0].to(device)
            
            predicted_class_label = model(input_data)
            
            test_predicted_list += list(np.argmax(np.array(predicted_class_label.detach().cpu()), axis=1))
        
    
    

    result_mapping = {0:'p1', 1:'DRAW', 2:'p2'}
    test_converted_list = [result_mapping[item] for item in test_predicted_list]
        
    return test_converted_list

