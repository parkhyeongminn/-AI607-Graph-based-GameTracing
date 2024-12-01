import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

import task2_utils

def load_dataloader(survival_train, survival_valid, args):
    _, train_player_game, train_score, _, _  = task2_utils.load_hyperedge_label(survival_train)
    _, valid_player_game, valid_score, _, _ = task2_utils.load_hyperedge_label(survival_valid)

    train_player_game = torch.LongTensor(train_player_game)
    valid_player_game = torch.LongTensor(valid_player_game)
    
    train_idx = torch.LongTensor(np.arange(len(train_player_game))) # ~ 252654
    valid_idx = torch.LongTensor(np.arange(len(train_player_game), len(train_player_game) + len(valid_player_game))) # 252655 ~ 270611

    train_dataset = TensorDataset(
        train_player_game[:, 0],  # player tensor
        train_player_game[:, 1],  # game tensor
        train_idx,  # Index tensor
        torch.LongTensor(train_score) # score tensor
    )

    # Create TensorDataset for validation data
    valid_dataset = TensorDataset(
        valid_player_game[:, 0],  # player tensor
        valid_player_game[:, 1],  # game is a tensor
        valid_idx,  # Index tensor
        torch.LongTensor(valid_score)  # score tensor
    )

    batch_size = args.batch_size  
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    infer_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    return train_dataloader, valid_dataloader, infer_dataloader

def load_dataloader_test(survival_train, survival_valid, survival_test, args):
    _, train_player_game, train_score, _, _  = task2_utils.load_hyperedge_label(survival_train)
    _, valid_player_game, valid_score, _, _ = task2_utils.load_hyperedge_label(survival_valid)
    _, test_player_game, _, _, _ = task2_utils.load_hyperedge_label(survival_test)

    train_player_game = torch.LongTensor(train_player_game)
    valid_player_game = torch.LongTensor(valid_player_game)
    test_player_game = torch.LongTensor(test_player_game)
    
    train_idx = torch.LongTensor(np.arange(len(train_player_game))) # ~ 270611  
    valid_idx = torch.LongTensor(np.arange(len(train_player_game) - len(valid_player_game), len(train_player_game))) # 252655 ~ 270611 
    test_idx = torch.LongTensor(np.arange(len(train_player_game), len(train_player_game) + len(test_player_game))) # 270612 ~

    train_dataset = TensorDataset(
        train_player_game[:, 0],  # player tensor
        train_player_game[:, 1],  # game tensor
        train_idx,  # Index tensor
        torch.LongTensor(train_score)  # score tensor
    )

    # Create TensorDataset for validation data
    valid_dataset = TensorDataset(
        valid_player_game[:, 0],   # player tensor
        valid_player_game[:, 1],  # game is a tensor
        valid_idx,  # Index tensor
        torch.LongTensor(valid_score)  # score tensor
    )

    test_dataset = TensorDataset(
        test_player_game[:, 0],   # player tensor
        test_player_game[:, 1],  # game is a tensor
        test_idx,  # Index tensor
    )

    batch_size = args.batch_size  # Adjust the batch size based on your needs
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    infer_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_dataloader, valid_dataloader, test_dataloader, infer_dataloader