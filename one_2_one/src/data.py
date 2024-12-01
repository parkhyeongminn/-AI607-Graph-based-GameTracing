import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import dgl
import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import copy

def gnn_train_data_preprocess(data_path, feature_path):
    train_data = pd.read_csv(data_path)
    merged_column = list(train_data['player 1'])
    unique_players = np.unique(np.array(merged_column))
    mapping_dict = {element: index for index, element in enumerate(unique_players)}
    train_data['player 1'] = train_data['player 1'].replace(mapping_dict)
    train_data['player 2'] = train_data['player 2'].replace(mapping_dict)
    train_data['winner'] = train_data['winner'].astype(int)
    
    


    
    feature_data =pd.read_pickle(feature_path)
    node_features = torch.zeros(feature_data.values.shape)
    for key in mapping_dict.keys():
        node_features[mapping_dict[key]] = torch.tensor(feature_data.loc[key,:])
    node_features = torch.tensor(node_features) + torch.tensor(node_features).mean(dim=0)
    
    return train_data, mapping_dict, \
        torch.tensor(train_data[['player 1', 'player 2', 'winner']].values), \
        torch.tensor(node_features, dtype=torch.float64)
        
def gnn_train_valid_data_preprocess(train_data_path, valid_data_path, feature_path):
    train_data = pd.read_csv(train_data_path)
    valid_data = pd.read_csv(valid_data_path)

    data = pd.concat([train_data, valid_data]).reset_index(drop=True)
    merged_column = list(data['player 1'])
    unique_players = np.unique(np.array(merged_column))
    mapping_dict = {element: index for index, element in enumerate(unique_players)}
    data['player 1'] = data['player 1'].replace(mapping_dict)
    data['player 2'] = data['player 2'].replace(mapping_dict)
    data['winner'] = data['winner'].astype(int)

    
    feature_data =pd.read_pickle(feature_path)
    node_features = torch.zeros(feature_data.values.shape)
    for key in mapping_dict.keys():
        node_features[mapping_dict[key]] = torch.tensor(feature_data.loc[key,:])
    node_features = torch.tensor(node_features) 

        
    
    return data, mapping_dict, \
        torch.tensor(data[['player 1', 'player 2', 'winner']].values), \
        torch.tensor(node_features, dtype=torch.float64)

def gnn_valid_data_preprocess(query_data_path, answer_data_path, mapping_dict, node_features):
    valid_data_query = pd.read_csv(query_data_path)
    valid_data_answer = pd.read_csv(answer_data_path)
    valid_data = pd.merge(valid_data_query, valid_data_answer, on = 'game')
    valid_data.loc[valid_data['winner']=='p1', 'winner'] = 0
    valid_data.loc[valid_data['winner']=='DRAW', 'winner'] = 1
    valid_data.loc[valid_data['winner']=='p2', 'winner'] = 2
    
    player_list = np.array(pd.concat([valid_data['player 1'], valid_data['player 2']]))
    
    missing_nodes = set(player_list) - set(mapping_dict.keys())
    updated_mapping_dict = mapping_dict.copy()
    updated_mapping_dict.update({node: max(mapping_dict.values()) + idx + 1 for idx, node in enumerate(missing_nodes)})
    valid_data['player 1'] = valid_data['player 1'].replace(updated_mapping_dict)
    valid_data['player 2'] = valid_data['player 2'].replace(updated_mapping_dict)
    valid_data['winner'] = valid_data['winner'].astype(int)
    
    
    
    additional_unique_players = [updated_mapping_dict[key] for key in missing_nodes]
    
    mean_tensor = node_features.mean(dim=0)
    repeated_mean_tensor = mean_tensor.repeat(len(additional_unique_players), 1)
    integrated_node_features = torch.vstack((node_features, repeated_mean_tensor))
    
    

    
    return valid_data, torch.tensor(valid_data[['player 1', 'player 2', 'winner']].values), \
        updated_mapping_dict, integrated_node_features
        
def gnn_test_data_preprocess(query_data_path, mapping_dict, node_features):
    test_data = pd.read_csv(query_data_path)
    player_list = np.array(pd.concat([test_data['player 1'], test_data['player 2']]))
    missing_nodes = set(player_list) - set(mapping_dict.keys())
    updated_mapping_dict = mapping_dict.copy()
    updated_mapping_dict.update({node: max(mapping_dict.values()) + idx + 1 for idx, node in enumerate(missing_nodes)})
    test_data['player 1'] = test_data['player 1'].replace(updated_mapping_dict)
    test_data['player 2'] = test_data['player 2'].replace(updated_mapping_dict)
    
       
    
    additional_unique_players = [updated_mapping_dict[key] for key in missing_nodes]
    
    mean_tensor = node_features.mean(dim=0)
    repeated_mean_tensor = mean_tensor.repeat(len(additional_unique_players), 1)
    integrated_node_features = torch.vstack((node_features, repeated_mean_tensor))
    
     

    
    return test_data, torch.tensor(test_data[['player 1', 'player 2']].values), \
        updated_mapping_dict, integrated_node_features
    
def graph_constructor(data, mapping_dict):
    
    num_nodes = max(mapping_dict.values()) + 1 
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for index, row in data.iterrows():
        adj_matrix[int(row['player 1']), int(row['player 2'])] = 1  
        adj_matrix[int(row['player 2']), int(row['player 1'])] = 1
    
    sparse_matrix = csr_matrix(adj_matrix)
    g = dgl.from_scipy(sparse_matrix)
    
    return g

def mf_read_data(train_path, valid_query_path, valid_answer_path, args):
    train_df = pd.read_csv(train_path)
    num_player1 = args.num_player
    num_player2 = args.num_player
    
    train_df.loc[train_df['winner'] == 'p1', 'winner'] = 2
    train_df.loc[train_df['winner'] == 'DRAW', 'winner'] = 1
    train_df.loc[train_df['winner'] == 'p2', 'winner'] = 0
    train_df['winner'] = train_df['winner'].astype(int)

    updated_train_df = train_df.copy()
    updated_train_df['player 1'], updated_train_df['player 2'] = train_df['player 2'], train_df['player 1']
    updated_train_df['winner'] = 2 - train_df['winner']
    
    preprocessed_train_df = pd.concat([train_df, updated_train_df], ignore_index=True)
    preprocessed_train_df = preprocessed_train_df.sort_values(by=["game"], ascending=[True])
    
    train_data = preprocessed_train_df.reset_index(drop=True)
    
    valid_query_df = pd.read_csv(valid_query_path)
    
    
    valid_answer_df = pd.read_csv(valid_answer_path)
    

    valid_answer_df.loc[valid_answer_df['winner'] == 'p1', 'winner'] = 2
    valid_answer_df.loc[valid_answer_df['winner'] == 'DRAW', 'winner'] = 1
    valid_answer_df.loc[valid_answer_df['winner'] == 'p2', 'winner'] = 0
    valid_answer_df['winner'] = valid_answer_df['winner'].astype(int)

    
    valid_data = valid_query_df.merge(valid_answer_df, how='left', on='game')
    
    return train_data, valid_data, num_player1, num_player2


class MFDataset(Dataset):
    def __init__(self, df):
        # class init
        self.df = df
        
        self.player1 = torch.tensor(self.df['player 1'].values)
        self.player2 = torch.tensor(self.df['player 2'].values)
        self.winner = torch.tensor(self.df['winner'].values)

    def __len__(self):
        return len(self.player1)

    def __getitem__(self, idx):
        player1 = self.player1[idx]
        player2 = self.player2[idx]
        rating = self.winner[idx]
            
        return (player1, player2, rating.float())
    
class MFDataset_Test(Dataset):
    def __init__(self, df):
        # class init
        self.df = df
        
        self.player1 = torch.tensor(self.df['player 1'].values)
        self.player2 = torch.tensor(self.df['player 2'].values)

    def __len__(self):
        return len(self.player1)

    def __getitem__(self, idx):
        player1 = self.player1[idx]
        player2 = self.player2[idx]
            
        return (player1, player2)