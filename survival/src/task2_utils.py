import numpy as np
import pandas as pd
import random
import pickle
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict, Counter

import torch
import torch.backends.cudnn as cudnn


TRAIN_PATH = '../data/survival_training.csv'
VALID_QUERY_PATH = '../data/survival_valid_query.csv'
VALID_ANSWER_PATH = '../data/survival_valid_answer.csv'
TEST_QUERY_PATH = '../data/survival_test_query.csv'

def load_data():
    survival_train = pd.read_csv(TRAIN_PATH)
    survival_valid_q = pd.read_csv(VALID_QUERY_PATH)
    survival_valid_a = pd.read_csv(VALID_ANSWER_PATH)
    survival_test_q = pd.read_csv(TEST_QUERY_PATH)

    return survival_train, survival_valid_q, survival_valid_a, survival_test_q

def preprocess_valid_test(survival_valid_q, survival_valid_a, survival_test_q):
    # valid query, answer merge
    survival_valid = survival_valid_q.merge(survival_valid_a, how='inner', on='game')

    # test score column add
    for i in range(8):
        score_key = f'score {i}'

        survival_test_q[score_key] = pd.Series(dtype='float64')

    return survival_valid, survival_test_q

def load_node_feature(feature_name, mapping_dict, survival_train, survival_valid, args):
    # Open the node feature pickle file
    with open(f'../data/{feature_name}.p', 'rb') as file:
        data = pickle.load(file)

    # # Min max normalize
    if args.normalize:
        scaler = MinMaxScaler()

        # Fit and transform the data using MinMaxScaler
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns,  index=data.index)

    # label encoding
    data.index = data.index.map(mapping_dict)

    train_player_idx = get_players_idx(survival_train)
    val_player_idx = get_players_idx(survival_valid)

    # Create player features that are not in the train but only in the validation data
    only_valid_player = set(np.array(val_player_idx)) - set(np.array(train_player_idx))
    X_mean = data.mean() # Node without node features -> Use average of existing node features

    for player in tqdm(only_valid_player):
        data.loc[player * 1.0] = X_mean

    data = data.sort_index()
    return data

def make_hyperedge(train_player_game, valid_player_game):
    train_player_game = torch.LongTensor(train_player_game)
    valid_player_game = torch.LongTensor(valid_player_game)

    player_game = torch.LongTensor(np.concatenate([np.array(train_player_game), np.array(valid_player_game)]))

    return player_game

def preprocess_data(survival_train, survival_valid, survival_trval, mapping_dict, args):

    # Load hyperedge and label
    _, train_player_game, train_score, _, _ = load_hyperedge_label(survival_train)
    _, valid_player_game, valid_score, _, _ = load_hyperedge_label(survival_valid)

    train_player_game = torch.LongTensor(train_player_game)
    valid_player_game = torch.LongTensor(valid_player_game)

    # Make Hyperedge (player, game)
    player_game = make_hyperedge(train_player_game, valid_player_game)

    # Define number of nodes(players), number of edges(games), labels

    n_players = max(mapping_dict.values()) + 1
    n_games = max(survival_trval['game']) + 1
    if args.test: # train : train + valid, test: test(no score)
        score = train_score
    else: # train : train , test: valid
        score = train_score + valid_score

    # Preprocess node feature 

    feature_name = args.node_feature_name
    data = load_node_feature(feature_name, mapping_dict, survival_train, survival_valid, args)

    player_X = np.array(data.values)

    playerwt = torch.ones(n_players) # default 1 for each player
    gamewt = torch.ones(n_games) # default 1 for each game
    cls_l = list(set(score)) # 0 ~ 9

    args.input_dim = player_X.shape[-1]
    args.ne = n_games # edge num
    args.nv = n_players # node num
    ne = args.ne
    nv = args.nv
    args.n_cls = len(cls_l)
    
    args.all_labels = torch.LongTensor(score).to(args.device) # label to tensor

    # v shape = (player num, 6)
    # e shape = (game num , n_hidden)
    if isinstance(player_X, np.ndarray):
        args.v = torch.from_numpy(player_X.astype(np.float32)).to(args.device)
    else:
        args.v = torch.from_numpy(np.array(player_X.astype(np.float32).todense())).to(args.device)
    
    args.e = torch.zeros(args.ne, args.n_hidden).to(args.device)

    args.vidx = player_game[:, 0].to(args.device) # Extract only players from hyperedge
    args.eidx = player_game[:, 1].to(args.device) # Extract only games from hyperedge

    args.v_weight = torch.Tensor([(1/w if w > 0 else 1) for w in playerwt]).unsqueeze(-1).to(args.device) 
    args.e_weight = torch.Tensor([(1/w if w > 0 else 1) for w in gamewt]).unsqueeze(-1).to(args.device)
    assert len(args.v_weight) == args.nv and len(args.e_weight) == args.ne

    player2sum = defaultdict(list) # shape = (player num, )
    game2sum = defaultdict(list) # shape = (game num, )
    e_reg_weight = torch.zeros(len(player_game)) # shape = (hyperedge num, )
    v_reg_weight = torch.zeros(len(player_game)) # shape = (hyperedge num, )

    use_exp_wt = args.use_exp_wt

    for i, (player_idx, game_idx) in enumerate(player_game.tolist()):
        e_wt = args.e_weight[game_idx]
        e_reg_wt = torch.exp(args.alpha_e*e_wt) if use_exp_wt else e_wt**args.alpha_e 
        e_reg_weight[i] = e_reg_wt
        player2sum[player_idx].append(e_reg_wt)
            
        v_wt = args.v_weight[player_idx]
        v_reg_wt = torch.exp(args.alpha_v*v_wt) if use_exp_wt else v_wt**args.alpha_v
        v_reg_weight[i] = v_reg_wt
        game2sum[game_idx].append(v_reg_wt) 

    v_reg_sum = torch.zeros(nv) # shape = (player num, )
    e_reg_sum = torch.zeros(ne) # shape = (game num, )
    for player_idx, wt_l in player2sum.items():
        v_reg_sum[player_idx] = sum(wt_l)
    for game_idx, wt_l in game2sum.items():
        e_reg_sum[game_idx] = sum(wt_l)

    e_reg_sum[e_reg_sum==0] = 1
    v_reg_sum[v_reg_sum==0] = 1
    args.e_reg_weight = torch.Tensor(e_reg_weight).unsqueeze(-1).to(args.device) # shape = (hyperedge num, 1)
    args.v_reg_sum = torch.Tensor(v_reg_sum).unsqueeze(-1).to(args.device) # shape = (player num, 1)
    args.v_reg_weight = torch.Tensor(v_reg_weight).unsqueeze(-1).to(args.device) # shape = (hyperedge num, 1)
    args.e_reg_sum = torch.Tensor(e_reg_sum).unsqueeze(-1).to(args.device) # shape = (game num, 1)

    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

def get_players_idx(df):
    # get players index from dataframe
    play_counts = {}

    cols = ['player 0', 'player 1']

    for col in cols:
        df [col] = df [col].apply(lambda x: np.float64(x) if pd.notnull(x) else x)

    for column in ['player 0', 'player 1', 'player 2', 'player 3', 'player 4', 'player 5', 'player 6', 'player 7']:
        valid_rows = df[df[column].notna()]
        
        for player in valid_rows[column].unique():
            play_counts[player] = play_counts.get(player, 0) + valid_rows[valid_rows[column] == player].shape[0]

    players_list = list(play_counts.keys())

    players_idx = torch.tensor(players_list, dtype=torch.int32)

    return players_idx


def load_hyperedge_label(df):
    # load hyperedge(player, game) and label

    game_player =[]
    player_game = []
    score = []
    game_weight = defaultdict(int)
    player_weight = defaultdict(int)

    cols = ['player 0', 'player 1', 'score 0', 'score 1']
    for col in cols:
        df[col] = df[col].apply(lambda x: np.float64(x) if pd.notnull(x) else x)

    for index, row in df.iterrows():
        for i in range(8):
            player_key = f'player {i}'
            score_key = f'score {i}'
            if pd.notna(row[player_key]):
                game_player.append((row['game'], row[player_key]))
                player_game.append((row[player_key], row['game']))
                score.append(row[score_key])
                game_weight[row['game']] += 1
                player_weight[row[player_key]] += 1

    return game_player, player_game, score, game_weight, player_weight

def player_encoding(*dfs):
    # Encoding player index labels to create hypergraph
    df = pd.concat([*dfs], ignore_index=True)

    cols = ['player 0', 'player 1', 'score 0', 'score 1']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

    mapping_dict = {}
    total_players = np.array([])
    for i in range(8):
        player_key = f'player {i}'
        score_key = f'score {i}'

        valid_rows = df[pd.notna(df[player_key])]
        total_players = np.concatenate((total_players, np.unique(valid_rows[player_key])))
    
    unique_players = np.unique(total_players)

    for index, player in enumerate(unique_players):
        mapping_dict.setdefault(player, index)

    for i in range(8):
        player_key = f'player {i}'

        df[player_key] = df[player_key].map(mapping_dict).fillna(df[player_key]).astype(float)

    return mapping_dict, df

def calculate_score(df1, df2):
    # caculate score for task 2
    # df1 answer df2 pred
    accuracy = 0

    cols = ['player 0', 'player 1', 'score 0', 'score 1']
    for col in cols:
        df1[col] = df1[col].apply(lambda x: np.float64(x) if pd.notnull(x) else x)
        df2[col] = df2[col].apply(lambda x: np.float64(x) if pd.notnull(x) else x)
    for (index1, row1), (index2, row2) in tqdm(zip(df1.iterrows(), df2.iterrows()), total=len(df1)):
        avg_ans = 0
        avg_pred = 0
        ind_err = 0
        for i in range(8):
            score_key = f'score {i}'
            if pd.notna(row1[score_key]) and pd.notna(row2[score_key]):
                ind_err += abs(row1[score_key] - row2[score_key])
                avg_ans += row1[score_key]
                avg_pred += row2[score_key]
            else:
                ind_err /= (i * 9) # individual-score error
                avg_err = abs(avg_ans - avg_pred) / (i * 9) # average-score error
                break
        accuracy += ((2 - avg_err - ind_err) / 2) * (100 / len(df1))
        
    return accuracy