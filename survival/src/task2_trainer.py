import sys
from tqdm import tqdm

import task2_model

import torch
import torch.nn as nn


def trainer(train_dataloader, valid_dataloader, args):
    loss_fn = nn.CrossEntropyLoss() 
            
    hypergraph = task2_model.Hypergraph(args)

    optim = torch.optim.Adam(hypergraph.all_params(), lr=args.lr)

    milestones = [100*i for i in range(1, 4)]                                        
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=milestones, gamma=0.51)

    hypergraph = hypergraph.to_device(args.device)


    v_init = args.v
    e_init = args.e
    best_err = sys.maxsize
    train_best_err = sys.maxsize
    early_stop_count = 0

    for i in tqdm(range(args.n_epoch)):
        cur_epoch = i
        total_train_loss = 0
        total_valid_loss = 0

        # train
        print(f' epoch {i + 1} train')
        for idx, (player_idx, game_idx, data_idx, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            hypergraph.train()
            player_idx, game_idx, data_idx, label = player_idx.to(args.device), game_idx.to(args.device), data_idx.to(args.device), label.to(args.device)
        
            v, e, pred = hypergraph(v_init, e_init, player_idx, game_idx, data_idx)

            loss = loss_fn(pred, label)
            total_train_loss += loss.item()


            optim.zero_grad()
            loss.backward()            
            optim.step()
            scheduler.step()

        train_loss = total_train_loss / len(train_dataloader)
        
        # validation
        with torch.no_grad():
            print(f' epoch {i + 1} validation')
            for idx, (player_idx, game_idx, data_idx, label) in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
                hypergraph.eval()
                player_idx, game_idx, data_idx, label = player_idx.to(args.device), game_idx.to(args.device), data_idx.to(args.device), label.to(args.device)

                v, e, pred = hypergraph(v_init, e_init, player_idx, game_idx, data_idx)           
                loss = loss_fn(pred, label)
                total_valid_loss += loss.item()
                

        valid_loss = total_valid_loss / len(valid_dataloader)
        
        if train_loss < train_best_err:
            train_best_err = train_loss
            early_stop_count = 0
            if not args.test: # train : train, valid : valid
                print('model save')
                best_model = hypergraph
        else:
            early_stop_count += 1
            print(f'early_stop_count : {early_stop_count}')

        if valid_loss < best_err:
            best_err = valid_loss
            if args.test: # train : train + valid, valid : valid
                print('model save')
                best_model = hypergraph

        print(f'train_err = {train_loss}, test_err = {valid_loss}, best_err = {best_err}')

        if early_stop_count >= 10:
            break

    return best_model

def inference(survival_valid_inference, best_model, infer_dataloader, args):
    # inference for task2_main
    cur_game_idx = 0
    player_num = 0

    v_init = args.v
    e_init = args.e

    with torch.no_grad():
        for idx, (player_idx, game_idx, data_idx, label) in tqdm(enumerate(infer_dataloader), total=len(infer_dataloader)):
            best_model.eval()
            if game_idx != cur_game_idx:
                if idx != 0:
                    survival_valid_inference = survival_valid_inference.append(game_dict, ignore_index=True)
                player_num = 0
                cur_game_idx = game_idx
                game_dict = {'game' : int(game_idx.item())}

            player_idx, game_idx, data_idx, label = player_idx.to(args.device), game_idx.to(args.device), data_idx.to(args.device), label.to(args.device)
            
            v, e, pred = best_model(v_init, e_init, player_idx, game_idx, data_idx)
            
            game_dict[f'player {player_num}'] = player_idx.item()
            game_dict[f'score {player_num}'] = float(torch.argmax(pred).item())
            player_num += 1

    # last game add
    survival_valid_inference = survival_valid_inference.append(game_dict, ignore_index=True)

    return survival_valid_inference

def inference_test(survival_test_inference, best_model, infer_dataloader, args):
    # inference for task2_main_test
    cur_game_idx = 0
    player_num = 0

    v_init = args.v
    e_init = args.e

    with torch.no_grad():
        for idx, (player_idx, game_idx, data_idx) in tqdm(enumerate(infer_dataloader), total=len(infer_dataloader)):
            best_model.eval()
            if game_idx != cur_game_idx:
                if idx != 0:
                    survival_test_inference = survival_test_inference.append(game_dict, ignore_index=True)
                player_num = 0
                cur_game_idx = game_idx
                game_dict = {'game' : int(game_idx.item())}

            player_idx, game_idx, data_idx = player_idx.to(args.device), game_idx.to(args.device), data_idx.to(args.device)
            
            v, e, pred = best_model(v_init, e_init, player_idx, game_idx, data_idx)
            
            game_dict[f'score {player_num}'] = float(torch.argmax(pred).item())
            player_num += 1


    # last game add
    survival_test_inference = survival_test_inference.append(game_dict, ignore_index=True)

    return survival_test_inference