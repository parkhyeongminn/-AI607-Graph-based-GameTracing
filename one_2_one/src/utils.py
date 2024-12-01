import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import pandas as pd


class VALIDEarlyStopping(object):

    def __init__(self, patience, save_path, eps):
        self.max_score = -1
        self.patience = patience
        self.path = save_path
        self.eps = eps
        self.counter = 0

    def should_stop(self, model, previous_score, score, test_converted_list):
        if score >= self.max_score:
            self.max_score = score
            self.counter = 0
            torch.save(model.state_dict(), os.path.join(
                self.path, "best_model_epoch.ckpt"))
            print("the best model has been saved by early stopping")
        elif score > previous_score:
            self.counter = 0
        elif score <= previous_score:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False
    
class TESTEarlyStopping(object):
    

    def __init__(self, patience, save_path, eps):
        self.max_score = -1
        self.patience = patience
        self.path = save_path
        self.eps = eps
        self.counter = 0

    def should_stop(self, model, previous_score, score, test_converted_list):
        if score >= self.max_score:
            self.max_score = score
            self.counter = 0
            torch.save(model.state_dict(), os.path.join(
                self.path, "best_model_epoch.ckpt"))
            print("the best model has been saved by early stopping")
        elif score > previous_score:
            self.counter = 0
        elif score <= previous_score:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False



class BalancedCrossEntropyLoss(nn.Module):
    def __init__(self, weights = None, reduction = 'mean'):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.weights = weights
        self.reduction = reduction
  
        
    def forward(self, inputs, targets):
        if self.weights is not None:
            weights = self.weights.to(inputs.device)
            loss = F.nll_loss(inputs, targets.long(), weight=weights, reduction='none')
        else:
            loss = F.nll_loss(inputs, targets.long(), reduction='none')
            
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
