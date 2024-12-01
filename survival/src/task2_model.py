import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

import numpy as np

# source from https://github.com/twistedcubic/HNHN/blob/master/hypergraph.py

class HyperMod(nn.Module):
    
    def __init__(self, args, is_last=False):
        super(HyperMod, self).__init__()
        self.args = args
        self.v_weight = args.v_weight
        self.e_weight = args.e_weight
        self.nv, self.ne = args.nv, args.ne

        
        self.W_v2e = Parameter(torch.randn(args.n_hidden, args.n_hidden))
        self.W_e2v = Parameter(torch.randn(args.n_hidden, args.n_hidden))
        self.b_v = Parameter(torch.zeros(args.n_hidden))
        self.b_e = Parameter(torch.zeros(args.n_hidden))
        self.is_last_mod = is_last
        self.use_edge_lin = args.use_edge_lin
        # not use
        if is_last and self.use_edge_lin:
            self.edge_lin = torch.nn.Linear(args.n_hidden, args.final_edge_dim)
            
    def forward(self, v, e, player_idx, game_idx, idx):

        # update node
        if self.args.edge_linear:
            ve = torch.matmul(v, self.W_v2e) + self.b_v
        else:
            ve = F.relu(torch.matmul(v, self.W_v2e) + self.b_v)

        v_fac = 4 if self.args.predict_edge else 1
        v = v*self.v_weight*v_fac 

        # normalize node
        eidx = game_idx.unsqueeze(-1).expand(-1, self.args.n_hidden)
        e = e.clone()
        ve = (ve*self.v_weight)[player_idx]

        ve *= self.args.v_reg_weight[idx]
        e.scatter_add_(src=ve, index=eidx, dim=0)
        e /= self.args.e_reg_sum     

        # update hyperedge
        ev = F.relu(torch.matmul(e, self.W_e2v) + self.b_e)

        # normalize hyperedge
        vidx = player_idx.unsqueeze(-1).expand(-1, self.args.n_hidden)
        ev_vtx = (ev*self.e_weight)[game_idx]
        ev_vtx *= self.args.e_reg_weight[idx]
        v.scatter_add_(src=ev_vtx, index=vidx, dim=0)
        v /= self.args.v_reg_sum

        if not self.is_last_mod:
            v = F.dropout(v, self.args.dropout_p)
        if self.is_last_mod and self.use_edge_lin:
            ev_edge = (ev*torch.exp(self.e_weight)/np.exp(2))[game_idx]
            v2 = torch.zeros_like(v)
            v2.scatter_add_(src=ev_edge, index=vidx, dim=0)
            v2 = self.edge_lin(v2)
            v = torch.cat([v, v2], -1)

        # player, game indexing
        if self.is_last_mod:
            v = v[player_idx]
            e = e[game_idx]
        return v, e

    
class Hypergraph(nn.Module):
    def __init__(self, args):
        super(Hypergraph, self).__init__()
        self.args = args
        self.hypermods = []
        is_first = True
        for i in range(args.n_layers):
            is_last = True if i == args.n_layers-1 else False            
            self.hypermods.append(HyperMod(args, is_last=is_last))
            is_first = False

        if args.predict_edge:
            self.edge_lin = torch.nn.Linear(args.input_dim, args.n_hidden) 

        self.vtx_lin = torch.nn.Linear(args.input_dim, args.n_hidden)
        self.cls = nn.Linear(args.n_hidden, args.n_cls)


    def to_device(self, device):
        self.to(device)  
        for mod in self.hypermods:
            mod.to('cuda')
        return self
        
    def all_params(self):
        params = []
        for mod in self.hypermods:
            params.extend(mod.parameters())
        return params
        
    def forward(self, v, e, player_idx, game_idx, idx):
    
        v = self.vtx_lin(v) # v (batch_size , input_dim) -> (batch_size, n_hidden)

        # not used
        if self.args.predict_edge: 
            e = self.edge_lin(e)

        for mod in self.hypermods:
            # input v shape: (batch_size, n_hidden), e shape: (batch_size, n_hidden)
            # output v shape: (batch_size, n_hidden), e shape: (batch_size, n_hidden)
            v, e = mod(v, e, player_idx, game_idx, idx)
        
        pred = self.cls(v)
        return v, e, pred
    