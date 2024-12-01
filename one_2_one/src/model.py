import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

class Aspect_Representation(nn.Module):

    def __init__(self, num_aspects, hidden_dim, aspect_dim):
        super(Aspect_Representation, self).__init__()
        self.aspect_dim = aspect_dim
        self.num_aspects = num_aspects
        self.aspEmbed = nn.Linear(in_features=hidden_dim, out_features=num_aspects*aspect_dim,bias=False)
        nn.init.xavier_uniform_(self.aspEmbed.weight)
        if self.aspEmbed.bias is not None:
            self.aspEmbed.bias.data.fill_(0)
        self.linear = nn.Linear(self.aspect_dim, 1)


    def forward(self, node_emb):
        batch, out_dim = node_emb.size()
        node_asp_Rep_q = self.aspEmbed(node_emb)
        
        node_asp_Rep = node_asp_Rep_q.view(batch, self.num_aspects, self.aspect_dim)   
        
        attn_score = F.softmax(self.linear(node_asp_Rep), dim=1)

        node_asp_Rep = attn_score*node_asp_Rep
        
        return node_asp_Rep
    

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.attn_linear = nn.Linear(in_feats, in_feats)
        self.linear = nn.Linear(in_feats, out_feats)
        self.attn_weight_linear = nn.Linear(in_feats, 1)
        self.tanh = nn.Tanh()
    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['x'] = feature
            g.update_all(dgl.function.u_mul_v('x', 'x', 'm'), dgl.function.mean('m', 'x'))
            x = g.ndata['x']
            return self.linear(x)


class GCN(nn.Module):
    def __init__(self, args, g, feature, device):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(args.in_dim, args.hid_dim)
        self.gcn2 = GCNLayer(args.hid_dim, args.hid_dim)
        self.concat_linear = nn.Linear(args.num_aspect*2, args.num_classes)
        self.aspect_merge_linear = nn.Linear(args.aspect_dim, 1)
        self.Aspect_Representation = Aspect_Representation(args.num_aspect, 
                                                           args.hid_dim, args.aspect_dim)
        self.graph = g.to(device)
        self.node_feature = feature.to(device) + feature.to(device).mean(dim=0)
        self.dropout = nn.Dropout(0.4)
        self.batch_norm = nn.BatchNorm1d(num_features=args.num_aspect)
        self.relu = nn.ReLU()
        

    def forward(self, input):
        
        
        x = self.dropout(self.gcn1(self.graph, self.node_feature))
        
        # x = self.dropout(self.relu(self.gcn1(self.graph, self.node_feature)))
        # x = self.dropout(self.gcn2(self.graph, x))
        

        player_one_id = input[:,0]
        player_two_id = input[:,1]
        
        player_one = x[[player_one_id]]
        player_two = x[[player_two_id]]
        
        
        player_one_aspect = self.aspect_merge_linear(self.Aspect_Representation(player_one)).squeeze()
        player_two_aspect = self.aspect_merge_linear(self.Aspect_Representation(player_two)).squeeze()

        

        
        output = self.concat_linear(torch.cat((player_one_aspect, player_two_aspect), dim=1)).squeeze()
        
        

        game_result = F.log_softmax(output)
        

        
        return game_result
    
    
class MF(nn.Module):
    def __init__(self, num_factors, num_player1, num_player2):
        super(MF, self).__init__()
        self.P = nn.Embedding(num_player1, num_factors)
        self.Q = nn.Embedding(num_player2, num_factors)
        self.player1_bias = nn.Embedding(num_player1, 1)
        self.player2_bias = nn.Embedding(num_player2, 1)

    def forward(self, player1, player2):
        P_p1 = self.P(player1)
        Q_p2 = self.Q(player2)
        b_p1 = self.player1_bias(player1)
        b_p2 = self.player2_bias(player2)
        
        outputs = torch.sum((P_p1 * Q_p2), axis = 1) + torch.squeeze(b_p1) + torch.squeeze(b_p2)
        
        return outputs.flatten()