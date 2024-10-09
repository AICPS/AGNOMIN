import torch
import torch.nn as nn
from torch.nn import Sequential
from transformers import BertModel, BertConfig

from torch_geometric.nn import GCNConv, GATConv, global_add_pool

class Gemini(torch.nn.Module):
    def __init__(self):
        pass
    def forward(self, x, edge_index, batch, edge_index_cg):
        #import pdb; pdb.set_trace()
        outputs = [x]
        x = self.graph_embed(x)
        for layer_idx in range(self.num_layers):
            x = self.gconv_layers[layer_idx](x, edge_index).tanh()
            outputs.append(x)
        x = torch.cat(outputs, dim=-1)
        x = self.dropout(x)
        x = global_add_pool(x, batch)
        x = self.fc(x)
        return x
    def graph_embed(self):
        pass

class GeminiACFG(torch.nn.Module):
    def __init__(self, num_layers, layer_spec, initial_dim_bb, initial_dim_func, dropout, pcode2vec='None',num_func_keywords=0):
        super(GeminiACFG, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer_spec = layer_spec.split(',')
        self.num_features_bb = initial_dim_bb + 12 if 'bb' in pcode2vec else 12
        self.num_features_func = initial_dim_func if 'func' in pcode2vec else 0
        
        self.convs = []
        self.fc_dim = sum([int(dim) for dim in self.layer_spec]) + self.num_features_bb

        for layer_idx in range(num_layers):
            in_dim = int(self.layer_spec[layer_idx-1]) if layer_idx > 0 else self.num_features_bb
            out_dim= int(self.layer_spec[layer_idx])
            self.convs.append(GCNConv(in_dim, out_dim))
        self.fc = nn.Linear(self.fc_dim, self.fc_dim)
        self.gconv_layers = Sequential(*self.convs)
        self.dropout = nn.Dropout(p=self.dropout)
        self.fc_final = nn.Linear(self.fc_dim + self.num_features_func, self.fc_dim)
        
        self.num_func_keywords = num_func_keywords

        if self.num_func_keywords != 0:
            self.fc_pred_kws = nn.Linear(self.fc_dim, self.num_func_keywords)
            self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, edge_index, batch, edge_index_cg, func_pcode_x=None):
        # import pdb; pdb.set_trace()
        outputs = [x]
        # print(x.shape)
        for layer_idx in range(self.num_layers):
            
            x = self.gconv_layers[layer_idx](x, edge_index).tanh()
            outputs.append(x)
        x = torch.cat(outputs, dim=-1)
        x = self.dropout(x)
        x = global_add_pool(x, batch)
        x = self.fc(x)
        if func_pcode_x is not None:
            x = torch.cat((x, func_pcode_x.float()), dim=-1)
            x = self.fc_final(x)
        return x
    def func_predict(self, x): # this is used for token prediction 
        if self.num_func_keywords != 0:
            x = self.fc_pred_kws(x)
            x = self.sigmoid(x)
        return x

class AGNOMINEncoder(torch.nn.Module):
    def __init__(self, num_layers, layer_spec, initial_dim_bb, initial_dim_func, dropout, pcode2vec='None'):
        super(AGNOMINEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer_spec = layer_spec.split(',')
        self.num_features_bb = initial_dim_bb + 12 if 'bb' in pcode2vec else 12
        self.num_features_func = initial_dim_func if 'func' in pcode2vec else 0
        
        self.convs = []
        self.fc_dim = sum([int(dim) for dim in self.layer_spec])+self.num_features_bb
        for layer_idx in range(num_layers):
            in_dim = int(self.layer_spec[layer_idx-1]) if layer_idx > 0 else self.num_features_bb
            out_dim= int(self.layer_spec[layer_idx])
            self.convs.append(GCNConv(in_dim, out_dim))
        self.acg_embed = GATConv(self.fc_dim, self.fc_dim) # GATConv(64, 64, heads=1)
        self.fc = nn.Linear(2*self.fc_dim, self.fc_dim) # Linear(in_features=128, out_features=64, bias=True)
        self.fc_prefinal = nn.Linear(self.fc_dim + self.num_features_func, self.fc_dim)
        self.fc_final = nn.Linear(self.fc_dim, self.fc_dim)

        self.gconv_layers = Sequential(*self.convs) # Sequential( (0): GCNConv(12, 32) (1): GCNConv(32, 32))
        self.dropout = nn.Dropout(p=self.dropout) # Dropout(p=0.0, inplace=False)
    
    def forward(self, x, edge_index, batch, edge_index_cg, func_pcode_x=None):
        outputs = [x]
        for layer_idx in range(self.num_layers):
            x = self.gconv_layers[layer_idx](x, edge_index).tanh()
            outputs.append(x)
        x = torch.cat(outputs, dim=-1)
        x = self.dropout(x)
        x = global_add_pool(x, batch)
        x_context = self.acg_embed(x, edge_index_cg).tanh()
        x = torch.cat([x, x_context], axis=1)
        x = self.fc(x)
        if func_pcode_x is not None:
            x = torch.cat((x, func_pcode_x.float()), dim=-1)
            x = self.fc_prefinal(x).tanh()
        x = self.fc_final(x)
        return x


class cfg2vecGoG(torch.nn.Module):
    def __init__(self, num_layers, layer_spec, initial_dim, dropout, pcode2vec='None'):
        super(cfg2vecGoG, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer_spec = layer_spec.split(',') # ['32', '32']
        self.num_features_bb = initial_dim + 12 if 'bb' in pcode2vec else 12
        self.num_features_func = initial_dim if 'func' in pcode2vec else 0
        
        self.convs = []
        self.fc_dim = sum([int(dim) for dim in self.layer_spec])+self.num_features_bb

        for layer_idx in range(num_layers):
            in_dim = int(self.layer_spec[layer_idx-1]) if layer_idx > 0 else self.num_features_bb
            out_dim= int(self.layer_spec[layer_idx])
            self.convs.append(GCNConv(in_dim, out_dim))
        self.acg_embed = GATConv(self.fc_dim, self.fc_dim) # GATConv(64, 64, heads=1)
        self.fc = nn.Linear(2*self.fc_dim, self.fc_dim) # Linear(in_features=128, out_features=64, bias=True)

        self.gconv_layers = Sequential(*self.convs) # Sequential( (0): GCNConv(12, 32) (1): GCNConv(32, 32))
        self.dropout = nn.Dropout(p=self.dropout) # Dropout(p=0.0, inplace=False)
        self.fc_final = nn.Linear(self.fc_dim + self.num_features_func, self.fc_dim)
    
    def forward(self, x, edge_index, batch, edge_index_cg, func_pcode_x=None):
        outputs = [x]
        for layer_idx in range(self.num_layers):
            x = self.gconv_layers[layer_idx](x, edge_index).tanh()
            outputs.append(x)
        x = torch.cat(outputs, dim=-1)
        x = self.dropout(x)
        x = global_add_pool(x, batch)
        x_context = self.acg_embed(x, edge_index_cg).tanh()
        x = torch.cat([x, x_context], axis=1)
        x = self.fc(x)
        if func_pcode_x is not None:
            x = torch.cat((x, func_pcode_x.float()), dim=-1)
            x = self.fc_final(x)
        return x

