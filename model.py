import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import networkx as nx
from tqdm import tqdm
from torch.nn.modules.linear import Linear
from utils import *
from sklearn.decomposition import PCA


class VGAE(nn.Module):
    def __init__(self, hyper_params, node_cnt):
        super(VGAE, self).__init__()
        self.hyper_params = hyper_params
        self.dropout = nn.Dropout()
        self.base_gcn = GraphConvSparse(hyper_params['input_dim'], hyper_params['input_dim'])
        self.gcn_mean = GraphConvSparse(hyper_params['input_dim'], hyper_params['hidden2_dim'], activation=lambda x:x)
        self.gcn_logstddev = GraphConvSparse(hyper_params['input_dim'], hyper_params['hidden2_dim'], activation=lambda x:x)

        self.embedding = nn.Embedding(
            num_embeddings=node_cnt+1,
            embedding_dim=hyper_params['input_dim'],
            padding_idx=0
        )

    def encode(self, features, adj):
        hidden = self.base_gcn(features, adj)
        self.mean = self.gcn_mean(hidden, adj)
        self.logstd = self.gcn_logstddev(hidden, adj)
        gaussian_noise = torch.randn_like(self.logstd)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def encode_for_pooling(self, features, adj):
        agg_node_feature = self.base_gcn(features, adj)
        return agg_node_feature

    def get_kl(self):
        kl_divergence = - 0.5 / self.mean.size(0) * (1 + 2 * self.logstd - self.mean**2 - torch.exp(self.logstd)**2).sum(1).mean()
        return kl_divergence

    def forward(self, adj, nodes):
        features = self.embedding(nodes)
        Z = self.encode(features, adj)
        A_pred = dot_product_decode(Z)
        return A_pred

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation = F.leaky_relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim) 
        self.activation = activation

    def forward(self, inputs, adj):
        x = inputs
        x = torch.matmul(x, self.weight)
        x = torch.matmul(adj, x)
        # outputs = self.activation(x)
        outputs = F.leaky_relu(x, negative_slope=0.05, inplace=True)
        return outputs


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.transpose(-1,-2)))
    return A_pred

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim) 
        self.activation = activation
        
    def forward(self, inputs, adj):
        x = inputs
        x = torch.matmul(x, self.weight)
        x = torch.matmul(adj, x)
        outputs = self.activation(x)
        return outputs

class AGAE(nn.Module):
    """
        Adversarial GAE. (GAE with AVB framework)
    """
    def __init__(self, hyper_params, node_cnt):
        super(AGAE, self).__init__()
        self.hyper_params = hyper_params
        
        self.base_gcn = GraphConvSparse(hyper_params['input_dim'], hyper_params['hidden1_dim'])
        self.gcn = GraphConvSparse(hyper_params['hidden1_dim'], hyper_params['hidden2_dim'], activation=lambda x:x)
        self.embedding = nn.Embedding(
            num_embeddings=node_cnt+1,
            embedding_dim=hyper_params['input_dim'],
            padding_idx=0
        )

    def encode(self, features, adj):
        hidden = self.base_gcn(features, adj)
        self.mean = self.gcn(hidden, adj)
        return self.mean

    def add_epsilon(self, adj: torch.Tensor) -> torch.Tensor:
        epsilon_matrix = torch.randn_like(adj) * self.hyper_params['eps_std']
        adj = adj + epsilon_matrix
        return adj

    def forward(self, adj, nodes):
        features = self.embedding(nodes)
        Z = self.encode(features, self.add_epsilon(adj))
        A_pred = dot_product_decode(Z)
        return A_pred


class Pooling(nn.Module):
    def __init__(self, hyper_params):
        super(Pooling, self).__init__()
        self.hyper_params = hyper_params

        self.linear = nn.Sequential(
            nn.Linear(hyper_params['hidden2_dim'], hyper_params['hidden2_dim']),
            nn.ReLU(),
            nn.Linear(hyper_params['hidden2_dim'], hyper_params['hidden2_dim'])
        )

    def forward(self, z):
        z = torch.mean(z, dim=-2)
        out = self.linear(z)

        return out


class Classifier(nn.Module):
    def __init__(self, hyper_params, num_classes: int):
        super(Classifier, self).__init__()
        self.hyper_params = hyper_params
        self.linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hyper_params['hidden1_dim'], hyper_params['hidden2_dim']),
            nn.ReLU(),
            nn.Linear(hyper_params['hidden2_dim'], num_classes)
        )

    def forward(self, feature):
        result = self.linear(feature)
        return result

class Attention_Pooling(nn.Module):
    def __init__(self, hyper_params):
        super(Attention_Pooling, self).__init__()
        self.hyper_params = hyper_params
        self.attention1 = nn.TransformerEncoderLayer(d_model=hyper_params['hidden1_dim'], 
                                                    nhead=4, dim_feedforward=128, 
                                                    dropout=0.5)
        self.act = nn.ReLU()
    def forward(self, z):
        # z = self.attention(z.permute(1,0,-1), src_key_padding_mask=node_mask).permute(1,0,-1)
        z = self.act(self.attention1(z.permute(1,0,-1)).permute(1,0,-1))
        return z

class GRU_pooling(nn.Module):
    def __init__(self, hyper_params):
        super(GRU_pooling, self).__init__()
        self.hyper_params = hyper_params
        self.pooling = nn.GRU(input_size=hyper_params['position_dim'], hidden_size=int(hyper_params['hidden2_dim']/2), 
                              num_layers=2, batch_first=True, dropout = 0.5, bidirectional=True)
        self.base_gcn = GraphConvSparse(hyper_params['hidden1_dim'], hyper_params['hidden1_dim'])        
        self.dropout = nn.Dropout()
        # self.alpha = nn.Parameter(torch.ones(1) / hyper_params['max_subgraph_len'])
        self.aggregator = hyper_params['aggregator']        
    def forward(self, z, l):
        GRU_out, hidden = self.pooling(z)
        if self.aggregator == 'mean':
            GRU_out = torch.sum(GRU_out, dim=1) / l.unsqueeze(1)
        elif self.aggregator == 'last':
            GRU_out = GRU_out[:, -1, :]
        elif self.aggregator == 'hidden_last':
            GRU_out = torch.cat((hidden[-1,:,:], hidden[-2,:,:]), dim=1)
        else:
            raise NotImplementedError
        return GRU_out
    def neighbor_pooling(self, node_embedding, adj_norm):
        node_embedding = self.base_gcn(node_embedding, adj_norm)
        node_embedding = self.dropout(node_embedding, adj_norm)
        return node_embedding
    
class LSTM_pooling(nn.Module):
    def __init__(self, hyper_params):
        super(LSTM_pooling, self).__init__()
        self.hyper_params = hyper_params
        self.pooling = nn.LSTM(input_size=hyper_params['position_dim'], 
                               hidden_size=int(hyper_params['hidden2_dim']/2), 
                               num_layers=2, batch_first=True, dropout = 0.5, bidirectional=True)
        self.base_gcn = GraphConvSparse(hyper_params['hidden1_dim'], hyper_params['hidden1_dim'])
        self.dropout = nn.Dropout()

        self.aggregator = hyper_params['aggregator']
    def forward(self, z, l):
        lstm_out, (hidden, c_n) = self.pooling(z)
        if self.aggregator == 'mean':
            lstm_out = torch.sum(lstm_out, dim=1) / l.unsqueeze(1)
        elif self.aggregator == 'last':
            lstm_out = lstm_out[:, -1, :]
        elif self.aggregator == 'hidden_last':
            lstm_out = torch.cat((hidden[-1,:,:], hidden[-2,:,:]), dim=1)
        else:
            raise NotImplementedError
        return lstm_out
    def neighbor_pooling(self, node_embedding, adj_norm):
        node_embedding = self.base_gcn(node_embedding, adj_norm)
        node_embedding = self.dropout(node_embedding)
        return node_embedding
        
    
class PositionEncoding(nn.Module):
    def __init__(self, hyper_params, graph: nx.Graph):
        super(PositionEncoding, self).__init__()
        self.hyper_params = hyper_params
        self.dist_matrix: np.ndarray = None
        self.graph = graph
        self.node_cnt = len(self.graph)
        self.dropout = nn.Dropout()
        self.pca = self.hyper_params['pca_distance']
        if self.pca:
            self.embedding = nn.Linear(self.hyper_params['pca_dim'], self.hyper_params['position_dim'])
            self.pca_dim = self.hyper_params['pca_dim']
            self.pca = True
        else:
            self.embedding = nn.Linear(len(graph)+1, self.hyper_params['position_dim'])

        self.init_dist_matrix()


    def init_dist_matrix(self):
        try:
            self.dist_matrix = np.load(os.path.join('model_dat', self.hyper_params['data_name'], 'cosine_matrix.npy'))
            print('Loading precomputed cosine postion encoding files……')
        except:
            dist_path = os.path.join('model_dat', self.hyper_params['data_name'], 'dist_matrix.npy')
            if os.path.exists(dist_path):
                print('Loading distance matrix form file…… ')
                self.dist_matrix = np.load(dist_path)
            else:
                print('Calculating shortest paths...')
                self.dist_matrix = np.zeros([len(self.graph)+1, len(self.graph)+1])

                for i, values in tqdm(nx.all_pairs_shortest_path_length(self.graph)):
                    for j, length in values.items():
                        self.dist_matrix[i+1, j+1] = length
                np.save(os.path.join('model_dat',self.hyper_params['data_name'], 'dist_matrix.npy'), self.dist_matrix)
            self.dist_matrix /= np.nanmax(self.dist_matrix,axis=0, keepdims=True) # 优化
            self.dist_matrix = np.cos(self.dist_matrix * np.pi)
            self.dist_matrix[np.isnan(self.dist_matrix)] = - 1.5
            if(len(self.dist_matrix) == len(self.graph)):
                self.dist_matrix = np.vstack((np.zeros((1,self.node_cnt)),
                                              self.dist_matrix))
                self.dist_matrix = np.hstack((np.zeros((self.node_cnt+1,1)),
                                              self.dist_matrix))
                print('Saving padded cosine distance matrix to', os.path.join('model_dat', self.hyper_params['data_name'], 'cosine_matrix.npy'))
                np.save(os.path.join('model_dat',self.hyper_params['data_name'], 'cosine_matrix.npy'), self.dist_matrix)       
        print('Phase propagation finished')
        #if hyper_params['device'] == 'cuda':
        #    self.dist_matrix = self.dist_matrix.cuda()
        if self.pca:
            if os.path.exists(os.path.join('model_dat',self.hyper_params['data_name'], 'pca_dist.npy')):
                print('Loading PCA from', os.path.join('model_dat',self.hyper_params['data_name'], 'pca_dist.npy'))
                self.pca_matrix = np.load(os.path.join('model_dat',self.hyper_params['data_name'], 'pca_dist.npy'))
            else:
                print('Pre-computing PCA')
                self.pca = PCA(n_components= self.hyper_params['pca_dim'])
                self.pca_matrix = self.pca.fit_transform(self.dist_matrix)
                np.save(os.path.join('model_dat', self.hyper_params['data_name'], 'pca_dist.npy'), self.pca_matrix)
                print('Saving PCA to', os.path.join('model_dat', self.hyper_params['data_name'], 'pca_dist.npy'))
            tmp_matrix = torch.zeros([len(self.graph)+1, self.hyper_params['pca_dim']])    
            tmp_matrix[1:] = torch.from_numpy(self.pca_matrix).float()[1:]
        else:
            tmp_matrix = torch.zeros([len(self.graph)+1, len(self.graph)+1])
            tmp_matrix[1:, 1:] = torch.from_numpy(self.pca_matrix).float()[1:, 1:]
        self.pca_matrix = tmp_matrix
        self.pca_matrix /= self.pca_matrix.std()
        self.pca_matrix.to(self.hyper_params['device'])
    def forward(self, nodes):
        nodes_flat = nodes.reshape(-1)
        pe = self.pca_matrix[nodes_flat].reshape(nodes.shape[0], nodes.shape[1], -1).float().to(self.hyper_params['device'])
        out = self.dropout(self.embedding(pe))
        return out
    
def PCA_svd(X, k, center=True):
    n = X.size()[0]
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    X_center =  torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components  = v[:k].float().cpu().t()
    #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return components


class PClassifier(nn.Module):
    def __init__(self, hyper_params, num_classes: int):
        super(PClassifier, self).__init__()
        self.linear_str = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hyper_params['hidden2_dim'], num_classes)
        )
        self.linear_avg = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hyper_params['hidden1_dim'], hyper_params['hidden2_dim']),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hyper_params['hidden2_dim'], num_classes)
        )

    def forward(self, average_embedding, structure_embedding):
        result = self.linear_avg(average_embedding)
        result += self.linear_str(structure_embedding)
        return result
    
class PA_VGAE(nn.Module):
    """
    Subgraph VGAE with 'Position-Aware Prior' in 3 different modes.
    mode 'A': Concate Node Embedding X with Potion Encoding P, i.e. GCN([P,X], A)
    mode 'B': Position Encoding as the input of VGAE. i.e. GCN(P, A)
    mode 'C': Node embedding is trained with 'Position-Aware Prior', i.e. X = DW (where P = PCA(D))
    """
    def __init__(self, hyper_params, node_cnt, PE):
        super(PA_VGAE, self).__init__()
        self.hyper_params = hyper_params
        self.pa_mode = hyper_params['pa_mode']
        self.dropout = nn.Dropout()
        
        if self.pa_mode == 'A':
            self.base_gcn = GraphConvSparse(hyper_params['input_dim']+hyper_params['position_dim'], 
                                        hyper_params['input_dim'])
        elif self.pa_mode == 'B':
            self.base_gcn = GraphConvSparse(hyper_params['position_dim'], 
                                        hyper_params['input_dim'])
            '''elif self.pa_mode == 'C':
            self.base_gcn = GraphConvSparse(hyper_params['input_dim'], 
                                        hyper_params['input_dim'])'''
        else:
            raise NotImplementedError
            
        self.gcn_mean = GraphConvSparse(hyper_params['input_dim'], hyper_params['hidden2_dim'],
                                        activation=lambda x:x)
        self.gcn_logstddev = GraphConvSparse(hyper_params['input_dim'], hyper_params['hidden2_dim'], 
                                             activation=lambda x:x)
        self.embedding_layer = nn.Embedding(
            num_embeddings=node_cnt+1,
            embedding_dim=hyper_params['input_dim'],
            padding_idx=0
        )
        
        self.PE = PE.to(hyper_params['device'])
        
        self.dist_matrix = self.PE.dist_matrix
        # self.position_prior = nn.Linear(node_cnt+1, hyper_params['input_dim'])
        # nn.init.xavier_uniform_(self.position_prior.weight)
    def embedding(self, nodes):
        if self.pa_mode == 'A':
            out = torch.cat((self.PE(nodes), self.embedding_layer(nodes)), dim=-1)
        elif self.pa_mode == 'B':
            out = self.PE(nodes)
            '''elif self.pa_mode == 'C':
            nodes_flat = nodes.cpu().reshape(-1)
            pe = torch.from_numpy(self.dist_matrix[nodes_flat].reshape(nodes.shape[0], nodes.shape[1], -1)).float().to(self.hyper_params['device'])
            out = self.position_prior(pe)'''
        else:
            raise NotImplementedError
        return out


    def encode(self, features, adj):
        hidden = self.base_gcn(features, adj)
        self.mean = self.gcn_mean(hidden, adj)
        self.logstd = self.gcn_logstddev(hidden, adj)
        gaussian_noise = torch.randn_like(self.logstd)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def encode_for_pooling(self, features, adj):
        agg_node_feature = self.base_gcn(features, adj)
        return agg_node_feature

    def get_kl(self):
        kl_divergence = - 0.5 / self.mean.size(0) * (1 + 2 * self.logstd - self.mean**2 - torch.exp(self.logstd)**2).sum(1).mean()
        return kl_divergence

    def forward(self, adj, nodes):
        features = self.embedding(nodes)
        Z = self.encode(features, adj)
        A_pred = dot_product_decode(Z)
        return A_pred
    
class pseudo_poistion_embedding(nn.Module):
    def __init__(self, hyper_params, node_cnt):
        super(pseudo_poistion_embedding, self).__init__()
        self.hyper_params = hyper_params
        self.embedding = nn.Embedding(
            num_embeddings=node_cnt+1,
            embedding_dim=hyper_params['position_dim'],
            padding_idx=0
        )

    def forward(self, nodes):
        features = self.embedding(nodes)
        return features