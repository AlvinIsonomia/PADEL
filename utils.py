from networkx.classes import graph
import torch
from torch._C import dtype
import torch.utils.data as Data
import os
import random
from typing import Tuple
from tqdm import tqdm
import networkx as nx
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score


def sparse_eye(n: int) -> torch.Tensor:
    indices = torch.vstack([torch.arange(0, n)] * 2)
    values = torch.ones(indices.size(1), dtype=torch.long)
    result = torch.sparse_coo_tensor(indices, values, [n, n], dtype=torch.long)

    return result

def sparse_diag(val: torch.Tensor) -> torch.Tensor:
    n = val.size(0)
    indices = torch.vstack([torch.arange(0, n)[val!=0]] * 2)
    values = val[val!=0]
    result = torch.sparse_coo_tensor(indices, values, [n, n], dtype=torch.long)

    return result

# def sparse_mm(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#     """
#         Perform batch matmul for sparse tensor.
#         x1: b x m x n
#         x2: n x o
#     """
#     b, m, n = x1.shape
#     result = torch.mm(x1.reshape(b*m, n), x2).reshape(b, m, -1)

#     return result

def convert_to_norm(adj: torch.Tensor, l) -> torch.Tensor:
    n = adj.size(0)
    adj = adj[:l, :l]
    adj_diag = adj - torch.diag(torch.diag(adj)).to(adj.device) + torch.eye(l).to(adj.device)
    degree = torch.diag(adj_diag.sum(dim=1).float() ** -0.5).to(adj.device)
    adj_norm = degree @ adj_diag @ degree
    adj_norm_pad = torch.zeros([n, n], dtype=torch.float)
    adj_norm_pad[:l, :l] = adj_norm

    return adj_norm_pad

def get_sample(dataset: Data.Dataset, model, index: int, cnt: int) -> Tuple[torch.Tensor, torch.Tensor]:
    result = []
    result_nodes = []

    adj, adj_norm, adj_mask, nodes, l, label = [i.unsqueeze(0) for i in dataset[index]]

    # Get sample
    for _ in range(cnt):
        reconstruct = model(adj_norm, nodes).squeeze(0).to('cpu')
        reconstruct = (reconstruct >= 0.5).long()
        result.append(convert_to_norm(reconstruct, l.squeeze().item()))
        result_nodes.append(nodes.squeeze(0))

    result = torch.stack(result)
    result_nodes = torch.stack(result_nodes)

    return result, result_nodes

def get_negative_samples(dataset: Data.Dataset, model, index: int, cnt: int) -> Tuple[torch.Tensor, torch.Tensor]:
    result = []
    result_nodes = []

    # Get negative samples
    total_cnt = len(dataset)
    i = 0
    while i < cnt:
        nxt = random.randint(0, total_cnt-1)
        if nxt == index:
            continue
        cur, cur_nodes = get_sample(dataset, model, nxt, 1)
        result.append(cur.squeeze(0))
        result_nodes.append(cur_nodes.squeeze(0))
        i += 1

    result = torch.stack(result)
    result_nodes = torch.stack(result_nodes)

    return result, result_nodes

def get_nodes_dist(graph_adj: torch.Tensor):
    degree = torch.diag(graph_adj.sum(1)).float()
    graph_adj = torch.eye(graph_adj.size(0)) + graph_adj
    adj_norm = degree @ graph_adj

    nodes_dist = adj_norm.norm(dim=1)
    nodes_dist /= nodes_dist.sum()
    return nodes_dist

def get_random_samples(graph_adj: torch.Tensor, nodes_limit: int, cnt: int, nodes_dist=None):
    node_cnt = graph_adj.size(0)
    result_adj_norm = []
    result_nodes = []

    total_nodes = list(range(node_cnt))
    
    for _ in range(cnt):
        node_cnt = random.randint(nodes_limit//2, nodes_limit)
        if nodes_dist is None:
            nodes = sorted(np.random.choice(total_nodes, node_cnt, replace=False))
        else:
            nodes = sorted(np.random.choice(total_nodes, node_cnt, replace=False, p=nodes_dist.numpy()))
        nodes = torch.tensor(nodes)
        
        adj = graph_adj[nodes, :][:, nodes]
        adj_pad = torch.zeros([nodes_limit, nodes_limit], dtype=torch.float)
        adj_pad[:node_cnt, :node_cnt] = adj
        adj_norm = convert_to_norm(adj_pad, node_cnt)
        nodes_pad = torch.zeros(nodes_limit, dtype=torch.long)
        nodes_pad[:nodes.size(0)] = nodes
        
        result_adj_norm.append(adj_norm)
        result_nodes.append(nodes_pad)

    result_adj_norm = torch.stack(result_adj_norm)
    result_nodes = torch.stack(result_nodes)

    return result_adj_norm, result_nodes


def subgraph_encode(model, pooling, adj_norm: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    embedding = model.embedding(nodes)
    embedding = model.base_gcn(embedding, adj_norm)
    # result = pooling(node_encode)

    result = torch.mean(embedding, dim=-2)

    return result


def contrastive_score(model, pooling, real_adj_norm, real_nodes, contrast_adj_norm, contrast_nodes):
    """
        Calculate contrastive score.
        adj: b x l x l
        nodes: b x l
        return: b
    """
    # Get subgraph encode for real_encode and contrast encode
    real_encode = model.embedding(real_nodes)
    real_encode = model.base_gcn(real_encode, real_adj_norm)
    real_encode = torch.mean(real_encode, dim=-2)
    
    contrast_encode = model.embedding(contrast_nodes)
    contrast_encode = model.base_gcn(contrast_encode, contrast_adj_norm)
    contrast_encode = torch.mean(contrast_encode, dim=-2)

    # Calculate inner product
    score = real_encode * contrast_encode
    score = torch.sum(score, dim=-1) / torch.norm(score, dim=-1)

    return score

def contrastive_score_v2(model, pooling, real_adj_norm, real_nodes, contrast_adj_norm, contrast_nodes):
    """
        Calculate contrastive score.
        adj: b x l x l
        nodes: b x l
        return: b
    """
    real_mask = (real_nodes != 0)
    real_l = real_nodes.sum(-1)

    contrast_mask = (contrast_nodes != 0)
    contrast_l = contrast_mask.sum(-1)

    # Get subgraph encode for real_encode and contrast encode
    real_encode = model.embedding(real_nodes)
    real_encode = model.base_gcn(real_encode, real_adj_norm)
    real_encode = (real_encode * real_mask.unsqueeze(-1)).sum(-2) / real_l.unsqueeze(-1)
    
    contrast_encode = model.embedding(contrast_nodes)
    contrast_encode = model.base_gcn(contrast_encode, contrast_adj_norm)
    contrast_encode = (contrast_encode * contrast_mask.unsqueeze(-1)).sum(-2) / contrast_l.unsqueeze(-1)

    # Calculate inner product
    score = real_encode * contrast_encode
    real_norm = real_encode.norm(dim=-1)
    contrast_norm = contrast_encode.norm(dim=-1)
    score = torch.sum(score, dim=-1) / (real_norm * contrast_norm + 1e-9)

    return score

def contrastive_score_v3(model, P_model, pooling, real_adj_norm, real_nodes, contrast_adj_norm, contrast_nodes):
    """
        Calculate contrastive score.
        adj: b x l x l
        nodes: b x l
        return: b
    """
    
    real_mask = (real_nodes != 0)
    real_l = real_nodes.sum(-1)

    contrast_mask = (contrast_nodes != 0)
    contrast_l = contrast_mask.sum(-1)
    

    # Get subgraph encode for real_encode 
    real_P_embedding = P_model(real_nodes)
    real_N_embedding = model.embedding(real_nodes)    
    real_PN_embedding = torch.cat((real_P_embedding, real_N_embedding), dim=-1)    
    real_PN_embedding = pooling.neighbor_pooling(real_PN_embedding, real_adj_norm)
    
    real_PN_encode =  real_PN_embedding.mean(dim=-2)
    real_S_encode = pooling(real_P_embedding, real_l)
    
    # Get subgraph encode for contrast encode
    contrast_P_embedding = P_model(contrast_nodes)
    contrast_N_embedding = model.embedding(contrast_nodes)    
    contrast_PN_embedding = torch.cat((contrast_P_embedding, contrast_N_embedding), dim=-1)    
    contrast_PN_embedding = pooling.neighbor_pooling(contrast_PN_embedding, contrast_adj_norm)
    
    contrast_PN_encode = contrast_PN_embedding.mean(dim=-2)
    contrast_S_encode = pooling(contrast_P_embedding, contrast_l)
    
    # Calculate inner product
    PN_score = real_PN_encode * contrast_PN_encode    
    real_PN_norm = real_PN_encode.norm(dim=-1)
    contrast_PN_norm = contrast_PN_encode.norm(dim=-1)
    PN_score = torch.sum(PN_score, dim=-1) / (real_PN_norm * contrast_PN_norm + 1e-9)
    
    
    S_score = real_S_encode * contrast_S_encode
    real_S_norm = real_S_encode.norm(dim=-1)
    contrast_S_norm = contrast_S_encode.norm(dim=-1)
    S_score = torch.sum(S_score, dim=-1) / (real_S_norm * contrast_S_norm + 1e-9)
    
    return PN_score + S_score


def nce_loss(positive_score: torch.Tensor, negative_score: torch.Tensor, eps: float=1e-7) -> torch.Tensor:
    """
        Calculate NCE Loss.
        positive_score: 1
        negative_score: b
    """

    positive_score_e = torch.exp(positive_score).sum()
    negative_score_e = torch.exp(negative_score).sum()

    nce_loss_e = positive_score_e / (positive_score_e + negative_score_e + eps)
    nce_loss = -torch.log(nce_loss_e + eps)

    return nce_loss

def random_walk(graph: nx.classes.graph.Graph, lenth_mean = 15):
    node_cnt = graph.number_of_nodes()
    random_walk_res = set()
    walk_steps = random.randint(int(lenth_mean*0.9), int(lenth_mean*1.5))
    # Pick a start point randomly
    random_node_id = random.randint(0, node_cnt-1)
    for i in range(walk_steps):
        random_walk_res |= set([random_node_id])
        random_node_id = random.choice(list(graph[random_node_id]))
    # print(random_walk_res)
    return list(random_walk_res)


def calc_f1(logits, labels, avg_type='macro',  multilabel_binarizer=None):
    '''
    Calculates the F1 score (either macro or micro as defined by 'avg_type') for the specified logits and labelss
    '''
    if multilabel_binarizer is not None: #multi-label prediction
        # perform a sigmoid on each logit separately & use > 0.5 threshold to make prediction
        probs = torch.sigmoid(logits)
        thresh = torch.tensor([0.5]).to(probs.device)
        pred = (probs > thresh)
        score = f1_score(labels.cpu().detach(), pred.cpu().detach(), average=avg_type)
        
    else: # multi-class, but not multi-label prediction

        pred = torch.argmax(logits, dim=-1) #get predictions by finding the indices with max logits
        score = f1_score(labels.cpu().detach(), pred.cpu().detach(), average=avg_type)
    return torch.tensor([score])

def calc_accuracy(logits, labels,  multilabel_binarizer=None):
    '''
    Calculates the accuracy for the specified logits and labels
    '''
    if multilabel_binarizer is not None: #multi-label prediction
        # perform a sigmoid on each logit separately & use > 0.5 threshold to make prediction
        probs = torch.sigmoid(logits)
        thresh = torch.tensor([0.5]).to(probs.device)
        pred = (probs > thresh)
        acc = accuracy_score(labels.cpu().detach(), pred.cpu().detach())
    else:
        pred = torch.argmax(logits, 1) #get predictions by finding the indices with max logits
        acc = accuracy_score(labels.cpu().detach(), pred.cpu().detach())
    return torch.tensor([acc])

def get_reconstruct_norm(model, adj_norm, nodes, l, batch_size=11):
    reconstruct = model(adj_norm, nodes)
    reconstruct = (reconstruct >= 0.5).long().to(adj_norm.device)
    results_norm = []
    for i in range(batch_size):
        results_norm.append(convert_to_norm(reconstruct[i], l[i]))
    results_norm = torch.stack(results_norm)
    return results_norm

def get_encodes(model, P_model, pooling, adj_norm, nodes, l):
    # Get subgraph encode for real_encode 
    P_embedding = P_model(nodes)
    N_embedding = model.embedding(nodes)
    PN_embedding = torch.cat((P_embedding, N_embedding), dim=-1)    
    PN_embedding = pooling.neighbor_pooling(PN_embedding, adj_norm)
    
    PN_encode =  PN_embedding.mean(dim=-2)
    S_encode = pooling(P_embedding, l)
    
    return PN_encode, S_encode

def get_encodes_A(model, P_model, pooling, adj_norm, nodes, l):
    # Get subgraph encode for real_encode 
    P_embedding = P_model(nodes)
    N_embedding = model.embedding(nodes)
    # PN_embedding = torch.cat((P_embedding, N_embedding), dim=-1)    
    PN_embedding = pooling.neighbor_pooling(N_embedding, adj_norm)
    
    PN_encode =  PN_embedding.mean(dim=-2)
    S_encode = pooling(P_embedding, l)
    
    return PN_encode, S_encode
def get_encodes_C3(model, pooling, adj_norm, nodes, l):
    # Get subgraph encode for real_encode 
    PN_embedding = model.embedding(nodes)    
    PN_embedding = pooling.neighbor_pooling(PN_embedding, adj_norm)
    
    PN_encode =  PN_embedding.mean(dim=-2)
    S_encode = pooling(PN_embedding, l)
    
    return PN_encode, S_encode
def compute_encodes_score(real_PN_encode, real_S_encode, contrast_PN_encode, contrast_S_encode):
    # Calculate inner product
    PN_score = real_PN_encode * contrast_PN_encode    
    real_PN_norm = real_PN_encode.norm(dim=-1)
    contrast_PN_norm = contrast_PN_encode.norm(dim=-1)
    PN_score = torch.sum(PN_score, dim=-1) / (real_PN_norm * contrast_PN_norm + 1e-9)
    
    S_score = real_S_encode * contrast_S_encode
    real_S_norm = real_S_encode.norm(dim=-1)
    contrast_S_norm = contrast_S_encode.norm(dim=-1)
    S_score = torch.sum(S_score, dim=-1) / (real_S_norm * contrast_S_norm + 1e-9)
    
    return PN_score + S_score

def get_dataset_path(hyper_params, con=False):
    ''' 
    Automatically get the path to training dataset
    '''
    folder_path = os.path.join('model_dat',hyper_params['data_name'])
    if con:
        file_name = 'con_dataset'
    else:
        file_name = 'train_dataset'
    
    if hyper_params['diffuse']:
        if hyper_params['cut_rate']:
            file_name = file_name + '_' + hyper_params['cut_rate'] + '_diffuse.pkl'
        else:
            file_name = file_name + '_diffuse.pkl'                  
    else:
        if hyper_params['cut_rate']:
            file_name = file_name + '_' + hyper_params['cut_rate'] + '.pkl'
        else:
            file_name = file_name + '.pkl'  
            
        
    dataset_path = os.path.join(folder_path,file_name)
    return dataset_path

def get_diffused_mask(data_name = 'hpo_metab', subgraph_name = 'subgraphs_10.pth'):
    
    file_origin = open(os.path.join('data', data_name, subgraph_name ), 'rt') 
    file_diffuse = open(os.path.join('data', data_name, 'diffuse_'+subgraph_name), 'rt')
    origin_lines = file_origin.readlines()
    diffuse_lines = file_diffuse.readlines()
    file_origin.close()
    file_diffuse.close()
    node_masks = []
    for sub_idx in tqdm(range(len(origin_lines))):
        # get original/diffused sugraphs
        origin_nodes, label, mode = origin_lines[sub_idx].rstrip().split('\t')
        origin_nodes_num = len(list(map(int, origin_nodes.split('-'))))
        diffuse_nodes, label, mode = diffuse_lines[sub_idx].rstrip().split('\t')
        diffuse_nodes = list(map(int, diffuse_nodes.split('-')))
        sampled_nodes = sorted(diffuse_nodes[origin_nodes_num:])
        diffuse_nodes = sorted(diffuse_nodes)
        
        node_mask = np.ones_like(diffuse_nodes)
        sampled_idx = 0
        node_idx = 0
        while(sampled_idx < len(sampled_nodes)):
            if diffuse_nodes[node_idx] != sampled_nodes[sampled_idx]:
                node_idx += 1
            else:
                node_mask[node_idx] = 0
                node_idx += 1
                sampled_idx += 1
        node_masks.append(node_mask) 
    return node_masks