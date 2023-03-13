from networkx.classes import graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from dataset import GraphLoader
from dataset_multi import Multilabel_GraphLoader
from utils import get_dataset_path

from model import PositionEncoding, PA_VGAE
import os
import pickle


# Hyper Params and Preparations
hyper_params = {
    "data_path": "data/em_user/",
    "subgraph_file": "subgraphs.pth",
    "batch_size": 32,
    "device": 'cuda:5',
    "epochs": 500,
    "input_dim": 96,
    "position_dim": 32,
    "hidden1_dim": 32,
    "hidden2_dim": 32,
    "diffuse": True, # IMPORTANT! Using pre-diffused subgprahs as dataset!
    "max_subgraph_len": 400, # 400 For EM_USER, 50 for others, (+) means diffused subgraphs 
    "eps_std": 3.0,
    "feature_lr": 1e-3,
    "pooling_lr": 1e-3,
    "classifier_lr": 1e-3,
    "pca_distance": True, # As we add position encoding into our VGAE model
    "pca_dim": 128, # Reduced Position encoding dim
    "multi_label": False,
}

''' Get name of trainning dataset, and whether(how) to use splited dataset '''
hyper_params['data_name'] = hyper_params['data_path'].split('/')[1]
try:
    hyper_params['cut_rate'] = hyper_params['subgraph_file'].split('_')[1].split('.')[0]
except:
    hyper_params['cut_rate'] = None # Using entire trainning graph

train_dataset_path = get_dataset_path(hyper_params) # with enough information, path to training set found
    
''' To ensure that node embedding is same with input_dim '''
hyper_params['input_dim'] -= hyper_params['position_dim']
hyper_params["hidden1_dim"] = hyper_params["input_dim"] + hyper_params["position_dim"]

''' HPO_NEURO datasets is a multi_label datasets, which need specific graph loader file'''
if hyper_params['data_name'] == 'hpo_neuro':
    hyper_params['num_class'] = 10
    hyper_params['multi_label'] = True
    
    

# Training
def train(graph_loader: GraphLoader):
    # Define dataloader for training
    print('Train: Generating train dataset...')

    # Loading dataset
    try:
        train_dataset = torch.load(open(train_dataset_path, 'rb'))            
    except Exception as err:
        print(err)
        os.makedirs(os.path.join('model_dat', hyper_params['data_name']), exist_ok=True)
        train_dataset = graph_loader.generate_dataset(input_hyper_params= hyper_params, mode="train")
        torch.save(train_dataset, open(train_dataset_path, 'wb'))
       
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=hyper_params['batch_size'],
        shuffle=False
    )

    # Define model
    # POSITION_AWARE VGAE
    P_model = PositionEncoding(hyper_params, graph_loader.graph).to(hyper_params['device'])
    model = PA_VGAE(hyper_params, len(graph_loader.graph), P_model).to(hyper_params['device'])
    optimizer = torch.optim.AdamW(model.parameters())

    # start train
    print('Train: Start training...')
    for epoch in range(hyper_params['epochs']):
        total_reconstruct = 0
        total_kl = 0
        total_steps = 0

        for step, (adj, adj_norm, adj_mask, nodes, l, label) in enumerate(train_loader):
            adj = adj.to(hyper_params['device'])
            adj_norm = adj_norm.to(hyper_params['device'])
            adj_mask = adj_mask.to(hyper_params['device'])
            nodes = nodes.to(hyper_params['device'])
            l = l.to(hyper_params['device'])
            label = label.to(hyper_params['device'])

            # Reconstruct matrix
            reconstruct = model(adj_norm, nodes)

            # Calculate loss weight
            adj_masked = adj[adj_mask]
            loss_weight = torch.ones_like(adj_masked)
            loss_weight[adj_masked==1] = graph_loader.edge_weight

            # Calculate loss
            loss = F.binary_cross_entropy(reconstruct[adj_mask], adj[adj_mask].float(), weight=loss_weight)
            total_reconstruct += loss.item()
            kl_loss = model.get_kl() * 0.2
            total_kl += kl_loss.item()
            loss += kl_loss
            total_steps += 1

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch} Reconstruct: {total_reconstruct / total_steps} KL: {total_kl / total_steps}')
    
    if hyper_params['cut_rate']:
        save_path = os.path.join('model_dat', hyper_params['data_name'],
                             'pre_model_'+hyper_params['cut_rate']+'.pkl')
    else:
        save_path = os.path.join('model_dat', hyper_params['data_name'],
                             'pre_model'+'.pkl')
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    try:
        graph_loader = pickle.load(open(os.path.join('model_dat', hyper_params['data_name'],
                                                         'graph_loader.pkl'), 'rb'))
    except Exception as err:
        print(err)
        os.makedirs(os.path.join('model_dat',  hyper_params['data_name']), exist_ok=True)
        if hyper_params['multi_label']:
            graph_loader = Multilabel_GraphLoader(hyper_params)
        else:
            graph_loader = GraphLoader(hyper_params)
        pickle.dump(graph_loader, open(os.path.join('model_dat',  hyper_params['data_name'],
                                        'graph_loader.pkl'), 'wb'))

    train(graph_loader)
