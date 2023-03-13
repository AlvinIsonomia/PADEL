from networkx.classes import graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
from dataset_multi import Multilabel_GraphLoader
from dataset_con import Con_GraphLoader
from utils import get_dataset_path
from dataset import GraphLoader
from model import VGAE, PA_VGAE, Classifier, Attention_Pooling, PositionEncoding, PClassifier, GRU_pooling, LSTM_pooling
import os
import shutil
import pickle
from tqdm import tqdm
from utils import get_reconstruct_norm, get_encodes, get_encodes_A, compute_encodes_score, nce_loss,calc_accuracy
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


hyper_params = {
    "data_path": "data/em_user/",
    "subgraph_file": "subgraphs_30.pth",
    "batch_size": 32,
    "device": 'cuda:2',
    "epochs": 100,
    "input_dim": 96,
    "position_dim": 32,
    "hidden1_dim": 32,
    "hidden2_dim": 32,
    "diffuse": False, # IMPORTANT! Using pre-diffused subgprahs as dataset!
    "max_subgraph_len": 400, # 320(+80) For EM_USER, 40(+10) for others, (+) means diffused subgraphs 
    "eps_std": 3.0,
    "feature_lr": 2e-3,
    "pooling_lr": 2e-3,
    "pca_distance": True,
    "pca_dim": 128,
    "aggregator": "hidden_last",    
    "sample_cnt": 64,
    "multi_label": False
}

''' Get name of trainning dataset, and whether(how) to use splited dataset '''
hyper_params['data_name'] = hyper_params['data_path'].split('/')[1]
try:
    hyper_params['cut_rate'] = hyper_params['subgraph_file'].split('_')[1].split('.')[0]
except:
    hyper_params['cut_rate'] = None # Using entire trainning graph

train_dataset_path = get_dataset_path(hyper_params) # with enough information, path to training set found
con_dataset_path = get_dataset_path(hyper_params, con=True) # with enough information, path to training set found


hyper_params['input_dim'] -= hyper_params['position_dim']


hyper_params["hidden1_dim"] = hyper_params["input_dim"] + hyper_params["position_dim"]

''' HPO_NEURO datasets is a multi_label datasets, which need specific graph loader file'''
if hyper_params['data_name'] == 'hpo_neuro':
    hyper_params['num_class'] = 10
    hyper_params['multi_label'] = True

# Training
def sample(graph_loader: Con_GraphLoader):
    # Define dataloader for training
    print('Sample: Generating sample dataset...')

    # Loading dataset
    print('Loading:Training set……')
    try:
        train_graph_loader = pickle.load(open(os.path.join('model_dat', hyper_params['data_name'],
                                                           'graph_loader.pkl'), 'rb'))
    except Exception as err:
        print(err)
        os.makedirs(os.path.join('model_dat',  hyper_params['data_name']), exist_ok=True)
        if hyper_params['multi_label']:
            train_graph_loader = Multilabel_GraphLoader(hyper_params)
        else:
            train_graph_loader = GraphLoader(hyper_params)
        pickle.dump(train_graph_loader, open(os.path.join('model_dat',  hyper_params['data_name'],
                                        'graph_loader.pkl'), 'wb'))
    
    # Loading dataset
    try:
        class_dataset = torch.load(open(train_dataset_path, 'rb'))
    except Exception as err:
        print(err)
        os.makedirs(os.path.join('model_dat', hyper_params['data_name']), exist_ok=True)
        class_dataset = train_graph_loader.generate_dataset(hyper_params, "train")
        torch.save(class_dataset, open(train_dataset_path, 'wb'))


    class_loader = Data.DataLoader(
        dataset=class_dataset,
        batch_size=32,
        shuffle=True
    )    

    # contrastive leanring dataset & loader
    try:
        con_dataset = torch.load(open(con_dataset_path, 'rb'))
    except Exception as err:
        print(err)
        con_dataset = con_graph_loader.generate_dataset(hyper_params, "train")
        torch.save(con_dataset, open(con_dataset_path, 'wb'))

    con_loader = Data.DataLoader(
            dataset = con_dataset,
            batch_size = hyper_params['sample_cnt']+1,
            shuffle = True 
    )
    
    # Random walk loader, NOTE: RUN 00_random_walker firstly
    negative_dataset = torch.load(open(os.path.join('model_dat', hyper_params['data_name'],
                                                             'negative_dataset.pkl'), 'rb'))
    negative_loader = Data.DataLoader(
                        dataset=negative_dataset,
                        batch_size=hyper_params['sample_cnt']//2,
                        shuffle=True)
    
    # Position model
    P_model = PositionEncoding(hyper_params, train_graph_loader.graph).to(hyper_params['device'])
    
    # Load model
    model =  PA_VGAE(hyper_params, len(graph_loader.graph), P_model).to(hyper_params['device'])
    if hyper_params['cut_rate']:
        model.load_state_dict(torch.load(open(os.path.join('model_dat', hyper_params['data_name'],
                                                'pre_model_'+hyper_params['cut_rate']+'.pkl'), 'rb')))
    else:
        model.load_state_dict(torch.load(open(os.path.join('model_dat', hyper_params['data_name'],
                                                'pre_model_'+'.pkl'), 'rb')))

    P_model = model.PE
    
    # Pooling model 
    pooling = LSTM_pooling(hyper_params).to(hyper_params['device'])
    

    # Classifier
    classifier = PClassifier(hyper_params, len(train_graph_loader.labels)).to(hyper_params['device'])
    classifier.train()
    
    # Optimizer
    optimizer = torch.optim.AdamW([
        {
            "params": P_model.parameters(),
            "lr": hyper_params['feature_lr']
        },        
        {
            "params": pooling.parameters(),
            "lr": hyper_params['pooling_lr']
        }
    ])
    # Optimizer
    class_optimizer = torch.optim.AdamW([
        {
            "params": model.parameters(),
            "lr": hyper_params['feature_lr']*2
        },
        #{
        #    "params": P_model.parameters(),
        #    "lr": hyper_params['feature_lr']*2
        #},        
        {
            "params": pooling.parameters(),
            "lr": hyper_params['pooling_lr']*2
        },
        {
            "params": classifier.parameters(),
            "lr": hyper_params['pooling_lr']*2
        }
    ])
    
    # Loss function
    if hyper_params['multi_label']:
        loss_func = nn.BCEWithLogitsLoss()
    else:
        loss_func = nn.CrossEntropyLoss()
    
    for epoch in range(hyper_params['epochs']):
        total_loss = 0.0
        class_loss = 0.0
        total_steps = 0
        total_class_steps = 0
        total_positive_score = 0.0
        total_random_score = 0.0
        total_negative_score = 0.0
        #######################
        total_correct = 0
        total_cnt = 0
        #######################
        t = tqdm(con_loader)
        for step, (adj, adj_norm, adj_mask, nodes, l, label) in enumerate(t):
            adj = adj.to(hyper_params['device'])
            adj_norm = adj_norm.to(hyper_params['device'])
            adj_mask = adj_mask.to(hyper_params['device'])
            nodes = nodes.to(hyper_params['device'])
            l = l.to(hyper_params['device'])
            label = label.to(hyper_params['device'])

            new_adj_norm = get_reconstruct_norm(model, adj_norm, nodes, l,
                                                l.size(0)).to(hyper_params['device'])
            real_PN_encodes, real_S_encodes = get_encodes_A(model, P_model, pooling, 
                                                              adj_norm, nodes, l)
            aug_PN_encodes, aug_S_encodes = get_encodes_A(model, P_model, pooling, 
                                                            new_adj_norm, nodes, l)
            loss = 0
            for i in range(l.size(0)):
                # Compute Positive Score
                positive_score = compute_encodes_score(real_PN_encodes[i], real_S_encodes[i],
                                                       aug_PN_encodes[i], aug_S_encodes[i])

                # Compute Global-view(Random walk) Negtive Score
                ran_adj, ran_adj_norm, ran_adj_mask, ran_nodes, ran_l, ran_label = next(iter(negative_loader))
                ran_adj_norm = ran_adj_norm.to(hyper_params['device'])
                ran_nodes = ran_nodes.to(hyper_params['device'])
                ran_l = ran_l.to(hyper_params['device'])
                ran_PN_encodes, ran_S_encodes = get_encodes_A(model, P_model, pooling, ran_adj_norm,
                                                            ran_nodes, ran_l)
                random_score = compute_encodes_score(real_PN_encodes[i], real_S_encodes[i], 
                                                     ran_PN_encodes, ran_S_encodes)

                # Compute Local-view Negtive Score
                neg_PN_encodes = aug_PN_encodes[torch.arange(aug_PN_encodes.size(0))!=i] 
                neg_S_encodes = aug_S_encodes[torch.arange(aug_S_encodes.size(0))!=i] 
                negative_score = compute_encodes_score(real_PN_encodes[i], real_S_encodes[i],
                                                       neg_PN_encodes, neg_S_encodes)
                # Calculate NCE Loss
                negative_score = torch.cat([negative_score, random_score])
                loss += nce_loss(positive_score, negative_score)
                
                # Recording Statistics
                total_positive_score += positive_score.sum()
                total_random_score += random_score.sum()
                total_negative_score += negative_score.sum()
                total_steps += 1
            
            total_loss += loss.item()
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step()
            
            ###################################################################################################
            class_adj, class_adj_norm, class_adj_mask, class_nodes, class_l, class_label = next(iter(class_loader))
            class_adj_norm = class_adj_norm.to(hyper_params['device'])
            class_nodes = class_nodes.to(hyper_params['device'])
            class_l = class_l.to(hyper_params['device'])
            class_label = class_label.to(hyper_params['device'])            
            # Get node embedding
            node_position_embedding = P_model(class_nodes)
            node_neibor_embedding = model.embedding(class_nodes)
            # Position-Neiborhood embedding
            node_embedding = node_neibor_embedding
            node_embedding = pooling.neighbor_pooling(node_embedding, class_adj_norm)

            # LSTM pooling
            structure_embedding = pooling(node_position_embedding, class_l)
            
            # Final Position-Neiborhood Average pooling
            average_embedding = node_embedding.mean(dim=-2)
            
            # Get Classification
            pred = classifier(average_embedding, structure_embedding)

            # Calculate loss
            if hyper_params['multi_label']:
                class_loss = loss_func(pred.squeeze(1), class_label.type_as(pred)) 
            else:
                class_loss = loss_func(pred, class_label)
            total_class_steps += 1

            # Calculate correnct cnt
            if hyper_params['multi_label']:
                total_correct += calc_accuracy(pred.cpu().detach(), class_label.cpu().detach(), hyper_params['multi_label']).item() * pred.size(0)
                total_cnt += pred.size(0)
            else:
                total_correct += torch.sum(torch.argmax(pred, dim=-1) == class_label).item()
                total_cnt += pred.size(0)
                
            # Optimize
            class_optimizer.zero_grad()
            class_loss.backward()
            # nn.utils.clip_grad_norm_(pooling.parameters(), max_norm=5, norm_type=2)
            class_optimizer.step()
            ############################################################################################
            
            t.set_description(f'loss: {loss.item()}')
                
        print(f'Epoch: {epoch} Loss: {total_loss / total_steps}')
        try:
            print(f'Pos: {total_positive_score / total_steps} Ran: {total_random_score / total_steps} Neg: {total_negative_score / total_steps} Train ACC: {total_correct / total_cnt}')
        except:
            print(f'Pos: {total_positive_score / total_steps} Ran: {total_random_score / total_steps} Neg: {total_negative_score / total_steps} ')
        
        # Save model
        if hyper_params['cut_rate']:
            torch.save(P_model.state_dict(), os.path.join('model_dat', hyper_params['data_name'],
                                                     'con_position'+hyper_params['cut_rate']+'.pkl'))
            torch.save(model.state_dict(), os.path.join('model_dat', hyper_params['data_name'],
                                                'con_pavgae_'+hyper_params['cut_rate']+'.pkl'))
            torch.save(pooling.state_dict(), os.path.join('model_dat', hyper_params['data_name'],
                                                      'con_LSTM_Pooling'+hyper_params['cut_rate']+'.pkl'))
        else:
            torch.save(P_model.state_dict(), os.path.join('model_dat', hyper_params['data_name'],
                                                     'con_position.pkl'))
            torch.save(model.state_dict(), os.path.join('model_dat', hyper_params['data_name'],
                                                'con_pavgae_'+'.pkl'))
            torch.save(pooling.state_dict(), os.path.join('model_dat', hyper_params['data_name'],
                                                      'con_LSTM_Pooling.pkl'))
        try:
            torch.save(classifier.state_dict(), os.path.join('model_dat', hyper_params['data_name'],
                                                      'con_class'+hyper_params['cut_rate']+'.pkl'))
        except:
            pass

        
if __name__ == '__main__':
    try:
        con_graph_loader = pickle.load(open(os.path.join('model_dat', hyper_params['data_name'],
                                                     'con_graph_loader.pkl'), 'rb'))
    except Exception as err:
        print(err)
        os.makedirs(os.path.join('model_dat',  hyper_params['data_name']), exist_ok=True)
        con_graph_loader = Con_GraphLoader(hyper_params)
        pickle.dump(con_graph_loader, open(os.path.join('model_dat',  hyper_params['data_name'],
                                                    'con_graph_loader.pkl'), 'wb'))

    # Sample
    sample(con_graph_loader)