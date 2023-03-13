from networkx.classes import graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import copy
import numpy as np
from dataset import GraphLoader
from dataset_multi import Multilabel_GraphLoader
from model import VGAE, PA_VGAE, Classifier, Attention_Pooling, PositionEncoding, PClassifier, GRU_pooling, LSTM_pooling
import os
import shutil
import pickle
from tqdm import tqdm
from utils import get_sample, get_negative_samples, contrastive_score, nce_loss, subgraph_encode, calc_accuracy, calc_f1, get_dataset_path
import random
from evaluator import Evaluator

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

seed = 10
setup_seed(seed)
cuda_device_id = str(seed % 5)
# Hyper Params
hyper_params = {
    "data_path": "data/em_user/",
    "subgraph_file": "subgraphs_50.pth",
    "batch_size": 32,
    "device": 'cuda:'+cuda_device_id,
    "epochs": 100,
    "input_dim": 96,
    "position_dim": 32,
    "hidden1_dim": 32,
    "hidden2_dim": 32,
    "diffuse": False, # NEVER use diffuse in classifier.
    "max_subgraph_len": 320, # 320(+80) For EM_USER, 40(+10) for others, (+) means diffused subgraphs 
    "eps_std": 3.0,
    "feature_lr": 5e-3,
    "pooling_lr": 5e-3,
    "classifier_lr": 5e-3,
    "pca_distance": True,
    "pca_dim": 128,
    "aggregator": "hidden_last",
    'multi_label' : False,
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
def classifier(graph_loader: GraphLoader):
    # Define dataloader for training
    print('Classifier: Generating classifier dataset...')

    # Loading dataset
    try:
        train_dataset = torch.load(open(train_dataset_path, 'rb'))
    except Exception as err:
        print(err)
        os.makedirs(os.path.join('model_dat', hyper_params['data_name']), exist_ok=True)
        train_dataset = graph_loader.generate_dataset(hyper_params, "train")
        torch.save(train_dataset, open(train_dataset_path, 'wb'))
    
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=hyper_params['batch_size'],
        shuffle=True
    )

    # Position model
    model1 = PositionEncoding(hyper_params, graph_loader.graph).to(hyper_params['device'])
    
    # Load model
    model2 =  PA_VGAE(hyper_params, len(graph_loader.graph), model1).to(hyper_params['device'])

    
    
    if hyper_params['cut_rate']:
        try:
            model1.load_state_dict(torch.load(open(os.path.join('model_dat', hyper_params['data_name'], 'con_position'+hyper_params['cut_rate']+'.pkl'), 'rb')))
            print("!!!!!!!!!!!!!!!!!!!! Using contrast learning as Position embedding !!!!!!!!!!!!!!!!!!!!")
        except Exception as err:
            print("End-to-end Trainning!")
            pass
        try:
            model2.load_state_dict(torch.load(open(os.path.join('model_dat', hyper_params['data_name'],
                                                    'con_pavgae_'+hyper_params['cut_rate']+'.pkl'), 'rb')))       
            print("!!!!!!!!!!!!!!!!!!!! Using contrast learning AVB embedding as Neighbor embedding !!!!!!!!!!!!!!!!!!!!")
        except Exception as err:
            model2.load_state_dict(torch.load(open(os.path.join('model_dat', hyper_params['data_name'],
                                                    'pre_model_'+hyper_params['cut_rate']+'.pkl'), 'rb')))       
            print("Using Self-Supervised pre-trainning!")
            pass
    else:
        try:
            model1.load_state_dict(torch.load(open(os.path.join('model_dat', hyper_params['data_name'], 'con_position.pkl'), 'rb')))
            print("!!!!!!!!!!!!!!!!!!!! Using contrast learning as Position embedding !!!!!!!!!!!!!!!!!!!!")
        except Exception as err:
            print("End-to-end Trainning!")
            pass
        try:
            model2.load_state_dict(torch.load(open(os.path.join('model_dat', hyper_params['data_name'],
                                                    'con_pavgae_'+'.pkl'), 'rb')))       
            print("!!!!!!!!!!!!!!!!!!!! Using contrast learning AVB embedding as Neighbor embedding !!!!!!!!!!!!!!!!!!!!")
        except Exception as err:
            model2.load_state_dict(torch.load(open(os.path.join('model_dat', hyper_params['data_name'],
                                                    'pre_model_'+'.pkl'), 'rb')))       
            print("Using Self-Supervised pre-trainning!")
            pass
    
    model1 = model2.PE
    # Pooling model
    pooling = LSTM_pooling(hyper_params).to(hyper_params['device'])
    
    
    if hyper_params['cut_rate']:
        try:
            pooling.load_state_dict(torch.load(open(os.path.join('model_dat',  hyper_params['data_name'], 'con_LSTM_Pooling'+hyper_params['cut_rate']+'.pkl'), 'rb')))
            print("!!!!!!!!!!!!!!!!!!!! Using contrast learning LSTM pooling !!!!!!!!!!!!!!!!!!!!") 
        except Exception as err:
            print("End-to-end Pooling Trainning!")
            pass
    else:
        try:
            #pooling.load_state_dict(torch.load(open(os.path.join('model_dat',  hyper_params['data_name'], 'con_LSTM_Pooling.pkl'), 'rb')))
            print("!!!!!!!!!!!!!!!!!!!! Using contrast learning LSTM pooling !!!!!!!!!!!!!!!!!!!!") 
        except Exception as err:
            print("End-to-end Pooling Trainning!")
            pass
    
    # Classifier model
    classifier = PClassifier(hyper_params, len(graph_loader.labels)).to(hyper_params['device'])

    model1.train()
    model2.train()
    pooling.train()
    classifier.train()

    # Optimizer
    optimizer = torch.optim.AdamW([
        #{
        #    "params": model1.parameters(),
        #    "lr": hyper_params['feature_lr'],
        #},
        {
            "params": model2.parameters(),
            "lr": hyper_params['feature_lr'],
        },
        {
            "params": pooling.parameters(),
            "lr": hyper_params['pooling_lr'],
        },        
        {
            "params": classifier.parameters(),
            "lr": hyper_params['classifier_lr'],
        }
    ])

    # Loss function
    if hyper_params['multi_label']:
        loss_func = nn.BCEWithLogitsLoss()
    else:
        loss_func = nn.CrossEntropyLoss()

    # Evalutor
    val_evaluator = Evaluator(hyper_params, graph_loader, mode="val")
    test_evaluator = Evaluator(hyper_params, graph_loader, mode="test")

    # start training classifier
    print('Classifier: Start training classifier...')
    max_f1 = 0
    best_model = None

    for epoch in range(hyper_params['epochs']):
        total_loss = 0.0
        total_steps = 0

        total_correct = 0
        total_cnt = 0

        t = tqdm(train_loader)
        for step, (adj, adj_norm, adj_mask, nodes, l, label) in enumerate(t):
            adj = adj.to(hyper_params['device'])
            adj_norm = adj_norm.to(hyper_params['device'])
            adj_mask = adj_mask.to(hyper_params['device'])
            nodes = nodes.to(hyper_params['device'])
            l = l.to(hyper_params['device'])
            label = label.to(hyper_params['device'])
            node_mask = nodes==0
            node_mask.to(hyper_params['device'])
            
                        
            # Get node embedding
            node_position_embedding = model1(nodes)
            node_neibor_embedding = model2.embedding(nodes)
            # Position-Neiborhood embedding
            node_embedding = node_neibor_embedding
            node_embedding = pooling.neighbor_pooling(node_embedding, adj_norm)

            # LSTM pooling
            structure_embedding = pooling(node_position_embedding, l)
            
            # Final Position-Neiborhood Average pooling
            average_embedding = node_embedding.mean(dim=-2)
            
            # Get Classification
            pred = classifier(average_embedding, structure_embedding)

            # Calculate loss
            if hyper_params['multi_label']:
                loss = loss_func(pred.squeeze(1), label.type_as(pred)) 
            else:
                loss = loss_func(pred, label)
            total_loss += loss.item()
            total_steps += 1

            # Calculate correnct cnt
            if hyper_params['multi_label']:
                total_correct += calc_accuracy(pred.cpu().detach(), label.cpu().detach(), hyper_params['multi_label']).item() * pred.size(0)
                total_cnt += pred.size(0)
            else:
                total_correct += torch.sum(torch.argmax(pred, dim=-1) == label).item()
                total_cnt += pred.size(0)
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(pooling.parameters(), max_norm=5, norm_type=2)
            optimizer.step()

            t.set_description(f'loss: {loss.item()}')

        print(f'Epoch: {epoch} Loss: {total_loss / total_steps} Train HR: {total_correct / total_cnt}')
        val_f1 = val_evaluator.evaluate(model1, model2, pooling, classifier, verbose=0)

        if val_f1 > max_f1:
            max_f1 = val_f1
            best_model = {
                "model1": copy.deepcopy(model1.state_dict()),
                "model2":  copy.deepcopy(model2.state_dict()),
                "pooling": copy.deepcopy(pooling.state_dict()),
                "classifier": copy.deepcopy(classifier.state_dict())
            }

        # Save model
        torch.save(model1.state_dict(), os.path.join('model_dat', hyper_params['data_name'],
                                                     'position.pkl'))
        torch.save(model2.state_dict(), os.path.join('model_dat', hyper_params['data_name'],
                                                     'model.pkl'))
        #torch.save(pooling.state_dict(), os.path.join('model_dat', hyper_params['data_name'],
        #                                              'Attention_Pooling.pkl'))
        torch.save(pooling.state_dict(), os.path.join('model_dat', hyper_params['data_name'],
                                                      'GRU_Pooling.pkl'))
        torch.save(classifier.state_dict(),os.path.join('model_dat',hyper_params['data_name'],
                                                      'Pclassifier.pkl'))
    print('Max val f1 is:', max_f1)

    # Test result on best model
    print('Testing on best model...')
    model1.load_state_dict(best_model['model1'])
    model2.load_state_dict(best_model['model2'])
    pooling.load_state_dict(best_model['pooling'])
    classifier.load_state_dict(best_model['classifier'])

    test_evaluator.evaluate(model1, model2, pooling, classifier)
    print('Test finished.')


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

    # Sample
    classifier(graph_loader)
