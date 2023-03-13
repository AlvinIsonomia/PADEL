import pickle
import networkx as nx
import random
import numpy as np
import tqdm
import os
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from tqdm import tqdm
# =========== Our Method ===============
from dataset import GraphLoader
# =========== Our Method ===============


parser = argparse.ArgumentParser(description='Perform random walk with given data path and name(i.e. hpo_metab).\n You should also specify the average lenth and total numebr of sampled subgraphs. \n The results will be saved to /data_path/data_name/negative.pth')

parser.add_argument('--data-path', type=str, default='./data', help='path to SubGNN datasets')
parser.add_argument('--data-name', type=str, default='em_user', help='name of task to be sampled')
parser.add_argument('--num-negative', type=int, default=20000, help='total number of negative samples')
parser.add_argument('--lenth-mean', type=int, default=320, help='excepted node counts in each subgraph')



args = parser.parse_args()

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

model_dat_dir = os.path.join('model_dat', args.data_name)
data_dir = os.path.join(args.data_path, args.data_name)
num_negative = 20000

hyper_params = {
    "data_path": data_dir,
    "subgraph_file": "subgraphs.pth",
    "max_subgraph_len": 320,
    "diffuse":False,
}


try:
    graph_loader = pickle.load(open(os.path.join(model_dat_dir, 'graph_loader.pkl'), 'rb'))
except Exception as err:
    print(err)
    os.makedirs(model_dat_dir, exist_ok=True)
    graph_loader = GraphLoader(hyper_params)
    pickle.dump(graph_loader, open(os.path.join(model_dat_dir, 'graph_loader.pkl'), 'wb'))
    
graph = graph_loader.graph
mode = 'train'
label = 'negative'

with open(os.path.join(data_dir, 'negative.pth'), 'w') as f:
    for i in tqdm(range(num_negative)):
        random_node_list = random_walk(graph_loader.graph)
        random_node_str = '-'.join(list(map(str, random_node_list)))
        f.writelines('\t'.join([random_node_str, label, mode])+'\n')

print('Writing negative datasets again……')
hyper_params = {
    "data_path": data_dir,
    "subgraph_file": "negative.pth",
    "batch_size": 32,
    "device": 'cpu',
    "epochs": 500,
    "input_dim": 64,
    "hidden1_dim": 64,
    "hidden2_dim": 32,
    "diffuse": False,
    "max_subgraph_len": 40,
    "eps_std": 3.0,
    "feature_lr": 1e-3,
    "pooling_lr": 1e-3,
    "classifier_lr": 1e-3
}
hyper_params['data_name'] = hyper_params['data_path'].split('/')[-1]
graph_loader = GraphLoader(hyper_params)
train_dataset = graph_loader.generate_dataset(hyper_params, "train")

print(os.path.join('model_dat', hyper_params['data_name'], 'negative_dataset.pkl'))
torch.save(train_dataset, open(os.path.join('model_dat', hyper_params['data_name'], 'negative_dataset.pkl'), 'wb'))
