import pickle
import networkx as nx
import random
import numpy as np
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


parser = argparse.ArgumentParser(description='Perform diffusion with given data path and name(i.e. hpo_metab).\n You should also specify the average lenth and count of sampled neighbors. \n The results will be saved to /data_path/data_name/diffuse_subgraphs.pth')

parser.add_argument('--data-path', type=str, default='./data', help='path to SubGNN datasets')
parser.add_argument('--data-name', type=str, default='cut_ratio', help='name of task to be sampled')
parser.add_argument('--subgraph_name', type=str, default='subgraphs', help='base name of subraphs .pth file')
parser.add_argument('--sample-cnt', type=int, default=10, help='excepted smapled node counts in each subgraph')


subgraph_name = 'subgraphs'
cut_ratio = ['10', '20', '30', '40', '50']

args = parser.parse_args()

def get_diffused_nodes(nodes, sample_cnt, input_graph):
    '''
    get a diffused subgraphs with given sample counts
    INPUTS:
    nodes: input subgraph ('s node ids)
    sample_cnt: count of sample neighbors
    input_graph: networkx graph
    '''
    diffuse_set = set()
    nodes_set = set(nodes)
    for node in nodes:
        neighbors = input_graph.neighbors(node)
        for n in neighbors:
            diffuse_set.add(n)
        
    neighbors = diffuse_set - nodes_set
    diffused_nodes = nodes + random.sample(neighbors, sample_cnt)
    return diffused_nodes


model_dat_dir = os.path.join('model_dat', args.data_name)
data_dir = os.path.join(args.data_path, args.data_name)
sample_cnt = args.sample_cnt

hyper_params = {
    "data_path": data_dir,
    "subgraph_file": "subgraphs.pth",
    "max_subgraph_len": 320,
}

try:
    graph_loader = pickle.load(open(os.path.join(model_dat_dir, 'graph_loader.pkl'), 'rb'))
except Exception as err:
    print(err)
    os.makedirs(model_dat_dir, exist_ok=True)
    graph_loader = GraphLoader(hyper_params)
    pickle.dump(graph_loader, open(os.path.join(model_dat_dir, 'graph_loader.pkl'), 'wb'))
    
input_graph = graph_loader.graph

diffused_file = open(os.path.join(data_dir, "diffuse_"+ subgraph_name +".pth"), 'w')

with open(os.path.join(data_dir, subgraph_name + '.pth'), 'rt') as f:
    for line in tqdm(f.readlines()):    
        nodes, label, mode = line.rstrip().split('\t')
        nodes = list(map(int, nodes.split('-')))
        diffused_nodes = get_diffused_nodes(nodes, sample_cnt, input_graph)
        diffused_nodes = '-'.join(list(map(str, diffused_nodes)))
        diffused_file.writelines(diffused_nodes + '\t' + label + '\t' + mode + '\n')

for cut_rate in cut_ratio:
    diffused_file = open(os.path.join(data_dir, "diffuse_"+subgraph_name+'_'+cut_rate+".pth"), 'w')

    with open(os.path.join(data_dir, subgraph_name+'_'+cut_rate+'.pth'), 'rt') as f:
        for line in tqdm(f.readlines()):    
            nodes, label, mode = line.rstrip().split('\t')
            nodes = list(map(int, nodes.split('-')))
            diffused_nodes = get_diffused_nodes(nodes, sample_cnt, input_graph)
            diffused_nodes = '-'.join(list(map(str, diffused_nodes)))
            diffused_file.writelines(diffused_nodes + '\t' + label + '\t' + mode + '\n')

diffused_file.close()