import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import torch.utils.data as Data
import os
from typing import List, Tuple, Dict
from utils import *
from tqdm import tqdm
import random


class Multilabel_GraphLoader:
    def __init__(self, hyper_params):
        self.hyper_params = hyper_params
        self.graph: nx.Graph = nx.Graph()

        # Subgraphs contains (nodes of subgraph, label)
        self.sub_nodes: Dict[str, List[Tuple[List[int], int]]] = {
            "train": [],
            "val": [],
            "test": []
        }

        # List of labels
        self.labels: List[str] = []

        # Weight of edge
        self.edge_weight = 1.0

        # Load data from file
        self.__load_edges_list()
        self.__load_from_file(self.hyper_params)
    def __load_edges_list(self):
        # Load edges_list from file
        print('Graph Loader: Load from file...')
        with open(os.path.join(self.hyper_params['data_path'], "edge_list.txt"), 'rt') as f:
            for line in tqdm(f.readlines()):
                u, v = map(int, line.rstrip().split())
                self.graph.add_edge(u, v)
    
    def __load_from_file(self, input_hyper_params):
        print('Dataset: Load from file...') 
        self.sub_nodes: Dict[str, List[Tuple[List[int], int]]] = {
            "train": [],
            "val": [],
            "test": []
        }
        # Load subgraphs
        total_cnt, edge_cnt = 0, 0
        if input_hyper_params['diffuse']:
            temp_subgraph_path = os.path.join(input_hyper_params['data_path'],
                                              'diffuse_'+input_hyper_params['subgraph_file'])
        else:
            temp_subgraph_path = os.path.join(input_hyper_params['data_path'],
                                              input_hyper_params['subgraph_file'])
        with open(temp_subgraph_path, 'rt') as f:
            for line in tqdm(f.readlines()):
                nodes, label, mode = line.rstrip().split()
                nodes = sorted(list(map(int, nodes.split('-'))))
                label_list = label.split('-')
                label_vector = torch.zeros(input_hyper_params['num_class'])

                if len(nodes) == 0 or len(self.graph.subgraph(nodes).edges) == 0:
                    continue
                if len(nodes) > input_hyper_params['max_subgraph_len']:
                    continue
                for label in label_list:
                    if label not in self.labels:
                        self.labels.append(label)
                    label_idx = self.labels.index(label) 
                    label_vector[label_idx] = 1
                self.sub_nodes[mode].append((nodes, label_vector))

                total_cnt += len(nodes) ** 2
                edge_cnt += len(self.graph.subgraph(nodes).edges)
        
        self.edge_weight = (total_cnt - edge_cnt) / edge_cnt

        print('Dataset: Finished.')

    def generate_dataset(self, input_hyper_params, mode="train") -> Data.Dataset:
        self.__load_from_file(input_hyper_params)
        # Generate train dataset (torch)
        adj_list = []
        adj_norm_list = []
        adj_mask_list = []
        node_list = []
        node_cnt_list = []
        label_list = []
        n = input_hyper_params['max_subgraph_len']

        # Calculate graph adj
        total_adj = torch.from_numpy(nx.adjacency_matrix(self.graph).toarray())

        for sub, label in tqdm(self.sub_nodes[mode]):
            # Calculate adj and adj_norm
            # adj = torch.from_numpy(nx.adjacency_matrix(self.graph.subgraph(sub)).toarray())
            adj = total_adj[sub][:, sub]
            adj_diag = adj + torch.eye(adj.size(0))
            degree = torch.diag(adj_diag.sum(dim=1).float() ** -0.5)
            adj_norm = degree @ adj_diag @ degree
            l = len(sub)

            # Pad Zeros
            adj_pad = torch.zeros([n, n], dtype=torch.long)
            adj_pad[:l, :l] = adj_diag
            adj_norm_pad = torch.zeros([n, n], dtype=torch.float)
            adj_norm_pad[:l, :l] = adj_norm
            adj_mask_pad = torch.zeros([n, n], dtype=torch.bool)
            adj_mask_pad[:l, :l] = True
            node_pad = torch.zeros(n, dtype=torch.long)
            node_pad[:l] = torch.tensor([i+1 for i in sub], dtype=torch.long)

            # Append to list
            adj_list.append(adj_pad)
            adj_norm_list.append(adj_norm_pad)
            adj_mask_list.append(adj_mask_pad)
            node_list.append(node_pad)
            node_cnt_list.append(l)
            label_list.append(label)

        # Generate TensorDataset
        adj_list = torch.stack(adj_list)
        adj_norm_list = torch.stack(adj_norm_list)
        adj_mask_list = torch.stack(adj_mask_list)
        node_list = torch.stack(node_list)
        node_cnt_list = torch.tensor(node_cnt_list, dtype=torch.long)
        label_list = torch.stack(label_list)

        dataset = Data.TensorDataset(adj_list, adj_norm_list, adj_mask_list, node_list, node_cnt_list, label_list)
        return dataset
