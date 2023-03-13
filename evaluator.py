import torch
import torch.utils.data as Data
from dataset import GraphLoader
from model import Classifier, Attention_Pooling, PositionEncoding, VGAE, Classifier, GRU_pooling, LSTM_pooling, PClassifier
import os
import pickle
from tqdm import tqdm
from utils import subgraph_encode, calc_accuracy, calc_f1

# Hyper Params
hyper_params = {
    "data_path": "data/hpo_metab/",
    "batch_size": 32,
    "device": 'cpu',
    "epochs": 200,
    "input_dim": 32,
    "position_dim": 32,
    "hidden1_dim": 64,
    "hidden2_dim": 32,
    "max_subgraph_len": 40,
    "eps_std": 3.0,
    "pca_distance": True,
    "pca_dim": 128,
}
hyper_params["hidden1_dim"] = hyper_params["input_dim"] + hyper_params["position_dim"]
hyper_params['data_name'] = hyper_params['data_path'].split('/')[1]
# Evalutation
class Evaluator:
    def __init__(self, hyper_params, graph_loader: GraphLoader, mode="test"):
        self.hyper_params = hyper_params
        self.graph_loader = graph_loader

        # Define dataloader for training
        print('Evalutor: Generating classifier dataset...')

        # Loading dataset
        try:
            test_dataset = torch.load(open(os.path.join('model_dat',hyper_params['data_name'] ,f'{mode}_dataset.pkl'), 'rb'))
        except Exception as err:
            print(err)
            os.makedirs('model_dat', exist_ok=True)
            test_dataset = graph_loader.generate_dataset(hyper_params, mode)
            torch.save(test_dataset, open(os.path.join('model_dat', hyper_params['data_name'],f'{mode}_dataset.pkl'), 'wb'))

        self.train_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=hyper_params['batch_size'],
            shuffle=False
        )

    def evaluate(self, model1=None, model2=None, pooling=None, classifier=None, verbose=1):
        if pooling is None:
            pooling = LSTM_pooling(self.hyper_params)
            pooling.load_state_dict(torch.load(open(os.path.join('model_dat', self.hyper_params['data_name'],'GRU_Pooling.pkl'), 'rb')))
        if classifier is None:
            # Classifier model
            classifier = PClassifier(self.hyper_params, len(graph_loader.labels))
            classifier.load_state_dict(torch.load(open(os.path.join('model_dat',self.hyper_params['data_name'], 'classifier.pkl'), 'rb')))

        # start evalution
        if verbose:
            print('Evalutor: Start evalutation...')
        
        total_correct = 0
        total_cnt = 0
        total_f1 = 0
        model1.eval()
        model2.eval()
        pooling.eval()
        classifier.eval()
        model1.to(self.hyper_params['device'])
        model2.to(self.hyper_params['device'])
        pooling.to(self.hyper_params['device'])
        classifier.to(self.hyper_params['device'])
        if verbose:
            t = tqdm(self.train_loader)
        else:
            t = self.train_loader

        for step, (adj, adj_norm, adj_mask, nodes, l, label) in enumerate(t):
            adj = adj.to(self.hyper_params['device'])
            adj_norm = adj_norm.to(self.hyper_params['device'])
            adj_mask = adj_mask.to(self.hyper_params['device'])
            nodes = nodes.to(self.hyper_params['device'])
            l = l.to(self.hyper_params['device'])
            label = label.to(self.hyper_params['device'])
            node_mask = nodes==0
            node_mask.to(self.hyper_params['device'])
            
            # Get node embedding
            node_position_embedding = model1(nodes)
            node_neibor_embedding = model2.embedding(nodes)
            # Position-Neiborhood embedding
            if self.hyper_params['pa_mode'] != 'A':
                node_embedding = torch.cat((node_position_embedding, node_neibor_embedding), dim=-1)
            else:
                node_embedding = node_neibor_embedding
            node_embedding = pooling.neighbor_pooling(node_embedding, adj_norm)

            # LSTM pooling
            structure_embedding = pooling(node_position_embedding, l)
            
            # Final Position-Neiborhood Average pooling
            average_embedding = node_embedding.mean(dim=-2)
            
            # Get Classification
            pred = classifier(average_embedding, structure_embedding)
            
            if self.hyper_params['multi_label']:
                total_correct += calc_accuracy(pred.cpu().detach(), label.cpu().detach(), self.hyper_params['multi_label']).item() * pred.size(0)
                total_cnt += pred.size(0)
                total_f1 += calc_f1(pred.cpu().detach(), label.cpu().detach(), 'micro', self.hyper_params['multi_label']).item() * pred.size(0)
            else:
                total_correct += torch.sum(torch.argmax(pred, dim=-1) == label).item()
                total_cnt += pred.size(0)
                total_f1 += torch.sum(torch.argmax(pred, dim=-1) == label).item()

        print(f'Hit: {total_correct}/{total_cnt}, HR: {total_correct/total_cnt}, f1: {total_f1/total_cnt}')

        model1.train()
        model2.train()
        pooling.train()
        classifier.train()
        return total_f1/total_cnt


if __name__ == '__main__':
    try:
        graph_loader = pickle.load(open(os.path.join('model_dat', hyper_params['data_name'],
                                                     'graph_loader.pkl'), 'rb'))
    except Exception as err:
        print(err)
        os.makedirs('model_dat', exist_ok=True)
        graph_loader = GraphLoader(hyper_params)
        pickle.dump(graph_loader, open(os.path.join('model_dat',hyper_params['data_name'],
                                                    'graph_loader.pkl'), 'wb'))

    # Print basic info
    print(f'labels: {len(graph_loader.labels)}, test size: {len(graph_loader.sub_nodes["test"])}')

    # Sample
    evalutor = Evaluator(hyper_params, graph_loader)
    evalutor.evaluate()