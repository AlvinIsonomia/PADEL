import os
from tqdm import tqdm
import argparse
import pickle

parser = argparse.ArgumentParser(description='Process all tasks in your dataset path, generate cold start dataset with given ratio.')

parser.add_argument('--data-path', type=str, default='./data', help='path to SubGNN datasets')
parser.add_argument('--cold-ratio', type=float, default=0.1, help='cold_ratio must be a number between (0,1]')

args = parser.parse_args()


data_path = args.data_path
cold_ratio = args.cold_ratio


print('======= Please run data_split.py fisrt! =======')

task_names = os.listdir(data_path)
task_names.remove('tensorboard')
task_names.remove('.ipynb_checkpoints')

print(task_names)

if cold_ratio > 1 or cold_ratio <= 0:
    raise ValueError("cold_ratio must be a number between (0,1]")

for task in task_names:
    val_path = os.path.join(data_path, task, "val_subgraphs.pth")
    test_path = os.path.join(data_path, task, "test_subgraphs.pth")
    
    val_file = open(val_path,'rt')
    test_file = open(test_path,'rt')
    cold_path = os.path.join(data_path, task,'subgraphs' + '_' +str(round(cold_ratio * 100)) + '.pth')
    
    with open(cold_path, 'w') as f:
        f.write(val_file.read())
        f.write(test_file.read())
        
    label_path = os.path.join(data_path, task, "label_dict.pkl")
    with open(label_path, 'rb') as f:
        label_dict = pickle.load(f)
    
    label_cold_dict = {k: int(v * cold_ratio) for k, v in label_dict.items()}

    for label in label_cold_dict:
        # print(label, label_cold_dict[label])
        train_path = os.path.join(data_path, task, 'train_label_subgraphs',
                                  'train_' + label + '_subgraphs.pth')
        train_file = open(train_path, 'rt')
        with open(cold_path, 'a') as f:
            for i in range(label_cold_dict[label]):
                f.writelines(train_file.readline())
            
        train_file.close()
    
    
    print('save file to', cold_path)
    val_file.close()
    test_file.close()
    
    
    