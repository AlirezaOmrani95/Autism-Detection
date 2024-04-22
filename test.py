import torch, torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from utils import load_model
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

torch.cuda.manual_seed(1)
torch.manual_seed(1)
np.random.seed(1)

def test(model, dataloader, accuracy, device):
    model.eval()
    acc_lst = []
    with torch.inference_mode():
        for batch_idx, batch_data in enumerate(dataloader,1):
            inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.to(device)
            
            logits = model(inputs).squeeze()
            preds_probs = torch.softmax(logits,dim=1)
            preds = torch.argmax(preds_probs,dim=1)
            
            acc = accuracy(labels,preds)
                    
            acc_lst.append(acc.item())
    return np.mean(acc_lst)

if __name__ == '__main__':
    general_params = {
        'device':'cuda' if torch.cuda.is_available() else 'cpu',
        'root_path': r'./dataset',
        'weight_path': './weight/best_model.pth'
    }

    model_params = {
        'accuracy': Accuracy('multiclass',num_classes=2).to(general_params['device']),

        'batch size': 32
    }

    print(general_params['device'])

    model,auto_transform = load_model(2)

    dataset = tv.datasets.ImageFolder(root = os.path.join(general_params['root_path'],'test'), transform = auto_transform)
    data_loader = DataLoader(dataset = dataset, batch_size= model_params['batch size'])
    
    model = model.to(general_params['device'])
    model.load_state_dict(torch.load(general_params['weight_path']))

    test_acc = test(model, data_loader, model_params['accuracy'],general_params['device'])

    print(f'test set:\nacc: {test_acc:0.2f}')



