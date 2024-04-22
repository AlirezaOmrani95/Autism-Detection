import torch, torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from utils import plot_history, load_model
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

torch.cuda.manual_seed(1)
torch.manual_seed(1)
np.random.seed(1)

def train(model, optimizer, criterion, dataloader, device):
    model.train()
    loss_lst = []
    acc_lst = []
    for batch_idx, batch_data in enumerate(dataloader,1):
        inputs, labels = batch_data
        inputs, labels = inputs.to(general_params['device']), labels.to(general_params['device'])
        
        logits = model(inputs).squeeze()
        preds_probs = torch.softmax(logits,dim=1)
        preds = torch.argmax(preds_probs,dim=1)
        
        loss = criterion(logits,labels)
        acc = accuracy_fn(labels,preds)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_lst.append(loss.item())
        acc_lst.append(acc.item())
    return np.mean(loss_lst), np.mean(acc_lst)

def validation(model, criterion, dataloader,epoch_num):
    model.eval()
    loss_lst = []
    acc_lst = []
    with torch.inference_mode():
        for batch_idx, batch_data in enumerate(dataloader,1):
            inputs, labels = batch_data
            inputs, labels = inputs.to(general_params['device']), labels.to(general_params['device'])
            
            logits = model(inputs).squeeze()
            preds_probs = torch.softmax(logits,dim=1)
            preds = torch.argmax(preds_probs,dim=1)
            
            loss = criterion(logits,labels)
            acc = accuracy_fn(labels,preds)
                    
            loss_lst.append(loss.item())
            acc_lst.append(acc.item())
        if epoch_num == 0:
            highest_acc = np.mean(acc_lst)
            lowest_loss = np.mean(loss_lst)
        elif highest_acc < np.mean(acc_lst):
            highest_acc = np.mean(acc_lst)
            lowest_loss = np.mean(loss_lst)
            torch.save(model.state_dict(),'./weight/best_model.pth')

    return np.mean(loss_lst), np.mean(acc_lst),highest_acc,lowest_loss

if __name__ == "__main__":
    general_params = {'device':'cuda' if torch.cuda.is_available() else 'cpu',
                  'root_path': './dataset'
                  }

    model_params = {'accuracy': Accuracy('multiclass',num_classes=2).to(general_params['device']),
                    'loss function': nn.CrossEntropyLoss(),
                    'batch size': 32,
                    'epochs':150}

    print(general_params['device'])

    model,auto_transform = load_model(2)

    datasets = {
        'train':tv.datasets.ImageFolder(root = os.path.join(general_params['root_path'],'train'), transform = auto_transform),
        'valid':tv.datasets.ImageFolder(root = os.path.join(general_params['root_path'],'valid'), transform = auto_transform),
    }
    
    dataloaders = {
        'train':DataLoader(dataset = datasets['train'], batch_size = model_params['batch size'], shuffle = True),
        'valid':DataLoader(dataset = datasets['valid'], batch_size = model_params['batch size']),
    }

    model_params ['optimizer'] = torch.optim.Adam(model.parameters(),lr=1e-4)

    model = model.to(general_params['device'])
    best_result = {'acc':0,
                   'loss':0}
    print('===============Training in Process===============')
    history = {'train_loss':[], 'train_acc':[],
            'validation_loss':[], 'validation_acc':[],
            'test_loss':[],'test_acc':[],
            'highest_acc':[]}

    for epoch in tqdm(range(model_params['epochs'])):
        with tqdm() as tepoch:
            train_loss, train_acc = train(model, model_params['optimizer'], model_params['loss function'], dataloaders['train'])
            
            valid_loss, valid_acc, best_result['acc'], best_result['loss'] = validation(model, model_params['loss function'], dataloaders['valid'])
            
            tepoch.set_postfix({'train_loss': train_loss, 'train_acc': train_acc, 'valid_loss': valid_loss, 'valid_acc':valid_acc})
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['validation_loss'].append(valid_loss)
            history['validation_acc'].append(valid_acc)
            
    plot_history(train_loss = history['train_loss'], train_acc = history['train_acc'],
                valid_loss = history['validation_loss'], valid_acc = history['validation_acc'],
                highest_acc = history['highest_acc'], epoch_num = model_params['epochs'])

    df = pd.DataFrame(history,columns=['train_loss','train_acc','validation_loss','validation_acc','highest_acc'])
    df.to_csv('history.csv',index=False)
    