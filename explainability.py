import torch, torch.nn as nn
from torch.utils.data import DataLoader
import torchvision as tv, torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import PIL.Image as Image
from utils import load_model, read_img, get_lime, get_rise, explainability_metric, plot_explainabilty

torch.cuda.manual_seed(1)
torch.manual_seed(1)
np.random.seed(1)

if __name__ == '__main__':
    #defining general variables
    general_params = {'device':'cuda' if torch.cuda.is_available() else 'cpu',
                  'file_path': './datasettest/autistic/001.jpg',
                  'label':torch.ones(1,dtype=torch.int8),
                  'weight_path': './best_model.pth'
                  }
    #loading model
    model,auto_transform = load_model(2)
    model = model.to(general_params['device'])
    model.load_state_dict(torch.load(general_params['weight_path']))

    #testing image
    img = read_img(general_params['file_path'])

    model.eval()
    logits = model(auto_transform(img).unsqueeze(0).to(general_params['device']))
    preds = torch.argmax(torch.softmax(logits,dim=1),dim=1)
    print(preds.item())

    lime_score = get_lime(img, model, transform = auto_transform, device = general_params['device'])
    rise_score = get_rise(img, model, transform = auto_transform, device = general_params['device'])

    #modes => del or ins
    mode = 'del'
    lime_metric_result, rise_metric_result = explainability_metric(model, img, auto_transform, lime_score, rise_score, mode = mode, device = general_params['device'])
    plot_explainabilty(lime_metric_result, rise_metric_result, mode = mode)