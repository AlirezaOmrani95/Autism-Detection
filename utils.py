import torch, torch.nn as nn
import torchvision as tv, torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
#importing libraries for explainability
from xailib.explainers.lime_explainer import LimeXAIImageExplainer
from xailib.explainers.rise_explainer import RiseXAIImageExplainer
from xailib.metrics.insertiondeletion import ImageInsDel
from skimage.color import label2rgb, gray2rgb, rgb2gray

from sklearn.metrics import auc
from scipy.ndimage.filters import gaussian_filter

def plot_history(train_loss,train_acc, valid_loss, valid_acc, test_loss, test_acc, highest_acc,epoch_num):
    plt.figure(figsize=(10,7))
    plt.subplot(121)
    plt.title('loss')
    plt.plot(range(epoch_num), train_loss, color = 'blue', label = 'train')
    plt.plot(range(epoch_num), valid_loss, color = 'orange', label = 'valid')
    plt.plot(range(epoch_num), test_loss, color = 'red', label='test')
    plt.legend()
    plt.subplot(122)
    plt.title('accuracy')
    plt.plot(range(epoch_num),train_acc,color='blue',label='train')
    plt.plot(range(epoch_num),valid_acc,color='orange',label='valid')
    plt.plot(range(epoch_num),test_acc,color='red',label='test')
    plt.plot(range(epoch_num),highest_acc,color='gray',label='highest acc')
    plt.legend()
    plt.savefig('./history_diagram_multi_aug.png')

def load_model(label_num):
    model_weight = tv.models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1
    model = tv.models.vit_h_14(weights = model_weight)
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    model.conv_proj.requires_grad = True
    
    model.heads.head = nn.Sequential(nn.Dropout(0.5,inplace=True),
                            nn.Linear(model.heads.head.in_features,label_num))

    auto_transform = model_weight.transforms()

    return (model,auto_transform)

#For Explainability
def read_img(path):
    #reading image for explainability
    img = torch.from_numpy(plt.imread(path)).permute(2,0,1)

    return img

def get_lime(image,model,transform,device,num_samples=5000):
    #This transform is used for displaying the picture
    display_transform = transforms.Compose([
        transforms.Resize((224,224)),
        ])
    def classifier_fn(images):
        with torch.no_grad():
            images = torch.tensor(images).permute(0,3,1,2)
            pred = torch.nn.functional.sigmoid(model(images.to(device))).detach().cpu().numpy()
            return pred
    # Create the Explainer
    lm = LimeXAIImageExplainer(model)
    
    # Fit the Explainer
    lm.fit()

    # Explain an Instance
    
    explanation = lm.explain(transform(image).permute(1,2,0).numpy(), classifier_fn, num_samples=num_samples)

    # Plot the results
    lm.plot_lime_values(display_transform(image).permute(1,2,0).numpy(), explanation)
    ind =  explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[ind])
    lime_score = np.vectorize(dict_heatmap.get)(explanation.segments)

    return lime_score

def get_rise(image, model, transform, device):
    class New_Model():
        def __init__(self, bb, input_size):
            self.model = bb
            self.input_size = input_size

        def predict(self, X):
            with torch.no_grad():
                images = torch.tensor(X).permute(0,3,1,2).float()
                return torch.nn.functional.softmax(model(images.to(device)),dim=1).cpu().detach().numpy()
    
    #This transform is used for displaying the picture
    display_transform = transforms.Compose([
        transforms.Resize((224,224)),
        ])

    model_new = New_Model(model, (224,224))
    rise = RiseXAIImageExplainer(model_new)
    
    N = 1000 # number of random masks
    s = 10 # cell_size = input_shape / s
    p1 = 0.9 # masking probability

    rise.fit(N, s, p1)

    explanation = rise.explain(transform(image).permute(1,2,0).numpy())
    rise_score = explanation[0,:]
    fig, ax = plt.subplots(1,3,figsize=(10,5))

    ax[0].imshow(display_transform(image).permute(1,2,0).numpy(),cmap='gray')
    ax[0].axis('off')

    ax[1].imshow(explanation[0,:],cmap='jet')
    ax[1].axis('off')

    ax[2].imshow(display_transform(image).permute(1,2,0).numpy(),cmap='gray')
    ax[2].imshow(explanation[0,:],cmap='jet',alpha=0.5)
    ax[2].axis('off')
    return rise_score

def gkern(klen, nsig):
    """
    Returns a Gaussian kernel array.
    Convolution with it results in image blurring.
    """
    CH = 3
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((CH, CH, klen, klen))
    for i in range(CH):
        kern[i, i] = k
    return torch.from_numpy(kern.astype('float32'))

def blur(image, klen=25, ksig=25):
    '''
    Function that blurs input image
    Returns the blurry version of the input image. 
    '''
    kern = gkern(klen, ksig)
    image = torch.tensor(image).float()
    return nn.functional.conv2d(image, kern, padding=klen//2)

def explainability_metric(model, image, transform, lime_score, rise_score, mode, device,step=224):
    def predict(image):
        with torch.no_grad():
            pred = torch.nn.functional.softmax(model(torch.from_numpy(image).to(device)),dim=1)
            return pred
    
    if mode == 'ins':
        metric = ImageInsDel(predict, mode, step, torch.zeros_like)
    else:
        metric = ImageInsDel(predict, mode, step, blur)

    rise_result = metric(transform(image).unsqueeze(0).numpy(), step, rise_score, rgb=True)
    lime_result = metric(transform(image).unsqueeze(0).numpy(), step, lime_score, rgb=True)
    
    return lime_result, rise_result

def plot_explainabilty(y_lime,y_rise,mode, step=224):
    x = np.arange(len(y_lime))/(224*224)*step
    x[-1] = 1.0

    for name, y in zip(['lime','rise'],[y_lime,y_rise]):
        plt.plot(x, y, label=f'{name}: {np.round(auc(x, y),4)}')
        plt.fill_between(x, y, alpha=0.4)
    if mode == 'del':
        plt.xlabel('Percentage of pixel removed',fontsize=20)
    else:
        plt.xlabel('Percentage of pixel inserted',fontsize=20)
        plt.ylabel('Accuracy of the model',fontsize=20)
        plt.legend(loc='lower left', fontsize=20)