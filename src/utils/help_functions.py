import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import torch

def Voting(name_original, name_sliced, pred, label, use_gpu=True):
    pred_original = [0]*len(name_original)
    label_original = [None]*len(name_original)
    for idx, name in enumerate(name_sliced):
        name = name.split('_')[0]
        indice = np.where(name_original == name)
        indice = int(indice[0])
        if pred[idx] == 1:
            pred_original[indice] += 1
        elif pred[idx] == 0:
            pred_original[indice] -= 1
        label_original[indice] = label[idx]

    pred_original = np.asarray(pred_original)
    label_original = np.asarray(label_original)
    above_zero = pred_original>0
    below_zero = pred_original<0
    pred_original[above_zero] = 1
    pred_original[below_zero] = 0
    
    if use_gpu==True:
        pred_original = torch.tensor(pred_original)
        label_original = torch.tensor(label_original)

    return pred_original, label_original


def ComputeUAR(pred, label, num_classes):
    correct_num = [None]*num_classes
    all_num = [None]*num_classes
    for i in range(num_classes):
        correct_num[i] = ((pred == label.squeeze()) & (label.squeeze()==(torch.zeros(len(label.squeeze()))+i))).sum()
        all_num[i] = len(label == i)
    
    recall = [correct_num[i]/all_num[i] for i in range(len(correct_num))]
    avg_uar = np.mean(recall)
    return avg_uar, recall

def compute_metrics_from_confusion_matrix(matrix, visualize=False, figsize=(6,8), title=''): 
    # compute recall
    intersection = np.diag(matrix)
    ground_truth_set = matrix.sum(axis=1)
    predicted_set = matrix.sum(axis=0)
    recall = intersection / ground_truth_set
    precison = intersection / predicted_set
    f1 = 2*intersection / (ground_truth_set + predicted_set)
    acc = np.sum(intersection)/np.sum(ground_truth_set)
    metrics = {'recall': recall, 'precision': precison, 'f1': f1}
    avg_uar = np.mean(recall[~np.isnan(recall)])

    if visualize:
        classes = matrix.shape[0]
        df_cfmatrix = pd.DataFrame(matrix/(np.sum(matrix, axis=1).reshape(-1,1)+1e-5), index = range(classes),
                  columns = range((classes)))
        df_metrics = pd.DataFrame([metrics['recall'], metrics['precision'], metrics['f1']], index = ['recall','precision', 'f1'],
                columns = range(classes))
        fig, axs = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        sns.heatmap(df_cfmatrix, annot=True, cmap='PuBu', ax=axs[0])
        axs[0].set_xlabel('predict')
        axs[0].set_ylabel('ground truth')
        axs[0].set_title(title)

        sns.heatmap(df_metrics, annot=True, cmap='PuBu', ax=axs[1])
        axs[1].set_xlabel('classes')
        axs[1].set_ylabel('metrics')
        return avg_uar, acc, metrics, fig

    return avg_uar, acc, metrics


def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

import matplotlib.pyplot as plt
def plot_loss(loss_train, loss_val, val_interval):
    plt.plot(range(0, len(loss_train)), loss_train, c='r', label='train')
    plt.plot(range(0, len(loss_val)*val_interval, val_interval), loss_val, c='b', label='val')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def use_gpu():
  cuda_dev = '0' #GPU device 0 (can be changed if multiple GPUs are available)
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")

  print('Device: ' + str(device))
  if use_cuda:
      print('GPU: ' + str(torch.cuda.get_device_name(int(cuda_dev)))) 

  set_gpu = lambda x=True: torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() and x else torch.FloatTensor)
  set_gpu()
  return device
