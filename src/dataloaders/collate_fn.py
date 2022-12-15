import torch
import random
import numpy as np

class BatchBalanceCollate(object):

    def __call__(self, batch):
        ecg = torch.stack([d['ecg'] for d in batch])
        label = torch.stack([d['label'] for d in batch])
        if len(label.size())>1:
          label = label.squeeze()
        bs = len(label)
        num_classes = 5
        num_major = int(bs * 1/num_classes)
        instance_weight = torch.zeros(bs)
        indices = [None]*num_classes
        for i in range(num_classes):
          indices[i] = torch.arange(bs)[label==i]

        major_indices = random.sample(indices[0].tolist(), num_major)

        instance_weight[major_indices] = 1
        instance_weight[indices[1]] = 1.2
        instance_weight[indices[2]] = 1.2
        instance_weight[indices[3]] = 2
        instance_weight[indices[4]] = 4
        
        batch = {'label':label, 'ecg':ecg}
        return batch, instance_weight
        

