import numpy as np
import torch

class Normalize(object):
  def __call__(self, sample):
    label, wavname, mfcc = sample['label'], sample['wavname'], sample['mfcc']
    mean = np.mean(mfcc[0])
    std = np.std(mfcc[0])
    if std>0:
      mfcc[0] = (mfcc[0]-mean)/std
    
    mean = np.mean(mfcc[1])
    std = np.std(mfcc[1])
    if std>0:
      mfcc[1] = (mfcc[1]-mean)/std
    
    mean = np.mean(mfcc[2])
    std = np.std(mfcc[2])
    if std>0:
      mfcc[2] = (mfcc[2]-mean)/std

    #padding = np.zeros(mfcc.shape[0])
    #mfcc = np.c_[mfcc, padding]

    sample = {'label':label, 'mfcc':mfcc, 'wavname':wavname}
    return sample

class ToTensor(object):
  def __call__(self, sample):
    label, ecg = sample['label'], sample['ecg']
    ecg = torch.from_numpy(ecg)
    label = torch.from_numpy(np.asarray(label))
    sample = {'label':label, 'ecg':ecg}
    return sample

class ImageNormalize(object):
  def __call__(self, sample):
    label, wavname, mfcc = sample['label'], sample['wavname'], sample['mfcc']
    mfcc = 0+(255-0)/(np.max(mfcc)-np.min(mfcc)) * (mfcc-np.min(mfcc))
    mfcc = np.repeat(mfcc[:, :, np.newaxis], 3, axis=2)
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    mfcc = transform(mfcc)
    sample = {'label':label, 'mfcc':mfcc, 'wavname':wavname}
    return sample