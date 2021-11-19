import os
import numpy as np
import pandas as pd

#Import torch stuff.
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import IPython.display as ipd
import matplotlib.pyplot as plt

from tqdm import tqdm
import wakeword_functions as wf
import time

from sklearn.metrics import accuracy_score

class WakewordDataset(Dataset):
    '''
    Construct a dataset with sound files.
    '''
    
    def __init__(self, dataframe, f, sr = 22000, normalize = True, transforms=None):
        #f is the function that takes audio and returns the spectrogram.
        self.paths = dataframe['outpath'].tolist()
        self.targets    = dataframe['class'].tolist()
        self.IDs        = dataframe['ID'].tolist()
        self.transforms = transforms
        self.f          = f
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path            = self.paths[idx]
        audio, sr, x    = wf.load_data(path, f = self.f, transforms=self.transforms)
        target          = self.targets[idx]
        ID              = self.IDs[idx] 
        
        return audio, sr, x, target, path, ID
    

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.sequential = nn.Sequential(
                              nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(num_features=16),
                              nn.ReLU(),
  
                              nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(num_features=16),
                              nn.ReLU(),
  
                              nn.MaxPool2d(kernel_size=2),
  
                              nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(num_features=32),
                              nn.ReLU(),
  
                              nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(num_features=32),
                              nn.ReLU(),
  
                              nn.MaxPool2d(kernel_size=2),
  
                              nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(num_features=64),
                              nn.ReLU(),
  
                              nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(num_features=64),
                              nn.ReLU(),
  
                              nn.MaxPool2d(kernel_size=2),
  
                              nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(num_features=128),
                              nn.ReLU(),
  
                              nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(num_features=128),
                              nn.ReLU(),
                            
                              nn.MaxPool2d(kernel_size=2),
            
                              nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=3, stride=1),
                              nn.BatchNorm2d(num_features=256),
                              nn.ReLU(),
            
                              nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=3, stride=1),
                              nn.BatchNorm2d(num_features=256),
                              nn.ReLU(),
            
                              nn.MaxPool2d(kernel_size=2),
            
                              nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=3, stride=1),
                              nn.BatchNorm2d(num_features=512),
                              nn.ReLU(),
            
                              #nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=3, stride=1),
                              #nn.BatchNorm2d(num_features=512),
                              #nn.ReLU(),
            
                              #Fully connected part
                              nn.Conv2d(in_channels=512, out_channels=512*2*2, kernel_size=(2,2)),
                              nn.ReLU(),
  
                              nn.Conv2d(in_channels=512*2*2, out_channels=512*2*2, kernel_size=1),
                              nn.ReLU(),
  
                              #Output - no softmax!
                              nn.Conv2d(in_channels=512*2*2, out_channels=2, kernel_size=1),
                            )    
        
    def forward(self, x):
        x = self.sequential(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        return x
    
def main():
    
    #Let's set the device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train, val, test = wf.get_splits()
    
    cnn = CNN()
    cnn.load_state_dict(torch.load('model.pth',  map_location=device))
    cnn.to(device)
    
    for ID in tqdm(test['ID'].unique()):
        
        if os.path.exists('temp.wav'):
            os.remove('temp.wav')
            
        wf.download(ID, 'temp.wav')
        
        audio, sr, x = wf.load_data('temp.wav', f = T.Spectrogram(), transforms = wf.TransformMono())
        
        #Only evaluate half of it.
        
        x = x[:, :, :int(np.round(x.shape[-1]/8))]
        
        x = x.unsqueeze(0).to(device)
        
        outputs = cnn(x)
        
        p = torch.softmax(outputs, dim=1).squeeze(0).detach().cpu().numpy()
        
        with open(f'{ID}.npy', 'wb') as f:
            np.save(f, p)
        
    
    return

if __name__ == '__main__':
    main()

