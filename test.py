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
    
    train, val, test = wf.read_metadata('./export')

    #Create the datasets.
    train_dataset = WakewordDataset(train, 
                                    T.Spectrogram(), 
                                    normalize=True, #normalize the audio when reading it with torchaudio.
                                    transforms = [wf.AudioAugment(reverb = 100, snr = 15, pitch = 150, p = [0.5, 0.5, 0.5]),
                                                  wf.TransformMono(), 
                                                  wf.Padder(44000)]
                                   )

    val_dataset   = WakewordDataset(val, T.Spectrogram(), normalize=True, transforms = [wf.TransformMono(), wf.Padder(44000)])
    test_dataset  = WakewordDataset(test, T.Spectrogram(), normalize=True, transforms = [wf.TransformMono(), wf.Padder(44000)])

    #Create the loaders.
    batch_size = 128
    train_loader  = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    val_loader    = DataLoader(val_dataset,   shuffle=True, batch_size=batch_size, num_workers=4)
    test_loader   = DataLoader(test_dataset,  shuffle=False, batch_size=batch_size, num_workers=4)
    
    cnn = CNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters())

    num_epochs = 30
    for epoch in tqdm(range(num_epochs), unit='epoch', desc="Epoch"):

        #Epochs start @ 1 now.
        epoch += 1

        #For each epoch
        train_correct = 0

        #Set it up for training.
        cnn.train()
        train_loss = 0
        predictions = []
        targets = []
        
        for data in train_loader:
            
            # get the inputs; data is a list of [inputs, labels]
            _, _, inputs, labels, paths, _ = data

            #Get that stuff on the GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn(inputs.float())
            #loss = criterion(outputs, labels)
            loss = criterion(outputs, labels.reshape(-1,).long())
            loss.backward()
            optimizer.step()

            #Add the batch loss
            train_loss += loss.item()

            #Save predictions and targets
            predictions += outputs.argmax(axis=1).tolist()
            targets     += labels.long().tolist()

        train_acc = accuracy_score(targets, predictions)

        
        #Set it up for evaluation on validation set.
        cnn.eval()  
        predictions = []
        targets = []
        val_loss = 0

        for data in val_loader:

            # get the inputs; data is a list of [inputs, labels]
            _, _, inputs, labels, paths, _ = data

            #Get that stuff on the GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn(inputs.float())

            #Calculate the loss
            loss = criterion(outputs, labels.long())
            val_loss += loss.item()

            #Save predictions and targets
            predictions += outputs.argmax(axis=1).tolist()
            targets     += labels.long().tolist()

        val_acc = accuracy_score(targets, predictions)
        
        #Normalize by the length of the datasets.
        train_loss = train_loss / len(train_dataset)
        val_loss   = val_loss / len(val_dataset)
        
        print('[%d]: Train: [%.4f | %.3f] Validation: [%.4f | %.3f]' % (epoch, train_loss, train_acc, val_loss, val_acc))
    
    #Evaluate on the test dataset.
    cnn.eval()  
    predictions = []
    targets = []
    paths = []
    test_loss = 0
    
    
    for data in test_loader:

        # get the inputs; data is a list of [inputs, labels]
        _, _, inputs, labels, path, _ = data

        #Get that stuff on the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cnn(inputs.float())

        #Calculate the loss
        loss = criterion(outputs, labels.long())
        val_loss += loss.item()

        #Save predictions and targets
        predictions += outputs.argmax(axis=1).tolist()
        targets     += labels.long().tolist()
        paths       += path

    test_acc = accuracy_score(targets, predictions)
    
    print(f'Final test accuracy: {test_acc}')
    
    torch.save(cnn.state_dict(), 'model.pth')
    
    #Saving the results.
    results = pd.DataFrame({'path' : paths, 'target' : targets, 'prediction' : predictions})
    results.to_csv('results.csv', index=False)
    
    return

if __name__ == '__main__':
    main()

