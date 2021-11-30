from ..wakeword import info_from_path, load_data
import os
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd

class WakewordDataset(Dataset):
    '''
    Construct a dataset with sound files.
    '''
    def __init__(self, f, folder, sr = 22050, normalize = True, transforms=None):
        #n can be used for debugging if we only want to load the first n files.

        self.paths  = [os.path.join(folder, x) for x in os.listdir(folder)]

        folderinfo  = [info_from_path(x) for x in self.paths] #Already here, it's shuffled.
        self.target = [x[3] for x in folderinfo]

        self.transforms = transforms
        self.f          = f
        self.normalize  = normalize

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path            = self.paths[idx]
        _, _, x         = load_data(path, f = self.f, transforms=self.transforms, normalize=self.normalize)
        target          = self.target[idx]
        return x, target, path


class WakewordDatasetRNN(Dataset):
    '''
    Construct a dataset with sound files.
    '''
    def __init__(self, f, folder, target_length, sr = 22050, transforms=None, normalize=True):
        #n can be used for debugging if we only want to load the first n files.

        self.paths  = [os.path.join(folder, x) for x in os.listdir(folder)]
        folderinfo  = [info_from_path(x) for x in self.paths] #Already here, it's shuffled.
        
        #Get the unique IDs.
        IDs = np.unique([x[0] for x in folderinfo])

        #Read in the timestamps.
        data = pd.read_csv('/zhome/55/f/127565/Desktop/01005WakeWord/csv/data.csv')
        timestamps = {}
        for ID, subset in data.groupby('ID'):
            timestamps[ID] = subset['t'].to_numpy()

        #Preallocate the target.
        targets = torch.zeros(size=(len(self.paths), target_length))

        #Set the utterance duration and the target duration.
        utterance_duration = 1
        target_duration    = 0.36
        
        for target, path in zip(targets, self.paths):
            #Fetch ID, t1, t2 and t_linspace. 
            ID, t1, t2, _ = info_from_path(path)
            t_linspace    = torch.linspace(t1, t2, len(target))

            for t_ut in timestamps[ID]:
                #A bit explanation here. The first boolean below means that we're only looking to the -right- of timestamps.
                #The second boolean makes sure that we shift the timestamp by half of the wakeword utterance duration and then we set all targets to 1 that are closer than 0.36s away
                #from this timestamp.
                target[(t_linspace >= (t_ut + utterance_duration/2.))&(np.abs(t_linspace - (t_ut + utterance_duration/2.)) <= target_duration)] = 1

        self.transforms = transforms
        self.f          = f
        self.normalize  = normalize
        self.target     = targets
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path            = self.paths[idx]
        _, _, x         = load_data(path, f = self.f, transforms=self.transforms, normalize=self.normalize)
        target          = self.target[idx,:]
        return x, target, path