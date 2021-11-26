from ..wakeword import info_from_path, load_data
import os
from torch.utils.data import Dataset, DataLoader

class WakewordDataset(Dataset):
    '''
    Construct a dataset with sound files.
    '''
    def __init__(self, f, folder, sr = 22050, normalize = True, transforms=None, n = None):
        #n can be used for debugging if we only want to load the first n files.

        self.paths  = [os.path.join(folder, x) for x in os.listdir(folder)]

        folderinfo  = [info_from_path(x) for x in self.paths] #Already here, it's shuffled.
        self.target = [x[3] for x in folderinfo]

        self.transforms = transforms
        self.f          = f
        self.normalize  = normalize

        if n is not None:
            #Only load the first n files.
            self.paths  = self.paths[:n]
            self.target = self.target[:n]
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path            = self.paths[idx]
        _, _, x         = load_data(path, f = self.f, transforms=self.transforms, normalize=self.normalize)
        target          = self.target[idx]
        return x, target, path

