from ..wakeword import info_from_path, load_data
import os
from torch.utils.data import Dataset, DataLoader

class WakewordDataset(Dataset):
    '''
    Construct a dataset with sound files.
    '''
    
    def __init__(self, f, folder, sr = 22050, normalize = True, transforms=None):
        
        self.paths  = [os.path.join(folder, x) for x in os.listdir(folder)]

        folderinfo  = [info_from_path(x) for x in self.paths] #Already here, it's shuffled.
        self.ID, self.t1, self.t2, self.target = [x[0] for x in folderinfo], [x[1] for x in folderinfo], [x[2] for x in folderinfo], [x[3] for x in folderinfo]

        self.transforms = transforms
        self.f          = f
        self.normalize  = normalize
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path            = self.paths[idx]
        audio, sr, x    = load_data(path, f = self.f, transforms=self.transforms, normalize=self.normalize)
        target          = self.target[idx]
        ID              = self.ID[idx] 
        t1         = self.t1[idx]
        t2         = self.t2[idx]
        
        return audio, sr, x, target, path, ID, t1, t2

