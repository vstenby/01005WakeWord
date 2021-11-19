import torch
import torch.nn as nn

class RNN_V1(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = (1, 15), stride = (1, 4))
        self.GRU  = nn.GRU(batch_first = True, input_size = 201, hidden_size = 128, num_layers = 2, dropout=0.2)
        
        self.fc1  = nn.Linear(in_features = 128, out_features = 64)
        self.fc2  = nn.Linear(in_features = 64,  out_features = 32)
        self.fc3  = nn.Linear(in_features = 32,  out_features = 1)
        
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2) 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.ReLU(x)
        x = self.dropout(x)

        x = x.squeeze(1)     
        x = x.permute(0,2,1) 

        output, _ = self.GRU(x)

        x = output.squeeze(0)

        x    = self.fc1(x)
        x    = self.ReLU(x)
        x    = self.fc2(x) 
        x    = self.ReLU(x)
        x    = self.fc3(x)
        
        return x