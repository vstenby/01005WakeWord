import torch
import torch.nn as nn

class CNN_V2(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
                              nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(num_features=16),
                              nn.ReLU(),
                              nn.Dropout2d(p=0.7),

                              nn.MaxPool2d(kernel_size=2),

                              nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(num_features=32),
                              nn.ReLU(),
                              nn.Dropout2d(p=0.7),
                            
                              nn.MaxPool2d(kernel_size=2),

                              nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(num_features=64),
                              nn.ReLU(),
                              nn.Dropout2d(p=0.7),

                              nn.MaxPool2d(kernel_size=2),

                              nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm2d(num_features=128),
                              nn.ReLU(),
                              nn.Dropout2d(p=0.7),

                              nn.MaxPool2d(kernel_size=2),

                              nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=3, stride=1),
                              nn.BatchNorm2d(num_features=256),
                              nn.ReLU(),
                              nn.Dropout2d(p=0.7),

                              nn.MaxPool2d(kernel_size=2),

                              nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=3, stride=1),
                              nn.BatchNorm2d(num_features=512),
                              nn.ReLU(),
                            )

        self.fc1 = nn.Conv2d(in_channels = 512, out_channels = (512*3*14), kernel_size=(3, 14))
        self.fc2 = nn.Conv2d(in_channels = (512*3*14), out_channels = 1, kernel_size=(1,1))

        #self.fc2 = nn.Linear(in_features = 1024,  out_features = 2)

        self.ReLU = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        x = self.ReLU(x)

        #No activation function after the last layer.
        x = self.fc2(x)
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)
        return x

    def __repr__(self):
        return f'CNN.V1: Convolutional neural network that takes inputs of size () ... '