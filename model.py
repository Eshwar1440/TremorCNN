import torch
import torch.nn as nn

class TremorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential( 
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 17 x 6

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) # 8 x 3              
            )
        
        self.fc = nn.Sequential(
            nn.Linear(704, 64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), - 1) # batch-size, 1-D
        x = self.fc(x)
        return x