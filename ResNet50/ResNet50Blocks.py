import torch 
import torch.nn as nn


class ResNetBlock(nn.Module):

    def __init__(self,
                 in_channels, 
                 out_channels, 
                 stride,
                 downsample = None):
        super().__init__()

        self.__conv1 = nn.Conv2d(
            in_channels = in_channels, 
            
        )
    
    def forward(self, x):
        return x 