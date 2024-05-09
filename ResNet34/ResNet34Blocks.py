import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super().__init__()
        self.__conv1 = nn.Sequential(
                        nn.Conv2d(
                            in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1
                            ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.__conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, 
                                  out_channels, 
                                  kernel_size = 3, 
                                  stride = 1, 
                                  padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.__downsample = downsample
        self.__relu = nn.ReLU()
        
        
    def forward(self, x):
        residual = x
        out = self.__conv1(x)
        out = self.__conv2(out)
        if self.__downsample is not None:
            residual = self.__downsample(x)
        out += residual
        out = self.__relu(out)
        return out
