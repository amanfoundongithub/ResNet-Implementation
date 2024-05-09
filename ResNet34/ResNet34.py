import torch.nn as nn

from ResNet34.ResNet34Blocks import ResidualBlock

class ResNet(nn.Module):

    def __init__(self,
                 layers : list, 
                 num_classes = 10):

        super().__init__()

        self.__in_planes = 64

        self.__conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.__max_pooling = nn.MaxPool2d(
            kernel_size = 3, stride = 2, padding = 1
        )

        self.__layer0 = self.__make_layer(
            planes = 64, no_of_blocks = layers[0], stride = 1 
        )

        self.__layer1 = self.__make_layer(
            planes = 128, no_of_blocks = layers[1], stride = 2
        )

        self.__layer2 = self.__make_layer(
            planes = 256, no_of_blocks = layers[2], stride = 2
        )

        self.__layer3 = self.__make_layer(
            planes = 512, no_of_blocks = layers[3], stride = 2
        )

        self.__avgpool = nn.AvgPool2d(7, stride=1)

        self.__fc = nn.Linear(512, num_classes)

        self.__initialize_weights() 

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    

    def __make_layer(self, 
                     planes, 
                     no_of_blocks, 
                     stride = 1):
        
        # Create a ResNet residual layer 
        
        downsample = None 

        if stride != 1 or self.__in_planes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.__in_planes, 
                    planes,
                    kernel_size = 1, 
                    stride = stride,
                ),
                nn.BatchNorm2d(planes)
            )
        
        layers = []
        layers.append(
            ResidualBlock(
                self.__in_planes, 
                planes, 
                stride, 
                downsample
            )
        )

        self.__in_planes = planes

        for j in range(1, no_of_blocks):
            layers.append(
                ResidualBlock(
                    self.__in_planes, 
                    planes 
                )
            )
        
        return nn.Sequential(
            *layers
        )

    def forward(self, x):
        # Initial convolution
        x = self.__max_pooling(self.__conv1(x))

        # Now pass via the resnet layers
        x = self.__layer0(x)
        x = self.__layer1(x)
        x = self.__layer2(x)
        x = self.__layer3(x) 

        # Average pooling 
        x = self.__avgpool(x) 

        #  Final output via linear network 
        x = x.view(x.size(0), -1)
        x = self.__fc(x)

        return x 




