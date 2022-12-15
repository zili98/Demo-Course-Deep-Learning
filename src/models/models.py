import torch
import torch.nn as nn
import torch.nn.functional as F

# define resnet building blocks
class ResidualBlock(nn.Module): 
    def __init__(self, inchannel, outchannel, stride=1): 
        
        super(ResidualBlock, self).__init__() 
        
        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3, 
                              stride=stride, padding=1, bias=False), 
                      nn.BatchNorm2d(outchannel), 
                      nn.ReLU(inplace=True), 
                      nn.Conv2d(outchannel, outchannel, kernel_size=3, 
                              stride=1, padding=1, bias=False), 
                      nn.BatchNorm2d(outchannel)) 
        
        self.shortcut = nn.Sequential() 
        
        if stride != 1 or inchannel != outchannel: 
            
            self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel, 
                                    kernel_size=1, stride=stride, 
                                    padding = 0, bias=False), 
                            nn.BatchNorm2d(outchannel) ) 
            
    def forward(self, x): 
        
        out = self.left(x) 
        
        out += self.shortcut(x) 
        
        out = F.relu(out) 
        
        return out


    
    # define resnet

class ResNet(nn.Module):
    
    def __init__(self, ResidualBlock, num_classes = 2):
        
        super(ResNet, self).__init__()
        
        self.inchannel = 64
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size = 3, stride = 1,
                                padding = 1, bias = False), 
                      nn.BatchNorm2d(64), 
                      nn.ReLU(),
                      nn.MaxPool2d(2),)
        
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride = 1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride = 2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride = 2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride = 2)
        self.maxpool = nn.MaxPool2d(kernel_size=(3,2), stride=2, padding=0)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=512*9, out_features=100, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(100),
        )
        self.fc2 = nn.Linear(in_features=100, out_features=num_classes, bias=True)
        
    
    def make_layer(self, block, channels, num_blocks, stride):
        
        strides = [stride] + [1] * (num_blocks - 1)
        
        layers = []
        
        for stride in strides:
            
            layers.append(block(self.inchannel, channels, stride))
            
            self.inchannel = channels
            
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)       
        x = self.maxpool(x)
        f1 = x.view(x.size(0), -1)
        f2 = self.fc1(f1)  
        x = self.fc2(f2)      
        return x, f2
   
def ResNet18():
    return ResNet(ResidualBlock)


class ResidualBlock1D(nn.Module): 
    def __init__(self, inchannel, outchannel, stride=1): 
        
        super(ResidualBlock1D, self).__init__() 
        
        self.left = nn.Sequential(nn.Conv1d(inchannel, outchannel, kernel_size=3, 
                              stride=stride, padding=1, bias=False), 
                      nn.BatchNorm1d(outchannel), 
                      nn.LeakyReLU(inplace=True), 
                      nn.Conv1d(outchannel, outchannel, kernel_size=3, 
                              stride=1, padding=1, bias=False), 
                      nn.BatchNorm1d(outchannel)) 
        
        self.shortcut = nn.Sequential() 
        
        if stride != 1 or inchannel != outchannel: 
            
            self.shortcut = nn.Sequential(nn.Conv1d(inchannel, outchannel, 
                                    kernel_size=1, stride=stride, 
                                    padding = 0, bias=False), 
                            nn.BatchNorm1d(outchannel) ) 
            
    def forward(self, x): 
        
        out = self.left(x) 
        
        out += self.shortcut(x) 
        
        out = F.relu(out) 
        
        return out

class ResNet1D(nn.Module):
    
    def __init__(self, ResidualBlock1D, num_classes = 5):
        
        super(ResNet1D, self).__init__()
        
        self.inchannel = 64
        self.conv1 = nn.Sequential(nn.Conv1d(12, 64, kernel_size = 3, stride = 1,
                                padding = 1, bias = False), 
                      nn.BatchNorm1d(64), 
                      nn.LeakyReLU(),
                      nn.MaxPool1d(2),
                    )
        
        self.layer1 = self.make_layer(ResidualBlock1D, 64, 2, stride = 1)
        self.layer2 = self.make_layer(ResidualBlock1D, 128, 2, stride = 2)
        self.layer3 = self.make_layer(ResidualBlock1D, 256, 2, stride = 2)
        self.layer4 = self.make_layer(ResidualBlock1D, 512, 2, stride = 2)
        self.maxpool = nn.MaxPool1d(kernel_size=6)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=512*10, out_features=1000, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(1000),
        )
        self.fc2 = nn.Linear(in_features=1000, out_features=num_classes, bias=True)
        
    
    def make_layer(self, block, channels, num_blocks, stride):
        
        strides = [stride] + [1] * (num_blocks - 1)
        
        layers = []
        
        for stride in strides:
            
            layers.append(block(self.inchannel, channels, stride))
            
            self.inchannel = channels
            
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)       
        x = self.maxpool(x)
        f1 = x.view(x.size(0), -1)
        f2 = self.fc1(f1)  
        x = self.fc2(f2)      
        return x
def ResNet18_1D(num_classes=2):
    return ResNet1D(ResidualBlock1D, num_classes)