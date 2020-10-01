import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):

    def __init__(self, in_, out_, k_size, pad, activation):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_, out_, kernel_size=k_size, stride=1, padding=pad)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.activation = activation
        
        assert self.activation in ['sigmoid', 'relu', 'tanh'], "Activation not found"

    def forward(self, x):
        
        if self.activation == 'sigmoid':
            return torch.sigmoid(self.max_pool(self.conv(x)))
        elif self.activation == 'relu':
            return F.relu(self.max_pool(self.conv(x)))
        elif self.activation == 'tanh':
            return torch.tanh(self.max_pool(self.conv(x)))

        
class SoftmaxLayer(nn.Module):

    def __init__(self, class_num):
        super(SoftmaxLayer, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*32, class_num)
        )

    def forward(self, x):
        
        return self.layer(x)


class FeedForwardNet():

    def __init__(self, class_num, activation, dataset):
        self.dims = [
            16*16*3,
            8*8*16,
            4*4*32,
            1
        ]

        self.activation = activation
        self.class_num = class_num
        self.dataset = dataset


    def network(self):
        return [
            DenseLayer(in_=1 if self.dataset == 'mnist' else 3, out_=3, k_size=3, pad=1, activation=self.activation).cuda(),
            DenseLayer(in_=3, out_=16, k_size=3, pad=1, activation=self.activation).cuda(),
            DenseLayer(in_=16, out_=32, k_size=3, pad=1, activation=self.activation).cuda(),
            SoftmaxLayer(self.class_num).cuda()
        ]


class ResNet():
    
    def __init__(self):
        pass
    
    def network(self):
        # Return separated layers in an array
        pass
    
    
class DenseNet():
    
    def __init__(self):
        pass
    
    def network(self):
        # Return separated layers in an array
        pass
    
    