import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class mnist_a( nn.Module ):
    '''
    MNIST-A, based on Madry et al. (2018)
    '''
    name = 'mnist_a'
    
    def __init__( self ):
        super( mnist_a, self ).__init__()
        self.conv1 = nn.Conv2d( 1, 32, (5,5), padding=2 )
        self.conv2 = nn.Conv2d( 32, 64, (5,5), padding=2 )
        self.fc1 = nn.Linear( 7*7*64, 1024 )
        self.fc2 = nn.Linear( 1024, 10 )
        self._init_weights()
    
    def forward( self, x ):
        x = F.max_pool2d( F.relu( self.conv1(x) ), (2,2) )
        x = F.max_pool2d( F.relu( self.conv2(x) ), (2,2) )
        x = x.view( x.size(0), -1 )
        x = F.relu( self.fc1(x) )
        x = self.fc2(x)
        return x
    
    def _init_weights( self ):
        self.conv1.weight.data.normal_(std=0.1)
        self.conv2.weight.data.normal_(std=0.1)
        self.fc1.weight.data.normal_(std=0.1)
        self.fc2.weight.data.normal_(std=0.1)
        self.conv1.bias.data.fill_(0.1)
        self.conv2.bias.data.fill_(0.1)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)

class mnist_b( nn.Module ):
    '''
    MNIST-B, based on Tramer et al. (2018)
    '''
    name = 'mnist_b'
    
    def __init__( self ):
        super( mnist_b, self ).__init__()
        self.conv1 = nn.Conv2d( 1, 128, (3,3), padding=1 )
        self.conv2 = nn.Conv2d( 128, 64, (3,3), padding=1 )
        self.conv2_drop = nn.Dropout2d( p=0.25 )
        self.fc1 = nn.Linear( 28*28*64, 128 )
        self.fc1_drop = nn.Dropout( p=0.5 )
        self.fc2 = nn.Linear( 128, 10 )
        self._init_weights()
    
    def forward( self, x ):
        x = F.relu( self.conv1(x) )
        x = F.relu( self.conv2(x) )
        x = self.conv2_drop(x)
        x = x.view( x.size(0), -1 )
        x = F.relu( self.fc1(x) )
        x = self.fc1_drop(x)
        x = self.fc2(x)
        return x
    
    def _init_weights( self ):
        self.conv1.weight.data.normal_(std=0.1)
        self.conv2.weight.data.normal_(std=0.1)
        self.fc1.weight.data.normal_(std=0.1)
        self.fc2.weight.data.normal_(std=0.1)
        self.conv1.bias.data.fill_(0.1)
        self.conv2.bias.data.fill_(0.1)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
