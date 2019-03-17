import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from deformation import ADef
from vector_fields import draw_vector_field
from models_cifar10 import LeNet, ResNet, BasicBlock

# ADef config
candidates = range(10)
max_iter = 100
max_norm = np.inf
sigma = 0.5 #how much it deforms
overshoot = 1.2
strong_targets = False
verbose = True

path_to_resources = 'resources/'
#path_to_model = path_to_resources + 'lenet_e100b50_clean_model.pt'
path_to_model = path_to_resources + 'resnet_e100b50_clean_model.pt'
#net = LeNet()
net = ResNet(block=BasicBlock, num_blocks=[2,2,2,2])

if os.path.exists( path_to_model ):
    net.load_state_dict( torch.load(path_to_model, map_location=lambda storage, loc: storage) )
else:
    print('Model not found. Run \'train_cifar10.py\' first!\nDeforming images w.r.t. an untrained model.')
    verbose = False
net.eval()
print('Model: ' + str(type(net)))

batch_size = 100
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
test = datasets.CIFAR10( path_to_resources, train=False, download=False, 
                        transform=transforms.Compose([transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, pin_memory=True)

# Get a random batch of images
batch, labels = next(iter( test_loader ))
print('actual_labels=', labels, labels.shape, batch.shape)

x = Variable( batch )
Fx = net.forward(x)
maxval, pred_labels = torch.max( Fx.data, 1 )

if (pred_labels != labels).any():
    print('Misclassified image(s).' )


# Deform image using ADef
def_batch, def_data = ADef( batch, net, ind_candidates=candidates,
                            max_norm=max_norm, max_iter=max_iter,
                            smooth=sigma, overshoot=overshoot,
                            targeting=strong_targets, verbose=verbose )


def_labels = def_data['deformed_labels']
vector_fields = def_data['vector_fields']
print('def_labels=', def_labels, def_labels.shape, def_batch.shape)

sample_size=4
fig, axs = plt.subplots( 2, sample_size )
for im_no in range(sample_size):
    #im = batch[ im_no, 0 ].numpy()
    im_r = batch[ im_no, 0 ].numpy()
    im_g = batch[ im_no, 1 ].numpy()
    im_b = batch[ im_no, 2 ].numpy()
    im = np.dstack((im_r, im_g, im_b))
    #def_im = def_batch[ im_no, 0 ].numpy()
    def_im_r = def_batch[ im_no, 0 ].numpy()
    def_im_g = def_batch[ im_no, 1 ].numpy()
    def_im_b = def_batch[ im_no, 2 ].numpy()
    def_im = np.dstack((def_im_r, def_im_g, def_im_b))
    
    axs[ 0, im_no ].imshow( im, vmin=0, vmax=1 )
    draw_vector_field( axs[ 0, im_no ], vector_fields[ im_no ], amp=3)
    if not pred_labels[im_no] == labels[im_no]:
        axs[ 0, im_no ].set_title( 'Misclf. as %d' % pred_labels[im_no], color='red' )
    else:
        axs[ 0, im_no ].set_title( '%d' % pred_labels[im_no] )
    axs[ 1, im_no ].imshow( def_im, vmin=0, vmax=1 )
    axs[ 1, im_no ].set_title( '%d' % def_labels[im_no] )

axs[0,0].set_ylabel('Original')
axs[1,0].set_ylabel('Deformed')
plt.show()

num_success_attack= len(labels)- torch.eq(labels, def_labels).sum().item()
success_rate = num_success_attack / len(labels)
print('attack_success_rate=', success_rate)