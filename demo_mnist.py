import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from deformation import ADef
from vector_fields import draw_vector_field

from models_mnist import mnist_a, mnist_b

# ADef config
candidates = range(10)
max_iter = 100
max_norm = np.inf
sigma = 0.5
overshoot = 1.2
strong_targets = False
verbose = True

path_to_resources = 'resources/'
path_to_model = path_to_resources + 'mnist_a_e5b50_clean_model.pt'

net = mnist_a()
if os.path.exists( path_to_model ):
    net.load_state_dict( torch.load(path_to_model, map_location=lambda storage, loc: storage) )
else:
    print('Model not found. Run \'train_mnist.py\' first!\nDeforming images w.r.t. an untrained model.')
    verbose = False
net.eval()
print('Model: ' + str(type(net)))

batch_size = 4
mnist_test = datasets.MNIST( path_to_resources, train=False, download=True, transform=transforms.ToTensor() )
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, pin_memory=False)

# Get a random batch of images
batch, labels = next(iter( test_loader ))

x = Variable( batch )
Fx = net.forward(x)
maxval, pred_labels = torch.max( Fx.data, 1 )

if (pred_labels != labels).any():
    print('Misclassified digit(s).' )


# Deform image using ADef
def_batch, def_data = ADef( batch, net, ind_candidates=candidates,
                            max_norm=max_norm, max_iter=max_iter,
                            smooth=sigma, overshoot=overshoot,
                            targeting=strong_targets, verbose=verbose )


def_labels = def_data['deformed_labels']
vector_fields = def_data['vector_fields']

fig, axs = plt.subplots( 2, batch_size )

for im_no in range(batch_size):
    im = batch[ im_no, 0 ].numpy()
    def_im = def_batch[ im_no, 0 ].numpy()
    axs[ 0, im_no ].imshow( im, cmap='Greys', vmin=0, vmax=1 )
    draw_vector_field( axs[ 0, im_no ], vector_fields[ im_no ], amp=3 )
    if not pred_labels[im_no] == labels[im_no]:
        axs[ 0, im_no ].set_title( 'Misclf. as %d' % pred_labels[im_no], color='red' )
    else:
        axs[ 0, im_no ].set_title( '%d' % pred_labels[im_no] )
    axs[ 1, im_no ].imshow( def_im, cmap='Greys', vmin=0, vmax=1 )
    axs[ 1, im_no ].set_title( '%d' % def_labels[im_no] )

axs[0,0].set_ylabel('Original')
axs[1,0].set_ylabel('Deformed')

plt.show()

