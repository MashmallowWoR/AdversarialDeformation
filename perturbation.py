import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter
import torch
import torch.nn as nn
from torch.autograd import Variable

import time

def PGD( batch, batch_labels, model, epsilon=0.3, iterations = 40, stepsize=0.01, random_start=False, clip=True ):
    '''
    Important: if clip, then batch should have values between 0 to 1.
    '''
    if len(batch.shape) < 4:
        # If input is not a batch, make it a batch of size 1.
        batch = torch.unsqueeze( batch, 0 )
    # Images after n iterations.
    batch_n = batch.clone()
    
    if random_start:
        batch_n += 2*epsilon*torch.rand_like(batch_n) - epsilon
    
    
    for n in range( iterations ):
        
        x = Variable( batch_n, requires_grad = True )
        
        output = model( x )
        loss = nn.CrossEntropyLoss()( output, Variable(batch_labels) )
        
        
        loss.backward()
        
        batch_n += stepsize * torch.sign( x.grad.data )
        
        batch_n = torch.max( batch_n, batch - epsilon )
        batch_n = torch.min( batch_n, batch + epsilon )
        if clip:
            batch_n.clamp_( 0, 1 )
        
    return batch_n




if __name__=='__main__':
    import torchvision.models as models
    import torchvision.transforms as transforms
    from vector_fields import draw_vector_field

    model = models.alexnet(pretrained=True)
    model.eval() # Turn off dropout

    b, c, h, w = ( 4, 3, 224, 224 )

    batch = torch.zeros( b, c, h, w )
    batch[0,0,:,112:] = 1
    batch[0,1,112:,:] = 1
    batch[0,2,:,:112:4] = 1
    batch[1] = torch.transpose( batch[0].clone(), 1, 2 )
    batch[2] = 0.5*(batch[0] + batch[1])
    batch[3,:,:,112:] = batch[2,:,:,:112]
    batch[3,:,:,:112] = batch[2,:,:,112:]

    Flabels, labels_tensor = torch.max(model(Variable( batch )),1)
    labels = labels_tensor.data.numpy()
    fig, axs = plt.subplots( 1, b )
    for im_no in range(b):
        image = batch[ im_no ]
        image_pil = transforms.ToPILImage()( image )
        axs[ im_no ].set_title(labels[im_no])
        axs[ im_no ].imshow( image_pil )

    plt.show()
    
    pertbatch = PGD( batch, labels_tensor, model, epsilon=16/255, iterations=10, random_start=True )
    
    Fpertlabels, pertlabels_tensor = torch.max(model(Variable(pertbatch)),1)
    pertlabels = pertlabels_tensor.data.numpy()
    fig, axs = plt.subplots( 1, b )
    for im_no in range(b):
        pertimage = pertbatch[ im_no ]
        pertimage_pil = transforms.ToPILImage()( pertimage )
        axs[ im_no ].set_title( '%d->%d' % (labels[im_no], pertlabels[im_no]) )
        axs[ im_no ].imshow( pertimage_pil )

    plt.show()

