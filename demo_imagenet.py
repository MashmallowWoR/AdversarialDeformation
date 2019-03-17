import numpy as np
import matplotlib.pyplot as plt
import os
import PIL.Image as Image

import torch
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms

from deformation import ADef
from vector_fields import draw_vector_field

path_to_resources = 'resources/'

Words = open(os.path.join(path_to_resources + 'synset_words.txt'), 'r').read().split('\n')

# Image is WIDTH x WIDTH x 3
WIDTH = 299

#net = models.resnet101(pretrained=True) # Use WIDTH = 224
net = models.inception_v3(pretrained=True) # Use WIDTH = 299
net.eval() # Turn off dropout and such
print('Model: ' + type(net).__name__)

# Image with min(width,height) >= WIDTH.
im_name = path_to_resources + 'im0.jpg'
image_PIL = Image.open(im_name)
print('Image: ' + im_name)

# Create tensor compatible with 'net'
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
normalize = transforms.Normalize( mean=mean, std=std )
image = transforms.Compose([
    transforms.Resize(WIDTH),
    transforms.CenterCrop(WIDTH),
    transforms.ToTensor(),
    normalize
    ])(image_PIL)

# 'net' accepts batches of images as Variable.
x = Variable( torch.unsqueeze( image, 0 ) )
Fx = net(x)
maxval, label = torch.max( Fx.data, 1 )
label = label.item()

# ADef config
candidates = 1
max_iter = 100
max_norm = np.inf
sigma = 1
overshoot = 1.1
strong_targets = False

# Deform image using ADef
def_image, def_data = ADef( image, net, ind_candidates=candidates,
                            max_norm=max_norm, max_iter=max_iter,
                            smooth=sigma, overshoot=overshoot,
                            targeting=strong_targets )

def_image = def_image[0]
def_label = def_data['deformed_labels'][0]
vec_field = def_data['vector_fields'][0]


# Get plottable images
unnormalize = transforms.Compose([
    transforms.Normalize( [0,0,0], 1/std ),
    transforms.Normalize( -mean, [1,1,1] )
    ])
def_image_PIL = transforms.Compose([
    unnormalize,
    transforms.ToPILImage()
    ])( def_image )
image_PIL = transforms.Compose([
    unnormalize,
    transforms.ToPILImage()
    ])( image )


# Size of perturbation:
pertl2 = np.linalg.norm( image.numpy().ravel() - def_image.numpy().ravel(), ord=2 )
pertlinf = np.linalg.norm( image.numpy().ravel() - def_image.numpy().ravel(), ord=np.inf )
# Size of vector field:
vecnorms = np.sqrt( vec_field[:,:,0]**2 + vec_field[:,:,1]**2 ).numpy()
vfl2 = np.linalg.norm( vecnorms.ravel(), ord=2 )
vfT = np.linalg.norm( vecnorms.ravel(), ord=np.inf )


# Plot results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
fig.set_size_inches(7,8)

ax1.imshow( image_PIL )
ax2.imshow( def_image_PIL )
bg_image = np.array(def_image_PIL)//2 + 128
ax3.imshow( bg_image )
draw_vector_field( ax3, vec_field, amp=4, tol=0.01 )
pert = 1.*np.array(def_image_PIL) - 1.*np.array(image_PIL)
ax4.imshow( pert )

original_title = Words[label][10:].split(',')[0]
deformed_title = Words[def_label][10:].split(',')[0]
ax1.set_title( 'Original: ' + original_title )
ax2.set_title( 'Deformed: ' + deformed_title )
ax3.set_title( 'Vector field' )
ax4.set_title( 'Perturbation' )

ax3.set_xlabel(r'$\ell^2$-norm: %.3f,  $T$-norm: %.3f' %( vfl2, vfT ) )
ax4.set_xlabel(r'$\ell^2$-norm: %.3f,  $\ell^\infty$-norm: %.3f' %( pertl2, pertlinf ) )

plt.show()
