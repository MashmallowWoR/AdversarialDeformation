import time
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F



def spatial_grad( func ):
    """
    Approximate derivatives of the functions func[b,c,:,:].
    
    dfdx, dfdy = spatial_grad( func )

    In:
    func: torch.FloatTensor
        of shape BxCxhxw with B >= 1 (batch size), C = 1 or C = 3 (color channels),
        h,w >= 3, and [type] is 'Float' or 'Double'.
        Contains the values of functions f_b: R^2 -> R^C, b=1,...,B,
        on the grid {0,...,h-1}x{0,...,w-1}.

    Out:
    dfdx: torch.FloatTensor
    dfdy: torch.FloatTensor
        of shape BxCxhxw contain the x and y derivatives of f_1, ..., f_B
        at the points on the grid, approximated by central differences (except on boundaries):
        For b=0,...,B-1, c=0,...,C, i=1,...,h-2, j=1,...,w-2
        dfdx[b,c,i,j] = (func[b,c,i,j+1] - func[b,c,i,j-1])/2
        dfdx[b,c,i,j] = (func[b,c,i+1,j] - func[b,c,i-1,j])/2

    positive x-direction is along rows from left to right.
    positive y-direction is along columns from above to below.
    """

    # Derivative in x direction (rows from left to right)
    dfdx = torch.zeros_like( func )
    # forward difference in first column
    dfdx[:,:,:,0] = func[:,:,:,1] - func[:,:,:,0]
    # backwards difference in last column
    dfdx[:,:,:,-1] = func[:,:,:,-1] - func[:,:,:,-2]
    # central difference elsewhere
    dfdx[:,:,:,1:-1] = 0.5*(func[:,:,:,2:] - func[:,:,:,:-2] )

    # Derivative in y direction (columns from above to below)
    dfdy = torch.zeros_like( func )
    # forward difference in first row
    dfdy[:,:,0,:] = func[:,:,1,:] - func[:,:,0,:]
    # backwards difference in last row
    dfdy[:,:,-1,:] = func[:,:,-1,:] - func[:,:,-2,:]
    # central difference elsewhere
    dfdy[:,:,1:-1,:] = 0.5*(func[:,:,2:,:] - func[:,:,:-2,:] )

    return dfdx.detach(), dfdy.detach()


def compose( func, flow ):
    """
    Calculate the composition of the function func with the vector
    field flow by interpolation.
    
    new_func = compose( func, flow )

    In:
    func: torch.FloatTensor
        of shape B*C*H*W
        func[b] contains the values of a function f_b:R^2 -> R^C
        on the grid {0,...,H-1}x{0,...,W-1}.
    flow: torch.FloatTensor
        of shape B*H*W*2
        flow[b] contains the values of a vector field u_b:R^2 -> R^2
        on the grid {0,...,H-1}x{0,...,W-1}.
    
    positive x-direction is along rows from left to right
    positive y-direction is along columns from above to below
    
    flow[b,y,x,0] = x-coordinate of the vector flow[b,y,x]
    flow[b,y,x,1] = y-coordinate of the vector flow[b,y,x]
    
    Out:
    new_func: torch.FloatTensor
        of shape B*C*H*W
        new_func[b] contains the values of the function f_b(id + u_b)
        on the grid {0,...,H-1}x{0,...,W-1}.
    """
    
    B,C,H,W = func.shape
    device = func.device
    
    hrange = torch.range( 0, H-1, device=device )
    wrange = torch.range( 0, W-1, device=device )
    gridx = wrange.repeat( H, 1 ).unsqueeze_(2)
    gridy = hrange.view( -1, 1).repeat( 1, W ).unsqueeze_(2)
    integer_grid = torch.cat([gridx, gridy], 2 )
    
    # Grid normalized to [-1,1]^2
    scale = 2 / torch.tensor([ W-1., H-1. ], device=device )
    grid = scale * integer_grid - 1
    
    # Deformed grid for b = 1,2,...,B
    grid = grid.repeat(B,1,1,1) + scale * flow
    
    # Return bilinear interpolation of func on new grid.
    return F.grid_sample( func, grid, padding_mode='border' ).detach()


def gaussian_filter( sigma=1, channels=1, device='cpu' ):
    '''
    A 2D Gaussian smoothing operator sending torch.FloatTensor of shape
    B*channels*H*W to torch.FloatTensor of shape B*channels*H*W for
    any B,H,W >= 1.
    
    In:
    sigma: float > 0
        standard deviation of the Gaussian kernel.
    channels: int >= 1
        number of input channels
    '''
    # Use n_sd sigmas
    n_sd = 4
    # odd size so padding results in correct output size
    size = 2*int( n_sd*sigma + 0.5 )+1
    pad = (size-1)//2
    mean = (size - 1.)/2.
    
    tt = (torch.range(0,size-1,device=device)-mean)/sigma
    gauss = torch.exp( -0.5*tt**2 ).view( size, 1 )
    gauss = gauss/gauss.sum()

    kernel = torch.mm( gauss, gauss.transpose(0,1) )
        
    filt = nn.Conv2d( channels, channels, groups=channels, kernel_size=size, bias=False, padding=pad )
    
    kernel = kernel.view( 1, 1, size, size )
    kernel = kernel.repeat( channels, 1, 1, 1 )
    
    filt.weight.data = kernel
    filt.weight.requires_grad = False
    
    return filt

def create_tau( fval, gradf, d1x, d2x, smoothing_operator=None ):
    """
    tau = create_tau( fval, gradf, d1x, d2x )

    In:
    fval: torch.FloatTensor
        of shape B
    gradf: torch.FloatTensor
        of shape B*C*H*W
    d1x: torch.FloatTensor
        of shape B*C*H*W
    d2x: torch.FloatTensor
        of shape B*C*H*W
    smoothing_operator: function
        A self-adjoint smoothing operator sending torch.FloatTensor
        of shape B*2*H*W to torch.FloatTensor of shape B*2*H*W.

    Out:
    tau: torch.FloatTensor
        of shape B*H*W*2
    """
    
    B,C,H,W = gradf.shape
    
    # Sum over color channels
    alpha1 = torch.sum( gradf*d1x, 1).unsqueeze_(1)
    alpha2 = torch.sum( gradf*d2x, 1).unsqueeze_(1)
    
    # stack vector field components into shape B*2*H*W
    tau = torch.cat([alpha1,alpha2], 1)
    
    # Smoothing
    if smoothing_operator:
        tau = smoothing_operator( tau )
        # torch can't sum over multiple axes.
        norm_squared_alpha = (tau**2).sum(1).sum(1).sum(1)
        # In theory, we need to apply the filter a second time.
        tau = smoothing_operator( tau )
    else:
        # torch can't sum over multiple axes.
        norm_squared_alpha = (tau**2).sum(1).sum(1).sum(1)
        
    scale = -fval/norm_squared_alpha
    tau *= scale.view(B,1,1,1)

    # rearrange for compatibility with compose(), B*2*H*W -> B*H*W*2
    return tau.permute( 0, 2, 3, 1 ).detach()

def Tnorm( vector_fields ):
    return (vector_fields**2).sum(-1).view(vector_fields.shape[0],-1).max(1)[0].sqrt()

def ADef( batch, model, ind_candidates = 1, max_iter = 50, max_norm = 'inf', overshoot = 1.0, smooth = 0., targeting = False, verbose = True ):
    '''
    Find an adversarial deformation of each image in batch w.r.t model.

    deformed_batch, out_data = ADef( batch, model, ... )
    
    In:
    batch: torch.FloatTensor
        of shape B*C*H*W (batch) or C*H*W (image).
    model: torch.nn.Module
        The classifier w.r.t. which we search for adversarial deformations.
        model takes as input a torch.Tensor of shape B*C*H*W,
        and returns a torch.Tensor of shape B*L where L is the total number of labels.
    ind_candidates: int or array_like of int
        The indices of labels to target in the ordering of descending confidence.
        For example:
        - ind_candidates = [1,2,3] to target the top three labels.
        - ind_candidates = 5 to to target the fifth best label.
        For l=0,...,len(ind_candidates) it should hold that
        0 < ind_candidates[l] <= L where L is the total number of labels.
    max_iter: int > 0
        Maximum number of iterations (default max_iter = 50).
    max_norm: float or 'inf'
        Maximum T-norm of vector fields (default max_norm = 'inf').
        T-norm of a vector field tau:R^2->R^2 is defined by
            || tau ||_T := max{ ||tau(p)||_2 : p in R^2 }
    overshoot: float >= 1
        Multiply the resulting vector field by this number,
        if deformed image is still correctly classified
        (default is overshoot = 1 for no overshooting).
    smooth: float >= 0
        Width of the Gaussian kernel used for smoothing.
        (default is smooth = 0 for no smoothing).
    targeting: bool
        targeting = False (default) to stop as soon as model misclassifies input
        targeting = True to stop only once a candidate label is achieved.
    verbose: bool
        verbose = True (default) to print progress,
        verbose = False for silence.
    '''
    batch_device = batch.device
    # Use instead of 'print'.
    vprint = print if verbose else lambda *a, **k: None    
    max_norm = float( max_norm )
    
    if len(batch.shape) < 4:
        # If input is not a batch, make it a batch of size 1.
        batch = torch.unsqueeze( batch, 0 )
    
    B, C, H, W = batch.shape # batch size, colors, height, width
    ind_images = list(range( B )) # keep track of images that are still to be deformed
    try_overshoot = [True]*B # keep track of unsuccessful deformations
    
    # Include the correct label (index 0) in the list of targets.
    # Remove duplicates and sort the label indices.
    ind_candidates = torch.tensor(ind_candidates).view(-1)
    ind_candidates = torch.cat([ ind_candidates, torch.tensor([0]) ])
    ind_candidates = torch.unique( ind_candidates, sorted=True ) # unique is currently CPU-only, and lacks CUDA support
    n_candidates = ind_candidates.nelement()
    
    smoother = gaussian_filter( sigma=smooth, channels=2, device=batch_device ) if smooth else None
    
    # Images after n iterations.
    batch_n = batch.clone()
    batch_n.requires_grad = True
    ones = torch.ones( B, device=batch_device ) # used for differentiation
    F_n = model( batch_n )
    
    
    # Indices of the 'n_candidates' highest values in descending order:
    candidates = F_n.sort(dim=1, descending=True)[1][:,ind_candidates]
    original_labels = candidates[:,0]
    current_labels = original_labels.clone()
    
    vprint('Deforming %d images.' % B )
    vprint('Labels:\tImage\tOrig.\tCandidates')
    for im_no in ind_images:
        vprint('\t' + str(im_no) + '\t' + str(original_labels[im_no].item()) + '\t' + str( candidates[ im_no, 1: ] ) )
    
    # f_n[b,l] is negative if the model prefers the original label
    # for batch[b] over the label l. 
    f_n = F_n - F_n[ range(B), original_labels ].view( B, -1 )
    
    iterations = torch.zeros( B, device=batch_device ) + max_iter
    tau_full = torch.zeros( B, H, W, 2, device=batch_device )
    norm_full = torch.zeros( B, device=batch_device )

    vprint('Iterations finished: 0')
    vprint('Images left: %d' % len(ind_images) )
    vprint('\tCurrent labels: ' + str(current_labels) )
    vprint('\tf(x0) = ')
    for im_no in ind_images:
        vprint('\t' + str(im_no) + '\t' + str( f_n[ im_no, candidates[im_no] ] ) )
    vprint('\tnorm(tau) = ' + str( norm_full ))
    
    n = 0 # iteration number
    time0 = time.time()
    while len(ind_images) > 0 and n < max_iter:
        n += 1
        
        # Differentiate batch:
        d1x, d2x = spatial_grad( batch_n[ind_images] )
        
        # Differentiate model:
        # gradient=ones to get derivative of Fx w.r.t each image in batch.
        # retain_graph allows repeated use of backward.
        F_n[ ind_images, current_labels[ind_images] ].backward( gradient=ones[ind_images], retain_graph=True )
        DF_current = batch_n.grad.clone()
        batch_n.grad.zero_()
        
        # Find vector fields for each image and each candidate label.
        # Keep the smallest vector field for each image.
        norm_min = torch.zeros_like( norm_full ) + max_norm
        tau_min = torch.zeros_like( tau_full )

        for ind_target in range( 1, n_candidates ):
            targets = candidates[ ind_images, ind_target ]
            F_n[ ind_images, targets ].backward( gradient=ones[ind_images], retain_graph=True )
            DF_target = batch_n.grad.clone()
            batch_n.grad.zero_()
            
            f_target = f_n[ ind_images, targets ]
            # Derivative of the binary classifier 'f_target = F_target - F_current'
            Df_target = DF_target[ ind_images ] - DF_current[ ind_images ]
            
            tau_target = create_tau( f_target, Df_target, d1x, d2x, smoother )
            tau_target += tau_full[ind_images]
            
            norm_target = Tnorm( tau_target )
            ind_update = (norm_target < norm_min[ind_images])
            norm_min[ torch.tensor(ind_images)[ind_update] ] = norm_target[ ind_update ]
            tau_min[ torch.tensor(ind_images)[ind_update] ] = tau_target[ ind_update ]
        
        # Quick proxy for vector field update.
        changes_made = ( norm_min - norm_full ).abs() > 1e-10
        changes_made *= ( norm_min < max_norm )
        new_ind_images = ind_images.copy()
        for im_no in ind_images:
            if not changes_made[im_no]:
                vprint('No changes made to image %d.' % im_no)
                new_ind_images.remove( im_no )
                iterations[im_no] = n-1
        ind_images = new_ind_images.copy()
        
        tau_full[ ind_images ] = tau_min[ ind_images ]
        norm_full[ ind_images ] = norm_min[ ind_images ]
        batch_n = compose( batch, tau_full )
        batch_n.requires_grad = True
        
        F_n = model( batch_n )
        current_labels = F_n.max(1)[1]
        f_n = F_n - F_n[ range(B), current_labels ].view( B, -1 )

        vprint('Iterations finished: %d' % n)
        vprint('Images left: %d' % len(ind_images) )
        vprint('\tCurrent labels: ' + str(current_labels[ind_images]) )
        vprint('\tf(x0) = ')
        for im_no in ind_images:
            vprint('\t' + str(im_no) + '\t' + str( f_n[ im_no, candidates[im_no] ] ) )
        vprint('\tnorm(tau) = ' + str( norm_full[ind_images] ))
    
        # See if we have been successful.
        boundary = (f_n > -1e-6).sum(1)
        for im_no in ind_images:
            im_label = current_labels[im_no]
            successful = targeting and (im_label in candidates[im_no,1:])
            successful = successful or ((not targeting) and im_label != original_labels[im_no])
            if successful:
                vprint('Image %d successfully deformed from %d to %d.' % (im_no, original_labels[im_no].item(), current_labels[im_no].item() ))
                try_overshoot[ im_no ] = (boundary[im_no].item()>1) # overshoot if the image lies on a decision boundary
                new_ind_images.remove( im_no )
                iterations[im_no] = n
        ind_images = new_ind_images


    
    time1 = time.time()
    
    vprint('\nFinished!')
    vprint('\tTime: %.3fs' % (time1 - time0) )
    vprint('\tTime: %.3fs per iteration' % ((time1 - time0)/n) )
    vprint('\tTime: %.3fs per image-iteration' % ((time1 - time0)/iterations.sum().item()) )
    vprint('\tAvg. #iterations: %.3f' % iterations.mean().item())
    vprint('\tOriginal labels: ' + str(original_labels ) )
    vprint('\tCurrent labels: ' + str(current_labels) )
    vprint('\tf(x0) = ')
    for im_no in range(B):
        vprint('\t' + str(im_no) + '\t' + str( f_n[ im_no, candidates[im_no] ] ) )
    vprint('\tnorm(tau) = ' + str( norm_full ))


    if overshoot > 1:
        vprint('\nOvershooting...')
        vprint('\t... on images ' + str([ im_no for im_no in range(B) if try_overshoot[im_no] ]))
        # Overshoot unsuccessful deformations, but do not exceed max_norm.
        os = (max_norm/norm_full).clamp( 1, overshoot ).view(B,1,1,1)
        try_overshoot = torch.tensor(try_overshoot)
        tau_full[ try_overshoot ] *= os[ try_overshoot ]
        norm_full = Tnorm( tau_full )
        batch_n = compose( batch, tau_full )
        F_n = model( batch_n )
        current_labels = F_n.max(1)[1]
        f_n = F_n - F_n[ range(B), current_labels ].view( B, -1 )
        
        vprint('\tCurrent labels: ' + str(current_labels) )
        vprint('\tf(x0) = ')
        for im_no in range(B):
            vprint('\t' + str(im_no) + '\t' + str( f_n[ im_no, candidates[im_no] ] ) )
        vprint('\tnorm(tau) = ' + str( norm_full ))
    

    data = {}
    data['vector_fields'] = tau_full
    data['iterations'] = iterations
    data['norms'] = norm_full
    data['original_labels'] = original_labels
    data['deformed_labels'] = current_labels
    data['overshot'] = try_overshoot

    return batch_n, data


def example_batch( b, c, h, w ):
    sigm = 0.05
    batch = sigm*torch.randn(b,c,h,w) + 0.5
    batch[:2,:,:,:] = 0.5
    batch[0,0,:,w//2:] += 0.5
    batch[0,1,h//2:,:] += 0.5
    batch[0,2,:,:w//2:4] += 0.5
    wh = min(w,h)
    batch[1,:,:wh,:wh] = torch.transpose( batch[0].clone(), 1, 2 )[:,:wh,:wh]
    
    if b > 3:
        for pp in range(wh):
            batch[2,0,pp,:] = np.sin(pp/6)#batch[2,0,pp,0]
            batch[2,2,pp,:] = np.sin(pp/8)#batch[2,0,pp,0]
            
            batch[2,0,pp,pp] = 1
            batch[2,1,pp,pp] = .5
            
            qq1 = int( wh*np.sin(pp/16)/3 ) + wh//2
            batch[3,0,qq1:(qq1+5),pp] = 1
            qq2 = int( wh*np.cos(pp/24)/3 ) + wh//2
            batch[3,2,qq2:(qq2+3),pp] = 1
            
            batch[3,1,pp,w-(pp+1)] += 0.4
            batch[3,2,pp,w-(pp+1)] -= 0.1
            
    
    for im_no in range(4,b):
        batch[im_no] = (batch[im_no]+0.2)**(im_no-1)
    
    batch = torch.clamp(batch, 0, 1)
    return batch


if __name__=='__main__':
    import matplotlib.pyplot as plt
    from vector_fields import draw_vector_field
    import torchvision.transforms as transforms
    import torchvision.models as models
    
    b, c, h, w = ( 6, 3, 224, 224 )
    batch = example_batch( b, c, h, w )
    
    model = models.alexnet(pretrained=True)
    model.eval() # Turn off dropout etc.
    
    ind_candidates = [5,7]
    max_iter = 50
    max_norm = 'inf'
    overshoot = 1.2
    smooth = 0.5
    targeting = False
    verbose = True
    

    Flabels, labels = torch.max(model( batch ),1)
    labels = labels.data.numpy()
    fig, axs = plt.subplots( 3, b )
    for im_no in range(b):
        image = batch[ im_no ]
        image_pil = transforms.ToPILImage()( image )
        axs[ 0, im_no ].set_title(labels[im_no])
        axs[ 0, im_no ].imshow( image_pil )

    defbatch, data = ADef( batch, model, targeting=targeting, ind_candidates=ind_candidates, smooth=smooth, max_iter=max_iter, max_norm=max_norm, overshoot=overshoot, verbose=verbose )

    deflabels = data['deformed_labels'].numpy()
    for im_no in range(b):
        defimage = defbatch[ im_no ]
        defimage_pil = transforms.ToPILImage()( defimage )
        axs[ 1, im_no ].set_title( deflabels[im_no] )
        axs[ 1, im_no ].imshow( defimage_pil )

    tau = data['vector_fields'].numpy()
    norms = data['norms'].numpy()

    for im_no in range(b):
        defimage = defbatch[ im_no ]
        defimage_pil = transforms.ToPILImage()( defimage/4+0.75 )

        axs[ 2, im_no ].imshow( defimage_pil )
        draw_vector_field( axs[2,im_no], tau[im_no], amp=4 )

    plt.show()
