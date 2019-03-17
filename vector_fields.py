
import numpy as np
import matplotlib.pyplot as plt



def draw_vector_field( ax, vec_field, skip=1, amp=1, tol=0, xmin=0, ymin=0 ):
    """
    Draw vec_field on ax.
    
    Example:
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1,1)
    >>> draw_vector_field( ax, vec_field, skip, amp, tol, xmin, ymin )
    
    with
    
    vec_field: numpy.array of shape (h,w,2)
    skip: int >= 1
    amp: float > 0
    tol: float >= 0
    xmin: int >= 0 and < vec_field.shape[1]
    ymin: int >=0 and < vec_field.shape[0]
    """
    
    h,w,d = vec_field.shape # height, width, dimension=2
    
    taux = np.zeros((h,w))
    tauy = np.zeros((h,w))
    
    taux[::skip,::skip] = vec_field[::skip,::skip,0]
    tauy[::skip,::skip] = vec_field[::skip,::skip,1]
    
    color_intensities = np.sqrt(taux**2 + tauy**2)
    
    hrange = np.arange(h) + ymin
    wrange = np.arange(w) + xmin
    MGx, MGy = np.meshgrid( wrange, hrange )
    
    locations = (color_intensities > tol)
    print('Drawing %d arrows' %locations.sum())
    
    MGx = MGx[ locations ]
    MGy = MGy[ locations ]
    taux = taux[ locations ]
    tauy = tauy[ locations ]
    color_intensities = color_intensities[ locations ]
    
    scale = 1/amp
    ax.quiver( MGx, MGy, taux, tauy, color_intensities, angles='xy', scale_units='xy', scale=scale, minlength=tol )
    
    
    ax.set_xlim(( xmin, xmin + w-1 ))
    ax.set_ylim(( ymin + h-1, ymin )) # Flip the y axis, otherwise the image is upside down
    ax.set_aspect('equal')
    



if __name__ == '__main__':
    from scipy.ndimage.filters import gaussian_filter
    h = 60
    w = 50
    vec_field = 5*np.random.randn(h, w, 2)
    vec_field[:,:,0] = gaussian_filter( vec_field[:,:,0], 2 )
    vec_field[:,:,1] = gaussian_filter( vec_field[:,:,1], 2 )
    
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)
    draw_vector_field( ax1, vec_field )
    ax1.set_title('raw')
    draw_vector_field( ax2, vec_field, amp=1.5 )
    ax2.set_title('amp=1.5')
    draw_vector_field( ax3, vec_field, tol=0.5 )
    ax3.set_title('tol=0.5')
    draw_vector_field( ax4, vec_field, skip=2 )
    ax4.set_title('skip=2')
    draw_vector_field( ax5, vec_field, xmin=10, ymin=10 )
    ax5.set_xlim([0,w-1])
    ax5.set_ylim([h-1,0])
    ax5.set_title('xmin=ymin=10')
    plt.show()















