
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage 
from matplotlib import gridspec
from skimage import io, color, measure
from skimage.transform import radon, rescale, resize
from skimage.morphology import skeletonize
from scipy.signal import medfilt



class DeformationMap():
    """ A class for importing Davis displacement data with 
    methods for returning deformation maps
    
    Requires: path and filename
    Returns: a deformation map object
    
    Usage:
    
    deformation_map=deformation_map('path','filename')
    
    """
    
    def __init__(self,path,fname) :
        
        self.path=path
        self.fname=fname
        self.data=np.loadtxt(self.path+self.fname,skiprows=1)
        self.xc=self.data[:,0] #x coordinates
        self.yc=self.data[:,1] #y coordinates
        self.xd=self.data[:,2] #x displacement
        self.yd=self.data[:,3] #y displacement
        
        self.xdim=(self.xc.max()-self.xc.min()
                  )/min(abs((np.diff(self.xc))))+1 #size of map along x
        self.ydim=(self.yc.max()-self.yc.min()
                  )/max(abs((np.diff(self.yc))))+1 #size of map along y
        self.x_map=self._map(self.xd,self.ydim,self.xdim) #u (displacement component along x) 
        self.y_map=self._map(self.yd,self.ydim,self.xdim) #v (displacement component along x) 
        self.f11=self._grad(self.x_map)[1]#f11
        self.f22=self._grad(self.y_map)[0]#f22
        self.f12=self._grad(self.x_map)[0]#f12
        self.f21=self._grad(self.y_map)[1]#f21
        
        self.max_shear=np.sqrt((((self.f11-self.f22)/2.)**2)
                               + ((self.f12+self.f21)/2.)**2)# max shear component
        self.mapshape=np.shape(self.max_shear)
        
    def _map(self,data_col,ydim,xdim):
        data_map=np.reshape(np.array(data_col),(int(ydim),int(xdim)))
        return data_map
    
    def _grad(self,data_map) :
        grad_step=min(abs((np.diff(self.xc))))
        data_grad=np.gradient(data_map,grad_step,grad_step)
        return data_grad

def strain_map_step(xdmap,ydmap,step):
    """
    Data coarsening
    Strain can be calculated over any range:
    """
    d_x=np.gradient(xdmap[::step,::step],8*step,8*step)
    d_y=np.gradient(ydmap[::step,::step],8*step,8*step) 
    return d_x,d_y


def plot_profile(def_map,start_point,end_point):
    """ Nice script to plot a profile (hacked from Stack Exchange):
    """
    
    y0, x0  = start_point[0], start_point[1] # These are in _pixel_ coordinates!!
    y1, x1 = end_point[0], end_point[1]
    profile_length=np.sqrt((y1-y0)**2+(x1-x0)**2)
    num = np.round(profile_length)
    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)

    # Extract the values along the line, using cubic interpolation
    zi = scipy.ndimage.map_coordinates(np.transpose(def_map), np.vstack((x,y)))
    plt.figure(figsize=(10,3))
    gs = gridspec.GridSpec(1, 2, height_ratios=[1]) 
    ax0=plt.subplot(gs[0])
    ax1=plt.subplot(gs[1])
    ax0.imshow(def_map,vmin=-0.00,vmax=0.05,interpolation='bilinear',cmap='viridis');
    ax0.plot([x0, x1], [y0, y1], 'rx-',lw=2);
    ax0.axis('image');
    ax1.plot(zi)
    ax1.axis('tight')




def def_hist(def_map,values_range=(),plot=False):
    """ Create histogram from a deformation map """
    if values_range==():
        xs=np.histogram(def_map.flatten(),bins=100,normed=1);
        if plot==True:
            plt.hist(def_map.flatten(),bins=100,normed=1);
    else:
        xs=np.histogram(def_map.flatten(),bins=100,normed=1, range=values_range);
        if plot==True:
            plt.hist(def_map.flatten(),bins=100,normed=1, range=values_range);
    if plot is True:
        plt.xlabel('Deformation component');
        plt.ylabel('Normalized frequency');
    return(xs)



def plot_hist_line(xs):
    """ Plot histogram using points and lines"""
    yvals=xs[0]
    xvals=0.5*(xs[1][1:]+xs[1][:-1])
    plt.plot(xvals,yvals,'-o');
    plt.ylabel('Normalized frequency')
    plt.xlabel('Map Values')

def plot_hist_log(xs):
    """ Plot log(y) vs. x histogram using points and lines. """
    yvals=np.log(xs[0])
    xvals=0.5*(xs[1][1:]+xs[1][:-1])
    plt.plot(xvals,yvals,'-o');
    plt.ylabel('Normalized frequency')
    plt.xlabel('Map Values')

def acorr_map(def_map,c_range=[]):
    """Calculate aoutocorrelation map for any deformation map"""
    acorr=(np.fft.fft2(def_map)*np.conjugate(np.fft.fft2(def_map)))
    ashift=np.fft.fftshift(acorr)
    corr_map=np.log(np.fft.fftshift((np.abs(np.fft.ifft2(ashift)))))
    if c_range==[]:
        plt.imshow(corr_map, interpolation='nearest', cmap='viridis');
    else:
        plt.imshow(corr_map, interpolation='nearest', cmap='viridis',
                   vmin=c_range[0], vmax=c_range[1]);
    return corr_map


def sb_angle(def_map,threshold=None,median_filter=None):
    """Uses Radon transform to calculate alignment of slip lines
    Returns profile of max intensity of the sinogram in degrees. 
    Threshold can be used to filter data.
    """
    if threshold is not None:
        def_map_filt=def_map>threshold
        strain_title='Threshold: {:2.3f}'.format(threshold)
    else:
        def_map_filt=def_map
        strain_title='Deformation map: no threshold'
    
    if median_filter is not None:    
        def_map_filt=medfilt(def_map_filt,median_filter)
    
    sin_map = radon(def_map_filt)
    profile_filt=np.max(sin_map,axis=0)
    
    plt.figure(figsize=(13,5))
    gs = gridspec.GridSpec(1, 3) 
    ax0=plt.subplot(gs[0])
    ax1=plt.subplot(gs[1])
    ax2=plt.subplot(gs[2])
    ax0.imshow(def_map_filt,cmap='viridis')
    
    ax0.set_title(strain_title)
    ax1.imshow(sin_map,cmap='viridis')
    ax1.set_title('Sinogram')
    ax2.plot(profile_filt)
    ax2.set_title('Band angle distribution')
    ax2.set_xlabel(r'Angle in degrees')
    ax2.set_ylabel(r'Intensity')
    return profile_filt
    

def get_grain_labels(def_map, mask_file):
    """Create grain labels from an image mask file"""
    mask_im=io.imread(mask_file)
    mask_im=color.rgb2gray(mask_im)
    gb_resized=resize(mask_im,np.shape(def_map))
    gb_binary=(gb_resized<1)
    grain_labels=measure.label(gb_binary)
    return grain_labels

def show_grain_labels(grain_labels):
    grain_boundaries=np.zeros(np.shape(grain_labels))
    grain_boundaries[np.where(grain_labels==0)]=1
    plt.imshow(grain_boundaries,cmap='binary')
    measurements=measure.regionprops(grain_labels)
    for grain in measurements:
        plt.text(grain.centroid[1],grain.centroid[0],
        '{:}'.format(grain.label))

def get_grain(def_map,glabels,glabel):
    """Use grain labels to mask individual grains"""
    grain=np.ma.array(def_map,mask=glabels!=glabel)
    return grain

