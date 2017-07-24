import numpy as np
import sys
import os
import numpy as np
from astropy.io import fits
from astropy import wcs
import warnings
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import colors
warnings.simplefilter('ignore', wcs.FITSFixedWarning)
import matplotlib.patches as patches

#==============================================================================#
# USER INPUT

# Input values for core SCOUSE stages
datadirectory    =  '../../scousepy/examples/'
# The data cube to be analysed
filename         =  'CMZ_3mm_HNCO_60'
# Fits extension
fitsfile         =  datadirectory+filename+'.fits'
# The range in velocity, x, and y over which to fit
ppv_vol          =  [0.0,0.0,0.0,0.0,0.0,0.0]
# Radius for the spectral averaging areas. Map units.
rsaa             =  [2.0,5.0,8.0]
# Enter an approximate rms value for the data.
rms_approx       =  0.05
# Threshold below which all channel values set to 0.0
sigma_cut        =  3.0

#==============================================================================#

def read_cube(fitsfile):
    """
    Read in the data cube
    """

    hdu = fits.open(fitsfile)
    header = hdu[0].header
    data = hdu[0].data
    hdu.close()
    data = np.squeeze(data)

    return data, header

def get_velocity(header):
    """
    Generate & return the velocity axis from the fits header.
    """
    mywcs = wcs.WCS(header)
    specwcs = mywcs.sub([wcs.WCSSUB_SPECTRAL])
    return specwcs.wcs_pix2world(np.arange(header['NAXIS{0}'.format(mywcs.wcs.spec+1)]), 0)

def mom_0(data, header, x, y, v, rms_approx, sigma_cut):
    """
    Generates zeroth moment
    """

    channel_spacing = header['CDELT3']/1000.0
    momzero = np.zeros((len(x), len(y)))
    keep = (data >= rms_approx*sigma_cut)

    for i in range(len(x)):
        for j in range(len(y)):
            momzero[i,j] = np.sum(channel_spacing*data[i,j,(keep[i,j,:]==True)])

    return momzero

def define_coverage(x, y, momzero, rsaa):

    rows, cols = np.where(momzero != 0.0)

    rangex = [np.min(rows), np.max(rows)]
    sizex = np.abs(np.min(rows)-np.max(rows))
    rangey = [np.min(cols), np.max(cols)]
    sizey = np.abs(np.min(cols)-np.max(cols))
    spacing = rsaa/2.

    nposx = int((sizex/rsaa)+1.0)
    nposy = int((sizey/rsaa)+1.0)

    cov_x = np.max(rangex)-rsaa*np.array(range(nposx))
    cov_y = np.min(rangey)+rsaa*np.array(range(nposy))

    nareas = 0.0
    coverage = []
    for i in range(len(cov_x)):
        for j in range(len(cov_y)):
            idx = np.squeeze(np.where( ( x >= cov_x[i]-spacing ) & ( x <= cov_x[i]+spacing) ))
            idy = np.squeeze(np.where( ( y >= cov_y[j]-spacing ) & ( y <= cov_y[j]+spacing) ))
            if (np.size(idx) != 0) & (np.size(idy) !=0):
                indx = np.squeeze( np.where( momzero[min(idx):max(idx), min(idy): max(idy)] != 0.0 ) )
                if np.size(indx) != 0.0:
                    tot_non_zero = float(np.size(indx))
                else:
                    tot_non_zero = 0.0

                fraction = tot_non_zero / (float(np.size(idx))*float(np.size(idy)))
                if fraction >= 0.5:

                    coverage.append([cov_x[i], cov_y[j]])

    coverage = np.array(coverage)

    return coverage

#####

data, header = read_cube(fitsfile)
data[np.isnan(data)] =0.0
data = np.transpose(data)

shape = np.shape(data)
x = range(shape[0])
y = range(shape[1])
v = get_velocity(header)
v = v[0]/1000.0

coverage_coordinates = {}
momzero = mom_0(data, header, x, y, v, rms_approx, sigma_cut)

for i in range(len(rsaa)):
    coverage_coordinates[i] = define_coverage(x, y, momzero, rsaa[i])

fig   = plt.figure(figsize=( 15.0, 5.0))
ax = fig.add_subplot(111)
ax.set_xlim([300,550])
plt.imshow(np.transpose(momzero), cmap='Greys')
cols = ['black','red','blue']
size = [0.5,1,2]
alpha = [1,0.8,0.5]

for i in range(len(rsaa)):
    covcoords = coverage_coordinates[i]
    for j in range(len(covcoords[:,0])):
        ax.add_patch(patches.Rectangle((covcoords[j,0], covcoords[j,1]), rsaa[i]*2.,rsaa[i]*2. ,facecolor='none', edgecolor=cols[i] ,lw=size[i], alpha=alpha[i]) )       # height

    #ax.scatter(covcoords[:,0], covcoords[:,1], marker='o', s=size[i], color=cols[i])

plt.show()
