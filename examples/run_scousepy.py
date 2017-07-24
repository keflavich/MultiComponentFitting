import numpy as np
import sys
import os
import itertools
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
datadirectory = './'
# The data cube to be analysed
filename         =  'CMZ_3mm_HNCO_60'
filename = 'CMZ_3mm_HNCO_60_SCOUSEcube'
# Fits extension
fitsfile         =  os.path.join(datadirectory, filename+'.fits')
# The range in velocity, x, and y over which to fit
ppv_vol          =  [0.0,0.0,0.0,0.0,0.0,0.0]
# Radius for the spectral averaging areas. Map units.
rsaa             =  [2.0,5.0,8.0]
# Enter an approximate rms value for the data.
rms_approx       =  0.05
# Threshold below which all channel values set to 0.0
sigma_cut        =  3.0

#==============================================================================#
def define_coverage(x, y, momzero, rsaa):

    cols, rows = np.where(momzero != 0.0)

    rangex = [np.min(rows), np.max(rows)]
    sizex = np.abs(np.min(rows)-np.max(rows))
    rangey = [np.min(cols), np.max(cols)]
    sizey = np.abs(np.min(cols)-np.max(cols))
    spacing = rsaa/2.

    nposx = int((sizex/rsaa)+1.0)
    nposy = int((sizey/rsaa)+1.0)

    cov_x = np.max(rangex)-rsaa*np.arange(nposx)
    cov_y = np.min(rangey)+rsaa*np.arange(nposy)

    coverage = []
    for cx,cy in itertools.product(cov_x, cov_y):
        momzero_cutout = momzero[int(cy-spacing):int(cy+spacing),
                                 int(cx-spacing):int(cx+spacing)]
        finite = np.isfinite(momzero_cutout)
        nmask = np.count_nonzero(finite)
        if nmask > 0:
            tot_non_zero = np.count_nonzero(np.isfinite(momzero_cutout) & (momzero_cutout!=0))
            fraction = tot_non_zero / nmask
            if fraction > 0.5:
                coverage.append([cx,cy])

    coverage = np.array(coverage)

    return coverage

#####

from spectral_cube import SpectralCube
from astropy import units as u

#data, header = read_cube(fitsfile)
cube = SpectralCube.read(fitsfile).with_spectral_unit(u.km/u.s)
#data[np.isnan(data)] =0.0
#data = np.transpose(data)

#shape = cube.shape[1:]
#x = range(shape[0])
#y = range(shape[1])
x = np.arange(cube.shape[2])
y = np.arange(cube.shape[1])
#v = get_velocity(header)
#v = v[0]/1000.0
#v = cube.spectral_axis.value

coverage_coordinates = {}
#momzero = mom_0(data, header, x, y, v, rms_approx, sigma_cut)
momzero = cube.with_mask(cube > u.Quantity(rms_approx*sigma_cut, cube.unit)).moment0(axis=0).value

for i in range(len(rsaa)):
    coverage_coordinates[i] = define_coverage(x, y, momzero, rsaa[i])

fig = plt.figure(1, figsize=(15.0, 5.0))
fig.clf()
ax = fig.add_subplot(111)
#ax.set_xlim([300,550])
#plt.imshow(np.transpose(momzero), cmap='Greys')
plt.imshow(momzero, cmap='Greys', origin='lower', interpolation='nearest',
           vmax=100)
cols = ['black','red','blue']
size = [0.5,1,2]
alpha = [1,0.8,0.5]

for i in range(len(rsaa)):
    covcoords = coverage_coordinates[i]
    for j in range(len(covcoords[:,0])):
        ax.add_patch(patches.Rectangle((covcoords[j,0]-rsaa[i], covcoords[j,1]-rsaa[i]),
                                       rsaa[i]*2., rsaa[i]*2.,facecolor='none',
                                       edgecolor=cols[i],lw=size[i],
                                       alpha=alpha[i]))       # height

    #ax.scatter(covcoords[:,0], covcoords[:,1], marker='o', s=size[i], color=cols[i])
plt.draw()
plt.show()
