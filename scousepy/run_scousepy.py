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
from spectral_cube import SpectralCube
from astropy import units as u
from stage_1 import *

#==============================================================================#
# USER INPUT

# Input values for core SCOUSE stages
datadirectory    =  '../../scousepy/examples/'
# The data cube to be analysed
filename         =  'CMZ_3mm_HNCO_60'
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

cube = SpectralCube.read(fitsfile).with_spectral_unit(u.km/u.s)
x = np.arange(cube.shape[2])
y = np.arange(cube.shape[1])

momzero = cube.with_mask(cube > u.Quantity(rms_approx*sigma_cut, cube.unit)).moment0(axis=0).value

coverage_coordinates = {}
saa_spectra = {}

for i in range(len(rsaa)):
    coverage_coordinates[i], saa_spectra[i] = define_coverage(cube, momzero, rsaa[i])


#fig = plt.figure(1, figsize=(15.0, 2.0))
#fig.clf()
#ax = fig.add_subplot(111)
#plt.imshow(momzero, cmap='Greys', origin='lower', interpolation='nearest', vmax=100)
#cols = ['black','red','blue']
#size = [0.5,1,2]
#alpha = [1,0.8,0.5]
#for i in range(len(rsaa)):
#    covcoords = coverage_coordinates[i]
#    for j in range(len(covcoords[:,0])):
#        if np.all(np.isfinite(covcoords[j,:])) ==True:
#            ax.add_patch(patches.Rectangle((covcoords[j,0]-rsaa[i], covcoords[j,1]-rsaa[i]),
#                                            rsaa[i]*2., rsaa[i]*2.,facecolor='none',
#                                            edgecolor=cols[i],lw=size[i],
#                                            alpha=alpha[i]))
#plt.draw()
#plt.show()
