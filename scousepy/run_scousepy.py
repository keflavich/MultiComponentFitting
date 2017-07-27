import numpy as np
import os
from astropy import wcs
import warnings
warnings.simplefilter('ignore', wcs.FITSFixedWarning)
from spectral_cube import SpectralCube
from astropy import units as u
from stage_1 import write_averaged_spectra, define_coverage, plot_rsaa

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

# get the coverage / average the subcube spectra
coverage_coordinates = {}
saa_spectra = {}
for i, r in enumerate(rsaa):
    coverage_coordinates[i], saa_spectra[i] = define_coverage(cube, momzero, r)

# write fits files for all the averaged spectra
write_averaged_spectra(cube.header, saa_spectra, rsaa)

# plot multiple coverage areas
plot_rsaa(coverage_coordinates, momzero, rsaa)

# TODO: stage 2 begins here:
#           - multicube the hell out of the fits files
#           - plot the guesses suggested
#           - maybe add DE?
#           - wrap the guesses back to the full cube...
#           - run pyspeckit on the whole thing

# TODO: merge into the package strucutre @keflavich put together
