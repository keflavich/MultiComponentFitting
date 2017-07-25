import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from cube_io import get_dummy_header
from astropy.utils.console import ProgressBar
from astropy.io import fits
import pandas as pd

try:
    #from .math_utils import planar_tilt, periodic_wiggle,
    from . import math_utils
    from . import make_circles
except SystemError:
    # forgive my non-pythonic blasphemy, but I like to %run my scripts
    import math_utils
    import make_circles
    #from math_utils import planar_tilt, periodic_wiggle, intensity_from_density


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

circles_kwargs = dict(n_samples=1000, noise=0.1,
        i_range=[np.pi/2, 3*np.pi/2], j_range=[-np.pi/2, np.pi/2],
        i_tilt=[1, .2, -.5], j_tilt=[0.5, -.4, .3],
        i_xy0=[.75, 0.], j_xy0=[-.75, 0.])

D, l = make_circles.two_circles(**circles_kwargs)

# plot both, scale size by intensity
ax.scatter(*D[l==0, :].T, s=D[l==0, 3]*1000)
ax.scatter(*D[l==1, :].T, s=D[l==1, 3]*1000)

plt.show()

# write a csv file for clustering analyses
toy = pd.DataFrame(D, columns=["ra", "dec", "vlsr", "peak"])
toy["comp"] = l
toy.to_csv("data/test-circ.dat", sep=" ", index=False)

# write fits files for gaussian paramenters
nbins = 150
_, int_map_1 = math_utils.intensity_from_density(
        toy[toy.comp==0].ra, toy[toy.comp==0].dec, nbins=nbins)
_, int_map_2 = math_utils.intensity_from_density(
        toy[toy.comp==1].ra, toy[toy.comp==1].dec, nbins=nbins)

ra_bins = np.linspace(toy.ra.min(), toy.ra.max(), nbins)
dec_bins = np.linspace(toy.dec.min(), toy.dec.max(), nbins)
ra_grid, dec_grid = np.meshgrid(ra_bins, dec_bins)

vlsr_map_1 = math_utils.planar_tilt(ra_grid, dec_grid,
        *circles_kwargs["i_tilt"])
vlsr_map_2 = math_utils.planar_tilt(ra_grid, dec_grid,
        *circles_kwargs["j_tilt"])

sig_map_1 = np.zeros_like(vlsr_map_1) + 0.2
sig_map_2 = np.zeros_like(vlsr_map_1) + 0.2

parcube = np.stack([int_map_1, vlsr_map_1, sig_map_1,
                    int_map_2, vlsr_map_2, sig_map_2])

# get dummy header
keylist = get_dummy_header()

# write out some values used to generate the cube:
test_header = fits.Header()
test_header.update(keylist)
test_hdu = fits.PrimaryHDU(data=parcube, header=test_header)
test_hdu.writeto("data/gauss_pars_x2.fits", overwrite=True, checksum=True)

# wait wait what's the spectral range for the cube?
xarr = np.linspace(toy.vlsr.min() - 0.2 * 5, toy.vlsr.max() + 0.2 * 5, 180)

f, axarr = plt.subplots(2, 3)

for i, (pararr, ax, parname) in enumerate(zip(parcube, axarr.ravel(),
         ['Amplitude', 'Velocity', 'Velocity dispersion']*2)):
    ax.imshow(pararr, interpolation='none')
    ax.set_title(parname + ' #{}'.format(i // 3 + 1))
    ax.axis('off')

plt.savefig('figs/summary-x2.png', dpi=130)

# generate a toy spectral cube
yy, xx = np.indices(parcube.shape[1:])
modelcube = np.empty(shape=(xarr.size, ) + parcube.shape[1:])

# okay this is getting too hacky, but just I want to get a toy cube fast...
# TODO refactor into pyspeckit spectral models / proper xarr...
gauss = lambda x, a, xoff, sig: a*np.exp(-(x - xoff)**2 / sig**2 / 2)
def gauss_x2(x, pars):
    p1, p2 = pars[:3], pars[3:]
    return gauss(x, *p1) + gauss(x, *p2)

def model_a_pixel(xy):
    x, y = int(xy[0]), int(xy[1])
    modelcube[:, y, x] = gauss_x2(xarr, pars=parcube[:, y, x])

for x, y in ProgressBar(list(np.ndindex(parcube.shape[1:]))):
    model_a_pixel([x, y])

# add noise to taste
snr = 42
noise_std = modelcube.max() / snr
white_noise = np.random.normal(scale=noise_std, size=modelcube.shape)
modelcube += white_noise

cube_header = fits.Header()
cube_header.update(get_dummy_header(update_header_dict={
    "CTYPE3": "VRAD",
    "CDELT3": np.diff(xarr)[0],
    "CUNIT3": "km/s",
    "CRVAL3": toy.vlsr.median(),
    "CRPIX3": xarr.size // 2,
    "RSTFRQ": 2.3e9,
    "RMSLVL": noise_std}))
cube_hdu = fits.PrimaryHDU(data=modelcube, header=cube_header)
cube_hdu.writeto("data/gauss_cube_x2.fits", overwrite=True, checksum=True)
