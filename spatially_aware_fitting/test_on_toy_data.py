import numpy as np
from spectral_cube import SpectralCube
import imp
import minicube_fit
imp.reload(minicube_fit)
from minicube_fit import unconstrained_fitter, fit_plotter
from astropy.io import fits
try:
    from ..toy_datasets.model_utils import make_model_cube
except SystemError:
    # cry a little....
    # TODO: remove this when we have a package....
    from astropy.utils.console import ProgressBar
    def make_model_cube(xarr, parcube):

        modelcube = np.empty(shape=(xarr.size, ) + parcube.shape[1:])

        # okay this is getting too hacky, but just I want to get a toy cube fast...
        # TODO refactor into pyspeckit spectral models / proper xarr...
        def gauss(x, a, xoff, sig):
            return a*np.exp(-(x - xoff)**2 / sig**2 / 2)

        def gauss_x2(x, pars):
            p1, p2 = pars[:3], pars[3:]
            return gauss(x, *p1) + gauss(x, *p2)

        def model_a_pixel(xy):
            x, y = int(xy[0]), int(xy[1])
            modelcube[:, y, x] = gauss_x2(xarr, pars=parcube[:, y, x])

        for x, y in ProgressBar(list(np.ndindex(parcube.shape[1:]))):
            model_a_pixel([x, y])

        return modelcube

cube = SpectralCube.read('gauss_cube_x2.fits')
minicube = cube[:,75:78,47:50]


parcube = fits.getdata('gauss_pars_x2.fits')
modelcube = make_model_cube(minicube.spectral_axis.value, parcube[:,75:78,47:50])

# with a terrible guess, it works terribly...
guess = {'amp': 0.1, 'ampdx': 0, 'ampdy': 0,
         'center': 0, 'centerdx': 0, 'centerdy': 0,
         'sigma': 2.5, 'sigmadx': 0, 'sigmady': 0,}

result = unconstrained_fitter(minicube.filled_data[:].value, minicube.spectral_axis.value, guess, npix=3)

print("LSQ Parameters:")
for par in result.params:
        print("{0}: {1}+/-{2}".format(par, result.params[par].value,
                                      result.params[par].stderr))


fit_plotter(result, minicube.filled_data[:].value,
            minicube.spectral_axis.value, modelcube=modelcube, npix=3)
