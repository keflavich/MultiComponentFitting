from spectral_cube import SpectralCube
import imp
import minicube_fit
imp.reload(minicube_fit)
from minicube_fit import unconstrained_fitter, fit_plotter

cube = SpectralCube.read('gauss_cube_x2.fits')
minicube = cube[:,72:75,44:47]

# with a terrible guess, it works terribly...
guess = {'amp': 0.1, 'ampdx': 0, 'ampdy': 0,
         'center': 0, 'centerdx': 0, 'centerdy': 0,
         'sigma': 2.5, 'sigmadx': 0, 'sigmady': 0,}

result = unconstrained_fitter(minicube.filled_data[:].value, minicube.spectral_axis.value, guess, npix=3)

print("LSQ Parameters:")
for par in result.params:
        print("{0}: {1}+/-{2}".format(par, result.params[par].value,
                                      result.params[par].stderr))

fit_plotter(result, minicube.filled_data[:].value, minicube.spectral_axis.value, npix=3)
