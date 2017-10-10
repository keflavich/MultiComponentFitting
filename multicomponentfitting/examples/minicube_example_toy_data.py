import numpy as np
from spectral_cube import SpectralCube
from astropy.io import fits
from multicomponentfitting.toy_datasets.model_utils import make_model_cube
from multicomponentfitting.spatially_aware_fitting.minicube_fit import \
    (constrained_fitter, fit_plotter, multicomp_minicube_model_generator)

npix = 5
pixx, pixy = 83,72
pixx, pixy = 40,76

cube = SpectralCube.read('gauss_cube_x2.fits')
minicube = cube[:,pixy-npix//2:pixy+npix//2+1,pixx-npix//2:pixx+npix//2+1]

parcube = fits.getdata('gauss_pars_x2.fits')
modelcube = make_model_cube(minicube.spectral_axis.value, parcube[:,pixy-npix//2:pixy+npix//2+1,pixx-npix//2:pixx+npix//2+1])

# with a terrible guess, it works terribly...
guess = {'amp0': 0.1, 'ampdx0': 0, 'ampdy0': 0,
         'center0': 0, 'centerdx0': 0, 'centerdy0': 0,
         'sigma0': 1.5, 'sigmadx0': 0, 'sigmady0': 0,}

result = constrained_fitter(minicube.filled_data[:].value, minicube.spectral_axis.value, guess, npix=npix)

print("LSQ Parameters:")
for par in result.params:
        print("{0}: {1}+/-{2}".format(par, result.params[par].value,
                                      result.params[par].stderr))


fit_plotter(result, minicube.filled_data[:].value,
            minicube.spectral_axis.value, modelcube=modelcube, npix=npix)

# multicomp
guess2 = guess.copy()
guess2.update({k.replace('0','1'):v for k,v in guess.items()})
guess2['center1'] += 1 # make sure they start as different comps
guess2['center0'] -= 1 # make sure they start as different comps
result = constrained_fitter(minicube.filled_data[:].value,
                            minicube.spectral_axis.value, guess2, npix=npix,
                            par_minima={'amp0':0.01, 'amp1':0.01,
                                        'center0':0.7, 'center1':-1.16,},
                            par_maxima={'center0':2.5, 'center1':0.7,
                                        'sigma0':0.5, 'sigma1':0.5,
                                       },
                            ncomps=2)

print("LSQ Parameters:")
for par in result.params:
        print("{0}: {1}+/-{2}".format(par, result.params[par].value,
                                      result.params[par].stderr))

fit_plotter(result=result,
            data=minicube.filled_data[:].value,
            xaxis=minicube.spectral_axis.value,
            modelfunc=multicomp_minicube_model_generator(npix=npix, ncomps=2),
            modelcube=modelcube, npix=npix)
