
import pymc3 as pm
import numpy as np
from minicube_fit import gaussian, plane


def minicube_pymc_fit(xax, data, guesses, **fit_kwargs):
    '''
    pymc fitting of a set of single Gaussians.
    '''

    basic_model = pm.Model()

    with basic_model:

        amp = pm.Normal('amp', mu=guesses['amp'], sd=5)
        ampdx = pm.Normal('ampdx', mu=guesses['ampdx'], sd=1)
        ampdy = pm.Normal('ampdy', mu=guesses['ampdy'], sd=1)

        center = pm.Normal('center', mu=guesses['center'], sd=5)
        centerdx = pm.Normal('centerdx', mu=guesses['centerdx'], sd=1)
        centerdy = pm.Normal('centerdy', mu=guesses['centerdy'], sd=1)

        sigma = pm.Normal('sigma', mu=guesses['sigma'], sd=1)
        sigmadx = pm.Normal('sigmadx', mu=guesses['sigmadx'], sd=1)
        sigmady = pm.Normal('sigmady', mu=guesses['sigmady'], sd=1)

        npix = data.shape[1]

        yy, xx = np.indices([npix, npix], dtype='float')

        amps = plane(xx, yy, amp, ampdx, ampdy, xcen=npix // 2, ycen=npix // 2)
        centers = plane(xx, yy, center, centerdx, centerdy,
                        xcen=npix // 2, ycen=npix // 2)
        sigmas = plane(xx, yy, sigma, sigmadx, sigmady,
                       xcen=npix // 2, ycen=npix // 2)

        model = gaussian(xax[:, None, None], amps, centers, sigmas)

        # mu = minicube_model(xax,
        #                     amp, ampdx, ampdy,
        #                     center, centerdx, centerdy,
        #                     sigma, sigmadx, sigmady,
        #                     npix=data.shape[1])

        sigma_n = pm.InverseGamma('sigma_n', alpha=1, beta=1)
        Y_obs = pm.Normal('Y_obs', mu=model, sd=sigma_n, observed=data)

        trace = pm.sample(**fit_kwargs)

    return trace
