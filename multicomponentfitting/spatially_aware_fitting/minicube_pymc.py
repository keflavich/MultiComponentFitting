
import pymc3 as pm
import theano as tt
import numpy as np
from minicube_fit import minicube_model


def minicube_pymc_fit(xax, data, guesses, **fit_kwargs):
    '''
    pymc fitting of a set of single Gaussians.
    '''

    basic_model = pm.Model()

    with basic_model:

        '''
        Widths of the priors need to be set automatically.
        For now, I've made assumptions with these intitial
        parameters.
        0.2, 0.5, -0.1,
        25, 1, -2,
        10, 0.1, 0.01
        '''

        on = pm.Bernoulli('on', p=0.9, shape=data.shape[1:])

        amp = pm.Normal('amp', mu=guesses['amp'], sd=0.2)
        ampdx = pm.Normal('ampdx', mu=guesses['ampdx'], sd=0.3)
        ampdy = pm.Normal('ampdy', mu=guesses['ampdy'], sd=0.3)

        center = pm.Normal('center', mu=guesses['center'], sd=5)
        centerdx = pm.Normal('centerdx', mu=guesses['centerdx'], sd=1)
        centerdy = pm.Normal('centerdy', mu=guesses['centerdy'], sd=1)

        # Create a bounded normal
        PositiveNormal = pm.Bound(pm.Normal, lower=0)
        sigma = PositiveNormal('sigma', mu=guesses['sigma'], sd=5)
        basic_model.sigma_lowerbound__ = pm.model.ObservedRV(tt.Variable(0))
        sigmadx = pm.Normal('sigmadx', mu=guesses['sigmadx'], sd=1)
        sigmady = pm.Normal('sigmady', mu=guesses['sigmady'], sd=1)

        model = on * minicube_model(xax,
                                    amp, ampdx, ampdy,
                                    center, centerdx, centerdy,
                                    sigma, sigmadx, sigmady,
                                    npix=data.shape[1],
                                    force_positive=False,
                                    check_isfinite=False)

        sigma_n = pm.InverseGamma('sigma_n', alpha=1, beta=1)
        Y_obs = pm.Normal('Y_obs', mu=model, sd=sigma_n, observed=data)

        trace = pm.sample(**fit_kwargs)

    medians = parameter_medians(trace)
    stddevs = parameter_stddevs(trace)

    return medians, stddevs, trace, basic_model


def parameter_medians(trace):
    '''
    Return the median for each continuous parameter.
    '''

    medians = {}

    for var in trace.varnames:
        medians[var] = np.median(trace.get_values(var), axis=0)

    return medians


def parameter_stddevs(trace):

    stddev = {}

    percs = [0.15865, 0.84135]

    for var in trace.varnames:
        bounds = np.percentile(trace.get_values(var), percs, axis=0)
        stddev[var] = (bounds[1] - bounds[0]) / 2.

    return stddev