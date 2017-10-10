
import pymc3 as pm
import theano as tt
import numpy as np
import scipy.optimize as opt
from astropy.convolution.kernels import Gaussian2DKernel
from astropy.convolution import convolve
from .minicube_fit import minicube_model


def minicube_pymc_fit(xax, data, guesses, ncomps=1, sample=True,
                      fmin=opt.fmin_bfgs, fmin_kwargs={}, **sampler_kwargs):
    '''
    pymc fitting of a set of single Gaussians.
    '''

    basic_model = pm.Model()

    with basic_model:

        params_dict = {}

        for i in range(ncomps):
            model_i, params_dict_i = \
                spatial_gaussian_model(xax, data, guesses, comp_num=i)
            if i == 0:
                model = model_i
            else:
                model += model_i

            params_dict.update(params_dict_i)

        sigma_n = pm.InverseGamma('sigma_n', alpha=1, beta=1)
        Y_obs = pm.Normal('Y_obs', mu=model, sd=sigma_n, observed=data)

        start = pm.find_MAP(fmin=fmin, **fmin_kwargs)
        # Use the initial guesses for the Bernoulli parameters
        for i in range(ncomps):
            start['on{}'.format(i)] = guesses['on{}'.format(i)]

        if sample:
            # An attempt to use variational inference b/c it would be
            # way faster. This fails terribly in every case I've tried
            # trace = pm.fit(500, start=start, method='svgd',
            #                inf_kwargs=dict(n_particles=100,
            #                                temperature=1e-4),
            #                ).sample(500)
            trace = pm.sample(start=start, **sampler_kwargs)

    if sample:
        medians = parameter_medians(trace)
        stddevs = parameter_stddevs(trace)

        return medians, stddevs, trace, basic_model
    else:
        return start, basic_model


def spatial_gaussian_model(spectral_axis, data, guesses,
                           prior_type='uniform', comp_num=0):
    '''
    Create a model for a single Gaussian that varies spatially.
    '''

    if prior_type not in ['uniform', 'normal']:
        raise ValueError("prior_type must be 'uniform' or 'normal'.")

    param_dict = {}

    if comp_num is None:
        comp_num = ""
    add_num = lambda name: "{0}{1}".format(name, comp_num)

    # guesses['p'] must match the shape of the data in the spatial dims
    assert guesses[add_num('p')].shape == data.shape[1:]

    param_dict[add_num("on")] = \
        pm.Bernoulli(add_num('on'), p=guesses[add_num('p')],
                     shape=data.shape[1:],
                     testval=guesses[add_num('on')])

    if prior_type == "normal":
        param_dict[add_num("amp")] = \
            pm.Normal(add_num('amp'), mu=guesses[add_num('amp')], sd=0.2)
        param_dict[add_num("ampdx")] = \
            pm.Normal(add_num('ampdx'), mu=guesses[add_num('ampdx')], sd=0.3)
        param_dict[add_num("ampdy")] = \
            pm.Normal(add_num('ampdy'), mu=guesses[add_num('ampdy')], sd=0.3)

        param_dict[add_num("center")] = \
            pm.Normal(add_num('center'), mu=guesses[add_num('center')], sd=5)
        param_dict[add_num("centerdx")] = \
            pm.Normal(add_num('centerdx'), mu=guesses[add_num('centerdx')], sd=1)
        param_dict[add_num("centerdy")] = \
            pm.Normal(add_num('centerdy'), mu=guesses[add_num('centerdy')], sd=1)

        # Create a bounded normal
        PositiveNormal = pm.Bound(pm.Normal, lower=0)
        param_dict[add_num("sigma")] = \
            PositiveNormal(add_num('sigma'), mu=guesses[add_num('sigma')], sd=5)

        param_dict[add_num("sigmadx")] = \
            pm.Normal(add_num('sigmadx'), mu=guesses[add_num('sigmadx')], sd=1)
        param_dict[add_num("sigmady")] = \
            pm.Normal(add_num('sigmady'), mu=guesses[add_num('sigmady')], sd=1)
    else:
        param_dict[add_num("amp")] = \
            pm.Uniform(add_num('amp'), lower=0, upper=1.2 * data.max())
        param_dict[add_num("ampdx")] = \
            pm.Normal(add_num('ampdx'), mu=guesses[add_num('ampdx')], sd=0.3)
        param_dict[add_num("ampdy")] = \
            pm.Normal(add_num('ampdy'), mu=guesses[add_num('ampdy')], sd=0.3)

        param_dict[add_num("center")] = \
            pm.Uniform(add_num('center'), lower=spectral_axis.min(),
                       upper=spectral_axis.max())
        param_dict[add_num("centerdx")] = \
            pm.Normal(add_num('centerdx'), mu=guesses[add_num('centerdx')], sd=1)
        param_dict[add_num("centerdy")] = \
            pm.Normal(add_num('centerdy'), mu=guesses[add_num('centerdy')], sd=1)

        param_dict[add_num("sigma")] = \
            pm.Uniform(add_num('sigma'), lower=0, upper=np.ptp(spectral_axis) / 2.)

        param_dict[add_num("sigmadx")] = \
            pm.Normal(add_num('sigmadx'), mu=guesses[add_num('sigmadx')], sd=1)
        param_dict[add_num("sigmady")] = \
            pm.Normal(add_num('sigmady'), mu=guesses[add_num('sigmady')], sd=1)

    model = param_dict[add_num("on")] * \
        minicube_model(spectral_axis,
                       param_dict[add_num("amp")],
                       param_dict[add_num("ampdx")],
                       param_dict[add_num("ampdy")],
                       param_dict[add_num("center")],
                       param_dict[add_num("centerdx")],
                       param_dict[add_num("centerdy")],
                       param_dict[add_num("sigma")],
                       param_dict[add_num("sigmadx")],
                       param_dict[add_num('sigmady')],
                       npix=data.shape[1],
                       force_positive=False,
                       check_isfinite=False)

    return model, param_dict


def parameter_medians(trace):
    '''
    Return the median for each continuous parameter.
    '''

    medians = {}

    for var in trace.varnames:
        if "__" in var:
            continue
        medians[var] = np.median(trace.get_values(var), axis=0)

    return medians


def parameter_stddevs(trace):

    stddev = {}

    percs = [0.15865, 0.84135]

    for var in trace.varnames:
        if "__" in var:
            continue
        bounds = np.percentile(trace.get_values(var), percs, axis=0)
        stddev[var] = (bounds[1] - bounds[0]) / 2.

    return stddev


def spatial_covariance_structure(on, cov_struct=Gaussian2DKernel,
                                 **cov_kwargs):
    '''
    Impose a covariance structure on the Bernoulli samples.

    For every on sample, the next sample the total probability
    of those that are turned on around it, weighted by the
    structure model.
    '''

    kernel = Gaussian2DKernel(**cov_kwargs)

    pvals = convolve(on, kernel)

    return pvals
