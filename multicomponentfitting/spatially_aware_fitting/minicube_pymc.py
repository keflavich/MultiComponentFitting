
import pymc3 as pm
import theano as tt
import numpy as np
from .minicube_fit import minicube_model


def minicube_pymc_fit(xax, data, guesses, ncomp=1, **fit_kwargs):
    '''
    pymc fitting of a set of single Gaussians.
    '''

    basic_model = pm.Model()

    with basic_model:

        params_dict = {}

        for i in range(ncomp):
            model_i, params_dict_i = \
                spatial_gaussian_model(xax, data, guesses, comp_num=i)
            if i == 0:
                model = model_i
            else:
                model += model_i

            params_dict.update(params_dict_i)

        sigma_n = pm.InverseGamma('sigma_n', alpha=1, beta=1)
        Y_obs = pm.Normal('Y_obs', mu=model, sd=sigma_n, observed=data)

        trace = pm.sample(**fit_kwargs)

    medians = parameter_medians(trace)
    stddevs = parameter_stddevs(trace)

    return medians, stddevs, trace, basic_model


def spatial_gaussian_model(spectral_axis, data, guesses,
                           prior_type='uniform', comp_num=0):
    '''
    Create a model ofr a single Gaussian the varies spatially.
    '''

    if prior_type not in ['uniform', 'normal']:
        raise ValueError("prior_type must be 'uniform' or 'normal'.")

    param_dict = {}

    if comp_num is None:
        comp_num = ""
    add_num = lambda name: "{0}{1}".format(name, comp_num)

    param_dict[add_num("on")] = \
        pm.Bernoulli(add_num('on'), p=0.9, shape=data.shape[1:])

    if prior_type == "normal":
        param_dict[add_num("amp")] = \
            pm.Normal(add_num('amp'), mu=guesses['amp'], sd=0.2)
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
        medians[var] = np.median(trace.get_values(var), axis=0)

    return medians


def parameter_stddevs(trace):

    stddev = {}

    percs = [0.15865, 0.84135]

    for var in trace.varnames:
        bounds = np.percentile(trace.get_values(var), percs, axis=0)
        stddev[var] = (bounds[1] - bounds[0]) / 2.

    return stddev
