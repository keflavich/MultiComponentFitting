import pymc3 as pm
import theano.tensor as TT
import matplotlib.pyplot as plt
import numpy as np


def BayesMCFit(Xdata, Ydata, y_error=None, max_ncomp=3, **fit_kwargs):

    ncomp = max_ncomp
    model = pm.Model()
    param_dict = {}

    with model:
        on = pm.Bernoulli("on", p=0.5, shape=(ncomp, ))
        for i in np.arange(ncomp):
            ii = i + 1
            param_dict['amp{}'.format(ii)] = \
                pm.Uniform('amp{}'.format(ii), lower=0,
                           upper=1.2 * Ydata.max())
            param_dict['vcen{}'.format(ii)] = \
                pm.Uniform('vcen{}'.format(ii), lower=Xdata.min(),
                           upper=Xdata.max())
            param_dict['sigv{}'.format(ii)] = \
                pm.Uniform('sigv{}'.format(ii), lower=1e-5,
                           upper=np.ptp(Xdata) / 2.)
            if i == 0:
                mu = theano_gaussian(Xdata,
                                     on[i],
                                     param_dict['amp{}'.format(ii)],
                                     param_dict['vcen{}'.format(ii)],
                                     param_dict['sigv{}'.format(ii)])
            else:
                mu += theano_gaussian(Xdata,
                                      on[i],
                                      param_dict['amp{}'.format(ii)],
                                      param_dict['vcen{}'.format(ii)],
                                      param_dict['sigv{}'.format(ii)])

        sigma = pm.InverseGamma('sigma', alpha=1, beta=y_error,
                                observed=y_error)
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma,
                          observed=Ydata)

        start = pm.find_MAP()
        trace = pm.sample(start=start, **fit_kwargs)
    return trace, model

def theano_gaussian(x, on, amp, vcen, sigv):
    return on * amp * TT.exp(- 0.5 * TT.power((x - vcen) / sigv, 2))


def MultiComponentFit(Xdata, Ydata, y_error=None, max_ncomp=3, min_p=0.5,
                      **fit_kwargs):
    '''
    Try choosing the number of components as its own variable.
    '''

    ncomp = max_ncomp

    while ncomp > 0:
        model = pm.Model()

        param_dict = {}

        with model:

            on = pm.Bernoulli("on", p=0.5, shape=(ncomp, ))

            for i in np.arange(ncomp):
                ii = i + 1
                param_dict['amp{}'.format(ii)] = \
                    pm.Uniform('amp{}'.format(ii), lower=0,
                               upper=1.2 * Ydata.max())
                param_dict['vcen{}'.format(ii)] = \
                    pm.Uniform('vcen{}'.format(ii), lower=Xdata.min(),
                               upper=Xdata.max())
                param_dict['sigv{}'.format(ii)] = \
                    pm.Uniform('sigv{}'.format(ii), lower=1e-5,
                               upper=np.ptp(Xdata) / 2.)
                if i == 0:
                    mu = theano_gaussian(Xdata,
                                         on[i],
                                         param_dict['amp{}'.format(ii)],
                                         param_dict['vcen{}'.format(ii)],
                                         param_dict['sigv{}'.format(ii)])
                else:
                    mu += theano_gaussian(Xdata,
                                          on[i],
                                          param_dict['amp{}'.format(ii)],
                                          param_dict['vcen{}'.format(ii)],
                                          param_dict['sigv{}'.format(ii)])

            sigma = pm.InverseGamma('sigma', alpha=1, beta=y_error,
                                    observed=y_error)
            Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma,
                              observed=Ydata)

            start = pm.find_MAP()
            trace = pm.sample(start=start, **fit_kwargs)

        # Get the samples for the on parameters
        on_trace = trace.get_values('on')
        on_frac = on_trace.sum(0) / float(on_trace.shape[0])

        if any(on_frac < min_p):
            ncomp -= 1
            continue
        else:
            # Now we're going to sample more with the model we're confident in
            # Start from where the chain left off and don't bother tuning
            # again.
            if fit_kwargs.get('tune') is not None:
                fit_kwargs.pop('tune')
            with model:
                trace = pm.sample(trace=trace, tune=0, **fit_kwargs)

            break

    return trace, model


def RealizedFit(trace, XdataIn, model, n_draws=10, with_noise=False):
    Xdata = np.expand_dims(XdataIn, axis=1)
    if with_noise:
        dd = pm.sampling.sample_ppc(trace, samples=n_draws, model=model)
    else:
        mu = np.zeros((model.Y_obs.shape.eval()[0], n_draws))
        i = 1
        while 'amp{0}'.format(i) in model.named_vars:
            nSamp = len(trace['amp{0}'.format(i)])
            draws = np.floor(np.random.rand(n_draws) * nSamp).astype(np.int)
            mu += (trace['on'][draws, i - 1]
                   * trace['amp{0}'.format(i)][draws]
                   * np.exp(-(Xdata - trace['vcen{0}'.format(i)][draws])**2
                            / (2*trace['sigv{0}'.format(i)][draws]**2)))
            i += 1
        return mu
