import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np

def BayesianMultiComponentFit(Xdata, Ydata, y_error=None, max_ncomp=3):
    """
    Multi-component fitting of lines.
    """

    model = pm.Model()
    if max_ncomp == 1:
        with model:
            amp1 = pm.Uniform('amp1', lower=0, upper=1.2 * Ydata.max())
            vcen1 = pm.Uniform('vcen1',lower=Xdata.min(),
                               upper=Xdata.max())
            sigv1 = pm.Uniform('sigv1', lower=1e-5, upper=np.ptp(Xdata) / 2.)
            on1 = pm.Bernoulli('on1', p=0.5)
            mu = ((on1 * amp1
                   * np.exp(-(Xdata - vcen1)**2 / (2*sigv1**2))))
            sigma = pm.InverseGamma('sigma', alpha=1, beta=Ydata.std(), observed=y_error)
            Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Ydata)

    if max_ncomp == 2:
        with model:
            amp1 = pm.Uniform('amp1', lower=0, upper=1.2 * Ydata.max())
            vcen1 = pm.Uniform('vcen1',lower=Xdata.min(),
                               upper=Xdata.max())
            sigv1 = pm.Uniform('sigv1', lower=1e-5, upper=np.ptp(Xdata) / 2.)
            on1 = pm.Bernoulli('on1', p=0.5)

            amp2 = pm.Uniform('amp2', lower=0, upper=1.2 * Ydata.max())
            vcen2 = pm.Uniform('vcen2',lower=Xdata.min(),
                               upper=Xdata.max())
            sigv2 = pm.Uniform('sigv2', lower=1e-5, upper=np.ptp(Xdata) / 2.)
            on2 = pm.Bernoulli('on2', p=0.5)

            mu = ((on1 * amp1
                   * np.exp(-(Xdata - vcen1)**2 / (2*sigv1**2)))
                  + (on2 * amp2
                     * np.exp(-(Xdata - vcen2)**2 / (2*sigv2**2))))
            sigma = pm.InverseGamma('sigma', alpha=1, beta=Ydata.std(), observed=y_error)
            Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Ydata)


    if max_ncomp == 3:
        with model:
            amp1 = pm.Uniform('amp1', lower=0, upper=1.2 * Ydata.max())
            vcen1 = pm.Uniform('vcen1',lower=Xdata.min(),
                               upper=Xdata.max())
            sigv1 = pm.Uniform('sigv1', lower=1e-5, upper=np.ptp(Xdata) / 2.)
            on1 = pm.Bernoulli('on1', p=0.5)

            amp2 = pm.Uniform('amp2', lower=0, upper=1.2 * Ydata.max())
            vcen2 = pm.Uniform('vcen2',lower=Xdata.min(),
                               upper=Xdata.max())
            sigv2 = pm.Uniform('sigv2', lower=1e-5, upper=np.ptp(Xdata) / 2.)
            on2 = pm.Bernoulli('on2', p=0.5)

            amp3 = pm.Uniform('amp3', lower=0, upper=1.2 * Ydata.max())
            vcen3 = pm.Uniform('vcen3',lower=Xdata.min(),
                               upper=Xdata.max())
            sigv3 = pm.Uniform('sigv3', lower=1e-5, upper=np.ptp(Xdata) / 2.)
            on3 = pm.Bernoulli('on3', p=0.5)

            mu = ((on1 * amp1
                   * np.exp(-(Xdata - vcen1)**2 / (2*sigv1**2)))
                  + (on2 * amp2
                     * np.exp(-(Xdata - vcen2)**2 / (2*sigv2**2)))
                  + (on3 * amp3
                     * np.exp(-(Xdata - vcen3)**2 / (2*sigv3**2))))
            sigma = pm.InverseGamma('sigma', alpha=1, beta=Ydata.std(), observed=y_error)
            Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Ydata)

    if max_ncomp > 3:
        with model:
            mustr = 'mu = ('
            for i in np.arange(max_ncomp):
                ii = i+1
                exec("amp{0} = pm.Uniform('amp{0}', lower=0, upper=1.2 * Ydata.max())".format(ii) )
                exec("vcen{0} = pm.Uniform('vcen{0}', lower=Xdata.min(), upper=Xdata.max())".format(ii))
                exec("sigv{0} = pm.Uniform('sigv{0}', lower=1e-5, upper=np.ptp(Xdata) / 2.)".format(ii))
                exec("on{0} = pm.Bernoulli('on{0}', p=0.5)".format(ii))
                if i > 0:
                    mustr += ' + '
                mustr += "(on{0} * amp{0} * np.exp(-(Xdata - vcen{0})**2 / (2*sigv{0}**2)))".format(ii)
            exec(mustr+')')
            sigma = pm.InverseGamma('sigma', alpha=1, beta=Ydata.std())
            Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Ydata)

    with model:
        trace = pm.sample()

    return(trace, model)

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
            mu += (trace['on{0}'.format(i)][draws]
                   * trace['amp{0}'.format(i)][draws]
                   * np.exp(-(Xdata - trace['vcen{0}'.format(i)][draws])**2
                            / (2*trace['sigv{0}'.format(i)][draws]**2)))
            i += 1
        return mu
