import numpy as np
import lmfit
import collections

def plane(xx, yy, value, dx, dy, xcen=1, ycen=1):
    z = ((xx-xcen) * dx + (yy-ycen) * dy) + value
    return z

def gaussian(xax, amp, cen, wid):
    return np.exp(-(xax-cen)**2/(2*wid**2)) * amp

def minicube_model(xax,
                   amp, ampdx, ampdy,
                   center, centerdx, centerdy,
                   sigma, sigmadx, sigmady,
                   npix=3,
                   func=gaussian,
                   force_positive=True,
                  ):

    for par in (amp, ampdx, ampdy, center, centerdx, centerdy,
                sigma, sigmadx, sigmady):
        assert np.isfinite(par)

    yy,xx = np.indices([npix, npix], dtype='float')

    amps = plane(xx, yy, amp, ampdx, ampdy, xcen=npix//2, ycen=npix//2)
    centers = plane(xx, yy, center, centerdx, centerdy, xcen=npix//2, ycen=npix//2)
    sigmas = plane(xx, yy, sigma, sigmadx, sigmady, xcen=npix//2, ycen=npix//2)

    model = func(xax[:,None,None], amps, centers, sigmas)

    if force_positive:
        model[model<0] = 0

    return model


def multicomp_minicube_model_generator(npix=3, func=gaussian, ncomps=1):

    ncomps_per = 9

    argnames = "amp, ampdx, ampdy, center, centerdx, centerdy, sigma, sigmadx, sigmady".split(", ")
    argdict = collections.OrderedDict([(kw+str(ii), None)
                                       for ii in range(ncomps) for kw in argnames])

    def minicube_modelfunc(xax, *args, **kwargs):

        generic_kwargs = {'npix': npix, 'func': func}

        for kw in kwargs:
            if kw in argdict:
                argdict[kw] = kwargs[kw]
            elif kw in ('npix', 'func', 'force_positive'):
                generic_kwargs[kw] = kwargs[kw]
            else:
                raise ValueError("Unrecognized parameter {0}".format(kw))

        kwarg_dict = {}
        for ii in range(ncomps):
            kwarg_dict[ii] = dict(zip(argnames,
                                      list(argdict.values())[ncomps_per*ii:ncomps_per*(ii+1)]))

        models = [minicube_model(xax,
                                 **kwarg_dict[ii],
                                 **generic_kwargs)
                  for ii in range(ncomps)]
        return np.sum(models, axis=0)


    minicube_modelfunc.argnames = ['xax']+[an+str(ii) for ii in range(ncomps) for an in argnames]
    minicube_modelfunc.kwargs = {}

    return minicube_modelfunc



def unconstrained_fitter(minicube, xax, input_parameters, **model_kwargs):
    """
    input_parameters should be a dict
    """

    model = lmfit.Model(multicomp_minicube_model_generator(**model_kwargs),
                        independent_vars=['xax'])

    params = model.make_params()

    for par in params:
        params[par].value = input_parameters[par]

    result = model.fit(minicube, xax=xax,
                       params=params)

    return result


def constrained_fitter(minicube, xax, input_parameters, **model_kwargs):
    """
    input_parameters should be a dict
    """

    model = lmfit.Model(multicomp_minicube_model_generator(**model_kwargs),
                        independent_vars=['xax'])

    params = model.make_params()

    for par in params:
        params[par].value = input_parameters[par]
        if 'amp' in par and par[3] != 'd':
            params[par].min = 0
        elif 'sigma' in par and par[5] != 'd':
            params[par].min = 0

    result = model.fit(minicube, xax=xax,
                       params=params)

    return result



def fit_plotter(result, data, xaxis, npix=3, fignum=1, clear=True,
                modelcube=None,
                modelfunc=minicube_model,
                figsize=(12,12)):
    # npix unfortunately has to be hand-specified...

    import pylab as pl

    params = result.params

    if clear:
        pl.figure(fignum, figsize=figsize).clf()
    fig, axes = pl.subplots(npix, npix, sharex=True, sharey=True, num=fignum,
                            figsize=figsize)

    fitcube = modelfunc(xaxis, *[x.value for x in params.values()])

    for ii,((yy,xx), ax) in enumerate(zip(np.ndindex((npix,npix)), axes.ravel())):
        ax.plot(xaxis, data[:,yy,xx], 'k-', alpha=0.5, zorder=-5, linewidth=1,
                drawstyle='steps-mid')
        ax.plot(xaxis, fitcube[:,yy,xx], 'b--', alpha=1, zorder=0,
                linewidth=1,
                drawstyle='steps-mid')
        ax.plot(xaxis, (data-fitcube)[:,yy,xx], 'r--', alpha=0.85, zorder=-10,
                linewidth=1,
                drawstyle='steps-mid')
        if modelcube is not None:
            ax.plot(xaxis, modelcube[:,yy,xx], 'k-', alpha=0.15, zorder=-10,
                    linewidth=3, drawstyle='steps-mid')

    pl.tight_layout()
    pl.subplots_adjust(hspace=0, wspace=0)
