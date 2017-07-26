import numpy as np
import lmfit

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

def minicube_model_generator(npix=3, func=gaussian,):

    def minicube_modelfunc(xax, amp, ampdx, ampdy, center, centerdx, centerdy,
                           sigma, sigmadx, sigmady,):
        return minicube_model(xax, amp, ampdx, ampdy, center, centerdx,
                              centerdy, sigma, sigmadx, sigmady, npix=npix,
                              func=func)

    return minicube_modelfunc


def unconstrained_fitter(minicube, xax, input_parameters, **model_kwargs):
    """
    input_parameters should be a dict
    """

    model = lmfit.Model(minicube_model_generator(**model_kwargs),
                        independent_vars=['xax'])

    params = model.make_params()

    for par in params:
        params[par].value = input_parameters[par]

    result = model.fit(minicube, xax=xax,
                       params=params)

    return result


def fit_plotter(result, data, xaxis, npix=3, fignum=1, clear=True,
                modelcube=None,
                figsize=(12,12)):
    # npix unfortunately has to be hand-specified...

    import pylab as pl

    params = result.params

    if clear:
        pl.figure(fignum, figsize=figsize).clf()
    fig, axes = pl.subplots(npix, npix, sharex=True, sharey=True, num=fignum,
                            figsize=figsize)

    fitcube = minicube_model(xaxis,
                             *[x.value for x in params.values()],
                             npix=npix)

    for ii,((yy,xx), ax) in enumerate(zip(np.ndindex((npix,npix)), axes.ravel())):
        ax.plot(data[:,yy,xx], 'k-', alpha=0.85, zorder=-5, linewidth=1,
                drawstyle='steps-mid')
        ax.plot(fitcube[:,yy,xx], 'b--', alpha=0.85, zorder=-5, linewidth=1,
                drawstyle='steps-mid')
        ax.plot((data-fitcube)[:,yy,xx], 'r--', alpha=0.85, zorder=-10, linewidth=1,
                drawstyle='steps-mid')
        if modelcube is not None:
            ax.plot(modelcube[:,yy,xx], 'k-', alpha=0.25, zorder=-10, linewidth=3,
                    drawstyle='steps-mid')

    pl.tight_layout()
